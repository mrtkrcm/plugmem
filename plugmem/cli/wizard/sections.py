"""Wizard sections: LLM, embedding, service."""
from __future__ import annotations

import secrets
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugmem.cli.config import PlugmemConfig, default_data_dir, EmbeddingConfig, LLMConfig
from plugmem.cli.wizard.probes import (
    OllamaInfo,
    detect_ollama,
    probe_embedding,
    probe_llm,
)
from plugmem.cli.wizard.ui import (
    error,
    header,
    info,
    prompt_action,
    prompt_choice,
    prompt_text,
    success,
    warn,
)

PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "needs_key": True,
        "default_llm_model": "gpt-4o-mini",
        "default_embed_model": "text-embedding-3-small",
    },
    "azure": {
        "base_url": "",
        "needs_key": True,
        "default_llm_model": "",
        "default_embed_model": "",
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "needs_key": False,
        "default_llm_model": "",
        "default_embed_model": "",
    },
    "other": {
        "base_url": "",
        "needs_key": False,
        "default_llm_model": "",
        "default_embed_model": "",
    },
}

PROVIDER_CHOICES = ["ollama", "openai", "azure", "vllm", "other"]


def _probe_loop(
    cfg_target,  # cfg.llm or cfg.embedding
    provider: str,
    detected: Optional[OllamaInfo],
    ask_fields,
    probe_func,
    *,
    kind_label: str,
) -> bool:
    while True:
        base_url, api_key, model = ask_fields(provider, detected, cfg_target)

        info("Validating {} endpoint\u2026".format(kind_label))
        ok, msg = probe_func(base_url, api_key, model)
        if ok:
            success("{} probe succeeded ({} @ {}).".format(kind_label.title(), model, base_url))
            cfg_target.base_url = base_url
            cfg_target.api_key = api_key
            cfg_target.model = model
            return True

        error("Validation failed: {}".format(msg))
        action = prompt_action()
        if action == "skip":
            warn("Skipping {} configuration.".format(kind_label))
            return False
        if action == "edit":
            cfg_target.base_url = base_url
            cfg_target.api_key = api_key
            cfg_target.model = model
            continue
        info("Retrying with the same values\u2026")
        ok, msg = probe_func(base_url, api_key, model)
        if ok:
            success("{} probe succeeded on retry.".format(kind_label.title()))
            cfg_target.base_url = base_url
            cfg_target.api_key = api_key
            cfg_target.model = model
            return True
        error("Still failing: {}".format(msg))


def run_llm_section(cfg: PlugmemConfig, *, ollama: Optional[OllamaInfo] = None) -> bool:
    header("LLM endpoint")
    detected = ollama if ollama is not None else detect_ollama()
    provider = _ask_provider(detected, kind="LLM")
    return _probe_loop(
        cfg.llm, provider, detected, _ask_llm_fields, probe_llm,
        kind_label="llm",
    )


def run_embedding_section(
    cfg: PlugmemConfig, *, ollama: Optional[OllamaInfo] = None
) -> bool:
    header("Embedding endpoint")
    detected = ollama if ollama is not None else detect_ollama()
    provider = _ask_provider(detected, kind="embedding")
    return _probe_loop(
        cfg.embedding, provider, detected, _ask_embedding_fields, probe_embedding,
        kind_label="embedding",
    )


def run_service_section(cfg: PlugmemConfig) -> bool:
    header("Service settings")

    cfg.service.host = prompt_text("host", default=cfg.service.host or "127.0.0.1")

    suggested_port = _next_free_port(cfg.service.port or 8080, host=cfg.service.host)
    if suggested_port != (cfg.service.port or 8080):
        warn("Port {} is in use; suggesting {}.".format(cfg.service.port or 8080, suggested_port))
    port_str = prompt_text("port", default=str(suggested_port))
    cfg.service.port = int(port_str)

    # Storage backend selection
    backend_choice = prompt_choice(
        "storage backend",
        choices=["chroma", "sqlite_vec"],
        default=cfg.service.storage_backend or "chroma",
    )
    cfg.service.storage_backend = backend_choice

    if backend_choice == "sqlite_vec":
        sqlite_default = cfg.service.sqlite_vec_path or str(default_data_dir().parent / "plugmem.db")
        sqlite_path = prompt_text("sqlite_vec path", default=sqlite_default)
        sqlite_dir = Path(sqlite_path).expanduser().parent
        if not _ensure_writable_dir(sqlite_dir):
            warn("Could not create or write to {}; using anyway, may fail at start.".format(sqlite_dir))
        cfg.service.sqlite_vec_path = sqlite_path
    else:
        data_dir_default = cfg.service.data_dir or str(default_data_dir())
        data_dir = prompt_text("chroma data_dir", default=data_dir_default)
        if not _ensure_writable_dir(Path(data_dir).expanduser()):
            warn("Could not create or write to {}; using anyway, may fail at start.".format(data_dir))
        cfg.service.data_dir = data_dir

    if not cfg.service.api_key:
        cfg.service.api_key = secrets.token_hex(16)
        success("Generated service api_key: {}".format(cfg.service.api_key))
        info("(Store this somewhere safe -- clients use it as `X-API-Key`.)")
    else:
        info("Reusing existing service api_key: {}...".format(cfg.service.api_key[:8]))

    cfg.service.log_level = prompt_choice(
        "log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=cfg.service.log_level or "INFO",
    )
    return True


def _ask_provider(detected: Optional[OllamaInfo], *, kind: str) -> str:
    if detected and detected.models:
        info("Detected Ollama at {} with {} model(s).".format(detected.base_url, len(detected.models)))
    elif detected:
        info("Detected Ollama at {} but no models pulled.".format(detected.base_url))
    else:
        info("No Ollama detected on 127.0.0.1:11434.")

    default = "ollama" if detected and detected.models else "openai"
    return prompt_choice("{} provider".format(kind.title()), PROVIDER_CHOICES, default=default)


def _ask_llm_fields(provider: str, detected: Optional[OllamaInfo], current) -> tuple[str, str, str]:
    return _ask_endpoint_fields(provider, detected, current, kind="llm")


def _ask_embedding_fields(provider: str, detected: Optional[OllamaInfo], current) -> tuple[str, str, str]:
    return _ask_endpoint_fields(provider, detected, current, kind="embedding")


def _ask_endpoint_fields(provider: str, detected: Optional[OllamaInfo], current, *, kind: str) -> tuple[str, str, str]:
    if provider == "ollama":
        if not detected:
            warn("Ollama wasn't detected; falling back to manual entry.")
            return _manual_endpoint_fields(provider, current, kind=kind)
        base_url = detected.base_url
        api_key = ""
        model = _ask_model_from_list(detected.models, current_model=getattr(current, "model", ""))
        return base_url, api_key, model

    return _manual_endpoint_fields(provider, current, kind=kind)


def _manual_endpoint_fields(provider: str, current: LLMConfig | EmbeddingConfig, *, kind: str) -> tuple[str, str, str]:
    preset: dict = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["other"])
    default_url: str | None = current.base_url or preset.get("base_url") or None
    base_url = prompt_text("base_url", default=default_url)

    api_key_default = current.api_key or None
    if preset["needs_key"]:
        api_key = prompt_text("api_key", default=api_key_default, password=True)
    else:
        api_key = prompt_text("api_key (leave blank if not required)", default=api_key_default, password=True, allow_empty=True)

    model_default: str | None = current.model or (preset.get("default_llm_model") if kind == "llm" else preset.get("default_embed_model")) or None
    model = prompt_text("model", default=model_default)
    return base_url, api_key, model


def _ask_model_from_list(models: List[str], *, current_model: str) -> str:
    if not models:
        return prompt_text("model")
    info("Pulled Ollama models:")
    for i, name in enumerate(models, start=1):
        info("  {}. {}".format(i, name))
    default = current_model if current_model in models else models[0]
    return prompt_choice("Model", choices=models, default=default)


def _next_free_port(start: int, *, host: str = "127.0.0.1", attempts: int = 20) -> int:
    for offset in range(attempts):
        port = start + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return port
        except OSError:
            continue
    return start


def _ensure_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".plugmem_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False
