"""TOML config schema + IO for the plugmem CLI."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    import tomli as toml_reader

import tomli_w


def _xdg(env_var: str, fallback: str) -> Path:
    raw = os.environ.get(env_var)
    base = Path(raw) if raw else Path.home() / fallback
    return base / "plugmem"


def default_config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config")


def default_config_path() -> Path:
    return default_config_dir() / "config.toml"


def default_data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "chroma"


def default_state_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state")


def default_pid_file() -> Path:
    return default_state_dir() / "plugmem.pid"


def default_log_file() -> Path:
    return default_state_dir() / "plugmem.log"


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    data_dir: str = ""
    log_level: str = "INFO"
    token_usage_file: str = ""

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be 1-65535, got {v}")
        return v


class LLMConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_retries: int = 5
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = "nomic-embed-text"
    max_text_len: int = 8192


class CodingConfig(BaseModel):
    """Coding-agent defaults used by ``plugmem coding`` commands."""
    default_graph: str = ""
    default_repo: str = ""
    default_branch: str = ""
    default_language: str = ""
    default_package_manager: str = ""
    default_tool_name: str = ""
    default_tool_version: str = ""
    default_os: str = ""
    default_component: str = ""
    source_filter: str = "correction,failure_delta"
    min_confidence: float = 0.3


class PlugmemConfig(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    coding: CodingConfig = Field(default_factory=CodingConfig)

    def with_defaults_applied(self) -> PlugmemConfig:
        c = self.model_copy(deep=True)
        if not c.service.data_dir:
            c.service.data_dir = str(default_data_dir())
        return c


def load_config(path: Optional[Path] = None) -> PlugmemConfig:
    p = path or default_config_path()
    if not p.exists():
        return PlugmemConfig()
    with open(p, "rb") as f:
        raw = toml_reader.load(f)
    return PlugmemConfig(**raw)


def save_config(cfg: PlugmemConfig, path: Optional[Path] = None) -> Path:
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump()
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        tomli_w.dump(payload, f)
    tmp.replace(p)
    return p


_CONFIG_TO_ENV: Dict[str, str] = {
    "service.api_key": "PLUGMEM_API_KEY",
    "service.data_dir": "CHROMA_PATH",
    "service.token_usage_file": "TOKEN_USAGE_FILE",
    "llm.base_url": "LLM_BASE_URL",
    "llm.api_key": "LLM_API_KEY",
    "llm.model": "LLM_MODEL",
    "llm.max_retries": "LLM_MAX_RETRIES",
    "llm.temperature": "LLM_TEMPERATURE",
    "llm.top_p": "LLM_TOP_P",
    "llm.max_tokens": "LLM_MAX_TOKENS",
    "embedding.base_url": "EMBEDDING_BASE_URL",
    "embedding.api_key": "EMBEDDING_API_KEY",
    "embedding.model": "EMBEDDING_MODEL",
    "embedding.max_text_len": "EMBEDDING_MAX_TEXT_LEN",
    "coding.default_graph": "CODING_DEFAULT_GRAPH",
    "coding.default_repo": "CODING_DEFAULT_REPO",
    "coding.default_branch": "CODING_DEFAULT_BRANCH",
    "coding.default_language": "CODING_DEFAULT_LANGUAGE",
    "coding.default_package_manager": "CODING_DEFAULT_PACKAGE_MANAGER",
    "coding.default_tool_name": "CODING_DEFAULT_TOOL_NAME",
    "coding.default_tool_version": "CODING_DEFAULT_TOOL_VERSION",
    "coding.default_os": "CODING_DEFAULT_OS",
    "coding.default_component": "CODING_DEFAULT_COMPONENT",
    "coding.source_filter": "CODING_SOURCE_FILTER",
    "coding.min_confidence": "CODING_MIN_CONFIDENCE",
}

# Build default values from the model constructors so we can skip them.
_DEFAULT_PLUGMEM = PlugmemConfig()
_DEFAULT_PAYLOAD = _DEFAULT_PLUGMEM.with_defaults_applied().model_dump()


def config_to_env(cfg: PlugmemConfig) -> Dict[str, str]:
    cfg = cfg.with_defaults_applied()
    out: Dict[str, str] = {}
    payload = cfg.model_dump()
    for dotted, env_name in _CONFIG_TO_ENV.items():
        section, field = dotted.split(".")
        v = payload[section][field]
        default_v = _DEFAULT_PAYLOAD[section][field]
        if v == default_v or v in (None, "", 0, 0.0):
            continue
        out[env_name] = str(v)
    return out


def apply_env_overrides(cfg: PlugmemConfig, env: Optional[Dict[str, str]] = None) -> PlugmemConfig:
    src = env if env is not None else os.environ
    payload = cfg.model_dump()
    for dotted, env_name in _CONFIG_TO_ENV.items():
        if env_name in src and src[env_name] != "":
            section, field = dotted.split(".")
            current = payload[section][field]
            raw = src[env_name]
            try:
                if isinstance(current, bool):
                    cast: Any = raw.lower() in ("1", "true", "yes")
                elif isinstance(current, int):
                    cast = int(raw)
                elif isinstance(current, float):
                    cast = float(raw)
                else:
                    cast = raw
            except ValueError:
                cast = raw
            payload[section][field] = cast
    return PlugmemConfig(**payload)
