"""Tests for CLI config module."""
from __future__ import annotations

from plugmem.cli.config import PlugmemConfig, config_to_env, load_config, save_config


def test_default_config_values():
    cfg = PlugmemConfig()
    assert cfg.service.host == "127.0.0.1"
    assert cfg.service.port == 8080
    assert cfg.service.log_level == "INFO"
    assert cfg.llm.model == ""
    assert cfg.embedding.model == "nomic-embed-text"


def test_config_to_env_skips_empty():
    cfg = PlugmemConfig()
    env = config_to_env(cfg)
    # Defaults with non-zero values (model, port limits, etc.) are emitted.
    assert "PLUGMEM_API_KEY" not in env
    assert "LLM_BASE_URL" not in env
    assert "LLM_API_KEY" not in env


def test_config_to_env_populated():
    cfg = PlugmemConfig(
        service={"api_key": "test-key"},
        llm={"base_url": "http://localhost:8000/v1"},
        embedding={"model": "test-model"},
    )
    env = config_to_env(cfg)
    assert env["PLUGMEM_API_KEY"] == "test-key"
    assert env["LLM_BASE_URL"] == "http://localhost:8000/v1"
    assert env["EMBEDDING_MODEL"] == "test-model"
    assert "LLM_API_KEY" not in env


def test_save_and_load_config(tmp_path):
    cfg = PlugmemConfig(
        service={"host": "0.0.0.0", "port": 9090, "api_key": "saved-key"},
        llm={"model": "gpt-4"},
    )
    p = tmp_path / "config.toml"
    save_config(cfg, p)
    assert p.exists()

    loaded = load_config(p)
    assert loaded.service.host == "0.0.0.0"
    assert loaded.service.port == 9090
    assert loaded.service.api_key == "saved-key"
    assert loaded.llm.model == "gpt-4"


def test_storage_backend_default():
    cfg = PlugmemConfig()
    assert cfg.service.storage_backend == "chroma"
    assert cfg.service.sqlite_vec_path == ""
    env = config_to_env(cfg)
    assert "STORAGE_BACKEND" not in env
    assert "SQLITE_VEC_PATH" not in env


def test_storage_backend_sqlite_vec_config_to_env():
    cfg = PlugmemConfig(
        service={"storage_backend": "sqlite_vec", "sqlite_vec_path": "/tmp/test.db"},
    )
    env = config_to_env(cfg)
    assert env["STORAGE_BACKEND"] == "sqlite_vec"
    assert env["SQLITE_VEC_PATH"] == "/tmp/test.db"


def test_storage_backend_round_trip(tmp_path):
    cfg = PlugmemConfig(
        service={"storage_backend": "sqlite_vec", "sqlite_vec_path": "/tmp/pm.db"},
    )
    p = tmp_path / "config.toml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.service.storage_backend == "sqlite_vec"
    assert loaded.service.sqlite_vec_path == "/tmp/pm.db"
