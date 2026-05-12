"""Tests for daemon management."""
from __future__ import annotations

from pathlib import Path

from plugmem.cli.config import PlugmemConfig
from plugmem.cli.daemon import _build_uvicorn_cmd, _clear_pid_file, _read_pid, _is_running, daemon_status


def test_build_uvicorn_cmd():
    cfg = PlugmemConfig(service={"host": "0.0.0.0", "port": 8080, "log_level": "INFO"})
    cmd = _build_uvicorn_cmd(cfg)
    assert "--host" in cmd
    assert "0.0.0.0" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--log-level" in cmd
    assert "info" in cmd


def test_pid_file_round_trip(tmp_path, monkeypatch):
    from plugmem.cli.config import default_pid_file

    monkeypatch.setattr("plugmem.cli.config.default_pid_file", lambda: tmp_path / "plugmem.pid")
    monkeypatch.setattr("plugmem.cli.daemon.default_pid_file", lambda: tmp_path / "plugmem.pid")

    assert _read_pid() is None

    pid_file: Path = tmp_path / "plugmem.pid"
    pid_file.write_text("12345")
    assert _read_pid() == 12345

    _clear_pid_file()
    assert not pid_file.exists()


def test_is_running():
    assert not _is_running(-1)
    assert not _is_running(0)
    assert not _is_running(999999999)


def test_daemon_status_stopped(monkeypatch):
    cfg = PlugmemConfig()

    def fake_read_pid():
        return None

    monkeypatch.setattr("plugmem.cli.daemon._read_pid", fake_read_pid)
    status = daemon_status(cfg)
    assert status["running"] is False
