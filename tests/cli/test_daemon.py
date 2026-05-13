"""Tests for daemon management."""
from __future__ import annotations

from pathlib import Path

import pytest

from plugmem.cli.config import PlugmemConfig, default_pid_file
from plugmem.cli.daemon import (
    HEALTH_PATH,
    DaemonError,
    _build_uvicorn_cmd,
    _clear_pid_file,
    _is_running,
    _quick_health,
    _read_pid,
    daemon_status,
    start_daemon,
    stop_daemon,
)


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
    monkeypatch.setattr("plugmem.cli.config.default_pid_file", lambda: tmp_path / "plugmem.pid")
    monkeypatch.setattr("plugmem.cli.daemon.default_pid_file", lambda: tmp_path / "plugmem.pid")

    assert _read_pid() is None
    pid_file = tmp_path / "plugmem.pid"
    pid_file.write_text("12345")
    assert _read_pid() == 12345
    _clear_pid_file()
    assert not pid_file.exists()


def test_is_running_with_invalid_pids():
    assert not _is_running(-1)
    assert not _is_running(0)
    assert not _is_running(999999999)


def test_daemon_status_stopped(monkeypatch):
    monkeypatch.setattr("plugmem.cli.daemon._read_pid", lambda: None)
    cfg = PlugmemConfig()
    status = daemon_status(cfg)
    assert status["running"] is False


def test_start_daemon_raises_if_already_running(monkeypatch):
    monkeypatch.setattr("plugmem.cli.daemon._read_pid", lambda: 12345)
    monkeypatch.setattr("plugmem.cli.daemon._is_running", lambda pid: True)
    with pytest.raises(DaemonError, match="already running"):
        start_daemon(PlugmemConfig())


def test_stop_daemon_returns_false_when_not_running(monkeypatch):
    monkeypatch.setattr("plugmem.cli.daemon._read_pid", lambda: None)
    assert stop_daemon() is False


def test_quick_health_uses_api_v1_health(monkeypatch):
    seen = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "ok"}

    def fake_get(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("plugmem.cli.daemon.requests.get", fake_get)
    cfg = PlugmemConfig(service={"host": "127.0.0.1", "port": 18081})

    data = _quick_health(cfg)
    assert data == {"status": "ok"}
    assert seen["url"] == f"http://127.0.0.1:18081{HEALTH_PATH}"
