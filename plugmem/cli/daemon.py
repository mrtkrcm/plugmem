"""Daemon process management."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests

from plugmem.cli.config import (
    PlugmemConfig,
    config_to_env,
    default_log_file,
    default_pid_file,
)


class DaemonError(Exception):
    pass


HEALTH_PATH = "/api/v1/health"


def _read_pid() -> Optional[int]:
    pid_file = default_pid_file()
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _clear_pid_file() -> None:
    try:
        default_pid_file().unlink(missing_ok=True)
    except OSError:
        pass


def daemon_status(cfg: PlugmemConfig) -> Dict[str, Any]:
    pid = _read_pid()
    if pid is None or not _is_running(pid):
        return {
            "running": False,
            "pid": None,
            "host": cfg.service.host,
            "port": cfg.service.port,
            "health": None,
        }
    return {
        "running": True,
        "pid": pid,
        "host": cfg.service.host,
        "port": cfg.service.port,
        "health": _quick_health(cfg),
    }


def _quick_health(cfg: PlugmemConfig) -> Optional[Dict[str, Any]]:
    url = f"http://{cfg.service.host}:{cfg.service.port}{HEALTH_PATH}"
    try:
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def _free_port(host: str, port: int) -> None:
    """If something is listening on host:port, SIGTERM then SIGKILL it."""
    import socket as _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) != 0:
            return  # port is free
    finally:
        s.close()

    # Port is occupied — find the PID and kill it
    import subprocess as _sp
    try:
        result = _sp.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().splitlines():
            try:
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, OSError):
                pass
        # Wait for the port to be released
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as check:
                check.settimeout(0.3)
                if check.connect_ex((host, port)) != 0:
                    return
            time.sleep(0.2)
        # Force kill
        for pid_str in result.stdout.strip().splitlines():
            try:
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, OSError):
                pass
        time.sleep(0.3)
    except Exception:
        pass  # best-effort


def start_daemon(
    cfg: PlugmemConfig,
    *,
    foreground: bool = False,
    wait_for_health: bool = True,
    health_timeout: float = 15.0,
) -> int:
    existing = _read_pid()
    if existing and _is_running(existing) and not foreground:
        raise DaemonError(f"Daemon already running with PID {existing}")
    if existing and not _is_running(existing):
        _clear_pid_file()

    # Ensure the port is free before spawning
    _free_port(cfg.service.host, cfg.service.port)

    log_file = default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file = default_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(config_to_env(cfg))

    cmd = _build_uvicorn_cmd(cfg)

    if foreground:
        os.execvpe(cmd[0], cmd, env)
        return 0

    log_fh = open(log_file, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    pid_file.write_text(str(proc.pid))

    if wait_for_health:
        if not _wait_for_health(cfg, timeout=health_timeout, expect_pid=proc.pid):
            raise DaemonError(
                f"Daemon spawned (PID {proc.pid}) but {HEALTH_PATH} did not respond "
                f"within {health_timeout:.0f}s. Check {log_file}."
            )

    return proc.pid


def _build_uvicorn_cmd(cfg: PlugmemConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "plugmem.api.app:app",
        "--host",
        cfg.service.host,
        "--port",
        str(cfg.service.port),
        "--log-level",
        cfg.service.log_level.lower(),
    ]


def _wait_for_health(cfg: PlugmemConfig, *, timeout: float, expect_pid: int) -> bool:
    url = f"http://{cfg.service.host}:{cfg.service.port}{HEALTH_PATH}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(expect_pid):
            return False
        try:
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.3)
    return False


def stop_daemon(*, timeout: float = 10.0) -> bool:
    pid = _read_pid()
    if pid is None or not _is_running(pid):
        _clear_pid_file()
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid_file()
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(pid):
            _clear_pid_file()
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(0.5)
    _clear_pid_file()
    return True
