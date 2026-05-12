"""Final probe: launch uvicorn briefly, hit /health, confirm everything wires up."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Tuple

import requests

from plugmem.cli.config import PlugmemConfig, config_to_env
from plugmem.cli.daemon import _build_uvicorn_cmd


def run_final_probe(cfg: PlugmemConfig, *, timeout: float = 30.0) -> Tuple[bool, str]:
    env = os.environ.copy()
    env.update(config_to_env(cfg))

    cmd = _build_uvicorn_cmd(cfg)

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        ok, msg = _poll_health(cfg, timeout=timeout, proc=proc)
    finally:
        _terminate(proc)

    return ok, msg


def _poll_health(cfg: PlugmemConfig, *, timeout: float, proc: subprocess.Popen) -> Tuple[bool, str]:
    url = "http://{}:{}/health".format(cfg.service.host, cfg.service.port)
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
            tail = stderr.splitlines()[-5:] if stderr else []
            return False, "Service exited before responding. Last log lines:\n  " + "\n  ".join(tail)
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                missing = [
                    k for k in ("llm_available", "embedding_available", "chroma_available")
                    if not data.get(k, False)
                ]
                if missing:
                    return False, "Service responded but the following are not available: " + ", ".join(missing)
                return True, "All checks passed."
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.5)
    return False, "Timed out after {:.0f}s. Last error: {}".format(timeout, last_err)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass
