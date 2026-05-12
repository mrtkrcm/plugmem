"""Connectivity probes for the wizard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import requests


@dataclass
class OllamaInfo:
    base_url: str
    models: List[str] = field(default_factory=list)


def detect_ollama(
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: float = 1.5,
) -> Optional[OllamaInfo]:
    url = "http://{}:{}/api/tags".format(host, port)
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        raw_models = data.get("models", [])
        names = [m.get("name") for m in raw_models if isinstance(m, dict)]
        names = [n for n in names if n]
    except (ValueError, AttributeError):
        return None
    return OllamaInfo(
        base_url="http://{}:{}/v1".format(host, port),
        models=sorted(names),
    )


def probe_llm(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer {}".format(api_key)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout:
        return False, "Timed out after {:.0f}s. Is the endpoint reachable?".format(timeout)
    except requests.RequestException as e:
        return False, "Connection error: {}".format(e)

    if resp.status_code == 200:
        try:
            data = resp.json()
            if "choices" in data and data["choices"]:
                return True, "OK"
        except ValueError:
            pass
        return False, "Got 200 but response shape was unexpected (no `choices`)."

    snippet = (resp.text or "")[:200].replace("\n", " ")
    return False, "HTTP {}: {}".format(resp.status_code, snippet)


def probe_embedding(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer {}".format(api_key)
    payload = {"model": model, "input": "ping"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout:
        return False, "Timed out after {:.0f}s.".format(timeout)
    except requests.RequestException as e:
        return False, "Connection error: {}".format(e)

    if resp.status_code == 200:
        try:
            data = resp.json()
            embeddings = data.get("data") or []
            if embeddings and embeddings[0].get("embedding"):
                return True, "OK"
        except (ValueError, AttributeError):
            pass
        return False, "Got 200 but response shape was unexpected."

    snippet = (resp.text or "")[:200].replace("\n", " ")
    return False, "HTTP {}: {}".format(resp.status_code, snippet)
