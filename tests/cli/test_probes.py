"""Tests for probe functions."""
from __future__ import annotations

from unittest.mock import patch

from plugmem.cli.wizard.probes import detect_ollama, probe_llm


def test_detect_ollama_no_server():
    result = detect_ollama(timeout=0.1)
    assert result is None


def test_detect_ollama_http_error():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 500
        result = detect_ollama(timeout=0.1)
        assert result is None


def test_probe_llm_connection_error():
    ok, msg = probe_llm("http://localhost:1", "", "test-model", timeout=0.1)
    assert ok is False
    assert msg != ""


def test_parse_candidates_no_candidates():
    from plugmem.inference.promotion import extract_coding_memories

    class FakeLLM:
        calls = []

        def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096):
            self.calls.append(messages)
            return "[]"

    llm = FakeLLM()
    result = extract_coding_memories(llm, [])
    assert result == []
    assert llm.calls == []
