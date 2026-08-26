import json
from pathlib import Path

import pytest

from llm_judge_summeval.ledger import execute_request, load_latest
from llm_judge_summeval.prompts import canonical_sha256, verify_final_prompt_manifest
from llm_judge_summeval.providers import AnthropicProvider, OpenAIProvider, ProviderError
from llm_judge_summeval.schemas import pointwise_schema, validate_output


VALID = {axis: {"score": 4, "rationale": "Evidence."} for axis in ("coherence", "consistency", "fluency", "relevance")}


def test_anthropic_adapter_builds_structured_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    seen = {}
    def transport(url, headers, payload, timeout):
        seen.update(payload)
        return {"id": "a", "model": payload["model"], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": json.dumps(VALID)}]}
    response = AnthropicProvider("claude", transport=transport).request("system", "user", pointwise_schema())
    assert response.parsed == VALID
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert seen["temperature"] == 0


def test_openai_adapter_builds_structured_request_and_disables_storage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    seen = {}
    def transport(url, headers, payload, timeout):
        seen.update(payload)
        return {"id": "o", "model": payload["model"], "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(VALID)}]}]}
    response = OpenAIProvider("gpt", transport=transport).request("system", "user", pointwise_schema())
    assert response.parsed == VALID
    assert seen["text"]["format"]["strict"] is True
    assert seen["store"] is False
    assert "temperature" not in seen


def test_validation_rejects_extra_axes_and_bad_score():
    with pytest.raises(ValueError):
        validate_output(VALID | {"extra": {}}, "pointwise")
    bad = json.loads(json.dumps(VALID)); bad["relevance"]["score"] = 6
    with pytest.raises(ValueError):
        validate_output(bad, "pointwise")


def test_ledger_is_resumable(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls = {"n": 0}
    def transport(url, headers, payload, timeout):
        calls["n"] += 1
        return {"id": "a", "model": "claude", "stop_reason": "end_turn", "usage": {},
                "content": [{"type": "text", "text": json.dumps(VALID)}]}
    provider = AnthropicProvider("claude", transport=transport)
    kwargs = dict(ledger_path=tmp_path / "ledger.jsonl", provider=provider, provider_name="anthropic", model="claude",
                  kind="pointwise", item_id="x", system="s", user="u", schema=pointwise_schema(), metadata={})
    first, second = execute_request(**kwargs), execute_request(**kwargs)
    assert first == second and calls["n"] == 1
    assert load_latest(tmp_path / "ledger.jsonl")[first["request_id"]]["status"] == "success"


def test_ledger_rejects_unexpected_served_model(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    def transport(url, headers, payload, timeout):
        return {"id": "a", "model": "different", "stop_reason": "end_turn", "usage": {},
                "content": [{"type": "text", "text": json.dumps(VALID)}]}
    result = execute_request(
        ledger_path=tmp_path / "ledger.jsonl", provider=AnthropicProvider("pinned", transport=transport),
        provider_name="anthropic", model="pinned", kind="pointwise", item_id="x",
        system="s", user="u", schema=pointwise_schema(), metadata={}, max_transport_attempts=1,
    )
    assert result["status"] == "provider_error"
    assert "did not match pinned model" in result["error"]


def test_amendment_allows_exactly_one_persisted_provider_retry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls = {"n": 0}
    def transport(url, headers, payload, timeout):
        calls["n"] += 1
        raise ProviderError("generic provider rejection", retryable=False, status=400)
    kwargs = dict(
        ledger_path=tmp_path / "ledger.jsonl", provider=AnthropicProvider("pinned", transport=transport),
        provider_name="anthropic", model="pinned", kind="pointwise", item_id="x",
        system="s", user="u", schema=pointwise_schema(), metadata={}, max_transport_attempts=1,
        allow_one_persisted_provider_retry=True,
    )
    execute_request(**kwargs)
    execute_request(**kwargs)
    execute_request(**kwargs)
    assert calls["n"] == 2


def test_heldout_gate_requires_matching_manifest(tmp_path: Path):
    prompts = tmp_path / "prompts.json"; prompts.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError):
        verify_final_prompt_manifest(prompts, tmp_path / "missing.json")


def test_heldout_gate_locks_config_and_primary_model(tmp_path: Path):
    prompts = tmp_path / "prompts.json"; prompts.write_text("{}", encoding="utf-8")
    config = {"judges": {"primary": {"provider": "anthropic", "model": "pinned"}}}
    manifest = tmp_path / "manifest.json"
    import hashlib
    manifest.write_text(json.dumps({"status": "frozen_after_dev_before_heldout",
                                    "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                                    "config_sha256": canonical_sha256(config),
                                    "provider": "anthropic", "model": "pinned"}), encoding="utf-8")
    assert verify_final_prompt_manifest(prompts, manifest, config)["model"] == "pinned"
    config["judges"]["primary"]["model"] = "changed"
    with pytest.raises(PermissionError):
        verify_final_prompt_manifest(prompts, manifest, config)
