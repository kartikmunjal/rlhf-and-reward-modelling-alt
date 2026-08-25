"""Append-only, resumable API request ledger."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from llm_judge_summeval.prompts import canonical_sha256
from llm_judge_summeval.providers import ProviderError
from llm_judge_summeval.schemas import validate_output


def load_latest(path: Path) -> dict[str, dict]:
    latest = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            latest[row["request_id"]] = row
    return latest


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute_request(*, ledger_path: Path, provider, provider_name: str, model: str, kind: str, item_id: str,
                    system: str, user: str, schema: dict, metadata: dict, max_transport_attempts: int = 5) -> dict:
    request_id = canonical_sha256({"provider": provider_name, "model": model, "kind": kind, "item_id": item_id,
                                  "system": system, "user": user, "schema": schema})
    prior = load_latest(ledger_path).get(request_id)
    if prior and prior["status"] == "success":
        return prior
    started = time.time()
    schema_attempt = 0
    transport_attempt = 0
    while schema_attempt <= 1:
        try:
            response = provider.request(system, user, schema)
            if response.model != model:
                raise ProviderError(f"Served model {response.model!r} did not match pinned model {model!r}")
            parsed = validate_output(response.parsed, kind)
            row = {"request_id": request_id, "status": "success", "provider": provider_name, "model": model,
                   "kind": kind, "item_id": item_id, "metadata": metadata, "parsed": parsed,
                   "response_id": response.response_id, "served_model": response.model, "stop_reason": response.stop_reason,
                   "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                   "schema_attempt": schema_attempt, "transport_attempts": transport_attempt + 1,
                   "started_unix": started, "completed_unix": time.time()}
            append_row(ledger_path, row)
            return row
        except (ValueError, json.JSONDecodeError) as error:
            if schema_attempt >= 1:
                row = {"request_id": request_id, "status": "invalid_output", "provider": provider_name, "model": model,
                       "kind": kind, "item_id": item_id, "metadata": metadata, "error": str(error),
                       "started_unix": started, "completed_unix": time.time()}
                append_row(ledger_path, row)
                return row
            schema_attempt += 1
        except ProviderError as error:
            transport_attempt += 1
            if not error.retryable or transport_attempt >= max_transport_attempts:
                row = {"request_id": request_id, "status": "provider_error", "provider": provider_name, "model": model,
                       "kind": kind, "item_id": item_id, "metadata": metadata, "error": str(error), "http_status": error.status,
                       "started_unix": started, "completed_unix": time.time()}
                append_row(ledger_path, row)
                return row
            time.sleep(min(60, 2 ** (transport_attempt - 1)) + random.random())
    raise AssertionError("Unreachable")
