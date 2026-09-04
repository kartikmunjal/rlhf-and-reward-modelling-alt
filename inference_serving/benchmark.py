"""Raw-trial HF and OpenAI-compatible streaming serving benchmarks."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from pathlib import Path

from inference_serving.data import article_id, prompt_text, read_jsonl, write_jsonl


def benchmark_prompts(articles_path: Path, config: dict, count: int) -> list[dict]:
    rows = read_jsonl(articles_path)
    if len(rows) < count:
        raise ValueError(f"Need {count} benchmark prompts, found {len(rows)}")
    return [{"article_id": article_id(row), "prompt": prompt_text(row["source"], config)} for row in rows[:count]]


def percentile(values: list[float], q: float) -> float:
    import numpy as np
    return float(np.percentile(values, q))


def aggregate_trial(requests: list[dict], wall_seconds: float, peak_gpu_memory_bytes: int) -> dict:
    output_tokens = sum(row["output_tokens"] for row in requests)
    return {
        "wall_seconds": wall_seconds,
        "requests": len(requests),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / wall_seconds,
        "request_throughput": len(requests) / wall_seconds,
        "ttft_ms_p50": percentile([row["ttft_ms"] for row in requests], 50),
        "ttft_ms_p95": percentile([row["ttft_ms"] for row in requests], 95),
        "itl_ms_p50": percentile([row["itl_ms"] for row in requests], 50),
        "itl_ms_p95": percentile([row["itl_ms"] for row in requests], 95),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "request_metrics": requests,
    }


def run_hf_trial(model, tokenizer, prompts: list[dict], *, concurrency: int, new_tokens: int) -> dict:
    """Naive synchronized greedy decode in fixed-size batches, with no engine scheduler."""
    import torch
    tokenizer.padding_side = "left"; tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats(); trial_started = time.perf_counter(); records = []
    for start in range(0, len(prompts), concurrency):
        chunk = prompts[start:start + concurrency]
        encoded = tokenizer([row["prompt"] for row in chunk], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024 - new_tokens).to(model.device)
        input_ids, attention = encoded["input_ids"], encoded["attention_mask"]
        batch_started = time.perf_counter(); intervals = []
        past = None
        for step in range(new_tokens):
            step_started = time.perf_counter()
            with torch.inference_mode():
                output = model(input_ids=input_ids if past is None else input_ids[:, -1:],
                               attention_mask=attention, past_key_values=past, use_cache=True)
                next_token = output.logits[:, -1].argmax(-1, keepdim=True); past = output.past_key_values
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention = torch.cat((attention, torch.ones_like(next_token)), dim=1)
            torch.cuda.synchronize(); intervals.append(time.perf_counter() - step_started)
        ttft_ms = intervals[0] * 1000
        itl_ms = (sum(intervals[1:]) / max(1, len(intervals) - 1)) * 1000
        for row in chunk:
            records.append({"article_id": row["article_id"], "ttft_ms": ttft_ms, "itl_ms": itl_ms,
                            "output_tokens": new_tokens, "batch_wall_ms": (time.perf_counter() - batch_started) * 1000})
    torch.cuda.synchronize()
    return aggregate_trial(records, time.perf_counter() - trial_started, torch.cuda.max_memory_allocated())


def prometheus_snapshot(metrics_url: str) -> dict[str, float]:
    try:
        text = urllib.request.urlopen(metrics_url, timeout=10).read().decode()
    except Exception:
        return {}
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line: continue
        name, raw = line.rsplit(" ", 1)
        if "spec" in name.lower() or "accept" in name.lower() or "draft" in name.lower():
            try: values[name] = float(raw)
            except ValueError: pass
    return values


async def _stream_request(session, url: str, model: str, row: dict, new_tokens: int, gate: asyncio.Semaphore) -> dict:
    payload = {"model": model, "prompt": row["prompt"], "max_tokens": new_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    intervals, token_count, usage_tokens, started, previous = [], 0, None, time.perf_counter(), None
    async with gate:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            async for raw in response.content:
                if not raw.startswith(b"data: "): continue
                body = raw[6:].strip()
                if body == b"[DONE]": break
                event = json.loads(body)
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                text = event.get("choices", [{}])[0].get("text", "")
                if not text: continue
                now = time.perf_counter(); intervals.append(now - (previous or started)); previous = now
                token_count += 1
    if not intervals:
        raise RuntimeError("Streaming request returned no timed token chunks")
    return {"article_id": row["article_id"], "ttft_ms": intervals[0] * 1000,
            "itl_ms": (sum(intervals[1:]) / max(1, len(intervals) - 1)) * 1000,
            "output_tokens": usage_tokens if usage_tokens is not None else token_count,
            "stream_chunks": token_count, "request_wall_ms": (time.perf_counter() - started) * 1000}


async def run_vllm_trial(base_url: str, model: str, prompts: list[dict], *, concurrency: int,
                         new_tokens: int, peak_gpu_memory_bytes: int = 0) -> dict:
    import aiohttp
    before = prometheus_snapshot(base_url.rstrip("/") + "/metrics")
    started = time.perf_counter(); gate = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        requests = await asyncio.gather(*[_stream_request(session, base_url.rstrip("/") + "/v1/completions",
                                                          model, row, new_tokens, gate) for row in prompts])
    wall = time.perf_counter() - started
    after = prometheus_snapshot(base_url.rstrip("/") + "/metrics")
    result = aggregate_trial(requests, wall, peak_gpu_memory_bytes)
    result["speculative_counter_delta"] = {name: after[name] - before.get(name, 0.0) for name in after}
    return result


def append_trial(path: Path, row: dict) -> None:
    existing = read_jsonl(path)
    write_jsonl(path, [*existing, row])
