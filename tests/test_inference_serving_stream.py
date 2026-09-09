import asyncio
import json

from inference_serving.benchmark import (
    _stream_request, benchmark_prompts, gpu_memory_used_bytes, prometheus_snapshot,
)
from inference_serving.data import write_jsonl


class _Content:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _Response:
    def __init__(self, events):
        self.content = _Content(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, events):
        self._events = events

    def post(self, *_args, **_kwargs):
        return _Response(self._events)


def _sse(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n"


def test_stream_request_accepts_usage_only_terminal_event():
    events = [
        _sse({"choices": [{"text": "a"}]}),
        _sse({"choices": [{"text": "b"}]}),
        _sse({"choices": [], "usage": {"completion_tokens": 2}}),
        b"data: [DONE]\n",
    ]
    result = asyncio.run(
        _stream_request(
            _Session(events), "http://example/v1/completions", "model",
            {"article_id": "a", "prompt": "p"}, 2, asyncio.Semaphore(1),
        )
    )
    assert result["output_tokens"] == 2
    assert result["stream_chunks"] == 2


class _WhitespaceTokenizer:
    def encode(self, text, **kwargs):
        values = text.split()
        maximum = kwargs.get("max_length")
        return values[:maximum] if kwargs.get("truncation") and maximum else values

    def decode(self, values, **_kwargs):
        return " ".join(values)


def test_benchmark_prompts_apply_locked_source_token_limit(tmp_path):
    articles = tmp_path / "articles.jsonl"
    write_jsonl(articles, [{"article_id": "a", "source": "one two three four five"}])
    config = {
        "generation": {
            "source_tokens": 3,
            "prompt_template": "Article: {source} Summary:",
        }
    }
    rows = benchmark_prompts(articles, config, 1, _WhitespaceTokenizer())
    assert rows[0]["prompt"] == "Article: one two three Summary:"
    assert rows[0]["source_tokens"] == 3
    assert rows[0]["prompt_tokens"] == 5


def test_gpu_memory_sampler_converts_mib_to_bytes(monkeypatch):
    class _Completed:
        stdout = "123.0\n"

    monkeypatch.setattr("inference_serving.benchmark.subprocess.run", lambda *args, **kwargs: _Completed())
    assert gpu_memory_used_bytes() == 123 * 1024 * 1024


def test_prometheus_snapshot_ignores_spec_in_label_values(monkeypatch):
    payload = b'vllm:num_requests_running{model_name="dpo-spec-k2"} 3\n' \
              b'vllm:spec_decode_num_draft_tokens_total{model_name="dpo-spec-k2"} 7\n'

    class _Response:
        def read(self):
            return payload

    monkeypatch.setattr("inference_serving.benchmark.urllib.request.urlopen", lambda *args, **kwargs: _Response())
    assert prometheus_snapshot("http://example/metrics") == {
        'vllm:spec_decode_num_draft_tokens_total{model_name="dpo-spec-k2"}': 7.0,
    }


def test_stream_latency_excludes_client_semaphore_queue_time():
    async def exercise():
        gate = asyncio.Semaphore(0)
        events = [
            _sse({"choices": [{"text": "a"}]}),
            _sse({"choices": [], "usage": {"completion_tokens": 1}}),
            b"data: [DONE]\n",
        ]
        task = asyncio.create_task(
            _stream_request(
                _Session(events), "http://example/v1/completions", "model",
                {"article_id": "a", "prompt": "p"}, 1, gate,
            )
        )
        await asyncio.sleep(0.05)
        gate.release()
        return await task

    result = asyncio.run(exercise())
    assert result["request_wall_ms"] < 25
