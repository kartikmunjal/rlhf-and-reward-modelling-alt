import asyncio
import json

from inference_serving.benchmark import _stream_request


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
