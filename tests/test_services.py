"""Speech-to-text and Ollama clients, with the network replaced by fakes."""
import io
import json
import wave

import pytest

from jarvis import llm, stt


class FakeResponse:
    def __init__(self, payload=None, lines=(), status=200):
        self.payload, self.lines, self.status = payload, [json.dumps(x).encode() for x in lines], status

    def raise_for_status(self):
        if self.status >= 400:
            import requests
            raise requests.HTTPError(f"{self.status}")

    def json(self):
        return self.payload

    def iter_lines(self):
        yield from self.lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --- speech to text ---------------------------------------------------------------------------
def test_stt_sends_a_16k_wav_and_returns_trimmed_text(monkeypatch, cfg, question):
    sent = {}

    def post(url, files, data, timeout):
        sent.update(url=url, data=data, wav=files["file"][1])
        return FakeResponse({"text": "  What is the tallest mountain?  "})
    monkeypatch.setattr(stt.requests, "post", post)

    text = stt.SpeechToText(cfg).transcribe(question.tobytes())
    assert text == "What is the tallest mountain?"
    assert sent["url"] == cfg["stt_url"] and sent["data"]["language"] == "en"
    with wave.open(io.BytesIO(sent["wav"])) as w:
        assert w.getframerate() == 16000 and w.getnframes() == len(question)


def test_stt_handles_missing_text(monkeypatch, cfg):
    monkeypatch.setattr(stt.requests, "post", lambda *a, **k: FakeResponse({"text": None}))
    assert stt.SpeechToText(cfg).transcribe(b"\0\0" * 100) == ""


def test_stt_raises_on_server_error(monkeypatch, cfg):
    import requests
    monkeypatch.setattr(stt.requests, "post", lambda *a, **k: FakeResponse({}, status=500))
    with pytest.raises(requests.HTTPError):
        stt.SpeechToText(cfg).transcribe(b"\0\0" * 100)


def test_stt_health_url(monkeypatch, cfg):
    seen = []
    monkeypatch.setattr(stt.requests, "get", lambda url, timeout: seen.append(url) or FakeResponse({"ok": True}))
    stt.SpeechToText(cfg).health()
    assert seen == ["http://stt.invalid:8000/health"]


# --- language model -----------------------------------------------------------------------------
def ollama_lines(*pieces):
    return [{"message": {"content": p}, "done": False} for p in pieces] + [{"message": {"content": ""}, "done": True}]


def test_brain_streams_pieces_and_remembers_the_conversation(monkeypatch, cfg):
    bodies = []

    def post(url, json, stream, timeout):
        bodies.append(json)
        return FakeResponse(lines=ollama_lines("Mount ", "Elbrus."))
    monkeypatch.setattr(llm.requests, "post", post)

    brain = llm.Brain(cfg)
    assert list(brain.ask("Tallest mountain in Europe?")) == ["Mount ", "Elbrus."]
    list(brain.ask("How high is it?"))

    second = bodies[1]
    assert second["model"] == "test-model" and second["stream"] is True
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == "Mount Elbrus."
    assert second["messages"][-1]["content"] == "How high is it?"


def test_brain_reset_forgets(monkeypatch, cfg):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(lines=ollama_lines("ok")))
    brain = llm.Brain(cfg)
    list(brain.ask("hi"))
    brain.reset()
    assert [m["role"] for m in brain.messages("again")] == ["system", "user"]


def test_brain_history_is_limited(cfg):
    cfg["max_history_turns"] = 2
    brain = llm.Brain(cfg)
    for i in range(5):
        brain.messages(f"q{i}")
        brain.history.append({"role": "assistant", "content": f"a{i}"})
    msgs = brain.messages("last")
    assert len(msgs) == 1 + 3                       # system + previous turn + this question
    assert msgs[-1]["content"] == "last" and msgs[1]["content"] == "q4"


def test_brain_raises_ollama_errors(monkeypatch, cfg):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: FakeResponse(lines=[{"error": "model 'x' not found"}]))
    with pytest.raises(RuntimeError, match="not found"):
        list(llm.Brain(cfg).ask("hi"))


def test_system_prompt_has_the_date():
    import time
    t = time.mktime((2026, 11, 19, 15, 30, 0, 0, 0, -1))
    prompt = llm.system_prompt(t)
    assert "Thursday 19 November 2026, 15:30" in prompt
    assert "Jarvis" in prompt and "no markdown" in prompt
