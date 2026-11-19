"""Shared test helpers: fake microphone, fake speech-to-text, fake model and a headless screen.

The tests never need a microphone, a display or the P5 server (the wake word tests download
openWakeWord's models into models/ once, and are skipped when offline).
Tests marked `live` do talk to the real P5 and are skipped unless you run:  pytest -m live
"""
import os
import sys
import wave

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # draw off-screen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server", "stt"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RATE = 16000


def load_question():
    """A real spoken question ("What is the tallest mountain in Europe?"), 16 kHz mono int16."""
    with wave.open(os.path.join(FIXTURES, "question.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def noise(seconds, level=60, seed=0):
    """Quiet background hiss, like a room."""
    return np.random.default_rng(seed).normal(0, level, int(seconds * RATE)).astype(np.int16)


def frames(signal, size=512):
    return [signal[i:i + size] for i in range(0, len(signal) - size + 1, size)]


class FakeMic:
    """Plays a scripted signal: each drain() hands out the next few frames, then endless silence."""
    def __init__(self, signal=None, per_drain=8):
        self.queue = frames(signal) if signal is not None else []
        self.per_drain = per_drain
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def clear(self):
        pass                      # keep the script: clear() happens between turns in the real app

    def drain(self):
        out = self.queue[:self.per_drain]
        self.queue = self.queue[self.per_drain:]
        while len(out) < self.per_drain:
            out.append(np.zeros(512, dtype=np.int16) + noise(0.032, seed=len(self.queue))[:512])
        return out

    def read(self, timeout=0.5):
        return self.drain()[0]

    def say(self, signal):
        """Queue more speech (e.g. a follow-up question)."""
        self.queue += frames(signal)


class FakeSTT:
    def __init__(self, texts=("What is the tallest mountain in Europe?",), error=None):
        self.texts = list(texts)
        self.error = error
        self.calls = []

    def transcribe(self, pcm):
        self.calls.append(len(pcm))
        if self.error:
            raise self.error
        return self.texts.pop(0) if self.texts else ""


class FakeBrain:
    def __init__(self, answers=("Mount Elbrus, at 5,642 metres.",), error=None):
        self.answers = list(answers)
        self.error = error
        self.questions = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def ask(self, text):
        self.questions.append(text)
        if self.error:
            raise self.error
        for word in (self.answers.pop(0) if self.answers else "OK.").split(" "):
            yield word + " "

    def warm_up(self):
        pass


class FakeClock:
    """Time that moves 0.1 s per call, so 'show this for 3 s' doesn't make tests wait."""
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 0.1
        return self.t


TEST_CONFIG = {
    "stt_url": "http://stt.invalid:8000/v1/audio/transcriptions",
    "ollama_url": "http://ollama.invalid:11434",
    "model": "test-model",
    "no_speech_timeout": 2,
    "follow_up_seconds": 2,
    "silence_end_ms": 600,
    "max_record_seconds": 15,
    "fps": 1000,
    "window_size": 270,
}


@pytest.fixture
def cfg():
    return dict(TEST_CONFIG)


@pytest.fixture
def question():
    return load_question()


@pytest.fixture
def make_app(cfg):
    from jarvis.app import Jarvis
    from jarvis.screen import Screen

    def make(mic=None, stt=None, brain=None, screen=None):
        return Jarvis(cfg, mic or FakeMic(), stt or FakeSTT(), brain or FakeBrain(), screen or Screen(cfg),
                      clock=FakeClock())
    return make


def pytest_collection_modifyitems(config, items):
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="talks to the real P5 server: run with  pytest -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
