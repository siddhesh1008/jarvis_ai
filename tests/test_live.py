"""Checks against the real P5 server. Skipped by default; run with:  py run.py --test -m live"""
import pytest

from jarvis.config import load_config
from jarvis.llm import Brain
from jarvis.stt import SpeechToText

pytestmark = pytest.mark.live


@pytest.fixture
def live_cfg():
    return load_config()


def test_ollama_has_the_configured_model(live_cfg):
    assert live_cfg["model"] in Brain(live_cfg).available_models()


def test_ollama_answers(live_cfg):
    answer = "".join(Brain(live_cfg).ask("Reply with just the word: ready"))
    assert "ready" in answer.lower()


def test_speech_server_transcribes_the_sample_question(live_cfg, question):
    text = SpeechToText(live_cfg).transcribe(question.tobytes()).lower()
    assert "tallest mountain" in text and "europe" in text
