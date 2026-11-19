"""Settings loading and the wake word."""
import json
import os
import sys
import wave

import numpy as np
import pytest

from conftest import FIXTURES, noise
from jarvis import config, wakeword

REQUIRED_KEYS = {"wake_engine", "wake_model", "wake_threshold", "stt_url", "ollama_url", "model",
                 "no_speech_timeout", "silence_end_ms", "follow_up_seconds", "max_record_seconds",
                 "fullscreen", "window_size"}


def test_committed_config_has_everything():
    with open(f"{config.ROOT}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert REQUIRED_KEYS <= set(cfg)
    assert cfg["wake_model"] in wakeword.KNOWN_WAKE_MODELS


def test_local_json_and_env_override(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "a", "input_device": None, "fullscreen": False}))
    (tmp_path / "local.json").write_text(json.dumps({"model": "b", "wake_threshold": 0.7}))
    cfg = config.load_config(root=str(tmp_path), argv=["--fullscreen"],
                             env={"JARVIS_MODEL": "c", "JARVIS_INPUT_DEVICE": "3"})
    assert cfg["model"] == "c" and cfg["wake_threshold"] == 0.7
    assert cfg["input_device"] == 3
    assert cfg["fullscreen"] is True


def test_works_without_local_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "a"}))
    assert config.load_config(root=str(tmp_path), env={}) == {"model": "a"}


def test_private_files_are_git_ignored():
    with open(f"{config.ROOT}/.gitignore", encoding="utf-8") as f:
        ignored = f.read().split()
    assert "local.json" in ignored and "models/" in ignored and "logs/" in ignored


# --- openWakeWord ------------------------------------------------------------------------------------
def load(name):
    with wave.open(os.path.join(FIXTURES, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


@pytest.fixture(scope="module")
def models_dir():
    """The real openWakeWord models (downloaded once into models/; skipped when offline)."""
    try:
        wakeword.ensure_models("hey_jarvis_v0.1", log=lambda m: None)
    except Exception as e:
        pytest.skip(f"wake word models not available: {e}")
    return wakeword.MODELS_DIR


def detect(wake, signal):
    """Feed a signal like the mic does (512-sample frames); return the frame indexes where it woke."""
    hits = []
    for i in range(0, len(signal) - 512 + 1, 512):
        if wake.process(signal[i:i + 512]):
            hits.append(i)
    return hits


@pytest.mark.parametrize("clip", ["hey_jarvis.wav", "jarvis.wav"])
def test_wakes_on_hey_jarvis_and_jarvis(models_dir, clip):
    wake = wakeword.OpenWakeWord("hey_jarvis_v0.1", threshold=0.5, models_dir=models_dir)
    hits = detect(wake, np.concatenate([noise(1.0), load(clip), noise(1.5)]))
    assert len(hits) == 1                                    # exactly once, not repeatedly
    assert 1.0 * 16000 <= hits[0] <= (1.0 + len(load(clip)) / 16000 + 0.6) * 16000


@pytest.mark.parametrize("clip", ["not_wake_word.wav", "question.wav"])
def test_ignores_other_speech(models_dir, clip):
    wake = wakeword.OpenWakeWord("hey_jarvis_v0.1", threshold=0.5, models_dir=models_dir)
    assert detect(wake, np.concatenate([noise(1.0), load(clip), noise(1.0)])) == []


def test_ignores_silence_and_hiss(models_dir):
    wake = wakeword.OpenWakeWord("hey_jarvis_v0.1", models_dir=models_dir)
    assert detect(wake, np.concatenate([np.zeros(16000 * 2, dtype=np.int16), noise(5.0, level=300)])) == []


def test_wakes_again_after_a_session(models_dir):
    wake = wakeword.OpenWakeWord("hey_jarvis_v0.1", models_dir=models_dir)
    clip = np.concatenate([noise(1.0), load("hey_jarvis.wav"), noise(1.0)])
    assert len(detect(wake, clip)) == 1
    assert len(detect(wake, clip)) == 1


def test_scores_come_every_80_ms_whatever_the_frame_size(models_dir):
    wake = wakeword.OpenWakeWord("hey_jarvis_v0.1", models_dir=models_dir)
    assert wake.scores(np.zeros(512, dtype=np.int16)) == []
    assert wake.scores(np.zeros(512, dtype=np.int16)) == []
    assert len(wake.scores(np.zeros(512, dtype=np.int16))) == 1      # 1536 samples: one 1280 step
    assert len(wake.scores(np.zeros(1280 * 3, dtype=np.int16))) == 3


def test_unknown_custom_model_is_a_clear_setup_error(tmp_path):
    with pytest.raises(wakeword.NoWakeSource, match="my_jarvis"):
        wakeword.ensure_models("my_jarvis", models_dir=str(tmp_path), log=lambda m: None)


# --- choosing wake sources -------------------------------------------------------------------------
class Detector:
    def __init__(self, fire_on, name):
        self.fire_on, self.name, self.seen, self.closed = fire_on, name, 0, False

    def process(self, frame):
        self.seen += 1
        return self.seen == self.fire_on

    def close(self):
        self.closed = True


def test_any_wake_fires_when_any_detector_fires_and_feeds_all():
    a, b = Detector(2, "a"), Detector(99, "b")
    wake = wakeword.AnyWake([a, b])
    frame = np.zeros(512, dtype=np.int16)
    assert wake.process(frame) is False
    assert wake.process(frame) is True
    assert a.seen == b.seen == 2                      # every detector sees every frame
    assert wake.name == "a or b"
    wake.close()
    assert a.closed and b.closed


def test_enter_key_wake_fires_once_per_press():
    import threading
    w = wakeword.EnterKeyWake.__new__(wakeword.EnterKeyWake)
    w.event = threading.Event()
    frame = np.zeros(512, dtype=np.int16)
    assert not w.process(frame)
    w.event.set()
    assert w.process(frame) and not w.process(frame)


def test_make_wake_uses_openwakeword_plus_enter_in_a_terminal(models_dir, monkeypatch):
    monkeypatch.setattr(wakeword.EnterKeyWake, "_loop", lambda self: None)       # don't read the real stdin
    wake = wakeword.make_wake({"wake_model": "hey_jarvis_v0.1"}, interactive=True)
    assert isinstance(wake, wakeword.AnyWake)
    assert wake.name.startswith("openWakeWord 'hey_jarvis_v0.1'") and wake.name.endswith("Enter key")
    assert isinstance(wakeword.make_wake({}, interactive=False), wakeword.OpenWakeWord)


def test_nothing_to_wake_with_is_a_clear_error():
    with pytest.raises(wakeword.NoWakeSource):
        wakeword.make_wake({"wake_engine": "off"}, interactive=False)


def test_setup_problem_exits_with_a_code_run_py_does_not_restart(monkeypatch):
    import jarvis.__main__ as cli
    import run

    class NoTTY:
        def isatty(self):
            return False
    monkeypatch.setattr(sys, "stdin", NoTTY())
    monkeypatch.setattr(cli, "build", lambda cfg: object())
    monkeypatch.setattr(cli, "load_config", lambda argv: {"wake_engine": "off"})
    assert cli.main([]) == cli.SETUP_NEEDED == run.SETUP_NEEDED
