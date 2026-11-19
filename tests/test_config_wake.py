"""Settings loading and wake word selection."""
import json
import sys

import numpy as np
import pytest

from jarvis import config, wakeword

REQUIRED_KEYS = {"wake_word", "stt_url", "ollama_url", "model", "no_speech_timeout", "silence_end_ms",
                 "follow_up_seconds", "max_record_seconds", "fullscreen", "window_size"}


def test_committed_config_has_everything_and_no_secrets():
    with open(f"{config.ROOT}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert REQUIRED_KEYS <= set(cfg)
    assert "picovoice_access_key" not in cfg          # the key belongs in local.json only


def test_local_json_and_env_override(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "a", "input_device": None, "fullscreen": False}))
    (tmp_path / "local.json").write_text(json.dumps({"model": "b", "picovoice_access_key": "file-key"}))
    cfg = config.load_config(root=str(tmp_path), argv=["--fullscreen"],
                             env={"PICOVOICE_ACCESS_KEY": "env-key", "JARVIS_INPUT_DEVICE": "3"})
    assert cfg["model"] == "b"
    assert cfg["picovoice_access_key"] == "env-key"
    assert cfg["input_device"] == 3
    assert cfg["fullscreen"] is True


def test_works_without_local_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "a"}))
    assert config.load_config(root=str(tmp_path), env={}) == {"model": "a"}


def test_local_json_is_git_ignored():
    with open(f"{config.ROOT}/.gitignore", encoding="utf-8") as f:
        assert "local.json" in f.read().split()


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
    w = wakeword.EnterKeyWake.__new__(wakeword.EnterKeyWake)
    import threading
    w.event = threading.Event()
    frame = np.zeros(512, dtype=np.int16)
    assert not w.process(frame)
    w.event.set()
    assert w.process(frame) and not w.process(frame)


def test_no_key_and_no_terminal_is_a_clear_error(monkeypatch):
    class NoTTY:
        def isatty(self):
            return False
    monkeypatch.setattr(sys, "stdin", NoTTY())
    with pytest.raises(wakeword.NoWakeSource, match="local.json"):
        wakeword.make_wake({}, access_key=None)


def test_missing_key_exits_with_setup_code_so_run_py_does_not_restart(monkeypatch, tmp_path):
    import jarvis.__main__ as cli
    import run

    class NoTTY:
        def isatty(self):
            return False
    monkeypatch.setattr(sys, "stdin", NoTTY())
    monkeypatch.setattr(cli, "build", lambda cfg: object())
    monkeypatch.setattr(cli, "load_config", lambda argv: {})
    assert cli.main([]) == cli.SETUP_NEEDED == run.SETUP_NEEDED


def test_porcupine_has_jarvis_for_pi_and_pc():
    import pvporcupine
    assert "jarvis" in pvporcupine.KEYWORDS
