"""The round screen (drawn off-screen) and the speech server's WAV decoding."""
import io
import math
import wave

import numpy as np
import pygame
import pytest

from jarvis import screen as scr
from wavdecode import BadAudio, wav_to_float

LONG = " ".join(["This is a long answer that keeps going so the text has to shrink and then scroll."] * 12)


@pytest.fixture
def screen(cfg):
    s = scr.Screen(cfg)
    s.open()
    yield s
    s.close()


@pytest.mark.parametrize("state,heard,answer", [
    ("listening", "", ""), ("transcribing", "", ""), ("thinking", "", ""),
    ("answering", "What's the weather on Mars?", "Cold and dusty."),
    ("follow_up", "Tell me a story", LONG),
    ("error", "", "I can't reach the speech service on the server."),
    ("bye", "", "Very well."),
    ("some_future_state", "", ""),
])
def test_every_state_draws(screen, state, heard, answer):
    screen.draw(state, heard, answer, level=0.5)
    assert screen.canvas.get_size() == (scr.SIZE, scr.SIZE)


@pytest.mark.parametrize("answer", ["Short.", "Medium answer " * 10, LONG])
def test_text_stays_inside_the_circle(screen, answer):
    lines = screen._layout("What is it?", answer, 48, scr.C - 400, scr.C + 390, scr.WHITE) \
        or screen._layout_tail("", answer, 36, scr.C - 400, scr.C + 390, scr.WHITE)
    assert lines
    y = scr.C - 400
    for surf, h, _ in lines:
        if surf is not None:
            w = surf.get_width()
            for edge_y in (y, y + h):          # both corners of the text line are inside the round screen
                assert math.hypot(w / 2, edge_y - scr.C) <= scr.R + 5, (w, edge_y)
        y += h
    assert y <= scr.C + 390 + 60


def test_long_answer_shows_the_newest_words(screen):
    lines = screen._layout_tail("", LONG + " FINALWORD", 36, scr.C - 400, scr.C + 390, scr.WHITE)
    assert "FINALWORD" in lines[-1][2]


def test_window_opens_and_closes_cleanly(cfg):
    s = scr.Screen(cfg)
    for _ in range(3):                         # one window per wake word, many times a day
        s.open()
        s.draw("listening")
        s.close()
        assert not pygame.display.get_init()


def test_escape_and_tap_dismiss(screen):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert screen.events() == "dismiss"
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10)))
    assert screen.events() == "dismiss"
    assert screen.events() is None


# --- speech server WAV decoding ---------------------------------------------------------------------
def make_wav(samples, rate=16000, channels=1, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def test_decode_16k_mono(question):
    out = wav_to_float(make_wav(question))
    assert out.dtype == np.float32 and len(out) == len(question)
    assert np.allclose(out[:100], question[:100] / 32768.0)


def test_decode_resamples_and_mixes_stereo():
    stereo = np.zeros((48000, 2), dtype=np.int16)
    out = wav_to_float(make_wav(stereo, rate=48000, channels=2))
    assert len(out) == 16000


def test_decode_rejects_bad_input():
    with pytest.raises(BadAudio):
        wav_to_float(b"not a wav")
    with pytest.raises(BadAudio):
        wav_to_float(make_wav(np.zeros(100, dtype=np.uint8), width=1))
