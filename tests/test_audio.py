"""Microphone frame handling and end-of-speech detection."""
import io
import wave

import numpy as np

from conftest import frames, noise
from jarvis.audio import Mic, Recorder, level, to_wav


def run(rec, signal):
    """Feed a signal; return the time (s) at which each new state was first reported."""
    seen = {}
    for i, f in enumerate(frames(signal)):
        state = rec.feed(f)
        seen.setdefault(state, round(i * 512 / 16000, 2))
        if state in ("done", "timeout"):
            break
    return seen


def test_detects_question_start_and_end(question):
    rec = Recorder(silence_end_ms=800)
    seen = run(rec, np.concatenate([noise(1.0), question, noise(3.0)]))
    assert "speaking" in seen and "done" in seen
    assert 0.9 <= seen["speaking"] <= 1.6                  # starts soon after the speech begins
    recorded = len(rec.pcm()) / 2 / 16000
    assert abs(recorded - len(question) / 16000) < 0.8     # keeps the whole question, not the silence after


def test_recording_keeps_the_beginning_of_the_question(question):
    rec = Recorder()
    run(rec, np.concatenate([noise(1.0), question, noise(2.0)]))
    pcm = np.frombuffer(rec.pcm(), dtype=np.int16)
    first_loud = np.argmax(np.abs(question) > 500)
    # the pre-roll means the first loud sample of the question is inside the recording
    assert np.abs(pcm).max() > 500 and len(pcm) >= len(question) - first_loud


def test_background_hiss_is_not_speech():
    rec = Recorder(no_speech_timeout=2)
    seen = run(rec, noise(4.0, level=60))
    assert "speaking" not in seen and "timeout" in seen


def test_louder_steady_hum_is_not_speech():
    t = np.arange(int(4 * 16000)) / 16000
    hum = (300 * np.sin(2 * np.pi * 50 * t)).astype(np.int16) + noise(4.0, level=40)
    seen = run(Recorder(no_speech_timeout=2), hum)
    assert "speaking" not in seen


def test_long_speech_is_cut_at_max_length(question):
    rec = Recorder(max_seconds=2.0)
    seen = run(rec, np.concatenate([noise(0.5)] + [question] * 4))
    assert "done" in seen
    assert len(rec.pcm()) / 2 / 16000 <= 2.5


def test_no_speech_timeout_counts_audio_not_wall_time():
    seen = run(Recorder(no_speech_timeout=1.0), noise(3.0))
    assert 0.9 <= seen["timeout"] <= 1.2


def test_level_is_between_0_and_1():
    assert level(np.zeros(512, dtype=np.int16)) == 0
    assert level(np.full(512, 32767, dtype=np.int16)) == 1.0
    assert level(np.zeros(0, dtype=np.int16)) == 0


def test_wav_roundtrip(question):
    data = to_wav(question.tobytes())
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.readframes(w.getnframes()) == question.tobytes()


def test_mic_rechunks_into_512_sample_frames():
    mic = Mic()
    block = np.arange(700, dtype=np.int16).reshape(-1, 1)
    mic._callback(block, 700, None, None)
    mic._callback(block, 700, None, None)
    got = mic.drain()
    assert [len(f) for f in got] == [512, 512]
    assert got[0][0] == 0 and got[1][0] == 512 % 700


def test_mic_resamples_48k_to_16k():
    mic = Mic()
    mic.native_rate = 48000
    one_second = np.zeros((48000, 1), dtype=np.int16)
    mic._callback(one_second, 48000, None, None)
    total = sum(len(f) for f in mic.drain()) + len(mic._buf)
    assert total == 16000


def test_mic_drops_oldest_frames_when_nobody_reads():
    mic = Mic()
    for i in range(mic.q.maxsize + 10):
        mic._callback(np.full((512, 1), i % 30000, dtype=np.int16), 512, None, None)
    got = mic.drain()
    assert len(got) == mic.q.maxsize
    assert got[-1][0] == (mic.q.maxsize + 9) % 30000            # newest audio kept
