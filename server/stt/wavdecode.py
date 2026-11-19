"""WAV decoding for the speech server. Kept separate (numpy only) so it can be tested without Whisper.

Decoding here avoids PyAV, whose newer versions break faster-whisper's own decoder."""
import io
import wave

import numpy as np


class BadAudio(ValueError):
    pass


def wav_to_float(data, rate_out=16000):
    """16-bit PCM WAV bytes -> mono float32 in [-1, 1] at rate_out."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                raise BadAudio("send 16-bit PCM WAV")
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError) as e:
        raise BadAudio(f"not a WAV file: {e}") from e
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != rate_out and len(pcm):
        n = int(len(pcm) * rate_out / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)
