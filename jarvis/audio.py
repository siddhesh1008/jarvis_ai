"""Microphone input and end-of-speech detection.

Mic delivers 16 kHz mono int16 frames of a fixed length (512 samples = 32 ms).
It uses pvrecorder (open source, Apache 2.0, no account or key needed): it ships its own audio library
for the Pi and Windows, so nothing has to be installed on the system, and it always records 16 kHz mono.

Recorder decides when the user has started and stopped talking (webrtcvad), and returns the
recorded speech as raw 16 kHz int16 bytes.
"""
import collections
import io
import queue
import threading
import wave

import numpy as np
import webrtcvad

RATE = 16000


def list_devices():
    from pvrecorder import PvRecorder
    return PvRecorder.get_available_devices()


def device_index(device, names=None):
    """config "input_device": null = default, a number = that index, text = first mic whose name contains it."""
    if device is None or device == "":
        return -1
    if isinstance(device, int):
        return device
    names = list_devices() if names is None else names
    for i, name in enumerate(names):
        if str(device).lower() in name.lower():
            return i
    raise ValueError(f"No microphone matching {device!r}. Available: {names}")


class Mic:
    def __init__(self, frame_length=512, device=None):
        self.frame_length = frame_length
        self.device = device
        self.q = queue.Queue(maxsize=400)          # ~13 s of audio; old frames are dropped if nobody reads
        self._buf = np.zeros(0, dtype=np.int16)
        self.recorder = None
        self._running = False
        self._thread = None
        self.name = ""

    def start(self):
        from pvrecorder import PvRecorder
        self.recorder = PvRecorder(frame_length=self.frame_length, device_index=device_index(self.device))
        self.name = self.recorder.selected_device
        self.recorder.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                samples = self.recorder.read()
            except Exception:
                if self._running:
                    raise
                return
            self._push(np.asarray(samples, dtype=np.int16))

    def _push(self, samples):
        self._buf = np.concatenate([self._buf, samples])
        while len(self._buf) >= self.frame_length:
            frame, self._buf = self._buf[:self.frame_length], self._buf[self.frame_length:]
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                try:
                    self.q.get_nowait()             # drop the oldest frame
                except queue.Empty:
                    pass
                self.q.put_nowait(frame)

    def read(self, timeout=0.5):
        """Next frame, or None if nothing arrived within timeout."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """All frames waiting right now (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def clear(self):
        self.drain()

    def stop(self):
        self._running = False
        if self.recorder:
            self.recorder.stop()
            if self._thread:
                self._thread.join(timeout=1)
            self.recorder.delete()
            self.recorder = None


def level(frame):
    """Loudness of a frame, 0..1 (for the on-screen pulse)."""
    if len(frame) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
    return min(1.0, rms / 4000.0)


class Recorder:
    """Feed it mic frames; it reports 'waiting', 'speaking', 'done' or 'timeout'."""
    VAD_FRAME = 480                                 # 30 ms at 16 kHz (webrtcvad accepts 10/20/30 ms)
    MIN_RMS = 80                                    # quieter than this is never speech

    def __init__(self, aggressiveness=2, no_speech_timeout=5.0, silence_end_ms=800, max_seconds=15.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.no_speech_frames = int(no_speech_timeout * 1000 / 30)
        self.silence_end_frames = max(1, int(silence_end_ms / 30))
        self.max_frames = int(max_seconds * 1000 / 30)
        self.pre_roll = collections.deque(maxlen=10)     # 300 ms kept from before speech started
        self.recent = collections.deque(maxlen=8)        # voiced flags of the last 240 ms
        self.audio = []
        self.pending = np.zeros(0, dtype=np.int16)
        self.started = False
        self.silent_run = 0
        self.frames_seen = 0
        self.noise = None
        self.state = "waiting"

    def feed(self, frame):
        if self.state in ("done", "timeout"):
            return self.state
        self.pending = np.concatenate([self.pending, frame])
        while len(self.pending) >= self.VAD_FRAME and self.state not in ("done", "timeout"):
            chunk, self.pending = self.pending[:self.VAD_FRAME], self.pending[self.VAD_FRAME:]
            self._step(chunk)
        return self.state

    def _step(self, chunk):
        self.frames_seen += 1
        # webrtcvad alone calls steady hiss (fans, mic noise) "speech", so a chunk only counts as
        # speech if it is also clearly louder than the background noise level
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if self.noise is None or rms < self.noise:
            self.noise = rms                                  # follow the background down quickly ...
        else:
            self.noise = 0.98 * self.noise + 0.02 * rms       # ... and up slowly
        loud = rms > max(self.MIN_RMS, self.noise * 2.5)
        voiced = loud and self.vad.is_speech(chunk.tobytes(), RATE)
        if not self.started:
            self.pre_roll.append(chunk)
            self.recent.append(voiced)
            if sum(self.recent) >= 5:                   # mostly speech for a moment: user started talking
                self.started = True
                self.state = "speaking"
                self.audio.extend(self.pre_roll)
                self.silent_run = 0
            elif self.frames_seen >= self.no_speech_frames:
                self.state = "timeout"
            return
        self.audio.append(chunk)
        self.silent_run = 0 if voiced else self.silent_run + 1
        if self.silent_run >= self.silence_end_frames or len(self.audio) >= self.max_frames:
            self.state = "done"

    def pcm(self):
        return np.concatenate(self.audio).tobytes() if self.audio else b""


def to_wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()
