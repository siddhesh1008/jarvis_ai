"""Wake word detection. Everything that knows about Porcupine lives in this file,
so another engine (e.g. openWakeWord) can be swapped in later without touching the rest.

Without a Picovoice AccessKey (or as an extra on a PC), pressing Enter in the terminal also
wakes Jarvis, which is handy for development.
"""
import sys
import threading


class PorcupineWake:
    def __init__(self, access_key, keyword="jarvis", sensitivity=0.6):
        import pvporcupine
        self.engine = pvporcupine.create(access_key=access_key, keywords=[keyword],
                                         sensitivities=[sensitivity])
        self.frame_length = self.engine.frame_length       # 512 samples
        self.name = f"Porcupine '{keyword}'"

    def process(self, frame):
        return self.engine.process(frame.tolist()) >= 0

    def close(self):
        self.engine.delete()


class EnterKeyWake:
    """Press Enter in the terminal to wake Jarvis (only when started from a terminal)."""
    frame_length = 512
    name = "Enter key"

    def __init__(self):
        self.event = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for _ in sys.stdin:
            self.event.set()

    def process(self, frame):
        if self.event.is_set():
            self.event.clear()
            return True
        return False

    def close(self):
        pass


class AnyWake:
    """Wakes when any of the given detectors fires."""
    def __init__(self, detectors):
        self.detectors = detectors
        self.frame_length = 512
        self.name = " or ".join(d.name for d in detectors)

    def process(self, frame):
        return any([d.process(frame) for d in self.detectors])   # list: every detector sees every frame

    def close(self):
        for d in self.detectors:
            d.close()


def make_wake(cfg, access_key):
    detectors = []
    if access_key:
        detectors.append(PorcupineWake(access_key, cfg.get("wake_word", "jarvis"),
                                       cfg.get("wake_sensitivity", 0.6)))
    if sys.stdin is not None and sys.stdin.isatty():
        detectors.append(EnterKeyWake())
    if not detectors:
        raise SystemExit("No Picovoice AccessKey and no terminal for the Enter key: nothing can wake Jarvis. "
                         "Put your key in local.json (see README).")
    return detectors[0] if len(detectors) == 1 else AnyWake(detectors)
