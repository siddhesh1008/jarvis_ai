"""Wake word detection. Everything that knows about the wake word engine lives in this file,
so another engine can be swapped in without touching the rest.

The engine is openWakeWord (open source, no account or key, runs offline on the Pi):
https://github.com/dscripka/openWakeWord. We run its ONNX models directly with onnxruntime instead of
installing the openwakeword package, because that package pulls in tflite-runtime (not available for
Python 3.13 on the Pi), scikit-learn and scipy. The pipeline is the same as openwakeword's:

    audio, 80 ms at a time ─▶ melspectrogram.onnx ─▶ embedding_model.onnx ─▶ hey_jarvis_v0.1.onnx ─▶ score 0..1

The models are downloaded once into models/ (git-ignored; openWakeWord's models are CC BY-NC-SA 4.0,
fine for personal use). Your own trained model (e.g. plain "jarvis") can go there too: set
"wake_model" in config.json / local.json to its file name without .onnx.

Pressing Enter in the terminal also wakes Jarvis, which is handy for development.
"""
import collections
import os
import sys
import threading
import urllib.request

import numpy as np

from .config import ROOT

MODELS_DIR = os.path.join(ROOT, "models")
RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
FEATURE_MODELS = ("melspectrogram", "embedding_model")
KNOWN_WAKE_MODELS = ("hey_jarvis_v0.1", "alexa_v0.1", "hey_mycroft_v0.1", "hey_rhasspy_v0.1")
CHUNK = 1280                         # openWakeWord works in 80 ms steps


class NoWakeSource(Exception):
    """Nothing can wake Jarvis. A setup problem, not worth restarting for."""


def model_path(name, models_dir=MODELS_DIR):
    return name if name.endswith(".onnx") and os.path.isabs(name) else os.path.join(models_dir, name + ".onnx")


def ensure_models(wake_model="hey_jarvis_v0.1", models_dir=MODELS_DIR, log=print):
    """Download any missing openWakeWord model files. Returns the wake model's path."""
    os.makedirs(models_dir, exist_ok=True)
    for name in (*FEATURE_MODELS, wake_model):
        path = model_path(name, models_dir)
        if os.path.exists(path):
            continue
        if name not in FEATURE_MODELS and name not in KNOWN_WAKE_MODELS:
            raise NoWakeSource(f"Wake word model {path} not found. Put your .onnx model there "
                               f"or set \"wake_model\" to one of {', '.join(KNOWN_WAKE_MODELS)}.")
        log(f"Downloading wake word model {name}.onnx ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(RELEASE + name + ".onnx", tmp)
        os.replace(tmp, path)
    return model_path(wake_model, models_dir)


class OpenWakeWord:
    """Streaming openWakeWord detector. Feed any frame size; scores are updated every 80 ms."""

    def __init__(self, wake_model="hey_jarvis_v0.1", threshold=0.5, models_dir=MODELS_DIR, patience=1):
        import onnxruntime as ort
        path = ensure_models(wake_model, models_dir)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1      # leave the CPU for Sky Radar
        cpu = ["CPUExecutionProvider"]
        self.melspec = ort.InferenceSession(model_path("melspectrogram", models_dir), opts, providers=cpu)
        self.embedder = ort.InferenceSession(model_path("embedding_model", models_dir), opts, providers=cpu)
        self.model = ort.InferenceSession(path, opts, providers=cpu)
        self.model_input = self.model.get_inputs()[0].name
        self.n_features = self.model.get_inputs()[0].shape[1]           # 16 embeddings = 1.28 s of audio
        self.threshold, self.patience = threshold, patience
        self.name = f"openWakeWord '{os.path.basename(path)[:-5]}'"
        self.frame_length = 512
        self.reset()

    # --- the openWakeWord feature pipeline ----------------------------------------------------------
    def _mel(self, samples):
        out = self.melspec.run(None, {"input": samples.astype(np.float32)[None, :]})[0]
        return np.squeeze(out) / 10 + 2                                  # same scaling as openwakeword

    def _embed(self, mel_window):
        return self.embedder.run(None, {"input_1": mel_window.astype(np.float32)[None, :, :, None]})[0].squeeze()

    def reset(self):
        """Forget recent audio (after a wake, so the same "Hey Jarvis" doesn't fire twice)."""
        self.raw = np.zeros(0, dtype=np.int16)            # samples not yet processed (< 80 ms)
        self.history = np.zeros(CHUNK + 480, dtype=np.int16)
        self.mels = np.ones((76, 32), dtype=np.float32)
        # openwakeword starts from embeddings of random noise, so the first second can't false-trigger
        noise = np.random.default_rng(0).integers(-1000, 1000, 16000 * 4).astype(np.int16)
        spec = self._mel(noise)
        self.features = np.array([self._embed(spec[i:i + 76]) for i in range(0, spec.shape[0] - 75, 8)])
        self.above = 0
        self.score = 0.0

    def _step(self, chunk):
        self.history = np.concatenate([self.history, chunk])[-(CHUNK + 480):]
        self.mels = np.vstack([self.mels, self._mel(self.history)])[-970:]
        self.features = np.vstack([self.features, self._embed(self.mels[-76:])])[-120:]
        x = self.features[-self.n_features:][None, :, :].astype(np.float32)
        self.score = float(self.model.run(None, {self.model_input: x})[0].squeeze())
        return self.score

    def scores(self, frame):
        """Feed audio; returns the scores of every completed 80 ms step (often none)."""
        self.raw = np.concatenate([self.raw, np.asarray(frame, dtype=np.int16)])
        out = []
        while len(self.raw) >= CHUNK:
            chunk, self.raw = self.raw[:CHUNK], self.raw[CHUNK:]
            out.append(self._step(chunk))
        return out

    def process(self, frame):
        for s in self.scores(frame):
            self.above = self.above + 1 if s >= self.threshold else 0
            if self.above >= self.patience:
                self.reset()
                return True
        return False

    def close(self):
        pass


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


def make_wake(cfg, interactive=None):
    interactive = sys.stdin is not None and sys.stdin.isatty() if interactive is None else interactive
    detectors = []
    if cfg.get("wake_engine", "openwakeword") == "openwakeword":
        try:
            detectors.append(OpenWakeWord(cfg.get("wake_model", "hey_jarvis_v0.1"),
                                          cfg.get("wake_threshold", 0.5), patience=cfg.get("wake_patience", 1)))
        except NoWakeSource:
            raise
        except OSError as e:                              # e.g. no internet for the first model download
            raise NoWakeSource(f"Couldn't load the wake word models ({e}). Check the internet connection "
                               "for the first download.") from e
    if interactive:
        detectors.append(EnterKeyWake())
    if not detectors:
        raise NoWakeSource('No wake word engine ("wake_engine" is off) and no terminal for the Enter key.')
    return detectors[0] if len(detectors) == 1 else AnyWake(detectors)
