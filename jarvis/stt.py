"""Speech-to-text client. Sends a WAV to an OpenAI-compatible /v1/audio/transcriptions endpoint
(our faster-whisper container on the P5, see server/), gets back the text."""
import requests

from .audio import to_wav


class SpeechToText:
    def __init__(self, cfg):
        self.url = cfg["stt_url"]
        self.language = cfg.get("stt_language", "en")
        self.timeout = cfg.get("stt_timeout", 30)

    def transcribe(self, pcm):
        files = {"file": ("speech.wav", to_wav(pcm), "audio/wav")}
        data = {"model": "whisper", "language": self.language, "prompt": "Jarvis"}
        r = requests.post(self.url, files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

    def health(self):
        base = self.url.split("/v1/")[0]
        r = requests.get(base + "/health", timeout=5)
        r.raise_for_status()
        return r.json()
