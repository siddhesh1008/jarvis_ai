"""Speech-to-text server for Jarvis, meant to run in Docker on the home server (Bosgame P5).

It speaks the same API as OpenAI's /v1/audio/transcriptions, so the Pi only sends a WAV file
and gets back {"text": "..."}. It uses faster-whisper on the CPU.

Settings (environment variables):
    WHISPER_MODEL    model name, default "small.en" (tiny.en / base.en are faster, medium.en more accurate)
    WHISPER_COMPUTE  compute type, default "int8"
    WHISPER_THREADS  CPU threads, default 0 = let faster-whisper decide
"""
import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

from wavdecode import BadAudio, wav_to_float

MODEL_NAME = os.environ.get("WHISPER_MODEL", "small.en")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
THREADS = int(os.environ.get("WHISPER_THREADS", "0"))

print(f"[stt] loading {MODEL_NAME} ({COMPUTE}) ...", flush=True)
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE, cpu_threads=THREADS)
print("[stt] ready", flush=True)

app = FastAPI(title="Jarvis STT")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: str = Form("en"),
                     model_name: str = Form("", alias="model"), prompt: str = Form("")):
    try:
        audio = wav_to_float(await file.read())
    except BadAudio as e:
        raise HTTPException(400, str(e))
    start = time.time()
    segments, _info = model.transcribe(
        audio,
        language=language or None,
        beam_size=1,                      # fastest; short voice commands don't gain much from bigger beams
        vad_filter=True,                  # skip leading/trailing silence
        initial_prompt=prompt or None,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"[stt] {time.time() - start:.2f}s: {text!r}", flush=True)
    return {"text": text}
