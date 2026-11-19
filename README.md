# Jarvis

A voice assistant for the Raspberry Pi 5 with the Waveshare 5" round display. It sits quietly next to
[Sky Radar](https://github.com/siddhesh1008/sky-radar): when you say **"Jarvis"**, it opens its own full-screen
window on top of the radar, listens to your question, shows the answer, and then closes again so the radar is back.

```
Sky Radar (always running, unchanged)
Jarvis (always running, no window, listening for "Jarvis")
   │  "Jarvis"
   ▼
Jarvis window on top: Listening → Got it → Thinking → answer
   │  you say nothing more for 6 s, say "thanks / never mind", or tap the screen
   ▼
window closes, Sky Radar visible again
```

The Pi only does the light work (wake word, recording, screen). The heavy work runs on the home server
(Bosgame P5, `192.168.2.163`):

| Step | What | Where |
|---|---|---|
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) "Hey Jarvis" model (free, offline, no account) | Pi |
| End of speech | webrtcvad + noise gate (stops ~0.8 s after you stop talking) | Pi |
| Speech to text | faster-whisper (`server/`, Docker, port 8000) | P5 |
| Answer | Ollama `llama3.2:3b` (port 11434) | P5 |
| Screen | pygame-ce, round 1080×1080 | Pi |

**Status: phase 1** (answers are shown as text, no speaker yet).
Next phases: tools (timers, weather), speaker + spoken answers, optional link to Sky Radar.

## 1. Speech server on the P5 (once)

Copy the `server/` folder to the P5 (or `git clone` this repo there), then on the P5:

```
cd server
docker compose up -d --build
docker logs -f jarvis-stt          # first start downloads the Whisper model, wait for "[stt] ready"
```

`http://192.168.2.163:8000/health` should then show `{"ok":true,...}`.
The model is set in `server/docker-compose.yml` (`small.en`; `base.en` is faster, `medium.en` more accurate).

## 2. Wake word

Nothing to set up: the first `run.py` downloads openWakeWord's models (about 4 MB) into `models/`.
Its "Hey Jarvis" model also reacts to a plain "Jarvis" in tests; if it misses you or wakes by itself,
change `wake_threshold` (lower = wakes more easily). You can train your own model later with
openWakeWord's training notebook and put it in `models/` (see `jarvis/wakeword.py`).
In a terminal, pressing **Enter** also wakes Jarvis.

## 3. Raspberry Pi

```
cd ~ && git clone https://github.com/siddhesh1008/jarvis_ai.git
cd jarvis_ai
python3 run.py --check          # mic, wake word, speech server, Ollama
python3 run.py
```

No `sudo` needed. The first `run.py` installs everything, adds Jarvis to the desktop autostart (next to
Sky Radar's line in `~/.config/labwc/autostart`) and starts it. Jarvis has no window until you say "Hey Jarvis".
When started at boot it writes its log to `logs/jarvis.log`.

**Update the Pi after pushing changes** (from your PC):

```
ssh jarvis@jarvis.local "python3 ~/jarvis_ai/run.py --update"
```

`--update` pulls from GitHub, installs anything new and restarts Jarvis on the Pi's screen.
To stop autostart, delete the Jarvis line in `~/.config/labwc/autostart`.

## 4. On a PC (development)

```
py run.py                 # Windows (Linux/macOS: python3 run.py)
py run.py --check
py run.py --list-mics     # pick another mic: put its number in config.json as "input_device"
py run.py --ask "What's the capital of Peru?"     # test screen + AI without the mic and speech server
```

## Using it

Say "Hey Jarvis" (or just "Jarvis"), wait for the window, then ask. After the answer, Jarvis listens for about 6 s for a follow-up
(no wake word needed). "Thanks", "never mind", "stop", silence, a tap on the screen or Esc closes it.

## Tests

```
py run.py --test               # all tests, no mic / screen / server needed (about 3 s)
py run.py --test -m live       # also checks the real P5: Ollama and the speech server
```

Run them before every push. GitHub runs them too on every push (`.github/workflows/tests.yml`).
They play whole conversations with a recorded question (`tests/fixtures/question.wav`) through fake
mic / speech / model parts and check: answering, follow-ups, dismissal phrases, silence, server errors,
taps, the window always closing, end-of-speech detection, text staying inside the round screen, and the
clients' requests to the P5.

## Settings (`config.json`)

| Key | Meaning |
|---|---|
| `wake_model` | openWakeWord model in `models/` (`hey_jarvis_v0.1`, or your own trained one) |
| `wake_threshold` | 0–1. Lower wakes more easily (and more often by mistake); default 0.5 |
| `wake_patience` | how many 80 ms steps in a row must be above the threshold (raise to 2–3 against false wakes) |
| `input_device` | mic number from `--list-mics`, `null` = default mic |
| `stt_url` | speech server on the P5 |
| `ollama_url`, `model` | Ollama on the P5 and the model (`llama3.2:3b` ≈ 2–3 s; `llama3.1:8b` ≈ 5 s, fewer made-up facts) |
| `no_speech_timeout` | seconds to wait for you to start talking after the wake word |
| `silence_end_ms` | how long a pause ends your question |
| `follow_up_seconds` | how long Jarvis waits for a follow-up after answering |
| `fullscreen`, `window_size` | full screen on the Pi (set by `run.py`); window size on a PC |

`local.json` overrides any of these per device. `qwen3:4b` is not recommended: on the P5 it thinks for ~25 s.

## Project layout

```
run.py                  set up + start (+ Pi autostart, --update, --test, crash restart)
config.json             default settings          local.json  private per-device settings (git-ignored)
jarvis/
  __main__.py           command line (python -m jarvis)
  config.py             loading settings
  app.py                the conversation state machine (wake → listen → transcribe → think → show → follow-up)
  wakeword.py           wake word: openWakeWord models run with onnxruntime (only file that knows the engine)
  audio.py              microphone + end-of-speech detection
  stt.py                speech-to-text client
  llm.py                Ollama client + Jarvis's personality (system prompt)
  screen.py             the round window
models/                 wake word models (downloaded by run.py, git-ignored)
server/                 Docker services for the P5 (speech-to-text now; text-to-speech later)
tests/                  automated tests (+ fixtures/: recorded "Hey Jarvis", "Jarvis", a question, other speech)
```

### Where new things go

| You want to… | Put it in |
|---|---|
| Change how Jarvis talks | `SYSTEM_PROMPT` in `jarvis/llm.py` |
| Add a skill (timer, weather, Sky Radar planes, …) | a new `jarvis/skills/` package, called from `jarvis/llm.py` (phase 2) |
| Speak answers | a new `jarvis/tts.py` + a `server/tts/` container, used in `app.py` when the answer is done |
| Another wake word / engine | a model file in `models/` + `wake_model`, or a new class in `jarvis/wakeword.py` with `process(frame)`, `close()`, `name` |
| A new screen state | `STATE_STYLE` in `jarvis/screen.py` and the state machine in `jarvis/app.py` |
| A new server service | `server/<name>/` + an entry in `server/docker-compose.yml` |

Every new part should come with a test in `tests/`; the fakes in `tests/conftest.py` let you test without hardware.
