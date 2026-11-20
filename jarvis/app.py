"""The conversation logic: wait for the wake word, then run one session in Jarvis's window.

A session is a small state machine, advanced once per screen frame:

    listening ──speech ended──▶ transcribing ──text──▶ thinking ──first words──▶ answering
        ▲                            │ "thanks"/"never mind"                          │ done
        │                            ▼                                                 ▼
        └──────── new speech ◀── follow_up (waits follow_up_seconds) ──silence──▶ window closes
    Any failure shows a message for a few seconds ("error"), then the window closes.
    A tap on the screen or Esc closes it at any time.

All parts (mic, wake word, speech-to-text, model, screen, clock) are passed in, so the tests can
swap in fakes and run whole conversations without hardware or network.
"""
import inspect
import queue
import re
import threading
import time

from .audio import Recorder, level

DISMISS = {"never mind", "nevermind", "stop", "cancel", "thats all", "that is all",
           "thank you", "thanks", "thank you jarvis", "thanks jarvis", "no", "nope", "no thanks",
           "no thank you", "goodbye", "bye", "bye jarvis", "goodbye jarvis", "nothing", "go away"}


def log(msg):
    print(f"[jarvis {time.strftime('%H:%M:%S')}] {msg}", flush=True)    # times help match the log to events


def is_dismissal(text):
    words = re.sub(r"[^a-z ]", "", text.lower().replace("'", "")).strip()
    return " ".join(words.split()) in DISMISS


def friendly_error(e, what):
    import requests
    if isinstance(e, requests.ConnectionError):
        return f"I can't reach the {what} on the server."
    if isinstance(e, requests.Timeout):
        return f"The {what} took too long to answer."
    return f"Something went wrong with the {what}."


class Job(threading.Thread):
    """Runs a slow (network) call off the screen thread. Generators stream their pieces into .pieces."""
    def __init__(self, fn, *args):
        super().__init__(daemon=True)
        self.fn, self.args = fn, args
        self.pieces = queue.Queue()
        self.result = None
        self.error = None
        self.finished = False
        self.start()

    def run(self):
        try:
            out = self.fn(*self.args)
            if inspect.isgenerator(out):
                for piece in out:
                    self.pieces.put(piece)
            else:
                self.result = out
        except Exception as e:                 # shown on screen, not a crash
            self.error = e
        finally:
            self.finished = True

    def take(self):
        out = []
        while True:
            try:
                out.append(self.pieces.get_nowait())
            except queue.Empty:
                return "".join(out)


class SessionResult:
    """What happened in one session (for the log and the tests)."""
    def __init__(self):
        self.turns = []            # (heard, answer)
        self.reason = ""           # why the window closed: no_follow_up, dismissed, touch, error, no_speech
        self.message = ""          # error / goodbye text that was shown

    def __repr__(self):
        return f"SessionResult(reason={self.reason!r}, turns={self.turns!r}, message={self.message!r})"


class Jarvis:
    def __init__(self, cfg, mic, stt, brain, screen, clock=time.time):
        self.cfg, self.mic, self.stt, self.brain, self.screen, self.clock = cfg, mic, stt, brain, screen, clock

    def recorder(self, no_speech_timeout):
        c = self.cfg
        return Recorder(aggressiveness=c.get("vad_aggressiveness", 2), no_speech_timeout=no_speech_timeout,
                        silence_end_ms=c.get("silence_end_ms", 800), max_seconds=c.get("max_record_seconds", 15))

    # --- always-on loop -----------------------------------------------------------------------
    def run(self, wake):
        self.mic.start()
        threading.Thread(target=self._warm_up, daemon=True).start()
        log(f"Ready. Wake with: {wake.name}.  (Ctrl+C to quit)")
        try:
            while True:
                frame = self.mic.read(timeout=0.5)
                if frame is not None and wake.process(frame):
                    log("Wake word heard.")
                    # get the model ready while you're still talking (it may have been unloaded)
                    threading.Thread(target=self._warm_up, daemon=True).start()
                    log(f"Session ended: {self.session()}")
                    self.mic.clear()
        except KeyboardInterrupt:
            log("Bye.")
        finally:
            self.mic.stop()
            wake.close()

    def _warm_up(self):
        try:
            self.brain.warm_up()
        except Exception as e:
            log(f"Couldn't preload the model ({e}). It will load on the first question.")

    # --- one conversation, from wake word until the window closes ----------------------------
    def session(self, first_text=None):
        cfg = self.cfg
        result = SessionResult()
        self.brain.reset()
        self.mic.clear()
        self.screen.open()
        state, heard, answer = "listening", "", ""
        rec = self.recorder(cfg.get("no_speech_timeout", 5))
        job, until, t_mark, follow_up = None, 0.0, self.clock(), False

        def show_then_close(new_state, message, seconds, reason):
            nonlocal state, answer, until
            state, answer, until = new_state, message, self.clock() + seconds
            result.message, result.reason = message, reason

        if first_text:
            heard, state, job = first_text, "thinking", Job(self.brain.ask, first_text)
        try:
            while True:
                if self.screen.events() == "dismiss":
                    result.reason = "touch"
                    break
                frames = self.mic.drain()
                lvl = max((level(f) for f in frames), default=0.0)
                now = self.clock()

                if state in ("listening", "follow_up"):
                    for f in frames:
                        rec.feed(f)
                    if rec.state == "speaking" and state == "follow_up":
                        state, heard, answer = "listening", "", ""          # new question: clear the screen
                    if rec.state == "done":
                        follow_up = bool(result.turns)              # not the first question of this session
                        state, job, t_mark = "transcribing", Job(self.stt.transcribe, rec.pcm()), now
                    elif rec.state == "timeout":
                        if state == "follow_up":
                            result.reason = "no_follow_up"                  # back to Sky Radar
                            break
                        show_then_close("error", "I didn't hear anything.", cfg.get("message_seconds", 2.5),
                                        "no_speech")

                elif state == "transcribing" and job.finished:
                    if job.error:
                        log(f"Speech-to-text failed: {job.error!r}")
                        show_then_close("error", friendly_error(job.error, "speech service"),
                                        cfg.get("error_seconds", 6), "error")
                    else:
                        text = job.result or ""
                        log(f"Heard ({now - t_mark:.1f}s): {text!r}")
                        if not text and follow_up:
                            result.reason = "no_follow_up"          # just a noise after the answer: close quietly
                            break
                        if not text:
                            show_then_close("error", "Sorry, I didn't catch that.", cfg.get("message_seconds", 2.5),
                                            "no_speech")
                        elif is_dismissal(text):
                            heard = ""
                            show_then_close("bye", "Very well.", cfg.get("bye_seconds", 1.2), "dismissed")
                        else:
                            heard, answer = text, ""
                            state, job, t_mark = "thinking", Job(self.brain.ask, text), now

                elif state in ("thinking", "answering"):
                    piece = job.take()
                    if piece:
                        if state == "thinking":
                            log(f"First words after {now - t_mark:.1f}s")
                        state, answer = "answering", answer + piece
                    if job.finished and job.pieces.empty():
                        if job.error:
                            log(f"Model failed: {job.error!r}")
                            show_then_close("error", friendly_error(job.error, "AI model"),
                                            cfg.get("error_seconds", 6), "error")
                        else:
                            log(f"Answer ({now - t_mark:.1f}s): {answer!r}")
                            result.turns.append((heard, answer.strip()))
                            state = "follow_up"
                            rec = self.recorder(cfg.get("follow_up_seconds", 6))
                            self.mic.clear()

                elif state in ("error", "bye") and now >= until:
                    break

                listening = state in ("listening", "follow_up")
                self.screen.draw(state, heard, answer.strip(), lvl if listening else 0.0)
        finally:
            self.screen.close()
        return result
