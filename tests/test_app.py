"""Whole conversations, end to end, with fake mic / speech-to-text / model and an off-screen window.

These protect the core behaviour: wake -> listen -> answer -> follow-up -> back to Sky Radar.
"""
import numpy as np
import pygame
import pytest
import requests

from conftest import FakeBrain, FakeMic, FakeSTT, noise
from jarvis.app import Job, friendly_error, is_dismissal


def speech(question, before=0.5, after=1.5):
    return np.concatenate([noise(before), question, noise(after)])


def test_question_answer_then_closes_when_no_follow_up(make_app, question):
    stt, brain = FakeSTT(), FakeBrain()
    result = make_app(mic=FakeMic(speech(question)), stt=stt, brain=brain).session()
    assert result.reason == "no_follow_up"
    assert result.turns == [("What is the tallest mountain in Europe?", "Mount Elbrus, at 5,642 metres.")]
    assert brain.questions == ["What is the tallest mountain in Europe?"]
    assert len(stt.calls) == 1 and stt.calls[0] > 16000 * 2 * 2      # more than 2 s of audio was sent


def test_follow_up_question_without_wake_word(make_app, question):
    mic = FakeMic(np.concatenate([speech(question), speech(question)]))
    stt = FakeSTT(["What is the tallest mountain in Europe?", "And in Africa?"])
    brain = FakeBrain(["Mount Elbrus.", "Kilimanjaro."])
    result = make_app(mic=mic, stt=stt, brain=brain).session()
    assert [t[0] for t in result.turns] == ["What is the tallest mountain in Europe?", "And in Africa?"]
    assert [t[1] for t in result.turns] == ["Mount Elbrus.", "Kilimanjaro."]
    assert brain.resets == 1                     # one session = one conversation memory


def test_silence_after_wake_word_shows_message_and_closes(make_app):
    stt, brain = FakeSTT(), FakeBrain()
    result = make_app(mic=FakeMic(), stt=stt, brain=brain).session()
    assert result.reason == "no_speech"
    assert result.message == "I didn't hear anything."
    assert stt.calls == [] and brain.questions == []


@pytest.mark.parametrize("phrase", ["Thank you.", "Never mind.", "No thanks!", "That's all."])
def test_dismissal_phrase_closes_without_asking_the_model(make_app, question, phrase):
    brain = FakeBrain()
    result = make_app(mic=FakeMic(speech(question)), stt=FakeSTT([phrase]), brain=brain).session()
    assert result.reason == "dismissed"
    assert brain.questions == []


def test_noise_during_follow_up_closes_quietly(make_app, question):
    """After an answer, a cough or a door (transcribed as nothing) should not show an error."""
    mic = FakeMic(np.concatenate([speech(question), speech(question)]))
    stt = FakeSTT(["What is the tallest mountain in Europe?", ""])
    result = make_app(mic=mic, stt=stt).session()
    assert result.reason == "no_follow_up" and result.message == ""
    assert len(result.turns) == 1


def test_empty_transcription_says_it_did_not_catch_that(make_app, question):
    result = make_app(mic=FakeMic(speech(question)), stt=FakeSTT([""])).session()
    assert result.reason == "no_speech"
    assert "didn't catch" in result.message


def test_speech_server_down_shows_error_instead_of_crashing(make_app, question):
    stt = FakeSTT(error=requests.ConnectionError("refused"))
    result = make_app(mic=FakeMic(speech(question)), stt=stt).session()
    assert result.reason == "error"
    assert result.message == "I can't reach the speech service on the server."


def test_model_error_shows_error_instead_of_crashing(make_app, question):
    brain = FakeBrain(error=requests.Timeout("slow"))
    result = make_app(mic=FakeMic(speech(question)), brain=brain).session()
    assert result.reason == "error"
    assert result.message == "The AI model took too long to answer."


def test_tap_closes_the_window_at_any_time(make_app):
    class TapScreen:
        opened = closed = 0
        def open(self): self.opened += 1
        def close(self): self.closed += 1
        def draw(self, *a): pass
        def events(self): return "dismiss"
    screen = TapScreen()
    result = make_app(screen=screen).session()
    assert result.reason == "touch"
    assert screen.opened == screen.closed == 1


def test_window_is_closed_even_if_something_crashes(make_app):
    class BrokenMic(FakeMic):
        def drain(self):
            raise RuntimeError("mic unplugged")
    app = make_app(mic=BrokenMic())
    with pytest.raises(RuntimeError):
        app.session()
    assert not pygame.display.get_init()          # window gone, Sky Radar visible again


def test_ask_mode_skips_listening(make_app):
    stt, brain = FakeSTT(), FakeBrain(["Hello, sir."])
    result = make_app(stt=stt, brain=brain).session(first_text="Who are you?")
    assert result.turns == [("Who are you?", "Hello, sir.")]
    assert stt.calls == []


def test_wake_loop_runs_a_session_after_the_wake_word(make_app, question):
    class WakeOnce:
        name = "test"
        calls = 0
        closed = False
        def process(self, frame):
            self.calls += 1
            if self.calls == 3:
                return True
            if self.calls > 3:
                raise KeyboardInterrupt          # stop the endless loop after the session
            return False
        def close(self): self.closed = True
    mic = FakeMic()
    brain = FakeBrain()
    app = make_app(mic=mic, brain=brain)
    app.session = lambda: "session ran"
    wake = WakeOnce()
    app.run(wake)
    assert wake.calls == 4 and wake.closed and not mic.started
    # the model is preloaded at start and again at every wake word, while you're still talking
    import time
    deadline = time.time() + 2
    while brain.warm_ups < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert brain.warm_ups == 2


@pytest.mark.parametrize("text,expected", [
    ("Thank you.", True), ("thanks Jarvis", True), ("NEVER MIND", True), ("Goodbye!", True),
    ("Thank you, what's the time?", False), ("Stop the timer", False), ("", False),
])
def test_is_dismissal(text, expected):
    assert is_dismissal(text) is expected


def test_friendly_errors():
    assert "can't reach" in friendly_error(requests.ConnectionError(), "AI model")
    assert "too long" in friendly_error(requests.Timeout(), "AI model")
    assert friendly_error(ValueError("x"), "AI model") == "Something went wrong with the AI model."


def test_job_streams_generator_and_captures_errors():
    def gen():
        yield "a"
        yield "b"
    job = Job(gen)
    job.join(2)
    assert job.finished and job.take() == "ab" and job.error is None

    def boom():
        raise ValueError("x")
    job = Job(boom)
    job.join(2)
    assert isinstance(job.error, ValueError)
