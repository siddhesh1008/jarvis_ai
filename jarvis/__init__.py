"""Jarvis voice assistant for the Raspberry Pi round display.

Package layout (each piece can be replaced or extended on its own):
    config.py     settings from config.json + local.json + environment
    app.py        the conversation: wake word -> listen -> transcribe -> think -> show -> follow-up
    wakeword.py   wake word engines (Porcupine, Enter key for testing)
    audio.py      microphone and end-of-speech detection
    stt.py        speech-to-text client (server on the P5)
    llm.py        language model client (Ollama on the P5) and the system prompt
    screen.py     the round full-screen window
    __main__.py   command line: python -m jarvis [--check | --list-mics | --ask "..."]
"""
__version__ = "0.1.0"
