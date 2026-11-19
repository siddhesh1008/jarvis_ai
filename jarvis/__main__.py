"""Command line.

    python -m jarvis                 listen for "Jarvis" (or press Enter in the terminal)
    python -m jarvis --ask "..."     skip wake word and speech: ask this right away (testing)
    python -m jarvis --check         check microphone, wake word key, speech server and Ollama
    python -m jarvis --list-mics     list microphones (use the number as "input_device" in config.json)
    --fullscreen                     full screen (the Pi); otherwise a window of "window_size" pixels
"""
import sys

from .config import load_config


def build(cfg):
    from .app import Jarvis
    from .audio import Mic
    from .llm import Brain
    from .screen import Screen
    from .stt import SpeechToText
    return Jarvis(cfg, Mic(device=cfg.get("input_device")), SpeechToText(cfg), Brain(cfg), Screen(cfg))


def list_mics():
    import sounddevice as sd
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "*" if i == default_in else " "
            print(f" {mark} {i:3d}  {d['name']}  ({int(d['default_samplerate'])} Hz)")
    print(" * = default. Put the number in config.json as \"input_device\" to use another mic.")


def check(cfg):
    ok = True

    def line(good, text):
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else '!!'}] {text}")

    print("Microphone:")
    try:
        import sounddevice as sd
        line(True, sd.query_devices(cfg.get("input_device"), "input")["name"])
    except Exception as e:
        line(False, f"no microphone found ({e})")
    print("Wake word:")
    key = cfg.get("picovoice_access_key")
    if not key:
        line(False, "no Picovoice AccessKey in local.json (the Enter key still works in a terminal)")
    else:
        try:
            from .wakeword import PorcupineWake
            PorcupineWake(key, cfg.get("wake_word", "jarvis")).close()
            line(True, "Porcupine started with your AccessKey")
        except Exception as e:
            line(False, f"Porcupine failed: {e}")
    print("Speech server:")
    try:
        from .stt import SpeechToText
        info = SpeechToText(cfg).health()
        line(True, f"{cfg['stt_url']} (model {info.get('model')})")
    except Exception as e:
        line(False, f"{cfg['stt_url']} not reachable ({e.__class__.__name__}). Start it on the server, see README")
    print("Ollama:")
    try:
        from .llm import Brain
        names = Brain(cfg).available_models()
        line(cfg["model"] in names, f"{cfg['ollama_url']} has {cfg['model']}" if cfg["model"] in names
             else f"{cfg['model']} not found; available: {', '.join(names)}")
    except Exception as e:
        line(False, f"{cfg['ollama_url']} not reachable ({e.__class__.__name__})")
    return ok


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv=args)
    if "--list-mics" in args:
        list_mics()
        return 0
    if "--check" in args:
        return 0 if check(cfg) else 1
    app = build(cfg)
    if "--ask" in args:
        i = args.index("--ask")
        question = args[i + 1] if i + 1 < len(args) else "Who are you?"
        app.mic.start()
        try:
            print(app.session(first_text=question))
        finally:
            app.mic.stop()
        return 0
    from .wakeword import make_wake
    app.run(make_wake(cfg, cfg.get("picovoice_access_key")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
