"""Command line.

    python -m jarvis                 listen for "Jarvis" (or press Enter in the terminal)
    python -m jarvis --ask "..."     skip wake word and speech: ask this right away (testing)
    python -m jarvis --check         check microphone, wake word key, speech server and Ollama
    python -m jarvis --list-mics     list microphones (use the number as "input_device" in config.json)
    --fullscreen                     full screen (the Pi); otherwise a window of "window_size" pixels
"""
import sys

from .config import load_config

SETUP_NEEDED = 2          # exit code: something must be set up first; run.py won't restart Jarvis for it


def build(cfg):
    from .app import Jarvis
    from .audio import Mic
    from .llm import Brain
    from .screen import Screen
    from .stt import SpeechToText
    return Jarvis(cfg, Mic(device=cfg.get("input_device")), SpeechToText(cfg), Brain(cfg), Screen(cfg))


def list_mics():
    from .audio import list_devices
    for i, name in enumerate(list_devices()):
        print(f"  {i:3d}  {name}")
    print('Set "input_device" in config.json or local.json to a number above, or to part of a name '
          '(e.g. "USB"). null = the system default.')


def check(cfg):
    ok = True

    def line(good, text):
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else '!!'}] {text}")

    print("Microphone:")
    try:
        import time
        from .audio import Mic, level
        mic = Mic(device=cfg.get("input_device"))
        mic.start()
        time.sleep(1.5)
        frames = mic.drain()
        mic.stop()
        loudest = max((level(f) for f in frames), default=0.0)
        line(len(frames) > 30, f"{mic.name}: {len(frames)} frames in 1.5 s, loudest {loudest:.3f}"
             + ("  (very quiet: is it the right mic?)" if loudest < 0.002 else ""))
    except Exception as e:
        line(False, f"microphone failed ({e})")
    print("Wake word:")
    try:
        import time
        import numpy as np
        from .wakeword import OpenWakeWord
        wake = OpenWakeWord(cfg.get("wake_model", "hey_jarvis_v0.1"), cfg.get("wake_threshold", 0.5))
        start = time.perf_counter()
        for _ in range(25):                                   # 2 s of silence
            wake.scores(np.zeros(1280, dtype=np.int16))
        per_step = (time.perf_counter() - start) / 25 * 1000
        line(per_step < 60, f"{wake.name} loaded, {per_step:.0f} ms of CPU per 80 ms of audio "
             f"({per_step / 80:.0%} of one core)")
    except Exception as e:
        line(False, f"openWakeWord failed: {e}")
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
    from .wakeword import NoWakeSource, make_wake
    try:
        wake = make_wake(cfg)
    except NoWakeSource as e:
        print(f"[jarvis] {e}", flush=True)
        return SETUP_NEEDED
    app.run(wake)
    return 0


if __name__ == "__main__":
    sys.exit(main())
