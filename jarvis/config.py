"""Settings: config.json (committed defaults) overlaid by local.json (private, git-ignored)
and a few environment variables."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_OVERRIDES = {                     # environment variable -> config key
    "JARVIS_MODEL": "model",
    "JARVIS_INPUT_DEVICE": "input_device",
}


def load_config(root=ROOT, argv=(), env=None):
    env = os.environ if env is None else env
    with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    local = os.path.join(root, "local.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            cfg.update(json.load(f))
    for var, key in ENV_OVERRIDES.items():
        if env.get(var):
            value = env[var]
            cfg[key] = int(value) if key == "input_device" and value.lstrip("-").isdigit() else value
    if "--fullscreen" in argv:
        cfg["fullscreen"] = True
    return cfg
