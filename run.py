#!/usr/bin/env python3
"""Jarvis: one file to set up and start, on Windows, Raspberry Pi / Linux and macOS.

    Windows:        double-click run.py   (or: py run.py)
    Pi / Linux:     python3 run.py

  1. creates a private Python environment in .venv (first run only)
  2. installs the packages from requirements.txt (only when that file changed)
  3. on a Raspberry Pi: adds Jarvis to the desktop autostart (next to Sky Radar), so it starts at boot
  4. starts Jarvis (full screen on a Pi) and restarts it if it crashes

Options:
    --update        download the newest version from GitHub (git pull), set up anything new and
                    restart Jarvis on the Pi's screen. From another computer:
                      ssh jarvis@jarvis.local "python3 ~/jarvis_ai/run.py --update"
    --setup-only    set everything up, but don't start Jarvis
    --windowed      start in a window, even on a Pi
    --no-autostart  don't add Jarvis to the Pi's autostart
    --test          install the test tools and run the automated tests
    anything else is passed to Jarvis, e.g. --check, --list-mics, --ask "What time is it?"
"""
import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
DEV_REQUIREMENTS = os.path.join(HERE, "requirements-dev.txt")
STAMP = os.path.join(VENV, ".requirements.sha256")
LOG_DIR = os.path.join(HERE, "logs")
LOG_FILE = os.path.join(LOG_DIR, "jarvis.log")
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
SETUP_NEEDED = 2          # Jarvis's exit code for "set something up first" (same as jarvis/__main__.py)
OWN_FLAGS = ("--update", "--setup-only", "--windowed", "--no-autostart", "--test")


def say(msg):
    print(f"[jarvis] {msg}", flush=True)


def fail(msg):
    say(msg)
    if IS_WINDOWS and sys.stdin and sys.stdin.isatty():
        input("Press Enter to close this window.")    # double-clicked: keep the message visible
    sys.exit(1)


def is_raspberry_pi():
    try:
        with open("/proc/device-tree/model", errors="ignore") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def apt_install(*packages):
    if not IS_LINUX or not shutil.which("apt-get"):
        return False
    say(f"Installing {' '.join(packages)} with apt (it may ask for your password) ...")
    return subprocess.call(["sudo", "apt-get", "install", "-y", *packages]) == 0


def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if IS_WINDOWS else os.path.join(VENV, "bin", "python")


def ensure_venv():
    if os.path.exists(venv_python()):
        return
    say("Creating a virtual environment in .venv ...")
    builder = venv.EnvBuilder(with_pip=True)
    try:
        builder.create(VENV)
    except Exception:
        shutil.rmtree(VENV, ignore_errors=True)
        if apt_install("python3-venv"):
            builder.create(VENV)
        else:
            fail("Couldn't create the virtual environment. On Raspberry Pi / Debian run: sudo apt install python3-venv")


def ensure_packages(requirements=REQUIREMENTS, stamp=STAMP):
    with open(requirements, "rb") as f:
        wanted = hashlib.sha256(f.read()).hexdigest()
    try:
        with open(stamp) as f:
            if f.read().strip() == wanted:
                return
    except OSError:
        pass
    py = venv_python()
    say(f"Installing packages from {os.path.basename(requirements)} (the first time can take a minute) ...")
    subprocess.call([py, "-m", "pip", "install", "--upgrade", "--quiet", "pip"])
    if subprocess.call([py, "-m", "pip", "install", "-r", requirements]) != 0:
        fail("Couldn't install the packages. Check the internet connection and run this again.")
    with open(stamp, "w") as f:
        f.write(wanted)


def ensure_wake_models():
    """Download the openWakeWord models into models/ (once; about 4 MB)."""
    code = ("from jarvis.config import load_config; from jarvis.wakeword import ensure_models; "
            "cfg = load_config(); "
            "ensure_models(cfg.get('wake_model', 'hey_jarvis_v0.1'), log=lambda m: print('[jarvis] ' + m))")
    if subprocess.call([venv_python(), "-c", code], cwd=HERE) != 0:
        say("Couldn't download the wake word models yet (no internet?). Jarvis will try again when it starts.")


def ensure_pi_autostart():
    """Start Jarvis when the Pi's desktop starts (Raspberry Pi OS Bookworm, labwc)."""
    path = os.path.expanduser("~/.config/labwc/autostart")
    line = f'python3 "{os.path.join(HERE, "run.py")}" &'
    try:
        with open(path) as f:
            if line in f.read():
                return
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")
    say(f"Added Jarvis to {path}, so it starts at boot. Delete that line to stop it.")


def update_from_github():
    if not (shutil.which("git") and os.path.isdir(os.path.join(HERE, ".git"))):
        fail("Can't update: this folder wasn't downloaded with git (git clone).")
    say("Downloading the newest version from GitHub ...")
    if subprocess.call(["git", "-C", HERE, "pull", "--ff-only"]) != 0:
        fail("git pull failed (see above). Files changed by hand in this folder can block it.")


def _other_launchers():
    """PIDs of other run.py processes for this folder (their watchdog would otherwise restart the old Jarvis)."""
    me, target = os.getpid(), os.path.join(HERE, "run.py")
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = f.read().decode(errors="ignore").split("\0")
        except OSError:
            continue
        if argv and "python" in os.path.basename(argv[0]) and target in argv and "--update" not in argv:
            pids.append(int(entry))
    return pids


def desktop_env():
    """Environment for starting Jarvis in the Pi's desktop session (also works from SSH)."""
    env = os.environ.copy()
    runtime = env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    if "WAYLAND_DISPLAY" not in env:
        sockets = sorted(f for f in os.listdir(runtime) if f.startswith("wayland-") and f[8:].isdigit()) \
            if os.path.isdir(runtime) else []
        if not sockets:
            return None
        env["WAYLAND_DISPLAY"] = sockets[0]
    env.setdefault("DISPLAY", ":0")
    return env


def restart_on_screen():
    for pid in _other_launchers():
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    subprocess.call(["pkill", "-f", f"{venv_python()} -m jarvis"])
    env = desktop_env()
    if env is None:
        say("No desktop running right now; the new version starts at the next boot.")
        return
    subprocess.Popen([sys.executable, os.path.join(HERE, "run.py")], cwd=HERE, env=env,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    say(f"Restarted Jarvis on the Pi's screen. Its log: {LOG_FILE}")


def run_tests(extra):
    ensure_packages(DEV_REQUIREMENTS, os.path.join(VENV, ".dev-requirements.sha256"))
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.exit(subprocess.call([venv_python(), "-m", "pytest", *extra], cwd=HERE, env=env))


def open_log():
    """When started without a terminal (autostart), write everything to logs/jarvis.log."""
    if sys.stdout is not None and sys.stdout.isatty():
        return None
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 2_000_000:
        os.replace(LOG_FILE, LOG_FILE + ".1")          # keep one old log
    return open(LOG_FILE, "a", buffering=1, encoding="utf-8")


def run_with_watchdog(cmd, env, log):
    """Start Jarvis and restart it if it crashes. Ctrl+C / SIGTERM (update, shutdown) end it for good."""
    recent = []
    while True:
        say("Starting Jarvis ...")
        if log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting Jarvis ===\n")
        proc = subprocess.Popen(cmd, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT if log else None)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return
        if code == 0:
            return
        if code == SETUP_NEEDED:
            say("Jarvis needs setting up first (see the message above / logs/jarvis.log). Not restarting.")
            return
        if code < 0 and -code in (signal.SIGTERM, signal.SIGINT):
            return
        recent = [t for t in recent if time.time() - t < 120] + [time.time()]
        if len(recent) > 5:
            fail("Jarvis keeps crashing (5 times in 2 minutes). Check logs/jarvis.log.")
        say(f"Jarvis stopped (code {code}) - restarting in 3 s.")
        time.sleep(3)


def main():
    args = sys.argv[1:]
    update, setup_only, windowed = "--update" in args, "--setup-only" in args, "--windowed" in args
    autostart, test = "--no-autostart" not in args, "--test" in args
    args = [a for a in args if a not in OWN_FLAGS]
    pi = is_raspberry_pi()
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is needed.")
    if update:
        update_from_github()
    say(f"System: {'Raspberry Pi' if pi else platform.system()}, Python {platform.python_version()}")

    ensure_venv()
    ensure_packages()
    ensure_wake_models()
    if test:
        run_tests(args)
    one_shot = any(a in args for a in ("--check", "--list-mics", "--ask"))
    if pi and autostart and not one_shot:
        ensure_pi_autostart()
    if setup_only:
        say("Setup complete.")
        return
    if update and IS_LINUX:
        restart_on_screen()
        return

    env = os.environ.copy()
    if pi and "WAYLAND_DISPLAY" not in env:
        env = desktop_env() or env                     # started over SSH: show the window on the Pi's screen
    if pi and not windowed and "--fullscreen" not in args and "--check" not in args and "--list-mics" not in args:
        args.append("--fullscreen")
    if IS_LINUX and env.get("WAYLAND_DISPLAY") and "SDL_VIDEODRIVER" not in env:
        env["SDL_VIDEODRIVER"] = "wayland"            # native Wayland, like Sky Radar (real touch events)
    cmd = [venv_python(), "-m", "jarvis", *args]
    if one_shot:
        sys.exit(subprocess.call(cmd, env=env, cwd=HERE))
    run_with_watchdog(cmd, env, open_log())


if __name__ == "__main__":
    main()
