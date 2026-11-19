"""Language model client (Ollama on the P5). Keeps the conversation history for one session."""
import json
import time

import requests

SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant living on a small round screen.
Your answers are read on that screen (and later spoken aloud), so:
- answer in one to three short sentences unless the user asks for more,
- no markdown, no bullet lists, no emojis, no code blocks,
- be friendly and a little witty, like a capable butler.
If you don't know something current (news, live prices), say so briefly.
Current local date and time: {now}."""


def system_prompt(now=None):
    now = now if now is not None else time.time()
    return SYSTEM_PROMPT.format(now=time.strftime("%A %d %B %Y, %H:%M", time.localtime(now)))


class Brain:
    def __init__(self, cfg):
        self.url = cfg["ollama_url"].rstrip("/") + "/api/chat"
        self.model = cfg.get("model", "llama3.2:3b")
        self.keep_alive = cfg.get("keep_alive", "30m")     # keep the model loaded so answers start fast
        self.max_turns = cfg.get("max_history_turns", 6)
        self.history = []

    def reset(self):
        self.history = []

    def messages(self, text):
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-(self.max_turns * 2 - 1):]   # last N turns, ending with this question
        return [{"role": "system", "content": system_prompt()}] + self.history

    def ask(self, text):
        """Yields the answer piece by piece as Ollama produces it."""
        body = {"model": self.model, "messages": self.messages(text), "stream": True,
                "keep_alive": self.keep_alive}
        answer = []
        with requests.post(self.url, json=body, stream=True, timeout=(5, 120)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                piece = (msg.get("message") or {}).get("content", "")
                if piece:
                    answer.append(piece)
                    yield piece
                if msg.get("done"):
                    break
        self.history.append({"role": "assistant", "content": "".join(answer)})

    def warm_up(self):
        """Load the model on the server so the first answer is quick."""
        requests.post(self.url, json={"model": self.model, "messages": [], "keep_alive": self.keep_alive},
                      timeout=120)

    def available_models(self):
        base = self.url.rsplit("/api/", 1)[0]
        tags = requests.get(base + "/api/tags", timeout=5).json()
        return [m["name"] for m in tags.get("models", [])]
