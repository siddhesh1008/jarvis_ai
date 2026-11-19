"""Jarvis screen: a round, full-screen window that only exists while Jarvis is active.

Everything is drawn on a 1080x1080 canvas (the Waveshare round display) and scaled to the window,
so on a PC it can run in a smaller window. Text is laid out to fit inside the circle.
"""
import math
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

SIZE = 1080
C = SIZE // 2
R = 500                       # radius of the area we draw in (screen edge is 540)

BG = (4, 8, 12)
CYAN = (60, 210, 255)
AMBER = (255, 176, 60)
RED = (255, 90, 80)
WHITE = (235, 245, 250)
GREY = (120, 140, 150)

STATE_STYLE = {                       # colour and label per state
    "listening": (CYAN, "Listening"),
    "transcribing": (CYAN, "Got it"),
    "thinking": (AMBER, "Thinking"),
    "answering": (CYAN, ""),
    "follow_up": (CYAN, "Anything else?"),
    "error": (RED, ""),
    "bye": (GREY, ""),
}


class Screen:
    def __init__(self, cfg):
        self.fullscreen = cfg.get("fullscreen", False)
        self.window_size = cfg.get("window_size", 720)
        self.fps = cfg.get("fps", 30)
        self.window = None
        self.canvas = None
        self.clock = None
        self._fonts = {}
        self.level = 0.0                  # smoothed mic level for the pulse

    # --- window -------------------------------------------------------------------------------
    def open(self):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Jarvis")
        if self.fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        else:
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.canvas = pygame.Surface((SIZE, SIZE))
        self.clock = pygame.time.Clock()
        self.level = 0.0

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None

    def events(self):
        """Returns 'dismiss' if the user tapped the screen or pressed Esc/Q."""
        result = None
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                result = "dismiss"
            elif e.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                result = "dismiss"
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                result = "dismiss"
        return result

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    # --- drawing ------------------------------------------------------------------------------
    def draw(self, state, heard="", answer="", level=0.0):
        t = pygame.time.get_ticks() / 1000.0
        self.level += (level - self.level) * 0.35
        colour, label = STATE_STYLE.get(state, (CYAN, ""))
        cv = self.canvas
        cv.fill(BG)
        self._rings(cv, t, colour, state)
        if heard or answer:
            self._text(cv, heard, answer, colour if state == "error" else WHITE)
            if label:
                self._label(cv, label, colour, y=C + 430, size=40)
        else:
            self._core(cv, t, colour, state)
            if label:
                dots = "." * (int(t * 2.5) % 4) if state in ("listening", "thinking") else ""
                self._label(cv, label + dots, colour, y=C + 230, size=64)
        self._present()
        self.clock.tick(self.fps)

    def _rings(self, cv, t, colour, state):
        speed = 2.2 if state == "thinking" else 0.6
        dim = tuple(int(c * 0.35) for c in colour)
        pygame.draw.circle(cv, dim, (C, C), R + 20, 2)
        for i, (radius, arc_len, width, direction) in enumerate(
                [(R, 1.9, 6, 1), (R - 30, 1.1, 3, -1), (R - 55, 2.6, 2, 1)]):
            start = (t * speed * direction + i * 2.1) % (2 * math.pi)
            for k in range(3):                                   # three arcs per ring
                a = start + k * 2 * math.pi / 3
                rect = pygame.Rect(C - radius, C - radius, radius * 2, radius * 2)
                pygame.draw.arc(cv, colour if i == 0 else dim, rect, a, a + arc_len / 3, width)

    def _core(self, cv, t, colour, state):
        breathe = 0.5 + 0.5 * math.sin(t * 3)
        r = 90 + self.level * 120 + (breathe * 10 if state != "listening" else 0)
        half = int(r * 2.2) + 1                                         # soft glow, one small surface
        glow = pygame.Surface((half * 2, half * 2), pygame.SRCALPHA)
        for k, alpha in ((2.2, 30), (1.6, 60), (1.25, 110)):
            pygame.draw.circle(glow, (*colour, alpha), (half, half), int(r * k))
        cv.blit(glow, (C - half, C - half))
        pygame.draw.circle(cv, colour, (C, C), int(r))
        pygame.draw.circle(cv, WHITE, (C, C), int(r * 0.45))

    def _label(self, cv, text, colour, y, size):
        surf = self.font(size).render(text, True, colour)
        cv.blit(surf, surf.get_rect(center=(C, y)))

    def _text(self, cv, heard, answer, answer_colour):
        top, bottom = C - 400, C + 390
        for size in (72, 62, 54, 48, 42, 36):
            lines = self._layout(heard, answer, size, top, bottom, answer_colour)
            if lines is not None:
                break
        else:
            # Too long even at the smallest size: show the newest part of the answer
            lines = self._layout_tail(heard, answer, 36, top, bottom, answer_colour)
        height = sum(h for _, h, _ in lines)
        y = top + max(0, (bottom - top - height) // 2)
        for surf, h, _ in lines:
            if surf is not None:
                cv.blit(surf, surf.get_rect(midtop=(C, y)))
            y += h

    def _chord(self, y, h):
        """Width available inside the circle for a line from y to y+h."""
        dy = max(abs(y - C), abs(y + h - C))
        return 2 * math.sqrt(max(0, (R - 40) ** 2 - dy ** 2))

    def _wrap(self, text, font, y, line_h, colour, limit_y):
        out = []
        words = text.split()
        line = ""
        while words:
            trial = (line + " " + words[0]).strip()
            if font.size(trial)[0] <= self._chord(y, line_h) or not line:
                line = trial
                words.pop(0)
                continue
            out.append((font.render(line, True, colour), line_h, line))
            y += line_h
            line = ""
            if y + line_h > limit_y:
                return out, y, False
        if line:
            out.append((font.render(line, True, colour), line_h, line))
            y += line_h
        return out, y, y <= limit_y

    def _layout(self, heard, answer, size, top, bottom, answer_colour):
        lines = []
        y = top
        if heard:
            f = self.font(int(size * 0.7))
            part, y, ok = self._wrap(f"“{heard}”", f, y, int(size * 0.75), GREY, bottom)
            if not ok:
                return None
            lines += part
            if answer:
                lines.append((None, size // 2, ""))
                y += size // 2
        if answer:
            part, y, ok = self._wrap(answer, self.font(size), y, int(size * 1.08), answer_colour, bottom)
            if not ok:
                return None
            lines += part
        return lines

    def _layout_tail(self, heard, answer, size, top, bottom, answer_colour):
        words = answer.split()
        while len(words) > 1:
            words = words[len(words) // 5 or 1:]                 # drop the oldest words until it fits
            lines = self._layout("", "… " + " ".join(words), size, top, bottom, answer_colour)
            if lines is not None:
                return lines
        return []

    def _present(self):
        w, h = self.window.get_size()
        side = min(w, h)
        if side == SIZE:
            self.window.blit(self.canvas, ((w - SIZE) // 2, (h - SIZE) // 2))
        else:
            scaled = pygame.transform.smoothscale(self.canvas, (side, side))
            self.window.fill(BG)
            self.window.blit(scaled, ((w - side) // 2, (h - side) // 2))
        pygame.display.flip()
