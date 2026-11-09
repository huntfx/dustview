"""Interactive From Dust water sandbox (Tkinter window) on top of watersim.Sim (version 0, textbook rules).

Run with the project venv:
  .venv/Scripts/python tools/watersim_live.py [MAP] [--scale 2]
MAP is a .sav path or part of a map name in the game's DUST/DustSave/Ingame folder
(e.g. "Map6", "Atlantis"); default ALPHA_Map1_Erosion.

Controls
  Left mouse        add (water or sand, see 1/2)
  Right mouse       remove
  Mouse wheel       brush size
  1 / 2             tool: water / sand
  Space             pause / resume
  + / -             simulation steps per frame (speed)
  R                 reset the map
  S                 save a screenshot to data/sim/
  Esc               quit
"""
import argparse, glob, os, sys, time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watersim as ws  # noqa: E402

INGAME = r'D:\Games\Steam\steamapps\common\From Dust\DUST\DustSave\Ingame'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = {'water': dict(rate=40.0, colour='#7fd4ff'), 'soft': dict(rate=12.0, colour='#f0d9a0')}


def resolve_map(arg):
    if not arg:
        return ws.DEFAULT_MAP
    if os.path.isfile(arg):
        return arg
    hits = sorted(glob.glob(os.path.join(INGAME, f'*{arg}*.sav')))
    if not hits:
        sys.exit(f'no map matching {arg!r} in {INGAME}')
    return hits[0]


class App:
    def __init__(self, root, path, scale, selftest=0):
        self.root, self.scale = root, scale
        self.selftest, self.frames = selftest, 0
        self.name = os.path.splitext(os.path.basename(path))[0]
        hard, soft, water = ws.load_map(path)
        self.sim = ws.Sim(hard, soft, water)
        self.rend = ws.Renderer(refresh_every=6)
        n = self.sim.shape[0]
        self.size = n * scale

        root.title(f'From Dust water sandbox – {self.name}')
        self.canvas = tk.Canvas(root, width=self.size, height=self.size, highlightthickness=0, bg='black',
                                cursor='none')
        self.canvas.pack()
        self.status = tk.Label(root, anchor='w', font=('Consolas', 10), bg='#202020', fg='#e0e0e0')
        self.status.pack(fill='x')
        self.photo = ImageTk.PhotoImage(Image.new('RGB', (self.size, self.size)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.ring = self.canvas.create_oval(0, 0, 0, 0, outline=TOOLS['water']['colour'], width=2)

        self.tool, self.radius, self.steps_per_frame = 'water', 10.0, 3
        self.paused, self.button, self.mouse = False, None, None
        self.fps, self.last = 0.0, time.perf_counter()

        c = self.canvas
        c.bind('<Motion>', self.on_move)
        c.bind('<Leave>', lambda e: (setattr(self, 'mouse', None), self.update_ring()))
        for b in (1, 3):
            c.bind(f'<ButtonPress-{b}>', lambda e, b=b: (setattr(self, 'button', b), self.on_move(e)))
            c.bind(f'<B{b}-Motion>', self.on_move)
            c.bind(f'<ButtonRelease-{b}>', lambda e: setattr(self, 'button', None))
        c.bind('<MouseWheel>', self.on_wheel)
        root.bind('<Key>', self.on_key)
        if selftest:                           # paint water on the highest inland point, no user input
            x, y = ws.default_source(self.sim.hard, self.sim.soft, self.sim.water)
            self.mouse, self.button = (x, y), 1
            self.update_ring()
        self.tick()

    # -- input
    def on_move(self, e):
        self.mouse = (e.x / self.scale, e.y / self.scale)
        self.update_ring()

    def on_wheel(self, e):
        self.radius = float(np.clip(self.radius * (1.15 if e.delta > 0 else 1 / 1.15), 2, 80))
        self.update_ring()

    def on_key(self, e):
        k = e.keysym.lower()
        if k == '1': self.tool = 'water'
        elif k == '2': self.tool = 'soft'
        elif k == 'space': self.paused = not self.paused
        elif k in ('plus', 'equal', 'kp_add'): self.steps_per_frame = min(20, self.steps_per_frame + 1)
        elif k in ('minus', 'kp_subtract'): self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif k == 'r': self.sim.reset(); self.rend.base = None
        elif k == 's': self.screenshot()
        elif k == 'escape': self.root.destroy(); return
        self.canvas.itemconfigure(self.ring, outline=TOOLS[self.tool]['colour'])

    def update_ring(self):
        if self.mouse is None:
            self.canvas.coords(self.ring, 0, 0, 0, 0); return
        x, y = self.mouse[0] * self.scale, self.mouse[1] * self.scale
        r = self.radius * self.scale
        self.canvas.coords(self.ring, x - r, y - r, x + r, y + r)

    def screenshot(self):
        out = os.path.join(HERE, 'data', 'sim', f'live_{self.name}_{time.strftime("%Y%m%d_%H%M%S")}.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        self.rend(self.sim, refresh=True).save(out)
        print('saved', out, flush=True)

    # -- main loop
    def tick(self):
        sim = self.sim
        dt_frame = sim.dt * self.steps_per_frame
        if self.button and self.mouse:
            sign = 1 if self.button == 1 else -1
            sim.brush(self.tool, self.mouse[0], self.mouse[1], self.radius,
                      sign * TOOLS[self.tool]['rate'] * dt_frame)
            if self.tool == 'soft':
                self.rend.base = None          # ground changed: refresh shading now
        if not self.paused:
            for _ in range(self.steps_per_frame):
                sim.step()
        img = self.rend(sim)
        if self.scale != 1:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        self.photo.paste(img)

        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last, 1e-6)
        self.last = now
        tool = 'water' if self.tool == 'water' else 'sand'
        self.status.configure(text=(
            f' tool: {tool:5s} (1/2)   brush: {self.radius:4.1f}   speed: {self.steps_per_frame} steps/frame (+/-)   '
            f'{"PAUSED" if self.paused else f"{self.fps:4.1f} fps"}   t = {sim.t:6.1f}   '
            f'water {float(sim.water.sum()) / 1e6:6.3f}M   |  left-click add, right-click remove, wheel size, R reset, S shot'))
        if self.selftest:
            self.frames += 1
            if self.frames >= self.selftest:
                print(f'selftest: {self.frames} frames, {self.fps:.1f} fps, t={sim.t:.2f}, '
                      f'water {float(sim.water.sum()) / 1e6:.4f}M', flush=True)
                self.screenshot(); self.root.destroy(); return
        self.root.after(1, self.tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map', nargs='?')
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--selftest', type=int, default=0, help=argparse.SUPPRESS)  # run N frames painting water, then exit
    a = ap.parse_args()
    try:                                   # crisp pixels on high-DPI Windows displays
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root, resolve_map(a.map), a.scale, a.selftest)
    root.mainloop()


if __name__ == '__main__':
    main()
