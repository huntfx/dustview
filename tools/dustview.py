"""DustView: live viewer for From Dust's simulation grid (read-only), first version.

Shows the running game's terrain, water and lava as a top-down map, refreshed several times a second,
with the level's springs / lava vents / drains drawn on top and a readout of the cell under the mouse.
The data source is pluggable (`Source.read()` returns a Frame): `LiveGame` reads the game's memory;
`ModelSource` runs the Python re-implementation (dustsim.tick: border ring, step, springs / vents /
drains) in a background thread, either taking over the live game's current state (key M) or starting
from a map file (--model MAP). The game's DLL can be added the same way, and a comparison mode can then
diff two sources.

Run with the project venv while a level is loaded:
  .venv/Scripts/python tools/dustview.py [--scale 2] [--flip] [--rotate N] [--model MAP]
--model MAP runs the Python model from a map's .sav (e.g. ALPHA_Map11_Void) with the generators that
are on at the start (no level script, no game needed).

Keys
  1 natural   2 height   3 water depth   4 lava temperature   5 sediment   6 flow speed
  M  switch to the Python model, starting from the state on screen (sources as last found)
  L  switch back to the live game       Space  pause / resume the model
  O  overlays (sources) on/off          V  flip vertically     R  rotate 90 degrees
  S  save a screenshot to data/live/    Esc quit
Mouse: hover for the cell's values (col/row, ground, soft ground, water or lava, sediment/temperature).

Reading the game: the simulator object (ISimulatorWater) is found by its parameter array
(live_check.find_sim). The frame counter (+0x92C) increments when a step starts; while step m runs,
buffer (m-1)&1 (pointers at +0x950) holds the complete result of step m-1 and is only read, until
step m+1 starts writing it. So: read the counter, read that buffer, and keep it only if the counter is
unchanged afterwards (always true while the game is paused).
Grid component layout: 0 ground surface (includes soft ground), 1 soft ground, 2 fluid (negative =
lava), 3 sediment (water) or temperature (lava, internal units), 4/5 flux pairs (u16, 1/256 units).
"""
import argparse, os, struct, sys, threading, time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, galileo, live_check, live_generators, dustsim  # noqa: E402

N = 512
GRID_BYTES = 8 * N * N * 4
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- data
class Frame:
    """One simulation state. Arrays are (512, 512), indexed [row, col]."""
    def __init__(self, tick, grid, params=None, sources=(), source_name=''):
        f = lambda c: grid[c].view(np.float32)
        self.grid = grid
        self.tick = tick
        self.ground = f(0)              # surface height including soft ground
        self.soft = f(1)                # soft ground thickness
        fluid = f(2)
        self.water = np.maximum(fluid, 0)
        self.lava = np.maximum(-fluid, 0)
        extra = f(3)
        self.sediment = np.where(fluid > 0, extra, 0)
        self.temperature = np.where(fluid < 0, extra, 0)
        lr, ud = grid[4], grid[5]
        self.flux = np.stack([lr & 0xFFFF, lr >> 16, ud & 0xFFFF, ud >> 16]).astype(np.float32) / 256
        self.params = params
        self.sources = list(sources)    # dicts with col, row, radius, rate, sediment, temperature
        self.source_name = source_name


def state_to_grid(G, like=None):
    """dustsim state dict -> (8, 512, 512) uint32 grid (components 6/7 copied from `like`)."""
    grid = np.zeros((8, N, N), np.uint32) if like is None else like.copy()
    for c, k in enumerate(('T', 'soft', 'W', 'sed')):
        grid[c] = np.asarray(G[k], np.float32).view(np.uint32)
    F = G['F'].astype(np.uint32)
    grid[4] = F[0] | (F[1] << 16)
    grid[5] = F[2] | (F[3] << 16)
    return grid


class Source:
    name = 'source'

    def read(self):
        """Return the newest Frame, or None if nothing new is available."""
        raise NotImplementedError

    def status(self):
        return ''


class LiveGame(Source):
    """The running game's simulation grid, read from its memory (read-only)."""
    name = 'live game'

    def __init__(self):
        self.p = memscan.Proc()
        self.base = live_check.find_sim(self.p)
        self.last_tick = None
        self.tick_times = []
        self.sources = []
        self.source_note = 'looking for sources...'
        self.paused = False
        threading.Thread(target=self._source_loop, daemon=True).start()

    def _counter(self):
        return struct.unpack('<I', self.p.read(self.base + 0x92C, 4))[0]

    def read(self):
        for _ in range(4):
            m = self._counter()
            if m == self.last_tick:
                self.paused = len(self.tick_times) > 1 and time.time() - self.tick_times[-1][1] > 0.5
                return None
            bufs = struct.unpack('<2I', self.p.read(self.base + 0x950, 8))
            raw = self.p.read(bufs[(m - 1) & 1], GRID_BYTES)
            params = np.frombuffer(self.p.read(self.base + 0x20, 224), np.float32).copy()
            if raw is None or len(raw) != GRID_BYTES or self._counter() != m:
                continue                                    # buffer was being overwritten: retry
            self.last_tick = m
            self.paused = False
            now = time.time()
            self.tick_times = [(t, s) for t, s in self.tick_times if now - s < 3] + [(m, now)]
            return Frame(m - 1, galileo.unpack(raw), params, self.sources, self.name)
        return None

    def ticks_per_second(self):
        if len(self.tick_times) < 2:
            return 0.0
        (t0, s0), (t1, s1) = self.tick_times[0], self.tick_times[-1]
        return (t1 - t0) / max(s1 - s0, 1e-6)

    def _source_loop(self):
        """Every ~15 s: find the generator objects, then their records in the simulation's source table
        (36-byte records: col, row, radius, ?, rate, ?, sediment, temperature, ?), matched by radius + rate."""
        delta = None
        while True:
            try:
                delta = live_generators.dustai_delta(self.p) if delta is None else delta
                gens = [g for g in live_generators.generators(self.p, delta) if g['mode'] != 0]
                pats = {struct.pack('<3f', g['radius'], 0.0, g['rate']): g for g in gens}
                found = {}          # address -> record; the table array is pooled, so stale copies of
                                    # earlier levels' records can match too: filtered below
                for base, size in self.p.regions():
                    if size > 64 << 20:
                        continue
                    data = self.p.read(base, size)
                    if not data:
                        continue
                    for pat, g in pats.items():
                        i = data.find(pat)
                        while i >= 0:
                            if i >= 8:
                                col, row = struct.unpack_from('<2f', data, i - 8)
                                rec = struct.unpack_from('<9f', data, i - 8)
                                if (0 <= col < N and 0 <= row < N and col == int(col) and row == int(row)
                                        and abs(rec[6] - g['sediment']) < 1e-4 and abs(rec[7] - g['temperature']) < 1e-3):
                                    found[base + i - 8] = dict(col=col, row=row, radius=rec[2], rate=rec[4], sediment=rec[6],
                                                               temperature=rec[7], activity=g['activity'], buried=g['activity'] < 1,
                                                               height_mode=g['height_mode'])
                            i = data.find(pat, i + 1)
                if found:                                   # keep the cluster around the median address
                    mid = sorted(found)[len(found) // 2]
                    found = {a: r for a, r in found.items() if abs(a - mid) < 36 * 256}
                self.sources = list({(r['col'], r['row']): r for r in found.values()}.values())
                self.source_note = '%d sources (%d generator objects active)' % (len(self.sources), len(gens))
            except Exception as e:                       # game closed, level changing, ...
                self.source_note = 'sources: %s' % e
            time.sleep(15)

    def status(self):
        state = 'paused' if self.paused else '%.0f ticks/s' % self.ticks_per_second()
        return '%s | %s | %s' % (self.name, state, self.source_note)


class ModelSource(Source):
    """The Python model (dustsim.tick) stepping in a background thread as fast as it can."""
    name = 'python model'

    def __init__(self, grid, params, sources, tick=0, label=''):
        self.P = {n: float(v) for n, v in zip(galileo.PARAM_NAMES, np.asarray(params, np.float32))}
        self.params = np.asarray(params, np.float32)
        self.sources = [dict(s) for s in sources]
        for s in self.sources:
            s['_cells'] = dustsim.disc(s['col'], s['row'], s['radius'])
        self.G = dustsim.from_grid(grid)
        self.extra = grid                       # components 6/7 (vegetation, fire) are carried along
        self.tick = tick
        self.label = label
        self.running = True
        self.paused = False
        self.published = None
        self.times = []
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05); continue
            G = dustsim.tick(self.G, self.P, self.sources)
            with self.lock:
                self.G, self.tick = G, self.tick + 1
                now = time.time()
                self.times = [t for t in self.times if now - t < 3] + [now]

    def read(self):
        with self.lock:
            if self.published == self.tick:
                return None
            self.published, G, t = self.tick, self.G, self.tick
        return Frame(t, state_to_grid(G, self.extra), self.params, self.sources, self.name)

    def status(self):
        rate = len(self.times) / max(self.times[-1] - self.times[0], 1e-6) if len(self.times) > 1 else 0
        state = 'paused' if self.paused else '%.1f ticks/s' % rate
        return '%s%s | %s | %d sources' % (self.name, ' (%s)' % self.label if self.label else '', state, len(self.sources))

    def stop(self):
        self.running = False


def model_from_map(name):
    """ModelSource for a map's .sav: its start grids, border values and the generators that are on at
    the start (mode 1 or 2). Parameters are the game's normal set (galileo.fluid_params)."""
    import compare_sim, sav, sav_sources
    T, So, W, C3, border = compare_sim.load_map(name)
    grid = np.zeros((8, N, N), np.uint32)
    for c, a in enumerate((T, So, W, C3)):
        grid[c] = a.astype(np.float32).view(np.uint32)
    _, u = sav.read_sav(os.path.join(compare_sim.SAVES, name + '.sav'))
    srcs = [dict(col=g['col'], row=g['row'], radius=g['radius'], rate=g['rate'], sediment=g['sediment'],
                 temperature=g['temperature']) for g in sav_sources.generators(u) if g['mode'] in (1, 2)]
    return ModelSource(grid, galileo.fluid_params(border=border), srcs, 0, name)


# ---------------------------------------------------------------- rendering
LIGHT = np.array([-0.6, -0.6, 0.55]) / np.linalg.norm([-0.6, -0.6, 0.55])
VIEWS = {'1': 'natural', '2': 'height', '3': 'water depth', '4': 'lava temperature', '5': 'sediment', '6': 'flow speed'}


def ramp_colours(x, stops):
    """Piecewise-linear colour map: stops = [(value, (r, g, b)), ...] in increasing value."""
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    out = np.empty(x.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(x, xs, cs[:, k])
    return out


def shade(height):
    gy, gx = np.gradient(height)
    L = LIGHT
    return np.clip((-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx * gx + gy * gy + 1.0), 0, 1) * 0.75 + 0.25


LAVA_STOPS = [(0, (40, 25, 20)), (12.5, (90, 20, 10)), (40, (170, 30, 10)), (90, (235, 90, 15)), (150, (255, 170, 40)), (255, (255, 245, 170))]


def render(fr, view):
    if view == 'natural':
        rock = np.array([118, 108, 98], np.float32); sand = np.array([214, 188, 138], np.float32)
        s = np.clip(fr.soft / 1.5, 0, 1)[..., None]
        col = (rock + (sand - rock) * s) * shade(fr.ground + fr.water + fr.lava)[..., None]
        d = fr.water
        t = np.exp(-d / 10.0)[..., None]
        wc = np.array([18, 52, 105], np.float32) * (1 - t) + np.array([70, 150, 185], np.float32) * t
        mud = np.clip(fr.sediment / (0.02 * d + 1e-6), 0, 1)[..., None] * 0.6
        wc += (np.array([150, 118, 80], np.float32) - wc) * mud
        a = (1 - np.exp(-d / 2.0))[..., None]
        col += (wc - col) * a
        lc = ramp_colours(fr.temperature, LAVA_STOPS)
        a = (1 - np.exp(-fr.lava / 0.5))[..., None]
        col += (lc - col) * a
    elif view == 'height':
        h = fr.ground
        lo, hi = np.percentile(h, 1), np.percentile(h, 99.5)
        col = ramp_colours((h - lo) / (hi - lo + 1e-6), [(0, (20, 40, 90)), (0.3, (60, 130, 80)), (0.6, (190, 170, 110)), (1, (250, 250, 250))])
        col *= shade(h)[..., None]
    elif view == 'water depth':
        col = ramp_colours(np.log1p(fr.water), [(0, (25, 25, 25)), (0.05, (40, 60, 90)), (1, (40, 120, 200)), (3, (120, 220, 255)), (5, (255, 255, 255))])
    elif view == 'lava temperature':
        col = np.where((fr.lava > 0)[..., None], ramp_colours(fr.temperature, LAVA_STOPS), 30.0)
        col = np.broadcast_to(col, fr.lava.shape + (3,)).astype(np.float32) * shade(fr.ground)[..., None]
    elif view == 'sediment':
        conc = fr.sediment / np.maximum(fr.water, 1e-6)
        col = ramp_colours(np.where(fr.water > 0.01, conc, -1), [(-1, (25, 25, 25)), (0, (30, 60, 110)), (0.2, (120, 110, 80)), (0.5, (200, 150, 70)), (1, (255, 230, 150))])
    else:  # flow speed
        h = np.maximum(fr.water + fr.lava, 1e-3)
        v = fr.flux.sum(0) / np.sqrt(h)
        col = ramp_colours(np.where(h > 0.01, v, 0), [(0, (20, 20, 30)), (0.2, (40, 70, 140)), (0.6, (60, 200, 200)), (1.5, (250, 250, 120)), (4, (255, 255, 255))])
    return np.clip(col, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- window
class App:
    def __init__(self, root, source, scale, flip, rotate):
        self.root, self.src, self.scale = root, source, scale
        self.live = source if isinstance(source, LiveGame) else None
        self.flip, self.rotate = flip, rotate % 4
        self.view, self.overlay = 'natural', True
        self.frame = None
        self.dirty = True
        self.mouse = None
        self.fps = []
        root.title('DustView')
        self.canvas = tk.Label(root, bg='black')
        self.canvas.pack()
        self.info = tk.Label(root, anchor='w', justify='left', font=('Consolas', 9), bg='#202020', fg='#e0e0e0')
        self.info.pack(fill='x')
        root.bind('<Key>', self.on_key)
        self.canvas.bind('<Motion>', self.on_move)
        self.canvas.bind('<Leave>', lambda e: self.set_mouse(None))
        self._make_index()
        self.tick()

    # orientation: display = flipud?(rot90(grid, k)); index maps translate display pixels to cells
    def orient(self, a):
        a = np.rot90(a, self.rotate)
        return a[::-1] if self.flip else a

    def _make_index(self):
        r, c = np.mgrid[0:N, 0:N]
        self.idx_r, self.idx_c = self.orient(r), self.orient(c)
        self.pos = {}
        disp = np.stack([self.idx_r, self.idx_c], -1)
        self.cell_to_disp = np.empty((N, N, 2), np.int32)
        ys, xs = np.mgrid[0:N, 0:N]
        self.cell_to_disp[disp[..., 0], disp[..., 1]] = np.stack([ys, xs], -1)

    def on_key(self, e):
        k = e.keysym.lower()
        if e.char in VIEWS:
            self.view = VIEWS[e.char]
        elif k == 'o':
            self.overlay = not self.overlay
        elif k == 'v':
            self.flip = not self.flip; self._make_index()
        elif k == 'r':
            self.rotate = (self.rotate + 1) % 4; self._make_index()
        elif k == 's':
            self.screenshot()
        elif k == 'm' and self.frame is not None and not isinstance(self.src, ModelSource):
            fr = self.frame
            self.src = ModelSource(fr.grid, fr.params, fr.sources, fr.tick, 'from live tick %d' % fr.tick)
        elif k == 'l' and isinstance(self.src, ModelSource):
            self.src.stop()
            try:
                self.live = self.live or LiveGame()
                self.src = self.live
                self.live.last_tick = None
            except SystemExit as e:
                self.info.configure(text=str(e))
        elif k == 'space' and isinstance(self.src, ModelSource):
            self.src.paused = not self.src.paused
        elif k == 'escape':
            self.root.destroy(); return
        self.dirty = True

    def on_move(self, e):
        self.set_mouse((min(N - 1, max(0, e.y // self.scale)), min(N - 1, max(0, e.x // self.scale))))

    def set_mouse(self, m):
        self.mouse = m
        self.update_info()

    def screenshot(self):
        if getattr(self, 'last_image', None) is not None:
            p = os.path.join(HERE, 'data', 'live', time.strftime('view_%Y%m%d_%H%M%S.png'))
            self.last_image.save(p)
            print('saved', p)

    def draw(self):
        fr = self.frame
        img = Image.fromarray(np.ascontiguousarray(self.orient(render(fr, self.view))), 'RGB')
        if self.scale != 1:
            img = img.resize((N * self.scale, N * self.scale), Image.NEAREST)
        if self.overlay and fr.sources:
            dr = ImageDraw.Draw(img)
            for s in fr.sources:
                y, x = self.cell_to_disp[int(s['row']), int(s['col'])]
                cx, cy, r = (x + 0.5) * self.scale, (y + 0.5) * self.scale, max(3, (s['radius'] + 1.5) * self.scale)
                colour = (255, 90, 40) if s['temperature'] else (255, 60, 220) if s['rate'] < 0 else (120, 230, 255)
                if s.get('buried'):
                    colour = (150, 150, 150)
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=2)
                dr.text((cx + r + 2, cy - 6), '%+.2f' % s['rate'], fill=colour)
        self.last_image = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)

    def update_info(self):
        fr = self.frame
        lines = ['%s | view: %s (1-6)   M model  L live  O overlay  V flip  R rotate  S screenshot' % (self.src.status(), self.view)]
        if fr is not None:
            fps = len(self.fps) / max(self.fps[-1] - self.fps[0], 1e-6) if len(self.fps) > 1 else 0
            border = fr.params[31] if fr.params is not None else float('nan')
            lines.append('tick %d | viewer %.1f fps | border water level %.2f | water %.0f, lava %.0f in total' % (
                fr.tick, fps, border, fr.water.sum(), fr.lava.sum()))
            if self.mouse is not None:
                r, c = self.idx_r[self.mouse], self.idx_c[self.mouse]
                if fr.lava[r, c] > 0:
                    fluid = 'lava %.2f at %.1f deg (editor %.0f)' % (fr.lava[r, c], fr.temperature[r, c], fr.temperature[r, c] * 8)
                elif fr.water[r, c] > 0:
                    fluid = 'water %.2f, sediment %.3f' % (fr.water[r, c], fr.sediment[r, c])
                else:
                    fluid = 'dry'
                lines.append('col %d row %d | ground %.2f (soft %.2f, hard %.2f) | %s | outflow L %.2f R %.2f U %.2f D %.2f' % (
                    c, r, fr.ground[r, c], fr.soft[r, c], fr.ground[r, c] - fr.soft[r, c], fluid, *fr.flux[:, r, c]))
        self.info.configure(text='\n'.join(lines))

    def tick(self):
        try:
            fr = self.src.read()
        except Exception as e:
            fr = None
            self.info.configure(text='read failed: %s' % e)
        if fr is not None:
            self.frame, self.dirty = fr, True
        if self.frame is not None and self.dirty:
            self.draw()
            self.dirty = False
            now = time.time()
            self.fps = [t for t in self.fps if now - t < 2] + [now]
        self.update_info()
        self.root.after(40, self.tick)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--flip', action='store_true', help='flip vertically (row 0 at the bottom)')
    ap.add_argument('--rotate', type=int, default=0, help='rotate by N x 90 degrees')
    ap.add_argument('--model', metavar='MAP', help='run the Python model from this map file instead of reading the game')
    a = ap.parse_args()
    root = tk.Tk()
    App(root, model_from_map(a.model) if a.model else LiveGame(), a.scale, a.flip, a.rotate)
    root.mainloop()


if __name__ == '__main__':
    main()
