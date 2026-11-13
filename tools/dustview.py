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
--model MAP runs the Python model from a map's .sav (e.g. ALPHA_Map11_Void) driven by its level script
(tools/level_schedule.py: sources switching on/off/moving, the tide); --villages "0:0,200:1,900:3" gives
the village counts the script checks. Waves and tsunamis are logged, not simulated. No game needed.

Keys
  1 natural   2 height   3 water depth   4 lava temperature   5 sediment   6 flow speed
  7 unexplained changes (live only): a background thread takes two consecutive game states, runs the
    game's own step (DLL via the simstep helper; Python if that fails) plus the known sources on the
    first and subtracts the result from the game's second state. What is left was done outside the
    simulation (creatures, volcanoes, waves, powers, script changes, unknown sources). Cyan = fluid
    gained, orange = fluid lost, yellow = ground raised, magenta = ground lowered; fades with a 1.5 s
    half-life. A few cells are always left (rounding, the small-vent timing).
  M  switch to the Python model, starting from the state on screen (sources as last found)
  L  switch back to the live game       Space  pause / resume the model
  O  overlays on/off: cyan spring, orange lava vent, violet water drain, dark red lava drain, grey = buried
     below 95% strength (labels show the amount applied per cell per tick, with % when reduced), white = RepulseFluid (village outposts: mode -1 pushes water, +1 lava; modelled)
  O  (toggle)          V  flip vertically     R  rotate 90 degrees
  D  terrain-change markers on/off (live only): terrain that changes in dry areas between frames,
     i.e. not caused by the simulation (moving dips, scripted terrain, powers). Magenta = lowered,
     yellow = raised; markers fade over 8 s; every spot is also logged to data/live/terrain_events_*.log
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
from live_sources import regions  # noqa: E402

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
        self.terrain_events = []        # (row, col, cells, total change) for terrain changed outside the sim
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
            self.refresh_sources()
            fr = Frame(m - 1, galileo.unpack(raw), params, self.sources, self.name)
            self._terrain_events(fr)
            return fr
        return None

    def _terrain_events(self, fr):
        """Terrain changes in dry areas since the previous frame. The simulation only changes terrain where
        fluid is (erosion, deposit, lava hardening) or next to it (lateral erosion, slumping into water),
        so a change where a cell and its 8 neighbours were dry in both frames comes from outside it."""
        fluid = (fr.water + fr.lava) > 1e-3
        prev = getattr(self, '_prev', None)
        self._prev = (fr.ground.copy(), fluid)
        if prev is None:
            return
        ground0, fluid0 = prev
        wet = fluid | fluid0
        near = wet.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= dustsim.shift(wet, dy, dx)
        dg = fr.ground - ground0
        mask = (np.abs(dg) > 0.004) & ~near
        mask[:33] = mask[479:] = False
        mask[:, :33] = mask[:, 479:] = False
        if mask.sum() > 5000:                       # level change / load: ignore
            return
        for ys, xs in regions(mask):
            fr.terrain_events.append((float(ys.mean()), float(xs.mean()), len(ys), float(dg[ys, xs].sum())))
        if fr.terrain_events:
            if not hasattr(self, '_elog'):
                self._elog = open(os.path.join(HERE, 'data', 'live', time.strftime('terrain_events_%Y%m%d_%H%M%S.log')), 'w', encoding='utf-8')
                self._elog.write('tick row col cells total_change (terrain changed in dry areas, outside the simulation)\n')
            for r, c, n, tot in fr.terrain_events:
                self._elog.write('%d %.1f %.1f %d %+.4f\n' % (fr.tick, r, c, n, tot))
            self._elog.flush()

    def enable_explain(self):
        """Start the background check for changes the simulation does not explain (view 7)."""
        if getattr(self, '_explain_on', False):
            return
        self._explain_on = True
        self.explained = None
        self.explain_note = 'unexplained: starting'
        threading.Thread(target=self._explain_loop, daemon=True).start()

    def _explain_loop(self):
        times = []
        while True:
            try:
                cap = live_check.capture_pair(self.p, self.base)
            except SystemExit:
                self.explain_note = 'unexplained: game paused'
                time.sleep(0.5)
                continue
            except Exception as e:
                self.explain_note = 'unexplained: %s' % e
                time.sleep(1)
                continue
            if not cap:
                continue
            n, A, B, prm = cap
            applied = list(getattr(self, 'entries', []))    # read right after the pair: same list as that tick
            GA, GB = dustsim.from_grid(A), dustsim.from_grid(B)
            if (np.abs(GA['W'] - GB['W']) > 1e-3).sum() > 100000:
                continue                                    # mistimed read
            Pa = np.frombuffer(prm[:224], np.float32).copy()
            try:
                X = dustsim.from_grid(galileo.run(A, Pa, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                                                  param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
                how = 'DLL'
            except Exception:
                X = dustsim.tick(GA, {k: float(v) for k, v in zip(galileo.PARAM_NAMES, Pa)}, [])
                how = 'python'
            dustsim.apply_modifiers(X, applied)
            # A torn read (a buffer copied while the game was writing it) shows as thousands of outflow
            # mismatches; normally ~300 (the village repulsion etc.), so skip such pairs.
            torn = int((GB['F'] != X['F']).any(0).sum())
            if torn > 3000:
                self.rejected = getattr(self, 'rejected', 0) + 1
                continue
            dT = GB['T'] - X['T']
            dF = np.abs(GB['W']) - np.abs(X['W'])
            for a in (dT, dF):
                a[:33] = a[479:] = 0
                a[:, :33] = a[:, 479:] = 0
                a[np.abs(a) < 1e-3] = 0
            now = time.time()
            times = [t for t in times if now - t < 3] + [now]
            cells = int(((dT != 0) | (dF != 0)).sum())
            self.explained = dict(tick=n, dT=dT, dF=dF, time=now, cells=cells, how=how)
            self.explain_note = 'unexplained: tick %d checked by %s, %d cells, %.1f checks/s, %d torn reads skipped' % (
                n, how, cells, len(times) / max(times[-1] - times[0], 1e-6) if len(times) > 1 else 0, getattr(self, 'rejected', 0))

    def ticks_per_second(self):
        if len(self.tick_times) < 2:
            return 0.0
        (t0, s0), (t1, s1) = self.tick_times[0], self.tick_times[-1]
        return (t1 - t0) / max(s1 - s0, 1e-6)

    def _source_loop(self):
        """Every ~10 s: find (once) the simulator's per-tick modifier list and cache the generator objects'
        addresses. The list itself is read every frame (refresh_sources), so script switches, burying,
        fade-in and Jellify show up at once."""
        delta = None
        while True:
            try:
                delta = live_generators.dustai_delta(self.p) if delta is None else delta
                gens = live_generators.generators(self.p, delta)
                self.gen_addrs = [int(g['address'], 16) for g in gens]
                if getattr(self, 'vec', None) is None:
                    self.source_note = 'looking for the source list...'
                    self.vec = live_generators.find_modifier_list(self.p, gens)
                if self.vec is None:
                    self.source_note = 'source list not found yet (needs a running, unburied source)'
                else:
                    self.refresh_sources()
            except Exception as e:                       # game closed, level changing, ...
                self.source_note = 'sources: %s' % e
            time.sleep(10)

    def refresh_sources(self, settle=0.01):
        """Read the modifier list now; pair each fluid entry with a generator object for its nominal rate,
        stored height and height mode (needed by the Python model's bury rule). Generators with identical
        settings are told apart by the effective amount (rate x activity x fade-in), then by stored height
        vs the ground at the entry's cell."""
        vec = getattr(self, 'vec', None)
        if vec is None:
            return
        entries = live_generators.read_modifier_list(self.p, vec, settle)
        if not entries:
            return
        gens = [g for g in (live_generators.read_generator(self.p, a) for a in getattr(self, 'gen_addrs', [])) if g and g['mode'] != 0]
        ground = self._prev[0] if getattr(self, '_prev', None) is not None else None
        sig = lambda d: (round(d['radius'], 2), round(d['sediment'], 3), round(d['temperature'], 1))
        free, out, other = list(gens), [], []
        for e in entries:
            if e['type'] != 0:
                other.append(e)
                continue
            src = dict(col=e['col'], row=e['row'], radius=e['radius'], rate=e['amount'], amount=e['amount'],
                       sediment=e['sediment'], temperature=e['temperature'])
            cands = [g for g in free if sig(g) == sig(e)]
            if cands:
                gz = float(ground[int(e['row']), int(e['col'])]) if ground is not None else 0.0
                eff = lambda g: g['rate'] * g['activity'] * (g['fade'] / g['fade_length'] if g['fade_length'] else 1)
                g = min(cands, key=lambda g: (round(abs(eff(g) - e['amount']), 3), abs(g['stored_z'] - gz)))
                free.remove(g)
                src.update(rate=g['rate'], activity=g['activity'], buried=g['activity'] < 1, height_mode=g['height_mode'],
                           z=g['stored_z'], bury_threshold=g['bury_threshold'])
            out.append(src)
        self.sources, self.other_modifiers, self.entries = out, other, entries
        self.source_note = '%d sources, %d other modifiers (live list)' % (len(out), len(other))

    def applied_sources(self):
        """The fluid entries as the game applies them this tick (effective amounts, no bury rule needed)."""
        return [dict(col=s['col'], row=s['row'], radius=s['radius'], rate=s['amount'], sediment=s['sediment'],
                     temperature=s['temperature']) for s in self.sources]

    def status(self):
        state = 'paused' if self.paused else '%.0f ticks/s' % self.ticks_per_second()
        extra = (' | ' + self.explain_note) if getattr(self, '_explain_on', False) else ''
        return '%s | %s | %s%s' % (self.name, state, self.source_note, extra)


class ModelSource(Source):
    """The Python model (dustsim.tick) stepping in a background thread as fast as it can."""
    name = 'python model'

    def __init__(self, grid, params, sources, tick=0, label='', repulsors=(), schedule=None):
        self.P = {n: float(v) for n, v in zip(galileo.PARAM_NAMES, np.asarray(params, np.float32))}
        self.params = np.asarray(params, np.float32)
        self.sources = [dict(s) for s in sources]
        self.other_modifiers = [dict(e) for e in repulsors if e.get('type') == 1]
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
        self.schedule = schedule                 # level_schedule.LevelSchedule: script-driven sources + tide
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05); continue
            sch = self.schedule
            if sch is not None:
                sch.tick(1 / 31, ground=self.G['T'])
                self.sources = sch.sources()
                self.P['BorderWaterSurfaceHeight'] = sch.border_level
            G = dustsim.tick(self.G, self.P, self.sources, self.other_modifiers)
            if sch is not None:
                sch.write_back(self.sources)
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
        extra = ''
        if self.schedule is not None:
            sch = self.schedule
            last = sch.events[-1] if sch.events else None
            extra = ' | level time %.0fs, border %.2f%s' % (sch.time, sch.border_level, (' | last script event at %.0fs: %s' % (last[0], last[1][:70])) if last else '')
        return '%s%s | %s | %d sources, %d repulsors%s' % (self.name, ' (%s)' % self.label if self.label else '', state, len(self.sources), len(self.other_modifiers), extra)

    def stop(self):
        self.running = False


def model_from_map(name, villages=None):
    """ModelSource for a map's .sav: its start grids, border values and the generators that are on at
    the start (mode 1 or 2). Parameters are the game's normal set (galileo.fluid_params)."""
    import compare_sim, sav, sav_sources
    T, So, W, C3, border = compare_sim.load_map(name)
    grid = np.zeros((8, N, N), np.uint32)
    for c, a in enumerate((T, So, W, C3)):
        grid[c] = a.astype(np.float32).view(np.uint32)
    _, u = sav.read_sav(os.path.join(compare_sim.SAVES, name + '.sav'))
    import level_schedule
    sch = level_schedule.LevelSchedule(name, villages=villages, border_level=border[0])
    return ModelSource(grid, galileo.fluid_params(border=border), sch.sources(), 0, name, schedule=sch)


# ---------------------------------------------------------------- rendering
LIGHT = np.array([-0.6, -0.6, 0.55]) / np.linalg.norm([-0.6, -0.6, 0.55])
VIEWS = {'1': 'natural', '2': 'height', '3': 'water depth', '4': 'lava temperature', '5': 'sediment', '6': 'flow speed',
         '7': 'unexplained changes'}


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


UNEXPLAINED = (('f_up', (0, 220, 255)), ('f_down', (255, 140, 0)), ('t_up', (255, 240, 0)), ('t_down', (255, 0, 255)))


def render_unexplained(fr, acc):
    col = render(fr, 'natural').astype(np.float32).mean(-1, keepdims=True).repeat(3, -1) * 0.4
    if acc:
        for name, colour in UNEXPLAINED:
            a = np.clip(np.log10(np.maximum(acc[name], 1e-3) / 1e-3) / 2.5, 0, 1)[..., None]   # 0.001 .. ~0.3
            col += (np.array(colour, np.float32) - col) * a
    return np.clip(col, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- window
class App:
    def __init__(self, root, source, scale, flip, rotate):
        self.root, self.src, self.scale = root, source, scale
        self.live = source if isinstance(source, LiveGame) else None
        self.flip, self.rotate = flip, rotate % 4
        self.view, self.overlay = 'natural', True
        self.show_events, self.events = True, []        # (time, row, col, cells, total)
        self.acc, self.acc_time, self.acc_tick = {}, time.time(), None   # view 7 accumulators
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

    def current_sources(self):
        """The source's current list (a paused game keeps returning no new frames, so the frame's own copy
        can predate the background source search)."""
        srcs = getattr(self.src, 'sources', None)
        return list(srcs) if srcs is not None else (list(self.frame.sources) if self.frame is not None else [])

    def on_key(self, e):
        k = e.keysym.lower()
        if e.char in VIEWS:
            self.view = VIEWS[e.char]
            if self.view == 'unexplained changes' and hasattr(self.src, 'enable_explain'):
                self.src.enable_explain()
        elif k == 'o':
            self.overlay = not self.overlay
        elif k == 'd':
            self.show_events = not self.show_events
        elif k == 'v':
            self.flip = not self.flip; self._make_index()
        elif k == 'r':
            self.rotate = (self.rotate + 1) % 4; self._make_index()
        elif k == 's':
            self.screenshot()
        elif k == 'm' and self.frame is not None and not isinstance(self.src, ModelSource):
            fr = self.frame
            srcs = self.current_sources()
            if not srcs and 'looking' in getattr(self.src, 'source_note', ''):
                self.info.configure(text='sources not found yet (the search takes a few seconds); press M again shortly')
                return
            self.src = ModelSource(fr.grid, fr.params, srcs, fr.tick, 'from live tick %d' % fr.tick,
                                   getattr(self.src, 'other_modifiers', []))
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
        rgb = render_unexplained(fr, self.acc) if self.view == 'unexplained changes' else render(fr, self.view)
        img = Image.fromarray(np.ascontiguousarray(self.orient(rgb)), 'RGB')
        if self.scale != 1:
            img = img.resize((N * self.scale, N * self.scale), Image.NEAREST)
        sources = self.current_sources()
        if self.overlay and sources:
            dr = ImageDraw.Draw(img)
            for s in sources:
                y, x = self.cell_to_disp[int(s['row']), int(s['col'])]
                cx, cy, r = (x + 0.5) * self.scale, (y + 0.5) * self.scale, max(3, (s['radius'] + 1.5) * self.scale)
                if s['rate'] < 0:          # drains: dark red = lava drain, violet = water drain
                    colour = (200, 40, 60) if s['temperature'] else (170, 90, 255)
                else:                      # orange = lava vent, cyan = spring
                    colour = (255, 140, 30) if s['temperature'] else (120, 230, 255)
                act = s.get('activity', 1.0)
                if act < 0.95:                         # noticeably buried: grey
                    colour = (150, 150, 150)
                amount = s.get('amount', s['rate'] * act)
                label = '%+.2f' % amount + (' (%d%%)' % round(act * 100) if act < 0.99 else '')
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=2)
                dr.text((cx + r + 2, cy - 6), label, fill=colour)
        if self.show_events and self.events:
            dr = ImageDraw.Draw(img)
            now = time.time()
            for t, row, col, n, tot in self.events:
                y, x = self.cell_to_disp[int(round(row)), int(round(col))]
                cx, cy = (x + 0.5) * self.scale, (y + 0.5) * self.scale
                f = max(0.25, 1 - (now - t) / 8)
                colour = tuple(int(v * f) for v in ((255, 0, 255) if tot < 0 else (255, 230, 0)))
                r = 4 * self.scale + n ** 0.5 * self.scale
                dr.line([cx - r, cy, cx + r, cy], fill=colour, width=2)
                dr.line([cx, cy - r, cx, cy + r], fill=colour, width=2)
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=1)
        others = getattr(self.src, 'other_modifiers', []) if self.overlay else []
        if others:
            dr = ImageDraw.Draw(img)
            for e in others:
                y, x = self.cell_to_disp[int(e['row']), int(e['col'])]
                cx, cy, r = (x + 0.5) * self.scale, (y + 0.5) * self.scale, e['radius'] * self.scale
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(235, 235, 235), width=1)
                dr.text((cx + r + 2, cy - 6), 'mod %+g/%+g' % (e['amount'], e['f5']), fill=(235, 235, 235))
        self.last_image = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)

    def update_info(self):
        fr = self.frame
        lines = ['%s | view: %s (1-6)   M model  L live  O overlay  V flip  R rotate  S screenshot' % (self.src.status(), self.view)]
        if fr is not None:
            fps = len(self.fps) / max(self.fps[-1] - self.fps[0], 1e-6) if len(self.fps) > 1 else 0
            border = fr.params[31] if fr.params is not None else float('nan')
            recent = [e for e in self.events if time.time() - e[0] < 8]
            lines.append('tick %d | viewer %.1f fps | border water level %.2f | water %.0f, lava %.0f in total | terrain changes outside the sim (8 s): %d lowered, %d raised (D)' % (
                fr.tick, fps, border, fr.water.sum(), fr.lava.sum(), sum(e[4] < 0 for e in recent), sum(e[4] > 0 for e in recent)))
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
                ex = getattr(self.src, 'explained', None)
                if self.view == 'unexplained changes' and ex is not None:
                    lines.append('unexplained at this cell (tick %d): ground %+.4f, fluid %+.4f' % (ex['tick'], ex['dT'][r, c], ex['dF'][r, c]))
        self.info.configure(text='\n'.join(lines))

    def tick(self):
        try:
            fr = self.src.read()
        except Exception as e:
            fr = None
            self.info.configure(text='read failed: %s' % e)
        if fr is not None:
            self.frame, self.dirty = fr, True
            now = time.time()
            self.events = [e for e in self.events if now - e[0] < 8] + [(now,) + e for e in fr.terrain_events]
        ex = getattr(self.src, 'explained', None)
        if self.view == 'unexplained changes':
            now = time.time()
            decay = 0.5 ** ((now - self.acc_time) / 1.5)
            self.acc_time = now
            new = ex if ex is not None and ex['tick'] != self.acc_tick else None
            if new is not None:
                self.acc_tick = new['tick']
            parts = {'f_up': lambda e: np.maximum(e['dF'], 0), 'f_down': lambda e: np.maximum(-e['dF'], 0),
                     't_up': lambda e: np.maximum(e['dT'], 0), 't_down': lambda e: np.maximum(-e['dT'], 0)}
            for k, f in parts.items():
                cur = self.acc.get(k, np.zeros((N, N), np.float32)) * decay
                self.acc[k] = np.maximum(cur, f(new)) if new is not None else cur
            self.dirty = self.dirty or new is not None or any(a.max() > 1e-3 for a in self.acc.values())
        if id(getattr(self.src, 'sources', None)) != getattr(self, '_drawn_sources', None):
            self._drawn_sources = id(getattr(self.src, 'sources', None))
            self.dirty = True
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
    ap.add_argument('--villages', metavar='T:N,...', help='with --model: villages built over time for the level script, e.g. "0:0,200:1,900:3"')
    a = ap.parse_args()
    root = tk.Tk()
    import level_schedule
    vil = level_schedule.parse_villages(a.villages) if a.villages else None
    App(root, model_from_map(a.model, vil) if a.model else LiveGame(), a.scale, a.flip, a.rotate)
    root.mainloop()


if __name__ == '__main__':
    main()
