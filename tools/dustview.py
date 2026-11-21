"""DustView: live viewer for From Dust's simulation grid (read-only), first version.

Shows the running game's terrain, water and lava as a top-down map, refreshed several times a second,
with the level's springs / lava vents / drains drawn on top and a readout of the cell under the mouse.
The data source is pluggable (`Source.read()` returns a Frame): `LiveGame` reads the game's memory;
`ModelSource` runs the Python re-implementation (dustsim.tick: border ring, step, springs / vents /
drains) in a background thread, either taking over the live game's current state (key M) or starting
from a map file (--model MAP). The game's DLL can be added the same way, and a comparison mode can then
diff two sources.

Run with the project venv while a level is loaded:
  .venv/Scripts/python tools/dustview.py [--scale 2] [--no-flip] [--rotate N] [--model MAP]
(or dustview.bat with the same options). The map is shown flipped vertically by default (row 0 at the
bottom), which matches the game according to play; --no-flip shows grid row 0 at the top.
--model MAP runs the Python model from a map's .sav (e.g. ALPHA_Map11_Void) driven by its level script
(tools/level_schedule.py: sources switching on/off/moving, the tide); --villages "0:0,200:1,900:3" gives
the village counts the script checks. Waves and tsunamis drive the edge lines (wave table). No game needed.

Control panel (right; tools/dustview_panel.py): source status with Live game / Continue in model, then
tabs Map (pick a level, then start it fresh or load one of its saves from any profile, or open a .sav), Model (pause, step,
speed, villages) and View (views, overlays, flip / rotate, zoom, screenshot). Without a running game the
viewer opens empty. The keys below do the same; the panel follows them.

Views (panel only): natural, height, water depth, lava temperature, sediment, flow speed, unexplained
  changes, engine texture (a dropdown of the engine's map-sized textures: the per-frame water upload,
  WaterParams with / without the edge fade, UV drag, edge masks, the level's colmap, grid layers incl. the
  raw vegetation / fire components; tools/engine_textures.py).
  Unexplained changes (live only): a background thread takes two consecutive game states, runs the
    game's own step (DLL via the simstep helper; Python if that fails) plus the known sources on the
    first and subtracts the result from the game's second state. What is left was done outside the
    simulation (creatures, volcanoes, waves, powers, script changes, unknown sources). Cyan = fluid
    gained, orange = fluid lost, yellow = ground raised, magenta = ground lowered; fades with a 1.5 s
    half-life. A few cells are always left (rounding, the small-vent timing).
  M  switch to the Python model, starting from the state on screen (sources as last found)
  C  compare: run the model alongside the live game, tick for tick, with the game's own per-tick list
     (again: re-sync the model to the game); G cycles game -> model -> differences (view 'model vs game':
     model minus game (cyan / orange fluid more / less, yellow /
     magenta ground higher / lower)
  L  switch to the live game (found afresh)    Space  pause / resume the model    N  one model tick
  T  model speed: 1x the game's 31 ticks/s (default), 2x, 4x, as fast as possible (the level script's
     clock runs per tick, so tides and events speed up with it). The model step is compiled with numba
     (tools/dustsim_nb.py, bit-identical to dustsim.step_numpy): a tick takes ~7-11 ms.
  Click a totem (model mode) to build / remove its village. Villages count for the level script;
     repulsion comes from the village's learned songs (repel water: radius 26, repel lava: 23.33), not
     from the totem; a village keeps the songs it has in the save / live game. Totems: gold = village. A level started with
     --model or from the panel has no villages (as shipped); --progress auto|PATH takes them from a save.
     Live: songs come from the song objects (the per-tick list has a song only while fluid is near it).
  O  overlays on/off: cyan spring, orange lava vent, violet water drain, dark red lava drain, grey = buried
     below 95% strength (labels show the amount applied per cell per tick, with % when reduced), white = RepulseFluid (village outposts: mode -1 pushes water, +1 lava; modelled)
  O  (toggle)          V  flip vertically     R  rotate 90 degrees
  F  flow texture on/off (natural view): the game's water rendering rules and constants as logged from
     the running game (tools/flowtex.py): ripple + foam textures dragged along the net outflow, foam =
     blurred 2 x |velocity|.
  D  terrain-change markers on/off (live only): terrain that changes in dry areas between frames,
     i.e. not caused by the simulation (moving dips, scripted terrain, powers). Magenta = lowered,
     yellow = raised; markers fade over 8 s; every spot is also logged to data/live/terrain_events_*.log
  Edit tab (model mode): tools inspect (click totems), water, lava, sand, rock; left button adds, right
     button removes, [ / ] brush size. Uses the game's own modifiers (Add/RemoveWater,
     Add/RemoveLava at temperature 255, SoftGround, HardGround); the fixed border band is not edited.
  S  save a screenshot to data/live/    Esc quit
Mouse wheel: zoom (1x-8x) about the cursor. Middle-drag (or left-drag with the inspect tool): pan.
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
try:
    import viewfast                     # compiled redraw helpers (numba)
except ImportError:
    viewfast = None

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
        if viewfast is not None:        # water, lava, sediment, temperature, flux in one compiled pass
            self.water, self.lava, self.sediment, self.temperature, self.flux = viewfast.frame_fields(
                np.ascontiguousarray(grid, np.uint32))
        else:
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
    if viewfast is not None:
        a = lambda k: np.ascontiguousarray(G[k], np.float32)
        viewfast.pack_grid(a('T'), a('soft'), a('W'), a('sed'), np.ascontiguousarray(G['F'], np.int32), grid)
        return grid
    for c, k in enumerate(('T', 'soft', 'W', 'sed')):
        grid[c] = np.asarray(G[k], np.float32).view(np.uint32)
    F = G['F'].astype(np.uint32)
    grid[4] = F[0] | (F[1] << 16)
    grid[5] = F[2] | (F[3] << 16)
    return grid


# Fluid repulsion at villages comes from the village's knowledge, not from the totem: a "repel water"
# object (entity 0x20400009 ENTITY_KNOWLEDGE_I) adds RepulseFluid mode -1 with radius trunc(40 x 2/3) = 26,
# a "repel lava" object (0x2040000A ENTITY_KNOWLEDGE_J) mode +1 with radius 35 x 2/3 = 23.33 (dustai
# 0x10134AC0 / 0x10135210). The object exists only while fluid threatens the village (logged live: it
# appears when water arrives, with a pulse, and is gone once the water is diverted), so saves and the live
# song scan only show the ones active at that moment. Assumed: every village repels water (seen on every
# village of Maps 3 and 10; possibly not on Map 2, not checked further); repel lava is per village. Totem
# dicts carry built, water, lava (radius, 0 = none).
# The model applies a known repel every tick: the same as the game's on-demand one except for the short
# window before it triggers and the pulse (TODO in findings/water-lava-sim.md).
SONG_RADIUS = {'water': 26.0, 'lava': 35 * 2 / 3}
SONG_MODE = {'water': -1, 'lava': +1}
def with_knowledge(totems, level=None):
    """Give every totem the repel-water knowledge (assumption: all villages repel water; the exact rule
    is in the game code, not read yet). Lava repel stays as found. Returns the totems (changed in place)."""
    for t in totems:
        if not t.get('water'):
            t['water'] = SONG_RADIUS['water']
    return totems


def totem_repulsors(totems, extra=()):
    """RepulseFluid entries for the built totems' songs, plus `extra` entries that belong to no totem."""
    out = [dict(e) for e in extra]
    for t in totems:
        if not t.get('built'):
            continue
        for kind in ('water', 'lava'):
            if t.get(kind):
                out.append(dict(type=1, col=t['col'], row=t['row'], radius=t[kind], amount=1.0,
                                f5=SONG_MODE[kind], totem=t['id']))
    return out


def songs_from_modifiers(totems, entries, reach=8, mark_built=False):
    """Give each totem the repel songs seen in a live modifier list (type-1 entries within `reach` cells
    of it). mark_built: a totem with a song has a village (the song objects only exist for villages).
    Returns the type-1 entries that belong to no totem."""
    left = []
    for t in totems:
        t.setdefault('water', 0.0)
        t.setdefault('lava', 0.0)
    for e in entries:
        if e.get('type') != 1:
            continue
        near = [t for t in totems if (e['col'] - t['col']) ** 2 + (e['row'] - t['row']) ** 2 <= reach ** 2]
        if not near:
            left.append(e)
            continue
        t = min(near, key=lambda t: (e['col'] - t['col']) ** 2 + (e['row'] - t['row']) ** 2)
        t['water' if e['f5'] < 0 else 'lava'] = e['radius']
        if mark_built:
            t['built'] = True
    return left


_LEVEL_TOTEMS = None


def live_repulsors(src):
    """A live game's RepulseFluid entries: the village song objects, every repel entry seen in the per-tick
    list since connecting (LiveGame.seen_repulsors), and the list's other type-1 entries, without
    duplicates. The game lists a repel only while fluid is near it (seen in play: a village repels water
    only while water touches it), so the current list alone makes repels blink in and out; and the song
    reader does not find every repel (Map 10: only p0's lava song, while p1 repels water in the list)."""
    cands = [dict(s) for s in (getattr(src, 'songs', None) or [])]
    cands += [dict(e) for e in (getattr(src, 'seen_repulsors', None) or {}).values()]
    cands += [dict(e) for e in (getattr(src, 'other_modifiers', None) or [])]
    same = lambda a, b: (abs(a['col'] - b['col']) < 1.5 and abs(a['row'] - b['row']) < 1.5
                         and (a['f5'] < 0) == (b['f5'] < 0))
    out = []
    for e in cands:
        if e.get('type') == 1 and any(o.get('type') == 1 and same(o, e) for o in out):
            continue
        out.append(e)
    return out


def level_number(src):
    """Story level number of what a source shows (or None): from a model's map name (ALPHA_Map8_...)
    or by matching its totems' handle IDs with the level files' totems (data/level_scripts)."""
    import re
    m = re.search(r'Map(\d+)_', getattr(src, 'label', '') or '')
    if m:
        return int(m.group(1))
    tots = getattr(src, 'totems', None)
    if not tots:
        return None
    global _LEVEL_TOTEMS
    if _LEVEL_TOTEMS is None:
        _LEVEL_TOTEMS = {}
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'level_scripts')
        for fn in os.listdir(d):
            mm = re.match(r'ALPHA_Map(\d+)_', fn)
            if not mm:
                continue
            txt = open(os.path.join(d, fn), encoding='utf-8').read()
            key = frozenset(re.findall(r'#\d+\s+col\s+\d+\s+row\s+\d+ \| power \d+ \| id ([0-9A-F]+)', txt))   # totem IDs (totems can move)
            if key:
                _LEVEL_TOTEMS[key] = int(mm.group(1))
    return _LEVEL_TOTEMS.get(frozenset(str(t.get('id', '')).upper() for t in tots))


def totem_label(t):
    songs = '+'.join(k for k in ('water', 'lava') if t.get(k))
    return 'totem p%d%s%s' % (t.get('power', -1), ' village' if t.get('built') else '',
                              (' repels ' + songs) if t.get('built') and songs else '')


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
        self.running = True
        threading.Thread(target=self._source_loop, daemon=True).start()
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._list_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _counter(self):
        return struct.unpack('<I', self.p.read(self.base + 0x92C, 4))[0]

    ROW = (N // 16) * 8 * 16 * 4                  # bytes per grid row in the game's packed layout
    PROBE_ROWS = (40, 136, 232, 328, 424, 472)

    def _capture_loop(self):
        """Reads each new grid right when the tick counter moves (measured 2026-11-18 on Map 4: at that
        moment buffer (m - 1) & 1 is complete, sources included; the game's next step starts overwriting it
        before the counter moves again, from ~4 ms later, usually ~25-60 ms). The display loop used to read
        at a random point in the tick and sometimes got a half-written next step (a spring's +amount
        missing, one-frame flicker). After the read a few rows are read again: if they changed, the next
        step had started during the read and the frame is dropped."""
        last, last_base, t_old = None, None, 0.0
        while self.running:
            try:
                base = self.base
                if base != last_base:
                    last, last_base = None, base
                m = struct.unpack('<I', self.p.read(base + 0x92C, 4))[0]
                now = time.perf_counter()
                if m == last:
                    t_old = now                          # last time the old value was seen
                    time.sleep(0.0005)
                    continue
                late = last is not None and now - t_old > 0.002   # change noticed late (other threads busy):
                last = m                                 # the buffer may already be overwritten, skip it
                if late and getattr(self, '_captured', None) is not None:
                    self.cap_late = getattr(self, 'cap_late', 0) + 1
                    continue
                bufs = struct.unpack('<2I', self.p.read(base + 0x950, 8))
                addr = bufs[(m - 1) & 1]
                R = self.ROW
                before = [self.p.read(addr + y * R, R) for y in self.PROBE_ROWS]
                raw = self.p.read(addr, GRID_BYTES)
                params = np.frombuffer(self.p.read(base + 0x20, 224), np.float32).copy()
                ok = raw is not None and len(raw) == GRID_BYTES and self._counter() == m
                if ok:                                   # sample rows the same before, in and after the read
                    ok = all(self.p.read(addr + y * R, R) == raw[y * R:(y + 1) * R] == b0
                             for y, b0 in zip(self.PROBE_ROWS, before))
                if ok:
                    self._captured = (m, raw, params)
                else:
                    self.cap_dropped = getattr(self, 'cap_dropped', 0) + 1
            except Exception:
                time.sleep(0.05)

    def read(self):
        cap = getattr(self, '_captured', None)
        if cap is None:
            return None
        m, raw, params = cap
        if self.last_tick is not None and m < self.last_tick:   # tick counter went back: a map (re)loaded
            self._relook = True
            self.source_note = 'map loaded: looking for the source list...'
        if m == self.last_tick:
            self.paused = bool(self.tick_times) and time.time() - self.tick_times[-1][1] > 0.5   # no new tick for 0.5 s
            return None
        self.last_tick = m
        self.paused = False
        now = time.time()
        self.tick_times = [(t, s) for t, s in self.tick_times if now - s < 3] + [(m, now)]
        self.refresh_sources()
        fr = Frame(m - 1, galileo.unpack(raw), params, self.sources, self.name)
        self._terrain_events(fr)
        return fr

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
        while self.running:
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
        """Every second until found, then every ~10 s (and at once after a map loads: tick counter going back):
        find the simulator's per-tick modifier list and cache the generator objects'
        addresses. The list itself is read every frame (refresh_sources), so script switches, burying,
        fade-in and Jellify show up at once."""
        delta = None
        while self.running:
            try:
                self._check_sim()
                if getattr(self, 'vec', None) is None:           # remembered address: no scan needed
                    self.vec = live_generators.known_modifier_list(self.p)
                    if self.vec is not None:
                        self.refresh_sources(paused_check=True)
                delta = live_generators.dustai_delta(self.p) if delta is None else delta
                gens = live_generators.generators(self.p, delta)
                self.gen_addrs = [int(g['address'], 16) for g in gens]
                try:
                    self.totems = live_generators.totems(self.p, delta)
                except Exception:
                    pass
                if getattr(self, 'vec', None) is None:
                    self.source_note = 'looking for the source list...'
                    self.vec = live_generators.known_modifier_list(self.p)        # remembered address first
                    if self.vec is None:
                        self.vec = live_generators.find_modifier_list(self.p, gens)
                        if self.vec is not None:
                            live_generators.remember_modifier_list(self.vec)
                try:                                     # village songs, readable without the modifier list
                    self.songs = live_generators.songs(self.p, delta)
                except Exception:
                    self.songs = []
                if self.vec is None:
                    self.source_note = 'source list not found yet (needs a running, unburied source)'
                    self.other_modifiers = [dict(s) for s in self.songs]   # repulsions from the song objects
                else:
                    self.refresh_sources(paused_check=True)
            except Exception as e:                       # game closed, level changing, ...
                self.source_note = 'sources: %s' % e
            for _ in range(10 if getattr(self, 'vec', None) is not None else 1):   # retry every second until found
                if getattr(self, '_relook', False):      # a new map loaded: search again at once
                    break
                time.sleep(1)
            if getattr(self, '_relook', False):
                self._relook = False
                self.vec, self.sources, self.gen_addrs, self._latest_entries = None, [], [], None
                self.__dict__.pop('seen_repulsors', None)   # the old level's repels
                self._last_count = None

    def _list_loop(self):
        """The game fills the per-tick modifier list during a tick and empties it afterwards, so a read at a
        random moment often finds it empty (sources then took 10+ s to appear). Check every 2 ms and keep
        the last non-empty list with its time (also covers a paused game: the last tick's list stays)."""
        while self.running:
            vec = getattr(self, 'vec', None)
            if vec is not None:
                try:
                    n = struct.unpack('<I', self.p.read(vec + 4, 4))[0]
                    if 0 < n <= 4096:
                        es = live_generators.read_modifier_list(self.p, vec, 0.002)
                        if es:
                            self._latest_entries = (time.time(), es)
                except Exception:
                    pass
            time.sleep(0.002)

    def _check_sim(self):
        """After a level change the game may use a new simulator object: when no tick has arrived for 3 s
        (paused, loading or a new level), look for it again and switch to it if it moved."""
        last = self.tick_times[-1][1] if self.tick_times else getattr(self, '_t_start', None)
        if last is None:
            self._t_start = time.time()
            return
        if time.time() - last < 3 or time.time() - getattr(self, '_sim_checked', 0) < 3:
            return
        self._sim_checked = time.time()
        try:
            base = live_check.find_sim(self.p)
        except SystemExit:
            return
        if base != self.base:
            self.base, self.last_tick, self.tick_times = base, None, []
            self._relook = True
            self.sim_changes = getattr(self, 'sim_changes', 0) + 1
            self.source_note = 'new level: looking for the source list...'

    def refresh_sources(self, settle=0.01, paused_check=False):
        """Read the modifier list now; pair each fluid entry with a generator object for its nominal rate,
        stored height and height mode (needed by the Python model's bury rule). Generators with identical
        settings are told apart by the effective amount (rate x activity x fade-in), then by stored height
        vs the ground at the entry's cell."""
        vec = getattr(self, 'vec', None)
        if vec is None:
            return
        lat = getattr(self, '_latest_entries', None)
        if lat is not None and (time.time() - lat[0] < 0.5 or self.paused):
            entries = lat[1]                             # caught by _list_loop
        else:
            entries = live_generators.read_modifier_list(self.p, vec, settle)
        from_buffer = False
        if entries:
            self._last_count = len(entries)              # the real list length, seen while the game ran
        elif paused_check:                               # paused game: the list is emptied after each tick
            m = self._counter()
            time.sleep(0.05)
            if self._counter() == m:                     # no tick in 50 ms: paused, so the buffer holds the last tick's list
                entries = live_generators.read_modifier_list(self.p, vec, 0, last_tick=True)
                if getattr(self, '_last_count', None):
                    entries = entries[:self._last_count]
                else:                                    # length unknown: older, longer lists leave stale entries behind
                    from_buffer = True                   # (seen: 4,080 on Story 10), so keep only generator-backed ones
        if not entries:
            return
        gens = [g for g in (live_generators.read_generator(self.p, a) for a in getattr(self, 'gen_addrs', [])) if g and g['mode'] != 0]
        ground = self._prev[0] if getattr(self, '_prev', None) is not None else None
        sig = lambda d: (round(d['radius'], 2), round(d['sediment'], 3), round(d['temperature'], 1))
        free, out, other = list(gens), [], []
        for e in entries:
            if e['type'] != 0:
                if not from_buffer:
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
            elif from_buffer:
                continue                                 # no generator left for it: stale
            out.append(src)
        if from_buffer:
            other = [dict(s) for s in getattr(self, 'songs', [])]   # repels from the song objects instead
            if not out:
                return
        self.sources, self.other_modifiers, self.entries = out, other, entries
        seen = self.__dict__.setdefault('seen_repulsors', {})   # repels stay known once seen (live_repulsors)
        for e in other:
            if e.get('type') == 1:
                seen[(round(e['col']), round(e['row']), e['f5'] < 0)] = dict(e)
        self.source_note = '%d sources, %d other modifiers (live list)' % (len(out), len(other))

    def applied_sources(self):
        """The fluid entries as the game applies them this tick (effective amounts, no bury rule needed)."""
        return [dict(col=s['col'], row=s['row'], radius=s['radius'], rate=s['amount'], sediment=s['sediment'],
                     temperature=s['temperature']) for s in self.sources]

    def status(self):
        state = 'paused' if self.paused else '%.0f ticks/s' % self.ticks_per_second()
        extra = (' | ' + self.explain_note) if getattr(self, '_explain_on', False) else ''
        return '%s | %s | %s%s' % (self.name, state, self.source_note, extra)


class NoSource(Source):
    """Nothing loaded yet (no game running): pick a map in the control panel."""
    name = 'nothing loaded'

    def __init__(self, note=''):
        self.note = note

    def read(self):
        return None

    def status(self):
        return 'nothing loaded%s: start a level or a save from the panel' % (' (%s)' % self.note if self.note else '')


class ModelSource(Source):
    """The Python model (dustsim.tick) stepping in a background thread as fast as it can."""
    name = 'python model'

    def __init__(self, grid, params, sources, tick=0, label='', repulsors=(), schedule=None, totems=None):
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
        self.pending, self.edits = [], 0         # queued edits (functions on the state dict), edits applied
        self.schedule = schedule                 # level_schedule.LevelSchedule: script-driven sources + tide
        self.totems = [dict(t) for t in totems] if totems else []
        self.loose_repulsors = list(self.other_modifiers)   # repulsors that belong to no totem: kept as they are
        if self.totems:                          # villages + their songs define the repulsors and the village count
            self.loose_repulsors = songs_from_modifiers(self.totems, self.other_modifiers, mark_built=True)
            with_knowledge(self.totems, level_number(self))
            self.other_modifiers = totem_repulsors(self.totems, self.loose_repulsors)
        if schedule is not None:
            schedule.villages = lambda t: sum(1 for x in self.totems if x.get('built'))
        threading.Thread(target=self._loop, daemon=True).start()

    GAME_RATE = 31.0                             # the game's ticks per second
    SPEEDS = (1, 2, 4, 0)                        # key T cycles; 0 = as fast as possible
    speed = 1
    step_req = 0                                 # ticks to run while paused (key N / Step button)

    def _loop(self):
        try:
            self._run_loop()
        finally:
            if self.veg is not None:
                self.veg.close()

    def _run_loop(self):
        due = time.perf_counter()
        while self.running:
            self._apply_edits()
            if self.paused and not self.step_req:
                time.sleep(0.01); due = time.perf_counter(); continue
            speed = self.speed                       # read once: the UI thread can change it meanwhile
            if self.paused:                          # single step while paused (no pacing)
                self.step_req -= 1
            elif speed:                              # pace to speed x the game's rate
                now = time.perf_counter()
                if due - now > 0:
                    time.sleep(due - now)
                due = max(due, time.perf_counter() - 0.25) + 1 / (self.GAME_RATE * speed)
            sch = self.schedule
            srcs = self.sources
            if sch is not None:
                sch.tick(1 / 31, ground=self.G['T'])
                srcs = sch.sources()                 # fresh dicts: 'activity' (burial) is set by the tick,
                self.P['BorderWaterSurfaceHeight'] = sch.border_level   # so publish them only after it
            G = dustsim.tick(self.G, self.P, srcs, self.other_modifiers)
            if sch is not None:
                sch.write_back(srcs)
                if sch.profiles:
                    dustsim.apply_edges(G, sch.border_level, sch.profiles)
            extra = self._vegetation_tick(G)
            with self.lock:
                self.G, self.tick, self.sources = G, self.tick + 1, srcs
                if extra is not None:
                    self.extra = extra
                now = time.time()
                self.times = [t for t in self.times if now - t < 3] + [now]

    def edit(self, fn):
        """Queue fn(G) (in-place change of the state dict); the model thread applies it before its next
        tick, or within ~10 ms while paused."""
        with self.lock:
            self.pending.append(fn)

    veg = None                                   # galileo.Resident running the game's vegetation step (job 4)
    veg_failed = False

    def _vegetation_tick(self, G):
        """One step of the game's own vegetation / fire job on the new state (ground, sand, water from G;
        plants from self.extra); returns a copy of self.extra with the new components 6 / 7, or None if
        vegetation is off or the helper is not available. Parameters: vegsim (species set per level:
        vegsim.level_set; ParamI and the propagate table partly guessed, findings/vegetation-fire-layers.md)."""
        if not VEGETATION_STEP or self.veg_failed:
            return None
        try:
            if self.veg is None:
                import vegsim
                vset = vegsim.level_set(getattr(self, 'level_name', '') or '')
                self.veg = galileo.Resident(galileo.param_image(self.params, 2, param_i=vegsim.param_i(),
                                                                param_u8=vegsim.param_u8(), ganja=vegsim.ganja_block(vset)))
                print('vegetation: game step running (species set %d)' % vset, flush=True)
            R, ex = self.veg, self.extra
            R.put({0: G['T'].view(np.uint32), 1: G['soft'].view(np.uint32), 2: G['W'].view(np.uint32),
                   3: G['sed'].view(np.uint32), 6: ex[6], 7: ex[7]}, (0, 1, 2, 3, 6, 7))
            R.run(4, 1)
            got = R.get((6, 7))
            new = ex.copy()
            new[6], new[7] = got[6], got[7]
            return new
        except Exception as e:
            print('vegetation step off:', e, flush=True)
            self.veg_failed = True
            return None

    def _apply_edits(self):
        with self.lock:
            todo, self.pending = self.pending, []
        if not todo:
            return
        G = {k: v.copy() for k, v in self.G.items()}   # a copy: read() may be converting the old state
        for fn in todo:
            fn(G)
        with self.lock:
            self.G, self.edits = G, self.edits + 1

    def ticks_per_second(self):
        times = self.times
        return (len(times) - 1) / max(times[-1] - times[0], 1e-6) if len(times) > 1 else 0.0

    def read(self):
        with self.lock:
            if self.published == (self.tick, self.edits):
                return None
            self.published, G, t = (self.tick, self.edits), self.G, self.tick
        return Frame(t, state_to_grid(G, self.extra), self.params, self.sources, self.name)

    def status(self):
        rate = len(self.times) / max(self.times[-1] - self.times[0], 1e-6) if len(self.times) > 1 else 0
        state = 'paused' if self.paused else '%.1f ticks/s (%s, T: speed)' % (
            rate, '%dx game speed' % self.speed if self.speed else 'as fast as possible')
        extra = ''
        if self.schedule is not None:
            sch = self.schedule
            last = sch.events[-1] if sch.events else None
            extra = ' | level time %.0fs, border %.2f%s' % (sch.time, sch.border_level, (' | last script event at %.0fs: %s' % (last[0], last[1][:70])) if last else '')
        note = getattr(self, 'progress_note', '')
        vil = (' | villages %d/%d%s (click a totem: village on/off)' % (sum(1 for t in self.totems if t.get('built')), len(self.totems),
                                                               (', from ' + note) if note else '')) if self.totems else ''
        return '%s%s | %s | %d sources, %d repulsors%s%s' % (self.name, ' (%s)' % self.label if self.label else '', state, len(self.sources), len(self.other_modifiers), vil, extra)

    def stop(self):
        self.running = False                     # the model thread closes the vegetation helper on its way out

    def adopt_totems(self, totems):
        """Totems that arrived after the takeover (the live totem search finished later)."""
        self.totems = [dict(t) for t in totems]
        self.loose_repulsors = songs_from_modifiers(self.totems, self.loose_repulsors, mark_built=True)
        with_knowledge(self.totems, level_number(self))
        self.other_modifiers = totem_repulsors(self.totems, self.loose_repulsors)
        if self.schedule is not None:
            self.schedule.villages = lambda t: sum(1 for x in self.totems if x.get('built'))

    def toggle_totem(self, t):
        """Build / remove the village at totem t. Its songs (from the save or the live game) are kept and
        only act while the village exists."""
        t['built'] = not t.get('built')
        self.other_modifiers = totem_repulsors(self.totems, self.loose_repulsors)


class CompareSource(Source):
    """The live game plus a copy of it in the Python model, run in lockstep: whenever the game advances N
    ticks the model advances N ticks too, with the game's own per-tick list (fluid sources at the amounts
    the game applied, RepulseFluid entries as listed), so what drifts apart is the simulation, not
    DustView's source / bury / village rules. Not modelled: tides / waves / weather, player powers,
    vegetation and fire (all of these show up as differences). read() returns the game's frame or the
    model's (show = 'game' / 'model'), each with .other = the other one; resync() copies the game into
    the model again. Anything not defined here (sources, totems, songs, ...) comes from the live reader."""
    name = 'model vs game'
    MAX_GAP = 600                                  # ticks behind (~20 s) before the model is re-synced instead

    def __init__(self, live, frame):
        self.live = live
        self.params = np.asarray(frame.params, np.float32)
        self.P = {n: float(v) for n, v in zip(galileo.PARAM_NAMES, self.params)}
        self.lock = threading.Lock()
        self.show = 'game'
        self.running = True
        self.game = frame
        self.game_params, self._params_obj = frame.params, None
        self.sim_changes = getattr(live, 'sim_changes', 0)
        self._sync_to(frame)
        threading.Thread(target=self._loop, daemon=True).start()

    def __getattr__(self, k):                     # sources, totems, songs, seen_repulsors, ... of the live reader
        if k == 'live':
            raise AttributeError(k)
        return getattr(self.live, k)

    def _sync_to(self, fr):
        with self.lock:
            self.G, self.extra = dustsim.from_grid(fr.grid), fr.grid
            self.model_tick, self.target, self.synced_at = fr.tick, fr.tick, fr.tick
            self.model_fr, self.published, self.pair, self._prev_diff = None, None, None, None
            self.gen = getattr(self, 'gen', 0) + 1
            self.note = 'in step since game tick %d' % fr.tick

    def resync(self):
        self._sync_to(self.game)

    def _loop(self):
        while self.running:
            with self.lock:
                G, behind, gen = self.G, self.target - self.model_tick, self.gen
            if behind <= 0:
                time.sleep(0.003)
                continue
            if behind > self.MAX_GAP:                 # fell far behind (or the game jumped): start again from the game
                self._sync_to(self.game)
                self.note = 'model was %d ticks behind: re-synced at game tick %d' % (behind, self.game.tick)
                continue
            live = self.live
            if getattr(live, 'vec', None) is None:       # source list not found yet: no model ticks (see read)
                time.sleep(0.01)
                continue
            if self._params_obj is not self.game_params:   # the game's current parameters (tide / border level, ...)
                self._params_obj = self.game_params
                self.P = {n: float(v) for n, v in zip(galileo.PARAM_NAMES, np.asarray(self.game_params, np.float32))}
            srcs = live.applied_sources()             # what the game applied this tick (fade-in, burial included)
            reps = [e for e in (getattr(live, 'other_modifiers', None) or []) if e.get('type') == 1]
            G = dustsim.tick(G, self.P, srcs, reps)
            ground = [e for e in (getattr(live, 'entries', None) or []) if e.get('type') in (2, 3)]
            if ground:                                # HardGround / SoftGround entries (e.g. a power adding sand)
                dustsim.apply_modifiers(G, ground, self.P)
            with self.lock:
                if self.gen == gen:                   # not re-synced meanwhile
                    self.G, self.model_tick = G, self.model_tick + 1

    def read(self):
        fr = self.live.read()
        changed = False
        if fr is not None and np.array_equal(fr.grid[2], self.game.grid[2]) and np.array_equal(fr.grid[0], self.game.grid[0]):
            # the tick counter moved but the grid did not (seen on Map 4: the counter kept running with the
            # whole map frozen, e.g. in the game's menu): treat it as paused, don't run the model
            self.frozen = getattr(self, 'frozen', 0) + 1
            fr = None
        elif fr is not None:
            self.frozen = 0
        if fr is not None:
            sim_changed = getattr(self.live, 'sim_changes', 0) != self.sim_changes
            if fr.tick < self.game.tick or sim_changed:   # the game went back or a new level loaded: start again
                self.sim_changes = getattr(self.live, 'sim_changes', 0)
                self.game = fr
                self._sync_to(fr)
                self.note = 'new level / map reload: re-synced at game tick %d' % fr.tick
            self.game, self.game_params, changed = fr, fr.params, True
            self.recent = [g for g in getattr(self, 'recent', []) if g.tick < fr.tick][-8:] + [fr]
            if getattr(self.live, 'vec', None) is None:     # no sources yet: keep the model on the game
                self._sync_to(fr)
                self.note = 'waiting for the source list (model held on the game until then)'
            elif self.note.startswith('waiting'):
                self.note = 'source list found: in step since game tick %d' % self.synced_at
            with self.lock:
                self.target = fr.tick
        with self.lock:
            key = self.model_tick
            if key != self.published:
                self.published, G, t = key, self.G, self.model_tick
                self.model_fr = Frame(t, state_to_grid(G, self.extra), self.params, self.live.sources, 'python model')
                changed = True
        # show the game frame of the model's tick (the model usually runs a tick behind the newest game
        # frame: pairing different ticks made every moving wave look like a difference)
        model = self.model_fr
        match = next((g for g in getattr(self, 'recent', []) if model is not None and g.tick == model.tick), None)
        if match is not None:
            if getattr(self, 'pair', None) is None or self.pair[1] is not model:
                self._diff(match, model)
            self.pair = (match, model)
        elif getattr(self, 'pair', None) is None or self.pair[0].tick > self.game.tick:
            self.pair = (self.game, model if model is not None else self.game)
        if not changed and getattr(self, '_shown', None) == (self.show, id(self.pair[0])):
            return None
        self._shown = (self.show, id(self.pair[0]))
        game, model = self.pair
        game.other, model.other = model, game
        game.role, model.role = 'game', 'model'
        return model if self.show == 'model' else game

    def _diff(self, game, model):
        """model - game for fluid (water + lava) and ground, kept only where the previous pair differed the
        same way too (by the smaller amount): one-frame blips (a game frame now and then a tick off) drop out."""
        dF = (model.water + model.lava) - (game.water + game.lava)
        dT = model.ground - game.ground
        prev, self._prev_diff = getattr(self, '_prev_diff', None), (dF, dT)
        if prev is not None:
            keep = lambda a, b: np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0)
            dF, dT = keep(dF, prev[0]), keep(dT, prev[1])
        game.diff = model.diff = (dF.astype(np.float32), dT.astype(np.float32))

    def status(self):
        lag = self.game.tick - self.model_tick
        frozen = ' | game grid not changing (paused / menu?): model held' if getattr(self, 'frozen', 0) > 3 else ''
        return 'model vs game (G: game / model / differences) | game tick %d, model tick %d%s%s | %s | %s' % (
            self.game.tick, self.model_tick, (' (%d behind)' % lag) if lag else '', frozen, self.note, self.live.status())

    def stop(self):                                 # the live reader stays (App.live)
        self.running = False


def render_compare(fr):
    """'model vs game': the game's natural view dimmed, with model minus game coloured like the
    unexplained-changes view: cyan = more fluid (water + lava) in the model, orange = less, yellow = model
    ground higher, magenta = lower (log scale, 0.01 .. ~3 units)."""
    other = getattr(fr, 'other', None)
    if other is None:
        return render(fr, 'natural')
    game, model = (fr, other) if getattr(fr, 'role', 'game') == 'game' else (other, fr)
    col = render(game, 'natural').astype(np.float32).mean(-1, keepdims=True).repeat(3, -1) * 0.4
    fm, fg = model.water + model.lava, game.water + game.lava
    dF = fm - fg
    dT = model.ground - game.ground
    if getattr(fr, 'diff', None) is not None:      # compare mode: steady differences only (CompareSource._diff)
        dF, dT = fr.diff
    # fluid: ignore the open sea's small wave noise (the game moves the sea, the model does not): only
    # differences above 0.05 and above 3 % of the local depth count
    floor = np.maximum(0.05, 0.03 * np.maximum(fm, fg))
    dF = np.sign(dF) * np.maximum(np.abs(dF) - floor, 0)
    for d, colour, lo in ((np.maximum(dF, 0), (0, 220, 255), 1e-2), (np.maximum(-dF, 0), (255, 140, 0), 1e-2),
                          (np.maximum(dT, 0), (255, 240, 0), 1e-2), (np.maximum(-dT, 0), (255, 0, 255), 1e-2)):
        a = np.clip(np.log10(np.maximum(d, lo) / lo) / 2.5, 0, 1)[..., None]
        col += (np.array(colour, np.float32) - col) * a
    return np.clip(col, 0, 255).astype(np.uint8)


def level_names():
    """Level files in the game's DustSave/Ingame folder (story maps first, then the challenge maps)."""
    import compare_sim, re
    names = [f[:-4] for f in os.listdir(compare_sim.SAVES) if f.lower().endswith('.sav')]
    import level_names as ln
    def key(n):                                  # story order, then challenges in the game's order
        m = re.match(r'ALPHA_Map(\d+)_', n)
        return (0, int(m.group(1)), n) if m else (1, ln.CHALLENGE.get(n, (99,))[0], n)
    return sorted(names, key=key)


def profile_names():
    """Player name per profile folder: Documents/Dust/profiles/profile_NN.xml has <PlayerData name="...">."""
    import glob, re, sav_sources
    out = {}
    for x in glob.glob(os.path.join(os.path.dirname(sav_sources.PROGRESS_DIR), 'profiles', 'profile_*.xml')):
        try:
            m = re.search(r'<PlayerData\s+name="([^"]*)"', open(x, encoding='utf-8', errors='replace').read())
        except OSError:
            continue
        if m:
            out[os.path.splitext(os.path.basename(x))[0]] = m.group(1)
    return out


def player_saves():
    """The player's progress saves, all profiles (Documents/Dust/DustSave/profile_*/ingame/MAP_nn.SAV,
    nn = story map - 1), newest first: dicts with path, level (level file name, by number; the Map2/Map3
    demo files get none), profile (folder), player (name from the profile, or the folder), time (mtime)
    and label."""
    import glob, re, sav_sources
    names = level_names()
    who = profile_names()
    out = []
    for p in glob.glob(os.path.join(sav_sources.PROGRESS_DIR, 'profile_*', 'ingame', 'MAP_*.SAV')):
        m = re.match(r'MAP_(\d+)\.SAV$', os.path.basename(p), re.I)
        lvl = [n for n in names if m and n.startswith('ALPHA_Map%d_' % (int(m.group(1)) + 1)) and 'Demo' not in n]
        prof = os.path.basename(os.path.dirname(os.path.dirname(p)))
        t = os.path.getmtime(p)
        player = who.get(prof, prof)
        out.append(dict(path=p, level=lvl[0] if lvl else None, profile=prof, player=player, time=t,
                        label='%s  %s  (%s)' % (player, time.strftime('%Y-%m-%d %H:%M', time.localtime(t)),
                                                os.path.basename(p)[:-4])))
    return sorted(out, key=lambda d: -d['time'])


def level_for_save(path):
    """The level file a player's save belongs to (its script, sources and border): MAP_nn = story map
    nn+1; where several level files have that number (the Map2/Map3 demos share totem IDs and border
    with the real levels) or the file is not named MAP_nn, the one whose hard ground (ground - soft)
    matches the save in the most cells (play changes soft ground, water and lava, rarely hard ground:
    MAP_02 matches Map3_Vallee in 97% of cells, Map3_Demo in 77%)."""
    import re, compare_sim
    names = level_names()
    m = re.match(r'MAP_(\d+)\.SAV$', os.path.basename(path), re.I)
    cands = [n for n in names if n.startswith('ALPHA_Map%d_' % (int(m.group(1)) + 1))] if m else names
    if len(cands) == 1:
        return cands[0]
    T, So = compare_sim.load_sav_grids(path)[:2]
    score = {}
    for n in cands:
        T2, So2 = compare_sim.load_map(n)[:2]
        score[n] = float((np.abs((T - So) - (T2 - So2)) < 0.01).mean())
    best = max(score, key=score.get) if score else None
    if best is None or score[best] < 0.5:
        raise ValueError('no level file matches %s (hard ground differs from every level)' % os.path.basename(path))
    return best


def progress_save_for(name):
    """The player's newest save of story level `name` (MAP_nn.SAV, nn = map number - 1, checked with
    level_for_save so the Map2/Map3 demo files get none). Totem positions are not used: totems can move
    in play (Map 6's did), so sav_sources.find_progress_save's position match misses those saves."""
    import re
    m = re.match(r'ALPHA_Map(\d+)_', name)
    if not m:
        return None
    want = 'MAP_%02d.SAV' % (int(m.group(1)) - 1)
    hits = [d['path'] for d in player_saves() if os.path.basename(d['path']).upper() == want and level_for_save(d['path']) == name]
    return max(hits, key=os.path.getmtime) if hits else None


def model_from_save(path, villages=None):
    """ModelSource starting from a player's save: its grids (terrain, water, lava as saved) and its
    villages and songs, with the sources and script of the level it belongs to. The level script starts
    from its beginning (the save's script state and the sources it had switched on are not read)."""
    return model_from_map(level_for_save(path), villages, progress=path, grid_path=path)


def model_from_map(name, villages=None, progress='none', grid_path=None):
    """ModelSource for a map's .sav: its start grids, border values and the generators that are on at
    the start (mode 1 or 2). Parameters are the game's normal set (galileo.fluid_params).
    progress: 'auto' = take which totems have a village and which villages know a repel song from the
    player's progress save of this level (found by totem positions), a path = that save, 'none' = no
    villages. Only villages and songs are taken from it; the grid comes from the level file, or from
    grid_path (a player's save) if given."""
    import compare_sim, sav, sav_sources
    T, So, W, C3, border = compare_sim.load_sav_grids(grid_path or os.path.join(compare_sim.SAVES, name + '.sav'))
    grid = np.zeros((8, N, N), np.uint32)
    for c, a in enumerate((T, So, W, C3)):
        grid[c] = a.astype(np.float32).view(np.uint32)
    _, u = sav.read_sav(os.path.join(compare_sim.SAVES, name + '.sav'))
    c6, c7 = sav.vegetation_components(sav.read_sav(grid_path)[1] if grid_path else u)   # plants (docs/sav-format.md)
    if c6 is not None:
        grid[6], grid[7] = c6, c7
    import level_schedule
    sch = level_schedule.LevelSchedule(name, villages=villages, border_level=border[0])
    st = sav_sources.statues(sav.read_sav(grid_path)[1] if grid_path else u)   # a save: totems where they are now
    tot = {t['index']: dict(id=t['id'], col=t['col'], row=t['row'], power=t['power'], built=False, water=0.0, lava=0.0)
           for t in st}
    note = ''
    path = (progress_save_for(name) or sav_sources.find_progress_save(st)) if progress == 'auto' else \
        (None if progress in (None, 'none') else progress)
    if path:
        _, pu = sav.read_sav(path)
        for i, d in sav_sources.progress_totems(pu).items():
            if i in tot:
                tot[i].update(d)
        note = '%s (%s)' % (os.path.basename(path), time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path))))
        print('villages and songs from', path, flush=True)
    label = name + (' from %s (script restarted)' % os.path.basename(grid_path) if grid_path else '')
    ms = ModelSource(grid, galileo.fluid_params(border=border), sch.sources(), 0, label, schedule=sch,
                     totems=list(tot.values()))
    ms.progress_note = note
    ms.level_name = name
    if villages is not None:                     # an explicit --villages schedule overrides the totem count
        sch.villages = villages
    return ms


# ---------------------------------------------------------------- editing (model mode)
TOOLS = ('inspect', 'water', 'lava', 'sand', 'rock')
LAVA_TEMPERATURE = 255.0                         # internal units, as the level files' lava vents
BRUSH_MIN, BRUSH_MAX = 1, 40
VEGETATION_STEP = True                           # model mode: run the game's vegetation / fire step each tick


def brush_edit(tool, col, row, radius, amount, P, temperature=LAVA_TEMPERATURE):
    """An edit for ModelSource.edit: the game's own modifiers on a disc (dustsim.fluid_modifier =
    Add/RemoveWater and Add/RemoveLava, dustsim.ground_modifier = SoftGround / HardGround), amount per cell
    (negative removes). Cells in the fixed border band (dustsim.SKIP) are left alone."""
    ys, xs = dustsim.disc(col, row, radius)
    lo, hi = dustsim.RING[0] + 1, dustsim.RING[1] - 1
    keep = (ys >= lo) & (ys <= hi) & (xs >= lo) & (xs <= hi)
    ys, xs = ys[keep], xs[keep]

    def fn(G):
        if not len(ys):
            return
        if tool == 'water':
            dustsim.fluid_modifier(G, ys, xs, amount, 0.0, 0.0, P)
        elif tool == 'lava':
            dustsim.fluid_modifier(G, ys, xs, amount, 0.0, temperature, P)
        elif tool in ('sand', 'rock'):
            dustsim.ground_modifier(G, ys, xs, amount, soft=(tool == 'sand'))
    return fn


# ---------------------------------------------------------------- rendering
LIGHT = np.array([-0.6, -0.6, 0.55]) / np.linalg.norm([-0.6, -0.6, 0.55])
VIEWS = ('natural', 'height', 'water depth', 'lava temperature', 'sediment', 'flow speed', 'unexplained changes',
         'model vs game', 'engine texture')         # model vs game: needs the compare source (C)                      # engine texture: one of engine_textures.ENTRIES (App.texture)
FLOW_TEXTURES = ('wp_velocity', 'wp_foam', 'uv_drag')   # engine textures that need the running flow layer


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


LAVA_STOPS = [(0, (40, 25, 20)), (12.5, (90, 20, 10)), (40, (170, 30, 10)), (90, (225, 70, 10)), (150, (250, 120, 20)), (255, (255, 180, 50))]
# DustView's own lava colours by temperature (the game's lava shader is not decoded); 255 = a vent's fresh lava


_LAVA_XS = np.array([s[0] for s in LAVA_STOPS], np.float32)
_LAVA_CS = np.array([s[1] for s in LAVA_STOPS], np.float32)


def add_plants(fr, rgb):
    """Untextured natural view: plants (grid component 6) blended in, fading under water and lava
    (the textured view puts them under its water layer instead, flowtex.shade_textured)."""
    import flowtex, ganja
    if not flowtex.SHOW_VEGETATION or fr.grid is None or not fr.grid[6].any():
        return rgb
    pc, cov = ganja.colours(fr.grid[6])
    cov = cov * np.exp(-fr.water / 0.5) * (fr.lava <= 0)
    pc = pc * (shade(fr.ground) * 1.2)[..., None]
    out = rgb.astype(np.float32)
    out += (pc - out) * cov[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render(fr, view):
    if view == 'natural' and viewfast is not None:
        out = np.empty((N, N, 3), np.uint8)
        viewfast.render_natural(fr.ground, fr.soft, fr.water, fr.lava, fr.sediment, fr.temperature,
                                LIGHT, _LAVA_XS, _LAVA_CS, out)
        return add_plants(fr, out)
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
        return add_plants(fr, np.clip(col, 0, 255).astype(np.uint8))
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


_TEXT_MASKS = {}


def draw_text(img, xy, text, colour):
    """ImageDraw.text(xy, text, fill=colour) with the rendered text cached per string (rendering the
    font is the slow part when many labels are drawn every frame)."""
    m = _TEXT_MASKS.get(text)
    if m is None:
        if len(_TEXT_MASKS) > 5000:
            _TEXT_MASKS.clear()
        probe = ImageDraw.Draw(Image.new('L', (1, 1)))
        l, t, r, b = probe.textbbox((0, 0), text)
        m = Image.new('L', (max(1, int(np.ceil(r))), max(1, int(np.ceil(b)))))
        ImageDraw.Draw(m).text((0, 0), text, fill=255)
        _TEXT_MASKS[text] = m
    x, y = int(np.floor(xy[0] + 0.5)), int(np.floor(xy[1] + 0.5))   # as Pillow places it (at exact
    img.paste(colour, (x, y, x + m.width, y + m.height), m)        # .5 positions it can be 1 px off)


# ---------------------------------------------------------------- window
class App:
    INFO_LINES = {1: 12, 2: 7}                   # info text height per zoom (lines wrap at the map's width)

    def __init__(self, root, source, scale, flip, rotate):
        self.root, self.src, self.scale = root, source, scale
        self.live = source if isinstance(source, LiveGame) else None
        self.flip, self.rotate = flip, rotate % 4
        self.view, self.overlay = 'natural', True
        self.flow_on, self.flow, self.flow_drawn = True, None, 0.0
        self.flow_time, self.flow_clock = 0.0, time.time()   # flow texture time: runs with the simulation
        self.texture, self.tex_read = 'wp_foam_nofade', None   # engine texture view: entry, cell readout
        self.tool, self.brush_radius, self.brush_strength = 'inspect', 6, 10.0   # editing (model mode)
        self.lava_temperature = LAVA_TEMPERATURE
        self.zoom, self.view_x0, self.view_y0 = 1.0, 0.0, 0.0   # zoom and the top-left display cell shown
        self.smooth = True                               # interpolate when zoomed (False: crisp cells)
        self.ground_tex = True                           # natural view: the game's sand / rock textures
        self.show_plants = True                          # natural view: plants / fire (flowtex.SHOW_VEGETATION)
        self.drag = None                                 # (x, y, view_x0, view_y0, moved) while panning
        self.painting, self.paint_time = None, None      # +1 add / -1 remove while a mouse button is held
        self.show_events, self.events = True, []        # (time, row, col, cells, total)
        self.acc, self.acc_time, self.acc_tick = {}, time.time(), None   # view 7 accumulators
        self.frame = None
        self.dirty = True
        self.mouse = None
        self.fps = []
        self.loading, self.loaded = None, None           # background load: label, result
        self.message = None                              # (time, text): shown in the info line for a while
        root.title('DustView')
        root.configure(bg='#202020')
        self.photo = ImageTk.PhotoImage(Image.new('RGB', (N * scale, N * scale)))
        self.canvas = tk.Label(root, bg='black', image=self.photo, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky='nw')
        self.info = tk.Label(root, anchor='nw', justify='left', font=('Consolas', 9), bg='#202020', fg='#e0e0e0',
                             wraplength=N * scale, height=self.INFO_LINES.get(scale, 6))   # fixed: no window resizing
        self.info.grid(row=1, column=0, sticky='new')
        root.rowconfigure(1, weight=1)                   # spare height (panel taller than the map) goes below the info
        import dustview_panel
        self.panel = dustview_panel.ControlPanel(root, self, sys.modules[__name__])   # this module, also when run as a script
        self.panel.frame.grid(row=0, column=1, rowspan=2, sticky='ns')
        root.bind('<Key>', self.on_key)
        self.canvas.bind('<Motion>', self.on_move)
        self.canvas.bind('<Leave>', lambda e: self.set_mouse(None))
        self.canvas.bind('<Button-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_b1_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<Button-3>', lambda e: self.start_paint(e, -1))
        self.canvas.bind('<B3-Motion>', self.on_move)
        self.canvas.bind('<ButtonRelease-3>', lambda e: self.stop_paint())
        self.canvas.bind('<Button-2>', self.start_drag)
        self.canvas.bind('<B2-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-2>', lambda e: self.end_drag())
        root.bind('<MouseWheel>', self.on_wheel)        # Windows sends the wheel to the focused widget
        self._make_index()
        self.tick()

    # orientation: display = flipud?(rot90(grid, k)); index maps translate display pixels to cells
    def orient(self, a):
        a = np.rot90(a, self.rotate)
        return a[::-1] if self.flip else a

    ZOOMS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

    def px_per_cell(self):
        return self.scale * self.zoom

    def to_px(self, y, x):
        """Canvas pixel (x, y) of the centre of display cell (y, x)."""
        k = self.px_per_cell()
        return (x + 0.5 - self.view_x0) * k, (y + 0.5 - self.view_y0) * k

    def cell_at(self, ex, ey):
        """Display cell (y, x) under canvas pixel (ex, ey)."""
        k = self.px_per_cell()
        return (int(min(N - 1, max(0, self.view_y0 + ey / k))), int(min(N - 1, max(0, self.view_x0 + ex / k))))

    def display_to_grid(self, yd, xd):
        """Continuous display coordinates -> grid (row, col) coordinates (flip / rotation are quarter
        turns and mirrors, so this is exact and maps rectangles to rectangles)."""
        r0, c0 = self.idx_r[0, 0], self.idx_c[0, 0]
        dy = (self.idx_r[1, 0] - r0, self.idx_c[1, 0] - c0)
        dx = (self.idx_r[0, 1] - r0, self.idx_c[0, 1] - c0)
        return (r0 + 0.5 + dy[0] * (yd - 0.5) + dx[0] * (xd - 0.5),
                c0 + 0.5 + dy[1] * (yd - 0.5) + dx[1] * (xd - 0.5))

    def _clamp_view(self):
        v = N / self.zoom
        self.view_x0 = min(max(self.view_x0, 0.0), N - v)
        self.view_y0 = min(max(self.view_y0, 0.0), N - v)

    def set_zoom(self, zoom, ex=None, ey=None):
        """Zoom about canvas pixel (ex, ey) (default: the centre); the cell there stays put."""
        size = N * self.scale
        ex = size / 2 if ex is None else ex
        ey = size / 2 if ey is None else ey
        k = self.px_per_cell()
        cx, cy = self.view_x0 + ex / k, self.view_y0 + ey / k
        self.zoom = float(min(max(zoom, self.ZOOMS[0]), self.ZOOMS[-1]))
        k = self.px_per_cell()
        self.view_x0, self.view_y0 = cx - ex / k, cy - ey / k
        self._clamp_view()
        self.dirty = True

    def zoom_step(self, d, ex=None, ey=None):
        i = min(range(len(self.ZOOMS)), key=lambda j: abs(self.ZOOMS[j] - self.zoom))
        self.set_zoom(self.ZOOMS[min(max(i + d, 0), len(self.ZOOMS) - 1)], ex, ey)

    def on_wheel(self, e):
        if self.root.winfo_containing(e.x_root, e.y_root) is not self.canvas:
            return                                   # wheel over the panel (lists): not a zoom
        ex, ey = e.x_root - self.canvas.winfo_rootx(), e.y_root - self.canvas.winfo_rooty()
        self.zoom_step(1 if e.delta > 0 else -1, ex, ey)

    def start_drag(self, e):
        self.drag = (e.x, e.y, self.view_x0, self.view_y0, False)

    def on_drag(self, e):
        if self.drag is None:
            return
        x, y, x0, y0, moved = self.drag
        if abs(e.x - x) + abs(e.y - y) > 3:
            moved = True
        k = self.px_per_cell()
        self.view_x0, self.view_y0 = x0 - (e.x - x) / k, y0 - (e.y - y) / k
        self._clamp_view()
        self.drag = (x, y, x0, y0, moved)
        self.dirty = True
        self.on_move(e)

    def end_drag(self):
        """Returns True if the mouse moved (a pan, not a click)."""
        moved = self.drag is not None and self.drag[4]
        self.drag = None
        return moved

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

    # ---- actions (keys and the control panel both call these)
    def say(self, text):
        self.message = (time.time(), text)

    def is_model(self):
        return isinstance(self.src, ModelSource)

    def set_view(self, view):
        self.view = view
        if view == 'unexplained changes':
            if hasattr(self.src, 'enable_explain'):
                self.src.enable_explain()
            else:
                self.say('the unexplained-changes view needs the live game (it compares the game with its own step)')
        self.dirty = True

    def set_texture(self, key):
        """Show one of the engine's map-sized textures (engine_textures.ENTRIES)."""
        self.texture = key
        self.set_view('engine texture')

    def set_tool(self, tool):
        self.tool = tool
        if tool != 'inspect' and not self.is_model():
            self.say('painting works on the Python model: press M (Continue in model) or load a map first')
        self.dirty = True

    def set_brush(self, radius=None, strength=None, temperature=None):
        if temperature is not None:
            self.lava_temperature = float(min(255.0, max(13.0, temperature)))
        if radius is not None:
            self.brush_radius = int(min(BRUSH_MAX, max(BRUSH_MIN, radius)))
        if strength is not None:
            self.brush_strength = float(strength)
        self.dirty = True

    def start_paint(self, e, sign):
        if self.tool == 'inspect':
            return
        if not self.is_model():
            self.say('painting works on the Python model: press M (Continue in model) or load a map first')
            return
        self.on_move(e)
        self.painting, self.paint_time = sign, None
        self.paint()

    def stop_paint(self):
        self.painting = None

    def paint(self):
        """Apply the brush at the mouse for the time since the last call (strength = amount per cell per
        second; the first call of a stroke applies one 30 fps frame's worth)."""
        if not self.painting or self.mouse is None or not self.is_model():
            return
        now = time.time()
        dt = self.FRAME if self.paint_time is None else min(now - self.paint_time, 0.1)
        self.paint_time = now
        r, c = self.idx_r[self.mouse], self.idx_c[self.mouse]
        self.src.edit(brush_edit(self.tool, int(c), int(r), self.brush_radius,
                                 self.painting * self.brush_strength * dt, self.src.P, self.lava_temperature))

    def set_flip(self, flip):
        self.flip = flip; self._make_index(); self.dirty = True

    def rotate_view(self):
        self.rotate = (self.rotate + 1) % 4; self._make_index(); self.dirty = True

    def set_scale(self, scale):
        if scale != self.scale:
            self.scale = scale
            self.info.configure(wraplength=N * scale, height=self.INFO_LINES.get(scale, 6))
            self._clamp_view()
            self.dirty = True

    def toggle_pause(self):
        if self.is_model():
            self.src.paused = not self.src.paused

    def set_speed(self, speed):
        if self.is_model():
            self.src.speed = speed

    def step_model(self, n=1):
        """One tick (or n) of the paused model."""
        if self.is_model():
            self.src.paused = True
            self.src.step_req += n

    def flow_rate(self):
        """How fast the flow texture's clock runs: 0 while the game or the model is paused (the game's
        own water texture stops too), the model's speed (measured when flat out), 1 for the live game."""
        src = self.src
        if self.frame is None or getattr(src, 'paused', False):
            return 0.0
        if isinstance(src, ModelSource):
            return float(src.speed) if src.speed else src.ticks_per_second() / ModelSource.GAME_RATE
        return 1.0 if isinstance(src, (LiveGame, CompareSource)) else 0.0

    def take_over(self):
        """M: continue in the Python model from the live state on screen."""
        if self.frame is not None and isinstance(self.src, LiveGame):
            self.pending_takeover = time.time()          # switch now if ready, else as soon as it is (tick)
            self.try_takeover()

    def compare(self):
        """C: run the model alongside the live game, tick for tick (CompareSource); again: re-sync the model."""
        if isinstance(self.src, CompareSource):
            self.src.resync()
            self.say('model re-synced to the game')
        elif self.frame is not None and isinstance(self.src, LiveGame):
            self.set_source(CompareSource(self.src, self.frame))
            if self.view not in ('natural', 'model vs game'):
                self.set_view('natural')
            self.say('comparing: G switches between game and model, view "model vs game" shows the differences')
        else:
            self.say('compare needs the live game (L)')

    def compare_mode(self):
        """What compare mode shows: 'game', 'model' or 'differences' (the model vs game view)."""
        if not isinstance(self.src, CompareSource):
            return None
        return 'differences' if self.view == 'model vs game' else self.src.show

    def toggle_compare_show(self):
        """G (compare mode): game -> model -> differences -> game. Both keep running throughout."""
        mode = self.compare_mode()
        if mode is None:
            return
        if mode == 'game':
            self.src.show = 'model'
        elif mode == 'model':
            self._view_before_diff = self.view
            self.src.show = 'game'
            self.set_view('model vs game')
        else:
            self.set_view(getattr(self, '_view_before_diff', 'natural') or 'natural')
            self.src.show = 'game'
        self.dirty = True

    def load(self, label, make):
        """Build a new source in the background (reading a map or finding the game can take seconds) and
        switch to it when ready (tick)."""
        if self.loading:
            self.say('still loading %s' % self.loading)
            return
        self.loading = label

        def run():
            try:
                res = make()
            except BaseException as e:               # SystemExit from the memory reader included
                res = e
            self.loaded = res
        threading.Thread(target=run, daemon=True).start()

    def load_level(self, name, progress='none'):
        self.load(name, lambda: model_from_map(name, progress=progress))

    def load_save(self, path):
        self.load(os.path.basename(path), lambda: model_from_save(path))

    def connect_live(self):
        """L: read the running game (found afresh: the level may have changed)."""
        self.load('the live game', LiveGame)

    def set_source(self, new):
        old = self.src
        if old is not new and isinstance(old, (ModelSource, CompareSource)):
            old.stop()
        if isinstance(new, LiveGame):
            if self.live is not None and self.live is not new:
                self.live.stop()
            self.live = new
        self.src = new
        self.frame, self.events, self.acc, self.acc_tick = None, [], {}, None
        self.pending_takeover = None
        if self.view == 'unexplained changes' and hasattr(new, 'enable_explain'):
            new.enable_explain()
        self.dirty = True

    def toggle_totem(self, t):
        if self.is_model():
            self.src.toggle_totem(t)
            self.dirty = True

    def on_key(self, e):
        if e.widget.winfo_class() in ('TCombobox', 'Entry', 'TEntry', 'Listbox'):
            return                                       # typing into a panel field
        k = e.keysym.lower()
        if k == 'bracketleft':
            self.set_brush(radius=self.brush_radius - 1)
        elif k == 'bracketright':
            self.set_brush(radius=self.brush_radius + 1)
        elif k == 'o':
            self.overlay = not self.overlay
        elif k == 'f':
            self.flow_on = not self.flow_on
        elif k == 'd':
            self.show_events = not self.show_events
        elif k == 'v':
            self.set_flip(not self.flip)
        elif k == 'r':
            self.rotate_view()
        elif k == 's':
            self.screenshot()
        elif k == 'm':
            self.take_over()
        elif k == 'c':
            self.compare()
        elif k == 'g':
            self.toggle_compare_show()
        elif k == 'l':
            self.connect_live()
        elif k == 'space':
            self.toggle_pause()
        elif k == 'n':
            self.step_model()
        elif k == 't' and self.is_model():
            sp = ModelSource.SPEEDS
            self.src.speed = sp[(sp.index(self.src.speed) + 1) % len(sp)]
        elif k == 'escape':
            self.root.destroy(); return
        self.dirty = True

    def takeover_ready(self):
        """Live sources found and the totem search done (or 20 s waited)."""
        src = self.src
        srcs_ready = bool(self.current_sources()) or 'looking' not in getattr(src, 'source_note', '')
        totems_ready = getattr(src, 'totems', None) is not None and getattr(src, 'songs', None) is not None
        return (srcs_ready and totems_ready) or time.time() - self.pending_takeover > 20

    def try_takeover(self):
        """M: copy the live state into a ModelSource (with sources, repulsors and totems)."""
        if self.frame is None or isinstance(self.src, ModelSource):
            self.pending_takeover = None
            return
        if not self.takeover_ready():
            self.say('waiting for the live sources / totems; switching to the model once ready...')
            return
        self.pending_takeover = None
        fr = self.frame
        self.src = ModelSource(fr.grid, fr.params, self.current_sources(), fr.tick, 'from live tick %d' % fr.tick,
                               live_repulsors(self.src), totems=getattr(self.src, 'totems', None))
        self.dirty = True

    def on_move(self, e):
        self.set_mouse(self.cell_at(e.x, e.y))
        if self.tool != 'inspect':
            self.dirty = True                    # move the brush outline

    def on_press(self, e):
        """Left button: paint (edit tools) or start a pan / click (inspect)."""
        if self.tool != 'inspect':
            self.start_paint(e, +1)
        else:
            self.start_drag(e)

    def on_b1_motion(self, e):
        if self.tool != 'inspect':
            self.on_move(e)
        else:
            self.on_drag(e)

    def on_release(self, e):
        if self.tool != 'inspect':
            self.stop_paint()
        elif not self.end_drag():
            self.on_click(e)                     # a click, not a pan

    def on_click(self, e):
        """Inspect tool, model mode: clicking within 8 cells of a totem builds / removes its village (songs
        unchanged)."""
        if not isinstance(self.src, ModelSource):
            if getattr(self.src, 'totems', None):
                self.say('totems can be toggled in model mode: press M (or "Continue in model") first')
            return
        if not self.src.totems:
            return
        y, x = self.cell_at(e.x, e.y)
        r, c = self.idx_r[y, x], self.idx_c[y, x]
        best = min(self.src.totems, key=lambda t: (t['row'] - r) ** 2 + (t['col'] - c) ** 2)
        if (best['row'] - r) ** 2 + (best['col'] - c) ** 2 <= 64:
            self.toggle_totem(best)

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
        flow_tex = self.view == 'engine texture' and self.texture in FLOW_TEXTURES
        if (self.view == 'natural' or flow_tex) and (self.flow is None or self.flow.s != self.scale):
            import flowtex
            self.flow = flowtex.FlowLayer(N, self.scale)
        if self.flow is not None:
            self.flow.set_level(level_number(self.src))
        if self.view == 'engine texture':
            import engine_textures
            if flow_tex:
                self.flow.update(fr.water, fr.flux, self.flow_time, fr.lava)
                self.flow_drawn = time.time()
            rgb, self.tex_read = engine_textures.image(self.texture, fr, self.flow if flow_tex else None, level_number(self.src))
        elif self.view == 'natural' and self.ground_tex:
            rgb = None                                   # drawn per pixel below
        else:
            rgb = (render_unexplained(fr, self.acc) if self.view == 'unexplained changes' else
                   render_compare(fr) if self.view == 'model vs game' else render(fr, self.view))
        size, v = N * self.scale, N / self.zoom          # canvas pixels; visible cells across
        x0, y0 = self.view_x0, self.view_y0
        if self.view == 'natural' and self.ground_tex:
            (ra, ca), (rb, cb) = self.display_to_grid(y0, x0), self.display_to_grid(y0 + v, x0 + v)
            hi = self.flow.shade_textured(fr, self.flow_time, min(ra, rb), min(ca, cb), self.px_per_cell(), size,
                                          LIGHT, _LAVA_XS, _LAVA_CS, water_tex=self.flow_on)
            img = Image.fromarray(np.ascontiguousarray(self.orient(hi)), 'RGB')
            self.flow_drawn = time.time()
        elif self.flow_on and self.view == 'natural':
            if self.zoom == 1:
                hi = self.flow.shade(rgb, fr.water, fr.flux, self.flow_time, fr.lava, sediment=fr.sediment)
            else:                                # render just the visible window at screen resolution
                (ra, ca), (rb, cb) = self.display_to_grid(y0, x0), self.display_to_grid(y0 + v, x0 + v)
                hi = self.flow.shade_window(rgb, fr.water, fr.flux, self.flow_time, min(ra, rb), min(ca, cb),
                                            self.px_per_cell(), size, fr.lava)
            img = Image.fromarray(np.ascontiguousarray(self.orient(hi)), 'RGB')
            self.flow_drawn = time.time()
        else:
            img = Image.fromarray(np.ascontiguousarray(self.orient(rgb)), 'RGB')
            if self.scale != 1 or self.zoom != 1:
                smooth = self.smooth and self.zoom != 1
                img = img.resize((size, size), Image.BILINEAR if smooth else Image.NEAREST, box=(x0, y0, x0 + v, y0 + v))
        k = self.px_per_cell()
        sources = self.current_sources()
        if self.overlay and sources:
            marks = []
            for s in sources:
                y, x = self.cell_to_disp[int(s['row']), int(s['col'])]
                cx, cy, r = *self.to_px(y, x), max(3, (s['radius'] + 1.5) * k)
                if s['rate'] < 0:          # drains: dark red = lava drain, violet = water drain
                    colour = (200, 40, 60) if s['temperature'] else (170, 90, 255)
                else:                      # orange = lava vent, cyan = spring
                    colour = (255, 140, 30) if s['temperature'] else (120, 230, 255)
                act = s.get('activity', 1.0)
                if act < 0.95:                         # noticeably buried: grey
                    colour = (150, 150, 150)
                amount = s.get('amount', s['rate'] * act)
                label = '%+.2f' % amount + (' (%d%%)' % round(act * 100) if act < 0.99 else '')
                marks.append((cx, cy, r, colour, label))
            dr = ImageDraw.Draw(img)
            for cx, cy, r, colour, label in marks:
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=2)
                draw_text(img, (cx + r + 2, cy - 6), label, colour)
        if self.show_events and self.events:
            dr = ImageDraw.Draw(img)
            now = time.time()
            for t, row, col, n, tot in self.events:
                y, x = self.cell_to_disp[int(round(row)), int(round(col))]
                cx, cy = self.to_px(y, x)
                f = max(0.25, 1 - (now - t) / 8)
                colour = tuple(int(v * f) for v in ((255, 0, 255) if tot < 0 else (255, 230, 0)))
                r = 4 * self.scale + n ** 0.5 * self.scale
                dr.line([cx - r, cy, cx + r, cy], fill=colour, width=2)
                dr.line([cx, cy - r, cx, cy + r], fill=colour, width=2)
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=1)
        tots = getattr(self.src, 'totems', None) if self.overlay else None
        if tots and not isinstance(self.src, ModelSource):      # live: songs from the game's modifier list
            tots = [dict(t) for t in tots]
            songs_from_modifiers(tots, live_repulsors(self.src))   # adds water / lava in place
            with_knowledge(tots, level_number(self.src))
        if tots:
            dr = ImageDraw.Draw(img)
            for t in tots:
                y, x = self.cell_to_disp[int(t['row']), int(t['col'])]
                cx, cy, r = *self.to_px(y, x), 4 * self.scale
                colour = (255, 205, 60) if t.get('built') else (150, 150, 150)
                dr.rectangle([cx - r, cy - r, cx + r, cy + r], outline=colour, width=2)
                dr.text((cx + r + 2, cy + 2), totem_label(t), fill=colour)
        others = (getattr(self.src, 'other_modifiers', []) if self.is_model() else live_repulsors(self.src)) if self.overlay else []
        if others:
            dr = ImageDraw.Draw(img)
            for e in others:
                y, x = self.cell_to_disp[int(e['row']), int(e['col'])]
                cx, cy, r = *self.to_px(y, x), e['radius'] * k
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(235, 235, 235), width=1)
                dr.text((cx + r + 2, cy - 6), 'mod %+g/%+g' % (e['amount'], e['f5']), fill=(235, 235, 235))
        if self.tool != 'inspect' and self.mouse is not None:
            y, x = self.mouse
            cx, cy, r = *self.to_px(y, x), (self.brush_radius + 0.5) * k
            colour = {'water': (120, 200, 255), 'lava': (255, 120, 40), 'sand': (240, 210, 140), 'rock': (200, 200, 200)}[self.tool]
            ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=1)
        self.last_image = img
        ph = getattr(self, 'photo', None)
        if ph is not None and (ph.width(), ph.height()) == img.size:
            ph.paste(img)                        # reuse the Tk image (cheaper than a new one each frame)
        else:
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.configure(image=self.photo)

    def update_info(self):
        fr = self.frame
        lines = ['%s | view: %s' % (self.src.status(), self.view)]
        if self.loading:
            lines.append('loading %s ...' % self.loading)
        if self.message and time.time() - self.message[0] < 6:
            lines.append(self.message[1])
        if fr is not None:
            fps = len(self.fps) / max(self.fps[-1] - self.fps[0], 1e-6) if len(self.fps) > 1 else 0
            border = fr.params[31] if fr.params is not None else float('nan')
            recent = [e for e in self.events if time.time() - e[0] < 8]
            if self.view == 'engine texture':
                import engine_textures
                lines.append('engine texture: %s' % engine_textures.LABELS.get(self.texture, self.texture))
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
                if self.view == 'engine texture' and self.tex_read is not None:
                    lines.append('texture here: ' + self.tex_read(r, c))
                oth = getattr(fr, 'other', None)
                if isinstance(self.src, CompareSource) and oth is not None:
                    g, m = (fr, oth) if getattr(fr, 'role', '') == 'game' else (oth, fr)
                    lines.append('model - game here: water %+.3f, lava %+.3f, ground %+.3f' % (
                        m.water[r, c] - g.water[r, c], m.lava[r, c] - g.lava[r, c], m.ground[r, c] - g.ground[r, c]))
                ex = getattr(self.src, 'explained', None)
                if self.view == 'unexplained changes' and ex is not None:
                    lines.append('unexplained at this cell (tick %d): ground %+.4f, fluid %+.4f' % (ex['tick'], ex['dT'][r, c], ex['dF'][r, c]))
        text = '\n'.join(lines)
        if text != self.info.cget('text'):
            self.info.configure(text=text)

    FRAME = 1 / 30                               # target redraw interval (the model ticks 31 times a second)

    def tick(self):
        t_start = time.perf_counter()
        res, self.loaded = self.loaded, None
        if res is not None:                      # a background load finished
            label, self.loading = self.loading, None
            if isinstance(res, BaseException):
                self.say('could not load %s: %s' % (label, res))
            else:
                self.set_source(res)
                self.say('loaded %s' % label)
        if getattr(self, 'pending_takeover', None):
            self.try_takeover()
        self.paint()
        src = self.src
        if isinstance(src, ModelSource) and not src.totems and src.label.startswith('from live') \
                and self.live is not None and getattr(self.live, 'totems', None):
            src.adopt_totems(self.live.totems)
            self.dirty = True
        try:
            fr = self.src.read()
        except Exception as e:
            fr = None
            self.say('read failed: %s' % e)
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
        now = time.time()
        rate = self.flow_rate()
        self.flow_time += min(now - self.flow_clock, 0.25) * rate
        self.flow_clock = now
        if rate and ((self.flow_on and self.view == 'natural') or (self.view == 'engine texture' and self.texture in FLOW_TEXTURES))                 and now - self.flow_drawn > self.FRAME * 0.9:
            self.dirty = True                    # keep the flow texture moving (not while paused)
        sig = tuple(id(getattr(self.src, k, None)) for k in ('sources', 'totems', 'other_modifiers'))
        if sig != getattr(self, '_drawn_sources', None):
            self._drawn_sources = sig
            self.dirty = True
        if self.frame is not None and self.dirty:
            self.draw()
            self.dirty = False
            now = time.time()
            self.fps = [t for t in self.fps if now - t < 2] + [now]
        self.panel.sync()
        self.update_info()
        spent = time.perf_counter() - t_start      # next frame FRAME after this one started
        self.root.after(max(1, int(1000 * (self.FRAME - spent))), self.tick)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--no-flip', dest='flip', action='store_false',
                    help='do not flip vertically (default: flipped, row 0 at the bottom, as in the game)')
    ap.add_argument('--rotate', type=int, default=0, help='rotate by N x 90 degrees')
    ap.add_argument('--model', metavar='MAP', help='run the Python model from this map file instead of reading the game')
    ap.add_argument('--villages', metavar='T:N,...', help='with --model: villages built over time for the level script, e.g. "0:0,200:1,900:3"')
    ap.add_argument('--progress', default='none', metavar='none|auto|PATH',
                    help="with --model: which totems have villages / repel songs: none (default, the level as shipped), from the player's save of this level (auto), or a given save")
    a = ap.parse_args()
    root = tk.Tk()
    import level_schedule
    vil = level_schedule.parse_villages(a.villages) if a.villages else None
    if a.model:
        src = model_from_map(a.model, vil, a.progress)
    else:
        try:
            src = LiveGame()
        except SystemExit as e:                  # no game / no level: start empty, load from the panel
            src = NoSource(str(e))
    App(root, src, a.scale, a.flip, a.rotate)
    root.mainloop()


if __name__ == '__main__':
    main()
