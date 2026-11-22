"""Vegetation / fire step of From Dust: run the game's own job (GalileoSimulatorWater_9_9.dll,
CallbackSimulatorGanja = job 4) on hand-made grids, with parameters built from the game data.

Library use:
  import vegsim as v
  grid = v.scene()                         # flat sand, no water, no plants
  v.plant(grid, rows, cols, type=1, status=3)
  out = v.run_dll(grid, steps=10)          # game's own step
  f = v.fields(out)                        # decoded fields (ganja.decode)

Parameter sources (findings/vegetation-fire-layers.md, "Parameters" and "Fire rules"):
  ganja block (512 bytes, per-type table) from SpecieVegetation*_0.ai set `vset` (vegetation_species.py);
  ParamI: LIVE_PARAM_I (read live; [5] = m_speedProgressBase, [7] = m_cellBurnedTime of VegetationInterface_0.ai);
  PropagateTable (ParamU8 bytes 0x20-0x2F, SimulatorParam+0x130).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galileo as g
import ganja

N = g.N

# Types 0 and 15 have no species file; their values as read live (The Big Wave 2026-11-16, Story 9 2026-11-21:
# the same on both). Keys: block offset in the 512-byte table.
LIVE_T0 = {0x00: 17, 0x20: 640, 0x40: 64, 0x60: 64, 0x80: 256, 0xA0: 0x800F, 0xC0: 5, 0xE0: 245, 0x120: 15,
           0x140: 640, 0x160: 32767, 0x1A0: 32767, 0x1C0: 15}
LIVE_T15 = {**LIVE_T0, 0x00: 37, 0x40: 16}


def _put(block, off, t, v):
    v = int(v) & 0xFFFF
    block[off + t] = v & 0xFF
    block[off + 0x10 + t] = v >> 8


def ganja_block(vset=0, overrides=None):
    """512-byte per-type table as the game builds it from species set `vset` (0-3).
    overrides: {(block_offset, type): value} applied last."""
    import vegetation_species as vs
    sp = vs.species()
    b = bytearray(512)
    for t in range(16):
        _put(b, 0x1E0, t, 8 * t)
    for t, s in sp.items():
        p = s['veg'][vset]
        slow = s['slow_fire'][vset]
        _put(b, 0x00, t, 8 * p['fireIgnition'])
        # types with erosionBlocking (9, 10) used the slow-fire duration in the live table
        _put(b, 0x20, t, slow['fireDuration'] if s['extra'].get('erosionBlocking') else p['fireDuration'])
        _put(b, 0x40, t, p['randomGrowingTime'])
        _put(b, 0x60, t, p['evolveDuration'])
        _put(b, 0x80, t, p['adultDuration'])
        _put(b, 0xA0, t, (p['nextVegetationType'] & 0xFF) | (0x8000 if p['lifeCycleType'] == 0 else 0))
        _put(b, 0xC0, t, p['floweringStartTime'])
        _put(b, 0xE0, t, p['floweringStopTime'])
        _put(b, 0x100, t, 128 * p['lifeWaterHeightMin'])
        _put(b, 0x120, t, 128 * p['lifeWaterHeightMax'])
        _put(b, 0x140, t, 10 * p['evolveDuration'])
        _put(b, 0x160, t, p['lifeFireResistanceWaterHeight'])
        _put(b, 0x180, t, 128 * p['spreadWaterHeightMin'])
        _put(b, 0x1A0, t, 128 * p['spreadWaterHeightMax'])
        _put(b, 0x1C0, t, 128 * p['spreadDeltaHeightMax'])
    for t, d in ((0, LIVE_T0), (15, LIVE_T15)):
        for off, v in d.items():
            _put(b, off, t, v)
    for (off, t), v in (overrides or {}).items():
        _put(b, off, t, v)
    return bytes(b)


# ParamI as read live (Story 9, 2026-11-21): each int32 holds the same 16-bit value twice (the step uses
# them as 16-bit lanes). [0] SoftGroundQuantityMin 10 (= 0.08 x 128), [4] DefaultGrowingTime 1,
# [5] FireSpeedProgressBase 10, [7] BurnedTime 246; [1] 2, [2] 10, [3] 1 (low half only), [6] 0: not used
# by the vegetation step as far as read (names unknown).
LIVE_PARAM_I = (0x000A000A, 0x00020002, 0x000A000A, 0x00000001, 0x00010001, 0x000A000A, 0, 0x00F600F6)


def param_i(soft_min=10, default_growing=1, fire_progress=10, burned_time=246, extra=None):
    """ParamI with the live values; arguments replace single values (written to both 16-bit halves)."""
    a = np.array(LIVE_PARAM_I, np.uint32)
    for k, v in ((0, soft_min), (4, default_growing), (5, fire_progress), (7, burned_time)):
        a[k] = (v & 0xFFFF) * 0x10001
    for k, v in (extra or {}).items():
        a[k] = v
    return a.view(np.int32)


def propagate_table():
    """PropagateTable guess: type t spreads into neighbours as m_spreadingVegetationType & 0xF of species t
    (5 -> 1, 10 -> 9, 11 -> 1, ...), types 0 and 15 -> 0. Identical to the live table (Story 9, 2026-11-21)."""
    import vegetation_species as vs
    tab = [0] * 16
    for t, sp in vs.species().items():
        tab[t] = sp['extra']['spreadingVegetationType'] & 0xF
    return bytes(tab)


def param_u8(propagate='species'):
    if isinstance(propagate, str):
        propagate = propagate_table()
    b = bytearray(48)
    b[25] = b[26] = 255                 # as live (Story 9); water-step bytes, no effect on the vegetation step
    if propagate is not None:
        b[0x20:0x30] = bytes(propagate)
    return bytes(b)


def scene(height=10.0, soft=1.0, water=0.0):
    grid = g.empty_grid()
    g.setf(grid, g.TERRAIN, np.full((N, N), height, np.float32))
    g.setf(grid, g.SOFT, np.full((N, N), soft, np.float32))
    g.setf(grid, g.WATER, np.full((N, N), water, np.float32))
    return grid


def plant(grid, rows, cols, type=1, status=3, fire=False, burned=False, time=0, fire_time=0, random=None):
    r = np.zeros((N, N), bool)
    r[rows, cols] = True
    w = grid[g.RANDGANJA]
    b0 = (type & 15) | ((status & 3) << 4) | (0x40 if fire else 0) | (0x80 if burned else 0)
    w[r] = (w[r] & ~np.uint32(0xFF)) | np.uint32(b0)
    if random is not None:
        w[r] = (w[r] & ~np.uint32(0xF00)) | np.uint32((random & 15) << 8)
    grid[g.GANJATIMEFIRE][r] = np.uint32((time & 0xFFFF) | ((fire_time & 0xFFFF) << 16))
    return grid


def from_sav(path):
    """Grid from a .sav (ground, water, plants), as DustView's model loads it."""
    import compare_sim, sav
    T, So, W, C3, border = compare_sim.load_sav_grids(path)
    grid = g.empty_grid()
    for c, a in enumerate((T, So, W, C3)):
        grid[c] = a.astype(np.float32).view(np.uint32)
    grid[6], grid[7] = sav.vegetation_components(sav.read_sav(path)[1])
    return grid


def run_dll(grid, steps=1, vset=0, P=None, pi=None, pu8=None, block=None, skip_border=2, every=False):
    if P is None:
        P = g.fluid_params()
    return g.run(grid, P, steps=steps, skip_border=skip_border, dump_every_step=every, job=4,
                 param_i=param_i() if pi is None else pi, param_u8=param_u8() if pu8 is None else pu8,
                 ganja=ganja_block(vset) if block is None else block)


# Species set (0-3) per level. How the game picks it is not found; these are the sets that matched the live
# per-type table (findings/vegetation-fire-layers.md). Other levels default to 0 (a guess).
LEVEL_SETS = {'Defi_006_BigWave': 0, 'ALPHA_Map7_TheSandSeas': 1, 'ALPHA_Map9_TheGreatCaldeira': 0}


def level_set(name):
    return LEVEL_SETS.get(name, 0)


def fields(grid):
    return ganja.decode(grid)
