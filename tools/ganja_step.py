"""From Dust's vegetation / fire step (GalileoSimulatorWater_9_9.dll job 4) re-implemented in numpy.

  python ganja_step.py [MapName|path.sav] [ticks]     # compare with the game's DLL, field by field

step(grid, block, ...) -> (c6, c7): the next components 6 / 7 from a grid (ground, sand, water,
plants). Written from the disassembly (tools/ganja_asm.py: Step1, Step2_ComputeViability, Step2)
and pinned with harness experiments; rules in findings/vegetation-fire-layers.md ("Growth rules",
"Fire rules", "Display fields"). The check predicts each tick from the DLL's previous tick (re-synced)
and counts mismatches per field.

Arithmetic follows the game's 16-bit lanes: values wrap at 16 bits and compares are signed 16-bit
(pcmpgtw) unless noted.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galileo as g
import vegsim

N = g.N
BORDER = 32          # skip_border 2 x 16 cells: the step leaves the outer 32 cells alone


def s16(x):
    """Reinterpret as signed 16-bit (for pcmpgtw compares)."""
    return ((np.asarray(x, np.int64) + 0x8000) & 0xFFFF) - 0x8000


def u16(x):
    return np.asarray(x, np.int64) & 0xFFFF


def table(block, off):
    b = np.frombuffer(bytes(block), np.uint8)
    return (b[off:off + 16].astype(np.int64) | (b[off + 16:off + 32].astype(np.int64) << 8))


def q128(a):
    """float -> x128 -> int16 like cvtps2dq (round to nearest even) + packssdw (saturate)."""
    return np.clip(np.rint(np.asarray(a, np.float32) * np.float32(128)), -32768, 32767).astype(np.int64)


def neighbours(a, fill=0):
    """The 4 direct neighbours of every cell: (up = row-1, down = row+1, left = col-1, right = col+1)."""
    up = np.full_like(a, fill); up[1:] = a[:-1]
    dn = np.full_like(a, fill); dn[:-1] = a[1:]
    lf = np.full_like(a, fill); lf[:, 1:] = a[:, :-1]
    rt = np.full_like(a, fill); rt[:, :-1] = a[:, 1:]
    return up, dn, lf, rt


TABLE_ORDER = (0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0, 0x100, 0x120, 0x140, 0x160, 0x180, 0x1A0, 0x1C0)
try:
    import ganja_step_nb
except ImportError:          # numba not installed: numpy only
    ganja_step_nb = None


_TAB_CACHE = {}


def step_arrays(c6, c7, T, So, W, block, propagate, soft_min=10, default_growing=1, fire_progress=10,
                burned_time=246):
    """Next (c6, c7) from separate arrays (c6 / c7 uint32, T / So / W float32, 512 x 512): what DustView's
    model uses. numba when installed, else the numpy step (~430 ms)."""
    if ganja_step_nb is None:
        grid = np.zeros((8, N, N), np.uint32)
        grid[g.TERRAIN], grid[g.SOFT], grid[g.WATER] = T.view(np.uint32), So.view(np.uint32), W.view(np.uint32)
        grid[6], grid[7] = c6, c7
        return step(grid, block, propagate, soft_min, default_growing, fire_progress, burned_time, fast=False)
    key = (bytes(block), bytes(propagate))
    if key not in _TAB_CACHE:
        _TAB_CACHE.clear()
        _TAB_CACHE[key] = (np.stack([table(block, off) for off in TABLE_ORDER]).astype(np.int64),
                           np.frombuffer(bytes(propagate), np.uint8).astype(np.int64))
    tab, prop = _TAB_CACHE[key]
    return ganja_step_nb.step_cells(c6, c7, T, So, W, tab, prop, rcp_table(), soft_min, default_growing,
                                    fire_progress, burned_time)


def step(grid, block, propagate, soft_min=10, default_growing=1, fire_progress=10, burned_time=246,
         height='terrain', fast=True):
    """Next (c6, c7). fast: the numba version (ganja_step_nb, bit-identical; ~100x faster) when installed."""
    if fast and ganja_step_nb is not None and height == 'terrain':
        tab = np.stack([table(block, off) for off in TABLE_ORDER]).astype(np.int64)
        prop = np.frombuffer(bytes(propagate), np.uint8).astype(np.int64)
        return ganja_step_nb.step_cells(grid[6], grid[7], grid[g.TERRAIN].view(np.float32),
                                        grid[g.SOFT].view(np.float32), grid[g.WATER].view(np.float32),
                                        tab, prop, rcp_table(), soft_min, default_growing, fire_progress, burned_time)
    c6_in = grid[6].astype(np.int64)
    c7_in = grid[7].astype(np.int64)
    rows = np.zeros((N, N), bool)                    # first / last processed row
    rows[[BORDER, N - 1 - BORDER], BORDER:N - BORDER] = True
    cols = np.zeros((N, N), bool)                    # first / last processed column
    cols[BORDER:N - BORDER, [BORDER, N - 1 - BORDER]] = True
    inside = np.zeros((N, N), bool)
    inside[BORDER:N - BORDER, BORDER:N - BORDER] = True
    # the ring (first / last processed row and column) is processed without its plant word (type, status,
    # random nibble: 0) but with its timers; cells outside the processed area count as empty neighbours.
    # Exception: the last processed row still counts with its plant word as the neighbour of the row above
    # (it is read before it is cleared); only its own output is cleared.
    first_row = np.zeros((N, N), bool)
    first_row[BORDER, BORDER:N - BORDER] = True
    c6 = np.where(cols | first_row | ~inside, 0, c6_in)
    c7 = c7_in
    T = grid[g.TERRAIN].view(np.float32)
    So = grid[g.SOFT].view(np.float32)
    W = grid[g.WATER].view(np.float32)
    TB = {name: table(block, off) for name, off in (
        ('FI', 0x00), ('DUR', 0x20), ('RGT', 0x40), ('EVO', 0x60), ('ADULT', 0x80), ('NEXT', 0xA0),
        ('FS', 0xC0), ('FT', 0xE0), ('LWMIN', 0x100), ('LWMAX', 0x120), ('ANIM', 0x140), ('RESIST', 0x160),
        ('SWMIN', 0x180), ('SWMAX', 0x1A0), ('SDM', 0x1C0))}
    prop = np.frombuffer(bytes(propagate), np.uint8).astype(np.int64)

    b0 = c6 & 0xFF
    b1 = (c6 >> 8) & 0xFF
    t = b0 & 15
    rnd = b1 & 15
    vt = c7 & 0xFFFF
    ft = c7 >> 16
    burned = (b0 & 0x80) != 0
    has = t != 0

    # ---- Step1: lava cells count as no water / no sand; a plant under lava counts as burning
    lava = W < 0
    Wz = np.where(lava, np.float32(0), W)
    Sz = np.where(lava, np.float32(0), So)
    fire_in = ((b0 & 0x40) != 0) | (has & lava)
    b0w = b0 | np.where(has & lava, 0x40, 0)        # the working copy: spread reads this too
    w = q128(np.abs(Wz))
    sand_ok = q128(Sz) > soft_min
    if height == 'ground':
        h = T + So
    elif height == 'terrain':
        h = T
    else:
        h = T + Sz

    # ---- Step2_ComputeViability
    tu, td, tl, tr = neighbours(t)
    same4 = has & (tu == t) & (td == t) & (tl == t) & (tr == t)
    life_ok = (TB['LWMIN'][t] <= w) & (w < TB['LWMAX'][t])
    viable = has & ((life_ok & sand_ok) | same4)
    n = sum(f.astype(np.int64) for f in neighbours(fire_in.astype(np.int64))) & 3
    spread = np.zeros_like(t)
    for nb_b0, nb_h in zip(neighbours(b0w), neighbours(h)):
        nt = nb_b0 & 15
        dh = q128(np.abs(h - nb_h))
        ok = ((nb_b0 & 0x70) == 0x30) & (dh < TB['SDM'][nt]) & (TB['SWMIN'][nt] <= w) & (w < TB['SWMAX'][nt]) & sand_ok
        spread = np.maximum(spread, np.where(ok, nt, 0))
    spread_t = prop[spread]

    # ---- Step2: heat / burn timer
    thr = TB['FI'][t] + 8 * rnd
    heating = ~burned & (s16(ft) < s16(thr))
    ft1 = np.where(heating, ft + 8 * n, ft + fire_progress)
    ft1 = np.where(heating & (n == 0), (ft + 1) >> 1, ft1)
    ft1 = u16(np.where(s16(TB['RESIST'][t]) > s16(w), ft1, 0))
    fire = ~burned & (s16(ft1) > s16(thr))

    # ---- Step2: vegetation time and life phases
    R = TB['RGT'][t] + rnd
    E = TB['EVO'][t]
    RE = u16(R + E)
    RAE = u16(RE + TB['ADULT'][t])
    END = u16(RAE + E)
    past_young = s16(vt) > s16(RE)
    dying = ~(s16(RAE) > s16(vt)) & past_young
    v1 = np.where(viable | dying, vt + default_growing, np.where(past_young, RAE, END - vt))
    v1 = u16(v1)
    entered = viable & ~dying & ~(s16(RAE) > s16(v1))
    cyc = (TB['NEXT'][t] & 0x8000) == 0
    v1 = u16(v1 - np.where(entered & cyc, TB['ADULT'][t] - 1, 0))
    v1 = np.where(has, v1, 0)
    status = np.where(s16(v1) > s16(RE), 0x20, 0x10)
    flower = ~(s16(TB['FS'][t] + RE) > s16(v1)) & (s16(TB['FT'][t] + RE) > s16(v1))
    status = status | np.where(flower, 0x10, 0)
    status = np.where(s16(RAE) > s16(v1), status, 0)
    dead = s16(v1) > s16(END)
    newb = t | status | np.where(fire, 0x40, 0)
    nxt = np.where(dead & (ft1 == 0), TB['NEXT'][t] & 0xFF, 0)
    nxt_ok = (TB['LWMIN'][nxt & 15] <= w) & (w < TB['LWMAX'][nxt & 15]) & (nxt != 0) & sand_ok
    nxt = np.where(nxt_ok, nxt, 0)
    newb = np.where(dead, 0, newb)
    v1 = np.where(dead, 0, v1)
    ft1 = np.where(dead, 0, ft1)

    # ---- fire end, burned phase
    anim_fire = fire & (s16(ft1) > s16(TB['DUR'][t] - TB['ANIM'][t]))
    v1 = np.where(anim_fire, np.maximum(v1, RAE), v1)
    burnout = s16(ft1) > s16(TB['DUR'][t])
    newb = np.where(burnout, 0, newb)
    B = burnout | burned
    ftb = np.where(burnout, 0, u16(ft + 1))
    burned_end = B & (ftb > burned_time)
    ft1 = np.where(B, np.where(burned_end, 0, ftb), ft1)
    newb = np.where(B & ~burned_end, 0x80, newb)

    # ---- empty cells: new plant from a dead plant's next type or from spreading neighbours
    empty = (newb & 0x8F) == 0
    born = np.where(nxt != 0, nxt, spread_t) | 0x10
    newb = np.where(empty, born, newb)
    v1 = np.where(empty, 0, v1)
    ft1 = np.where(empty, 0, ft1)

    # ---- byte 1: random nibble, die bit, static / index (display)
    die = ((has & ~viable) | ((b1 & 0x10) != 0))
    t2 = newb & 15
    anim_a, static, index = display_phase(t2, rnd, v1, TB)
    b1n = rnd | np.where(die, 0x10, 0) | np.where(t2 != 0, (static << 5) | (index << 6), 0)
    thr2 = TB['FI'][t2] + rnd
    anim_b = np.where(u16(ft1) > u16(thr2), np.minimum(u16(ft1 - thr2), 255), 0)
    c6n = newb | (b1n << 8) | (anim_a << 16) | (anim_b << 24)
    c7n = u16(v1) | (u16(ft1) << 16)
    c6n = np.where(rows, 0, c6n)                     # the ring rows come out cleared
    c7n = np.where(rows, 0, c7n)
    c6n = np.where(inside, c6n, c6_in)               # outside the processed area: unchanged
    c7n = np.where(inside, c7n, c7_in)
    return c6n.astype(np.uint32), c7n.astype(np.uint32)


_RCP = None
RCP_FILE = os.path.join(HERE, '..', 'data', 'vegetation', 'rcpps_table.bin')


def rcp_table():
    """rcpps of every integer 0..65535 as this CPU computes it (tools/simstep/rcptable.cpp writes the file;
    without it, the exact reciprocal is used and TimeAnim a can differ by 1 on some cells)."""
    global _RCP
    if _RCP is None:
        if os.path.exists(RCP_FILE):
            _RCP = np.fromfile(RCP_FILE, np.float32)
        else:
            with np.errstate(divide='ignore'):
                _RCP = (np.float32(1) / np.arange(65536, dtype=np.float32)).astype(np.float32)
    return _RCP


def display_phase(t, rnd, v, TB):
    """TimeAnim a (0-255 through the current phase), static bit, phase index 0-3 (young, adult,
    flowering, dying), as Step2 builds them for the new type t and time v."""
    E = TB['EVO'][t]
    RE = u16(E + rnd + TB['RGT'][t])
    FS = TB['FS'][t]
    FT = TB['FT'][t]
    X1 = u16(RE + FS)
    m1 = s16(v) > s16(X1)
    idx = m1.astype(np.int64)
    rel = np.where(m1, u16(v - X1), v)
    ln2 = u16(FT - FS) >> 3
    length = np.where(m1, ln2, RE)
    X2 = u16(RE + FT - ln2)
    m2 = s16(v) > s16(X2)
    idx += m2
    rel = np.where(m2, u16(v - X2), rel)
    length = np.where(m2, ln2, length)
    X3 = u16(RE + TB['ADULT'][t])
    m3 = s16(v) > s16(X3)
    idx += m3
    rel = np.where(m3, u16(v - X3), rel)
    length = np.where(m3, E, length)
    # rcpps (approximate reciprocal, CPU-dependent) looked up per u16 length; then x rel x 256,
    # cvtps2dq (NaN / inf / too big -> 0x80000000) and pminud 255
    rcp = rcp_table()[length]
    with np.errstate(over='ignore', invalid='ignore'):
        x = (rcp * rel.astype(np.float32)) * np.float32(256)
    ok = np.isfinite(x) & (x < 2.0 ** 31)
    xi = np.where(ok, np.rint(np.where(ok, x, 0)), 2.0 ** 31).astype(np.int64)
    xi = np.minimum(xi, 255)
    static = (s16(rel) > s16(length)).astype(np.int64)
    return xi, static, idx & 3


FIELDS = (('type', 6, 0, 4), ('status', 6, 4, 2), ('fire', 6, 6, 1), ('burned', 6, 7, 1), ('random', 6, 8, 4),
          ('die', 6, 12, 1), ('static', 6, 13, 1), ('index', 6, 14, 2), ('anim_a', 6, 16, 8), ('anim_b', 6, 24, 8),
          ('vtime', 7, 0, 16), ('ftime', 7, 16, 16))


def check(grid, ticks=100, vset=0, chunk=25, examples=3, fields=None, quiet=False, margin=0, **kw):
    block = vegsim.ganja_block(vset)
    prop = vegsim.propagate_table()
    inner = np.zeros((N, N), bool)
    inner[BORDER + margin:N - BORDER - margin, BORDER + margin:N - BORDER - margin] = True
    cur = grid.copy()
    done = 0
    tot = {f[0]: 0 for f in FIELDS}
    while done < ticks:
        k = min(chunk, ticks - done)
        outs = vegsim.run_dll(cur, steps=k, vset=vset, every=True)
        prev = cur
        for i, out in enumerate(outs):
            c6, c7 = step(prev, block, prop, **kw)
            mine = {6: c6, 7: c7}
            tick = done + i + 1
            line = []
            for name, comp, sh, bits in FIELDS:
                m = (1 << bits) - 1
                a = (mine[comp] >> sh) & m
                b = (out[comp] >> sh) & m
                bad = inner & (a != b)
                nb = int(bad.sum())
                tot[name] += nb
                if nb:
                    line.append(f'{name} {nb}')
                    if not quiet and (fields is None or name in fields):
                        for r, c in list(zip(*np.nonzero(bad)))[:examples]:
                            print(f'   t{tick} {name} r{r} c{c}: in c6 {int(prev[6][r, c]):08x} c7 {int(prev[7][r, c]):08x}'
                                  f' -> game {int(out[6][r, c]):08x} {int(out[7][r, c]):08x}'
                                  f' | model {int(c6[r, c]):08x} {int(c7[r, c]):08x}')
            if line or tick % 25 == 0 or tick == 1:
                print(f'tick {tick}: ' + (', '.join(line) if line else 'all fields exact'), flush=True)
            prev = out
        cur = outs[-1].copy()
        for c in range(6):
            cur[c] = grid[c]
        done += k
    print('total mismatching cell-ticks per field:', tot)
    return tot


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'ALPHA_Map6_Fire'
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    import compare_sim
    path = arg if arg.lower().endswith('.sav') else os.path.join(compare_sim.SAVES, arg + '.sav')
    name = os.path.splitext(os.path.basename(path))[0]
    grid = vegsim.from_sav(path)
    print(f'{name}: species set {vegsim.level_set(name)}, {ticks} ticks', flush=True)
    check(grid, ticks, vset=vegsim.level_set(name))


if __name__ == '__main__':
    main()
