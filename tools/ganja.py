"""Decode the vegetation / fire grid components (6 kRandomGanja, 7 kGanjaTimeFire) and the per-type
vegetation table ("ganja" = vegetation in the engine's naming).

Layout read from GalileoSimulatorWater.dll (SSE41): the per-cell debug readout near 0x10013AF0 prints
"Ganja = %u [s=%u][t=%u][Die =%u]", "Fire = %u [t=%u]", "Burned = %u [t=%u]", "Random = (%u,%u)",
"TimeAnim(%u,%u) Index(%u) Static(%u)" from these bits, and the masks in
SimulatorGanjaIterator::ConstParam (GalileoSimulatorWater_9_9.dll 0x10005656) agree (TypeMask 0x0F,
StatusYoung 0x10, StatusFlowering 0x30, FireMask 0x40, BurnedMask 0x80). Details:
findings/vegetation-fire-layers.md.

Component 6 (u32):  bits 0-3 type, 4-5 status, 6 fire, 7 burned, 8-11 random, 12 die, 13 static,
                    14-15 index, 16-23 TimeAnim a, 24-31 TimeAnim b
Component 7 (u32):  bits 0-15 growing time (t of Ganja), 16-31 fire / burned time

Per-type table: the 512-byte block at ISimulatorWater+0x150 (= live_check's prm[0x130:0x330]); each
parameter is 16 entries (one per type) of 16 bits, low bytes at +off and high bytes at +off+0x10.

Usage (game running, read-only):
  .venv/Scripts/python tools/ganja.py              field histograms + per-type table
  .venv/Scripts/python tools/ganja.py watch [SECS]  fire log once a second (burning / burned per type, and
                                                    the history of the first cells seen igniting) to
                                                    data/live/ganja_watch_*.log
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C6_FIELDS = (  # name, shift, bits
    ('type', 0, 4), ('status', 4, 2), ('fire', 6, 1), ('burned', 7, 1), ('random', 8, 4), ('die', 12, 1),
    ('static', 13, 1), ('index', 14, 2), ('time_anim_a', 16, 8), ('time_anim_b', 24, 8))
C7_FIELDS = (('grow_time', 0, 16), ('fire_time', 16, 16))
STATUS = {0: '0', 1: 'young (0x10)', 2: '2', 3: 'flowering (0x30)'}

# per-type parameters: offset of the low-byte table inside the ganja block (high bytes at +0x10). Names from
# the DLL's debug readout where it has one; every block is derived from the species files
# (tools/vegetation_species.py, findings/vegetation-fire-layers.md): x128 = metres of water / height x 128, floored.
TYPE_PARAMS = (('FireIgnition (8 x fireIgnition)', 0x00), ('FireDuration', 0x20), ('RandomGrowingTime', 0x40),
               ('EvolveDuration', 0x60), ('AdultDuration', 0x80), ('nextType | 0x8000 if lifeCycleType 0', 0xA0),
               ('FloweringStartTime', 0xC0), ('FloweringStopTime', 0xE0), ('lifeWaterHeightMin x128', 0x100),
               ('lifeWaterHeightMax x128', 0x120), ('evolveDuration x10 ("FireAnimation")', 0x140),
               ('lifeFireResistanceWaterHeight', 0x160), ('spreadWaterHeightMin x128', 0x180),
               ('spreadWaterHeightMax x128', 0x1A0), ('spreadDeltaHeightMax x128', 0x1C0), ('8 x type', 0x1E0))


def field(words, shift, bits):
    return (np.asarray(words, np.uint32) >> np.uint32(shift)) & np.uint32((1 << bits) - 1)


def decode(grid):
    """dict field name -> (512, 512) uint32 array for components 6 and 7 of a (8, 512, 512) grid."""
    out = {name: field(grid[6], s, b) for name, s, b in C6_FIELDS}
    out.update({name: field(grid[7], s, b) for name, s, b in C7_FIELDS})
    return out


def type_table(ganja):
    """Per-type parameters from the 512-byte ganja block: dict name -> list of 16 values."""
    g = bytes(ganja)
    return {name: [g[off + t] | (g[off + 0x10 + t] << 8) for t in range(16)] for name, off in TYPE_PARAMS}


def read_live(p, base):
    """(frame, grid, params) from the running game (read-only; the completed buffer)."""
    import live_check, galileo as g
    rd = lambda off, n: p.read(base + off, n)
    n = struct.unpack('<I', rd(0x92C, 4))[0]
    buf = struct.unpack('<2I', rd(0x950, 8))[(n - 1) & 1]
    return n, g.unpack(p.read(buf, live_check.GRID)), rd(0x20, 0x330)


def watch(seconds, log_path):
    """Log fire state once a second: counts of burning / burned cells and fire-time range per type, and
    the full history of a few cells that start burning (fields per second) to log_path."""
    import time, memscan, live_check
    p = memscan.Proc()
    base = live_check.find_sim(p)
    tracked = []                                       # (row, col) of cells seen igniting
    prev_fire = None
    with open(log_path, 'w', encoding='utf-8') as log:
        say = lambda s: (print(s, flush=True), log.write(s + '\n'), log.flush())
        t0 = time.time()
        while time.time() - t0 < seconds:
            n, grid, _ = read_live(p, base)
            f = decode(grid)
            fire, burned = f['fire'] == 1, f['burned'] == 1
            parts = []
            for k in np.unique(f['type'][fire | burned]):
                m = f['type'] == k
                ft = f['fire_time'][m & (fire | burned)]
                parts.append('type %d: fire %d, burned %d, fire_time %d-%d' % (k, (m & fire).sum(), (m & burned).sum(), ft.min(), ft.max()))
            say('t=%5.1f frame %d | burning %d, burned %d | %s' % (time.time() - t0, n, fire.sum(), burned.sum(), '; '.join(parts)))
            if prev_fire is not None and len(tracked) < 4:
                new = np.argwhere(fire & ~prev_fire)
                for rc in new[:4 - len(tracked)]:
                    tracked.append(tuple(int(x) for x in rc))
            prev_fire = fire
            for r, c in tracked:
                say('   cell %d,%d: %s' % (r, c, ', '.join('%s %d' % (k, int(v[r, c])) for k, v in f.items())))
            time.sleep(max(0.0, 1.0 - (time.time() - t0) % 1.0))


def time_name(fmt):
    import time
    return time.strftime(fmt)


def main():
    import memscan, live_check
    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'data', 'live', time_name('ganja_watch_%Y%m%d_%H%M%S.log'))
        print('logging to', path, flush=True)
        watch(secs, path)
        return
    p = memscan.Proc()
    base = live_check.find_sim(p)
    n, grid, prm = read_live(p, base)
    f = decode(grid)
    print('frame', n)
    for name, arr in f.items():
        vals, counts = np.unique(arr, return_counts=True)
        if len(vals) <= 16:
            print('%-12s' % name, ', '.join('%d: %d' % (v, c) for v, c in zip(vals, counts)))
        else:
            print('%-12s %d distinct values, min %d, median %d, max %d, zero %.1f%%' % (
                name, len(vals), arr.min(), np.median(arr), arr.max(), 100 * (arr == 0).mean()))
    t = type_table(prm[0x130:0x330])
    print('\nper-type table (type 0..15):')
    for name, v in t.items():
        print('%-20s %s' % (name, ' '.join('%5d' % x for x in v)))
    tp = f['type']
    print('\nby type: cells, fire, burned, status counts, mean grow_time')
    for k in np.unique(tp):
        m = tp == k
        st = np.bincount(f['status'][m].astype(np.int64), minlength=4)
        print('type %2d: %6d cells, fire %5d, burned %5d, status %s, grow_time mean %.0f' % (
            k, m.sum(), f['fire'][m].sum(), f['burned'][m].sum(), list(st), f['grow_time'][m].mean()))


if __name__ == '__main__':
    main()
