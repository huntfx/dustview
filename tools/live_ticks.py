"""Tick-by-tick watch of a small area of a running From Dust level, plus the parameter array (read-only).

Reads only the rows/16-cell groups covering the area from the latest complete buffer, every tick, and
polls the 56 simulation parameters every ~2 ms. Logs per tick: fluid (negative = lava), temperature
and terrain summed/maxed over the area, and flags ticks where they jump; logs every parameter change
with the tick it was seen on.

Usage: .venv/Scripts/python tools/live_ticks.py <row0> <row1> <col0> <col1> [seconds=120] [delay_ms=10]
(delay: wait after each tick change before reading; sources/vents are applied a few ms after the
counter moves, so reading at once misses them on some ticks.)
Log: data/live/ticks_<time>.log
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc
from galileo import PARAM_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r0, r1, c0, c1 = (int(v) for v in sys.argv[1:5])
    secs = float(sys.argv[5]) if len(sys.argv) > 5 else 120
    delay = float(sys.argv[6]) / 1000 if len(sys.argv) > 6 else 0.010
    os.makedirs(os.path.join(HERE, '..', 'data', 'live'), exist_ok=True)
    logp = os.path.join(HERE, '..', 'data', 'live', time.strftime('ticks_%Y%m%d_%H%M%S.log'))
    log = open(logp, 'w', encoding='utf-8')
    def out(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    p = memscan.Proc()
    base = lc.find_sim(p)
    g0, g1 = c0 >> 4, c1 >> 4                        # 16-cell groups covering the columns
    ngroups = g1 - g0 + 1
    rd = lambda off, n: p.read(base + off, n)
    frame = lambda: struct.unpack('<I', rd(0x92C, 4))[0]

    def area(buf):
        """(rows, 8 comps, cols) float32 for rows r0..r1 and the covered groups."""
        rows = []
        for y in range(r0, r1 + 1):
            raw = p.read(buf + ((y << 12) + g0 * 128) * 4, ngroups * 512)
            a = np.frombuffer(raw, np.float32).reshape(ngroups, 8, 16).transpose(1, 0, 2).reshape(8, ngroups * 16)
            rows.append(a)
        a = np.stack(rows, 0)                        # y, comp, x
        return a[:, :, c0 - g0 * 16:c1 - g0 * 16 + 1]

    out('area rows %d-%d cols %d-%d; log %s' % (r0, r1, c0, c1, logp))
    out('  tick | lava vol   lava cells  max lava depth  max temp | terrain sum      max   | flags')
    bufs = struct.unpack('<2I', rd(0x950, 8))
    prevP = np.frombuffer(rd(0x20, 224), np.float32).copy()
    prev = None
    last_tick = frame()
    t0 = time.time()
    while time.time() - t0 < secs:
        P = np.frombuffer(rd(0x20, 224), np.float32).copy()
        f = frame()
        for i in np.nonzero(P != prevP)[0]:
            out('  PARAM tick %d  %2d %-38s %g -> %g' % (f, i, PARAM_NAMES[i], prevP[i], P[i]))
        prevP = P
        if f == last_tick:
            time.sleep(0.002)
            continue
        if f != last_tick + 1:
            out('  (skipped %d ticks)' % (f - last_tick - 1))
        last_tick = f
        time.sleep(delay)
        if frame() != f:
            out('  (tick %d passed during the delay)' % f); continue
        a = area(bufs[(f - 1) & 1])                  # result of the previous step: complete now
        W, Tm, T = a[:, 2, :], a[:, 3, :], a[:, 0, :]
        lava = W < 0
        cur = dict(vol=float(-W[lava].sum()), cells=int(lava.sum()), maxd=float(-W.min()) if lava.any() else 0.0,
                   maxt=float(Tm[lava].max()) if lava.any() else 0.0, tsum=float(T.sum()), tmax=float(T.max()))
        flags = []
        if prev:
            if cur['tmax'] - prev['tmax'] > 0.5 or cur['tsum'] - prev['tsum'] > 5: flags.append('TERRAIN JUMP')
            if cur['vol'] - prev['vol'] > 2: flags.append('LAVA JUMP')
            if cur['maxd'] - prev['maxd'] > 1: flags.append('DEPTH JUMP')
        out('%6d | %8.2f   %5d      %7.2f       %7.2f | %10.2f  %7.2f | %s' % (
            f - 1, cur['vol'], cur['cells'], cur['maxd'], cur['maxt'], cur['tsum'], cur['tmax'], ' '.join(flags)))
        prev = cur
    out('done')


if __name__ == '__main__':
    main()
