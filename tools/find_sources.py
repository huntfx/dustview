"""Find the game's source objects (springs, lava vents) in a running From Dust level (read-only).

1. Measures the level's sources live (game's next state minus the game's own step): centre, cells,
   amount per cell per tick, temperature for lava (same method as live_sources.py).
2. For each source, builds candidate float32 values in plausible units (per tick, per second at 30/31
   ticks, total over the disc, radius guesses, temperature internal and x8 editor units, centre in
   grid units) and searches all readable memory for them.
3. Reports memory windows (default 256 bytes) holding several of one source's candidate values: likely
   the source object. Prints the window as floats for inspection.

Usage: .venv/Scripts/python tools/find_sources.py
(The window search below, search_windows(), was the first attempt; it found nothing because the
table stores values per cell per tick. The source table itself is found by each source's centre.)
"""
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, galileo as g, dustsim as m
from live_sources import regions

RADIUS_BY_CELLS = {9: 1.5, 21: 2.3, 37: 3.5, 49: 4, 69: 4.5, 81: 5}


def measure(p, base):
    while True:
        cap = lc.capture_pair(p, base)
        if not cap:
            continue
        n, A, B, prm = cap
        P = np.frombuffer(prm[:224], np.float32).copy()
        step = m.from_grid(g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                                 param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
        game = m.from_grid(B)
        d = np.abs(game['W']) - np.abs(step['W'])
        d[:33] = d[479:] = 0; d[:, :33] = d[:, 479:] = 0
        regs = regions(np.abs(game['W'] - step['W']) > 1e-3)
        if sum(len(r[0]) for r in regs) > 5000:          # mistimed read
            continue
        out = []
        for ys, xs in regs:
            add = d[ys, xs]
            if len(ys) < 5 or add.min() <= 0 or add.max() - add.min() > 1e-3:
                continue                                  # sources: positive, uniform discs
            lava = game['W'][ys, xs] < 0
            out.append(dict(cy=(ys.min() + ys.max()) / 2, cx=(xs.min() + xs.max()) / 2, cells=len(ys),
                            rate=float(add.mean()), lava=bool(lava.all()),
                            temp=float(game['sed'][ys, xs].max()) if lava.all() else None))
        return out


def candidates(s):
    r = s['rate']; c = []
    for k, v in [('rate', r), ('rate*30', r * 30), ('rate*31', r * 31), ('total/tick', r * s['cells']),
                 ('total/s30', r * s['cells'] * 30), ('cells', s['cells']),
                 ('radius', RADIUS_BY_CELLS.get(s['cells'], 0)), ('row', s['cy']), ('col', s['cx'])]:
        if v:
            c.append((k, np.float32(v)))
    if s['temp']:
        c += [('temp', np.float32(s['temp'])), ('temp*8', np.float32(s['temp'] * 8))]
    return c


FIELDS = ['col', 'row', 'radius', 'f3', 'rate', 'f5', 'sediment', 'temperature', 'f8']


def table(p, srcs):
    """Read each measured source's 36-byte record from the game's source table (found by its centre)."""
    out = []
    for s in srcs:
        for a in p.find(struct.pack('<2f', s['cx'], s['cy'])):
            rec = dict(zip(FIELDS, (float(v) for v in struct.unpack('<9f', p.read(a, 36)))))
            if abs(rec['rate'] - s['rate']) < 1e-4:
                rec.update(address=hex(a), kind='lava' if rec['temperature'] else 'water', measured_cells=s['cells'])
                out.append(rec)
    return out


def main():
    """Measure sources, read their table records, print and save them (data/live/source_table_<time>.json)."""
    import json, time
    sys.stdout.reconfigure(encoding='utf-8')
    p = memscan.Proc()
    base = lc.find_sim(p)
    P = np.frombuffer(p.read(base + 0x20, 224), np.float32)
    srcs = measure(p, base)
    recs = table(p, srcs)
    print('border water %.2f | %d sources measured, %d table records found' % (P[31], len(srcs), len(recs)))
    for r in recs:
        print('  %-5s col %6.1f row %6.1f radius %4.1f rate %.4f sediment %.3f temperature %6.2f | f3 %g f5 %g f8 %g | %s' % (
            r['kind'], r['col'], r['row'], r['radius'], r['rate'], r['sediment'], r['temperature'], r['f3'], r['f5'], r['f8'], r['address']))
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live', time.strftime('source_table_%Y%m%d_%H%M%S.json'))
    json.dump(dict(border_water=float(P[31]), sources=recs), open(outp, 'w'), indent=1)
    print('saved', outp)


if __name__ == '__main__':
    main()
