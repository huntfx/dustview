"""Log what a running From Dust level adds to the grid outside the simulation step (read-only).

Every few seconds: capture two consecutive simulation states (live_check.capture_pair), run the game's
own step (DLL harness) on the first, and subtract it from the second. What's left is what the level
itself added that tick: volcano lava, waterfall/spring water, bombs, tide edges, powers.
For each connected spot it logs position, size, fluid added per tick, and for lava the resulting
temperature (lava vents set it to a fixed value). About 1 in 15 samples is a mistimed read (huge
regions); ignore those.
Terrain changes (crater/cone edits) are logged the same way.

Usage: .venv/Scripts/python tools/live_sources.py [seconds=240] [interval=3]
Log: data/live/sources_<time>.log (also printed).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, galileo as g, dustsim as m

HERE = os.path.dirname(os.path.abspath(__file__))


def regions(mask):
    """8-connected components of a boolean mask -> list of (ys, xs)."""
    lab = np.zeros(mask.shape, np.int32)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        lab[y, x] = len(out) + 1
        st, cells = [(y, x)], []
        while st:
            cy, cx = st.pop()
            cells.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 512 and 0 <= nx < 512 and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = len(out) + 1
                        st.append((ny, nx))
        ys, xs = np.array(cells).T
        out.append((ys, xs))
    return out


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 240
    every = float(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(os.path.join(HERE, '..', 'data', 'live'), exist_ok=True)
    logp = os.path.join(HERE, '..', 'data', 'live', time.strftime('sources_%Y%m%d_%H%M%S.log'))
    log = open(logp, 'w', encoding='utf-8')
    def out(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    p = memscan.Proc()
    base = lc.find_sim(p)
    out('simulator at %s; logging to %s' % (hex(base), logp))
    t0 = time.time()
    while time.time() - t0 < secs:
        cap = lc.capture_pair(p, base)
        if not cap:
            continue
        n, A, B, prm = cap
        P = np.frombuffer(prm[:224], np.float32).copy()
        step = m.from_grid(g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                                 param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
        game = m.from_grid(B)
        inside = np.zeros((512, 512), bool); inside[33:479, 33:479] = True
        dW = np.where(inside, game['W'] - step['W'], 0)       # signed: lava is negative height
        dT = np.where(inside, game['T'] - step['T'], 0)
        out('--- t=%6.1fs tick %d | lava cells %d, water cells %d' % (time.time() - t0, n + 1, (game['W'] < 0).sum(), (game['W'] > 0).sum()))
        for ys, xs in regions(np.abs(dW) > 1e-3):
            gw, sw = game['W'][ys, xs], step['W'][ys, xs]
            lava = gw < 0
            add = np.abs(gw) - np.abs(sw)                        # added depth per cell (either fluid)
            line = 'fluid  %4d cells  centre (%3d,%3d) rows %3d-%3d cols %3d-%3d | added %+8.3f total, per cell %+.3f..%+.3f' % (
                len(ys), ys.mean(), xs.mean(), ys.min(), ys.max(), xs.min(), xs.max(), add.sum(), add.min(), add.max())
            if lava.any():
                gt = game['sed'][ys, xs][lava]
                # vents overwrite the temperature (cells end at one exact value), so report the result
                line += ' | LAVA %d cells, temperature after %.2f..%.2f' % (lava.sum(), gt.min(), gt.max())
            else:
                line += ' | water'
            out('  ' + line)
        for ys, xs in regions(np.abs(dT) > 1e-4):
            d = dT[ys, xs]
            out('  terrain %4d cells centre (%3d,%3d) rows %3d-%3d cols %3d-%3d | change %+.4f..%+.4f, total %+.3f' % (
                len(ys), ys.mean(), xs.mean(), ys.min(), ys.max(), xs.min(), xs.max(), d.min(), d.max(), d.sum()))
        time.sleep(every)
    out('done')


if __name__ == '__main__':
    main()
