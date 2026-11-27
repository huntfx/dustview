"""Log fire-tree water removal over time (running game, read-only): about once a second, for each given point,
the unexplained fluid loss (game's next frame minus the game's own DLL step) within 16 cells of it.

Per point and sample: total loss, cells hit (losing > 0.001), wet cells within 16, peak loss per cell, and the
reach (distance of the 3rd-farthest hit cell from the point; ~ the disc radius if the point is the tree).
Cells are assigned to the nearest point, so trees close together do not double count.

Usage: .venv/Scripts/python tools/live_firetree_watch.py "col,row;col,row;..." [seconds=300] [interval=1]
Log:   data/live/firetree_watch_<time>.tsv (one row per sample and point) + progress on stdout.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, galileo as g, dustsim as m

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pts = [tuple(float(v) for v in p.split(',')) for p in sys.argv[1].split(';') if p.strip()]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 300
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    os.makedirs(os.path.join(HERE, '..', 'data', 'live'), exist_ok=True)
    path = os.path.join(HERE, '..', 'data', 'live', time.strftime('firetree_watch_%Y%m%d_%H%M%S.tsv'))
    log = open(path, 'w', encoding='utf-8')
    log.write('t_s\ttick\tcol\trow\tloss\thit\twet\tpeak\treach\n')
    yy, xx = np.mgrid[:512, :512]
    D = np.stack([np.hypot(xx - c, yy - r) for c, r in pts])
    near = D.argmin(0)
    zones = [(near == i) & (D[i] <= 16) for i in range(len(pts))]
    p = memscan.Proc()
    base = lc.find_sim(p)
    print('logging %d points for %.0f s to %s' % (len(pts), secs, path), flush=True)
    t0 = time.time(); k = 0
    while time.time() - t0 < secs:
        try:
            cap = lc.capture_pair(p, base)
        except SystemExit:
            print('%6.0f s  (paused)' % (time.time() - t0), flush=True); time.sleep(1); continue
        if not cap:
            continue
        n, A, B, prm = cap
        P = np.frombuffer(prm[:224], np.float32).copy()
        X = m.from_grid(g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                              param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
        GA, GB = m.from_grid(A), m.from_grid(B)
        if int((GB['F'] != X['F']).any(0).sum()) > 3000:
            continue                                                    # torn read
        dF = np.abs(GB['W']) - np.abs(X['W'])
        wet = np.abs(GA['W']) > 0.05
        t = time.time() - t0
        parts = []
        for i, (c, r) in enumerate(pts):
            z = zones[i]
            q = -dF[z]
            hit = q > 1e-3
            dh = np.sort(D[i][z][hit])
            reach = dh[-3] if len(dh) >= 3 else (dh[-1] if len(dh) else 0)
            log.write('%.1f\t%d\t%.0f\t%.0f\t%.4f\t%d\t%d\t%.4f\t%.1f\n' % (
                t, n, c, r, q[hit].sum(), hit.sum(), (wet & z).sum(), q.max() if len(q) else 0, reach))
            parts.append('%5.1f@%4.1f' % (q[hit].sum(), reach))
        log.flush()
        k += 1
        if k % 10 == 0:
            print('%6.0f s tick %d | loss@reach per point: %s' % (t, n, ' '.join(parts)), flush=True)
        time.sleep(every)
    print('done', flush=True)


if __name__ == '__main__':
    main()
