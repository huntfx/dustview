"""Unexplained fluid change around given points (e.g. fire trees) from a live_firetree.py .npz.

Usage: .venv/Scripts/python tools/firetree_spots.py <file.npz> "col,row;col,row;..."
Per point: water depth at the point, unexplained change per tick summed within 4 / 8 / 12 / 16 cells, the
centre of the loss within 16 cells, and a radial profile (mean change per cell per tick by distance).
Then every loss spot on the map (connected cells losing > 0.001 per tick), so spots without a listed point show.
"""
import sys
import numpy as np


def main():
    d = np.load(sys.argv[1])
    dF = d['dF_sum'] / max(int(d['n_pairs']), 1)
    W = np.abs(d['water'])
    pts = [tuple(float(v) for v in p.split(',')) for p in sys.argv[2].split(';') if p.strip()]
    yy, xx = np.mgrid[:512, :512]
    print('%d pairs' % int(d['n_pairs']))
    print('  col  row | depth | change/tick within 4 / 8 / 12 / 16 | loss centre (col,row) | profile r=0..15 (x1000)')
    for c, r in pts:
        dist = np.hypot(xx - c, yy - r)
        sums = [dF[dist <= k].sum() for k in (4, 8, 12, 16)]
        sel = (dist <= 16) & (dF < 0)
        w = -dF[sel]
        cen = '%.1f,%.1f' % ((xx[sel] * w).sum() / w.sum(), (yy[sel] * w).sum() / w.sum()) if w.sum() > 0 else '-'
        prof = [dF[(dist >= k) & (dist < k + 1)].mean() * 1000 for k in range(16)]
        print('%5.0f %4.0f | %5.2f | %+7.2f %+7.2f %+7.2f %+7.2f | %-21s | %s' % (
            c, r, W[int(r), int(c)], *sums, cen, ' '.join('%4.0f' % v for v in prof)))
    lo = dF < -1e-3
    lab = np.zeros(dF.shape, int); k = 0
    for y, x in zip(*np.nonzero(lo)):
        if lab[y, x]:
            continue
        k += 1; st = [(y, x)]; lab[y, x] = k
        while st:
            a, b = st.pop()
            for u in range(max(a - 2, 0), min(a + 3, 512)):
                for v in range(max(b - 2, 0), min(b + 3, 512)):
                    if lo[u, v] and not lab[u, v]:
                        lab[u, v] = k; st.append((u, v))
    print('\nloss spots on the map (> 0.001 per cell per tick):')
    for i in range(1, k + 1):
        s = lab == i
        if s.sum() < 3:
            continue
        ys, xs = np.nonzero(s); w = -dF[s]
        cy, cx = (ys * w).sum() / w.sum(), (xs * w).sum() / w.sum()
        print('  col %5.1f row %5.1f | %3d cells | radius %4.1f | %6.2f per tick | peak %.3f per cell | depth %.2f' % (
            cx, cy, s.sum(), np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()), w.sum(), w.max(), W[s].mean()))


if __name__ == '__main__':
    main()
