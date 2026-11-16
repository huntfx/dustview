"""Look for the water shaders' constant values (WaterTime c36, WaterSedimentColor c37, UVScrollParams
c38, WaterNormalmapParams c40, WaterDetailParams c42) in the running game's memory (read-only).

Signature: the deferred water shader (fxcache 117) also gets the inverse view matrix f44_ViewI in
c50-c52: three float4 rows whose xyz parts are unit length and mutually perpendicular. If the engine
(or the D3D runtime) keeps a mirror of the pixel-shader constant registers, c36..c42 sit 14..8 float4s
before c50. The scan finds every such orthonormal 3-row block and prints what sits at those offsets.

Usage: python find_water_constants.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan

NAMES = {36: 'WaterTime', 37: 'WaterSedimentColor', 38: 'UVScrollParams', 39: '(c39)', 40: 'WaterNormalmapParams',
         41: '(c41)', 42: 'WaterDetailParams'}


def orthonormal_rows(f, tol=2e-3):
    """f: (n, 4) float32 rows (16-byte aligned). Index i where rows i, i+1, i+2 have unit, mutually
    perpendicular xyz."""
    with np.errstate(all='ignore'):
        x = f[:, :3].astype(np.float64)
        ln = np.einsum('ij,ij->i', x, x)
        ok = np.abs(ln - 1) < tol
        cand = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:])
        if not len(cand):
            return cand
        a, b, c = x[cand], x[cand + 1], x[cand + 2]
        d = np.maximum.reduce([np.abs((a * b).sum(1)), np.abs((a * c).sum(1)), np.abs((b * c).sum(1))])
        good = d < tol
        # the 4th column of an inverse view is the camera position: finite, not all zero
        w = f[cand, 3]
        good &= np.isfinite(w) & (np.abs(f[cand, 3]) + np.abs(f[cand + 1, 3]) + np.abs(f[cand + 2, 3]) > 1) \
            & (np.abs(f[cand, 3]) < 5000)
        return cand[good]


def main():
    p = memscan.Proc()
    t0 = time.time()
    found = []
    for base, size in p.regions():
        if size < 4096:
            continue
        for off in range(0, size, 1 << 24):
            n = min(1 << 24, size - off)
            raw = p.read(base + off, n)
            if raw is None:
                continue
            f = np.frombuffer(raw[: (len(raw) // 16) * 16], np.float32).reshape(-1, 4)
            for i in orthonormal_rows(f):
                if i >= 14:
                    found.append((base + off + 16 * int(i), f[i - 14:i + 3].copy()))
    print('scan %.0f s, %d orthonormal 3-row blocks with a translation' % (time.time() - t0, len(found)), flush=True)
    # keep blocks whose c36..c42 look like plausible parameters (finite, moderate)
    for addr, blk in found:
        c = blk[:7]
        if not np.all(np.isfinite(c)) or np.abs(c).max() > 1e4 or np.abs(c).max() == 0:
            continue
        print('c50 at %08x  (c36 at %08x)' % (addr, addr - 14 * 16))
        for k in range(7):
            print('   c%d %-22s %s' % (36 + k, NAMES[36 + k], ' '.join('%10.4f' % v for v in c[k])))
        print('   c50-52 rows:', ' | '.join(' '.join('%.3f' % v for v in blk[14 + r]) for r in range(3)))


if __name__ == '__main__':
    main()
