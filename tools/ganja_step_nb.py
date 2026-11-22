"""numba version of ganja_step.step (From Dust's vegetation / fire step), bit-identical to the numpy one.

ganja_step.step(..., fast=True) (the default when numba is installed) calls step_cells() here.
Check after any rule change: python ganja_step_nb.py (compares numpy vs numba on levels and random grids).
"""
import numpy as np
from numba import njit, prange

N = 512
BORDER = 32
# rows of the table array passed in (ganja_step.TABLE_ORDER)
FI, DUR, RGT, EVO, ADULT, NEXT, FS, FT, LWMIN, LWMAX, ANIM, RESIST, SWMIN, SWMAX, SDM = range(15)


@njit(inline='always')
def s16(x):
    return ((x + 0x8000) & 0xFFFF) - 0x8000


@njit(inline='always')
def q128(x):
    """float32 x 128 -> round to nearest even -> saturate to int16 (cvtps2dq + packssdw)."""
    v = np.float32(x) * np.float32(128.0)
    r = np.rint(v)
    if r > 32767.0:
        return 32767
    if r < -32768.0:
        return -32768
    return np.int64(r)


@njit(cache=True, nogil=True, parallel=True)
def step_cells(c6_in, c7_in, T, So, W, tab, prop, rcp, soft_min, default_growing, fire_progress, burned_time):
    lo, hi = BORDER, N - BORDER          # processed rows / cols [lo, hi)
    c6n = c6_in.copy()
    c7n = c7_in.copy()
    # working copies per cell: plant byte (b0w incl. the lava fire bit), w, sand_ok, fire_in
    b0w = np.zeros((N, N), np.int64)
    wq = np.zeros((N, N), np.int64)
    sok = np.zeros((N, N), np.bool_)
    fin = np.zeros((N, N), np.int64)
    for r in prange(lo, hi):
        for c in range(lo, hi):
            v6 = np.int64(c6_in[r, c])
            if c == lo or c == hi - 1 or r == lo:
                v6 = 0
            b0 = v6 & 0xFF
            t = b0 & 15
            lava = W[r, c] < np.float32(0.0)
            if lava and t != 0:
                b0 |= 0x40
            b0w[r, c] = b0
            fin[r, c] = 1 if (b0 & 0x40) != 0 else 0
            if lava:
                wq[r, c] = 0
                sok[r, c] = 0 > soft_min
            else:
                wq[r, c] = q128(abs(W[r, c]))
                sok[r, c] = q128(So[r, c]) > soft_min

    for r in prange(lo, hi):
        for c in range(lo, hi):
            if r == lo or r == hi - 1:   # ring rows come out cleared (the last one still serves as a neighbour above)
                c6n[r, c] = 0
                c7n[r, c] = 0
                continue
            v6 = np.int64(c6_in[r, c])
            ring = c == lo or c == hi - 1 or r == lo
            if ring:
                v6 = 0
            v7 = np.int64(c7_in[r, c])
            b0 = v6 & 0xFF
            b1 = (v6 >> 8) & 0xFF
            t = b0 & 15
            rnd = b1 & 15
            vt = v7 & 0xFFFF
            ft = v7 >> 16
            burned = (b0 & 0x80) != 0
            has = t != 0
            w = wq[r, c]
            sand_ok = sok[r, c]

            # ---- viability, burning neighbours, spread
            same4 = has
            nfire = 0
            spread = 0
            for k in range(4):
                rr = r - 1 if k == 0 else (r + 1 if k == 1 else r)
                cc = c - 1 if k == 2 else (c + 1 if k == 3 else c)
                inside = lo <= rr < hi and lo <= cc < hi
                nb = b0w[rr, cc] if inside else 0
                nt = nb & 15
                if nt != t:
                    same4 = False
                nfire += fin[rr, cc] if inside else 0
                if (nb & 0x70) == 0x30:
                    dh = q128(abs(T[r, c] - T[rr, cc]))
                    if dh < tab[SDM, nt] and tab[SWMIN, nt] <= w and w < tab[SWMAX, nt] and sand_ok:
                        if nt > spread:
                            spread = nt
            nfire &= 3
            life_ok = tab[LWMIN, t] <= w and w < tab[LWMAX, t]
            viable = has and ((life_ok and sand_ok) or same4)
            spread_t = prop[spread]

            # ---- heat / burn timer
            thr = tab[FI, t] + 8 * rnd
            heating = (not burned) and s16(ft) < s16(thr)
            if heating:
                ft1 = (ft + 1) >> 1 if nfire == 0 else ft + 8 * nfire
            else:
                ft1 = ft + fire_progress
            ft1 &= 0xFFFF
            if not (s16(tab[RESIST, t]) > s16(w)):
                ft1 = 0
            fire = (not burned) and s16(ft1) > s16(thr)

            # ---- vegetation time and phases
            E = tab[EVO, t]
            RE = (tab[RGT, t] + rnd + E) & 0xFFFF
            RAE = (RE + tab[ADULT, t]) & 0xFFFF
            END = (RAE + E) & 0xFFFF
            past_young = s16(vt) > s16(RE)
            dying = (not (s16(RAE) > s16(vt))) and past_young
            if viable or dying:
                v1 = vt + default_growing
            elif past_young:
                v1 = RAE
            else:
                v1 = END - vt
            v1 &= 0xFFFF
            entered = viable and (not dying) and not (s16(RAE) > s16(v1))
            if entered and (tab[NEXT, t] & 0x8000) == 0:
                v1 = (v1 - (tab[ADULT, t] - 1)) & 0xFFFF
            if not has:
                v1 = 0
            status = 0x20 if s16(v1) > s16(RE) else 0x10
            if (not (s16(tab[FS, t] + RE) > s16(v1))) and s16(tab[FT, t] + RE) > s16(v1):
                status |= 0x10
            if not (s16(RAE) > s16(v1)):
                status = 0
            dead = s16(v1) > s16(END)
            newb = t | status | (0x40 if fire else 0)
            nxt = 0
            if dead and ft1 == 0:
                nxt = tab[NEXT, t] & 0xFF
                n4 = nxt & 15
                if not (tab[LWMIN, n4] <= w and w < tab[LWMAX, n4] and nxt != 0 and sand_ok):
                    nxt = 0
            if dead:
                newb = 0
                v1 = 0
                ft1 = 0

            # ---- fire end, burned phase
            if fire and s16(ft1) > s16(tab[DUR, t] - tab[ANIM, t]):
                if RAE > v1:
                    v1 = RAE
            burnout = s16(ft1) > s16(tab[DUR, t])
            if burnout:
                newb = 0
            B = burnout or burned
            if B:
                ftb = 0 if burnout else (ft + 1) & 0xFFFF
                if ftb > burned_time:
                    ft1 = 0
                else:
                    ft1 = ftb
                    newb = 0x80

            # ---- empty: new plant
            if (newb & 0x8F) == 0:
                newb = (nxt if nxt != 0 else spread_t) | 0x10
                v1 = 0
                ft1 = 0

            # ---- byte 1 and display bytes
            die = (has and not viable) or (b1 & 0x10) != 0
            t2 = newb & 15
            E2 = tab[EVO, t2]
            RE2 = (E2 + rnd + tab[RGT, t2]) & 0xFFFF
            FS2 = tab[FS, t2]
            FT2 = tab[FT, t2]
            X1 = (RE2 + FS2) & 0xFFFF
            idx = 0
            rel = v1
            length = RE2
            ln2 = ((FT2 - FS2) & 0xFFFF) >> 3
            if s16(v1) > s16(X1):
                idx += 1
                rel = (v1 - X1) & 0xFFFF
                length = ln2
            X2 = (RE2 + FT2 - ln2) & 0xFFFF
            if s16(v1) > s16(X2):
                idx += 1
                rel = (v1 - X2) & 0xFFFF
                length = ln2
            X3 = (RE2 + tab[ADULT, t2]) & 0xFFFF
            if s16(v1) > s16(X3):
                idx += 1
                rel = (v1 - X3) & 0xFFFF
                length = E2
            x = (rcp[length] * np.float32(rel)) * np.float32(256.0)
            if np.isfinite(x) and x < np.float32(2147483648.0):
                ai = np.int64(np.rint(x))
                if ai > 255:
                    ai = 255
            else:
                ai = 255
            static = 1 if s16(rel) > s16(length) else 0
            b1n = rnd | (0x10 if die else 0)
            if t2 != 0:
                b1n |= (static << 5) | ((idx & 3) << 6)
            thr2 = tab[FI, t2] + rnd
            fu = ft1 & 0xFFFF
            bb = 0
            if fu > (thr2 & 0xFFFF):
                bb = (fu - thr2) & 0xFFFF
                if bb > 255:
                    bb = 255
            c6n[r, c] = np.uint32(newb | (b1n << 8) | (ai << 16) | (bb << 24))
            c7n[r, c] = np.uint32((v1 & 0xFFFF) | ((ft1 & 0xFFFF) << 16))
    return c6n, c7n


def main():
    import os
    import sys
    import time
    import vegsim
    import compare_sim
    import ganja_step as gs
    from stress_ganja import random_grid
    grids = [('Map6', vegsim.from_sav(os.path.join(compare_sim.SAVES, 'ALPHA_Map6_Fire.sav')), 0),
             ('BigWave', vegsim.from_sav(os.path.join(compare_sim.SAVES, 'Defi_006_BigWave.sav')), 0)]
    grids += [(f'random {s} set {s % 4}', random_grid(s), s % 4) for s in (10, 21, 32, 43)]
    ok = True
    for name, grid, vset in grids:
        blk = vegsim.ganja_block(vset)
        prop = vegsim.propagate_table()
        cur = grid
        for i in range(20):
            a6, a7 = gs.step(cur, blk, prop, fast=False)
            b6, b7 = gs.step(cur, blk, prop, fast=True)
            nb = int((a6 != b6).sum() + (a7 != b7).sum())
            if nb:
                ok = False
                r, c = np.argwhere((a6 != b6) | (a7 != b7))[0]
                print(f'{name} tick {i + 1}: {nb} differences, e.g. r{r} c{c} numpy {a6[r, c]:08x} {a7[r, c]:08x} numba {b6[r, c]:08x} {b7[r, c]:08x}')
                break
            cur = cur.copy()
            cur[6], cur[7] = a6, a7
        else:
            print(f'{name}: numpy and numba identical over 20 ticks')
    blk = vegsim.ganja_block(0)
    t0 = time.perf_counter()
    for _ in range(50):
        gs.step(grids[0][1], blk, prop, fast=True)
    print('numba step: %.1f ms' % ((time.perf_counter() - t0) / 50 * 1000))
    print('IDENTICAL' if ok else 'DIFFERENT')


if __name__ == '__main__':
    main()
