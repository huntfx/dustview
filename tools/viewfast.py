"""Compiled (numba) helpers for DustView's redraw: frame fields from a grid, grid packing, the natural
view and the flow texture's blur. Same results as the numpy code they replace in dustview.py /
flowtex.py (rendering, so float rounding may differ in the last bit; the simulation is not affected).
"""
import numba
import numpy as np

f32 = np.float32


@numba.njit(parallel=True, nogil=True, cache=True)
def frame_fields(grid):
    """(8, n, n) uint32 grid -> water, lava, sediment, temperature (n, n) float32 and flux (4, n, n)
    float32 in cells (1/256 units / 256), as dustview.Frame computed them."""
    n = grid.shape[1]
    fl = grid[2].view(np.float32)
    ex = grid[3].view(np.float32)
    water = np.empty((n, n), np.float32); lava = np.empty((n, n), np.float32)
    sed = np.empty((n, n), np.float32); temp = np.empty((n, n), np.float32)
    flux = np.empty((4, n, n), np.float32)
    for r in numba.prange(n):
        for c in range(n):
            f = fl[r, c]
            water[r, c] = max(f, f32(0))
            lava[r, c] = max(-f, f32(0))
            sed[r, c] = ex[r, c] if f > 0 else f32(0)
            temp[r, c] = ex[r, c] if f < 0 else f32(0)
            lr = grid[4, r, c]
            ud = grid[5, r, c]
            flux[0, r, c] = f32(lr & 0xFFFF) / f32(256)
            flux[1, r, c] = f32(lr >> 16) / f32(256)
            flux[2, r, c] = f32(ud & 0xFFFF) / f32(256)
            flux[3, r, c] = f32(ud >> 16) / f32(256)
    return water, lava, sed, temp, flux


@numba.njit(parallel=True, nogil=True, cache=True)
def pack_grid(T, So, W, sed, F, grid):
    """dustsim state arrays -> components 0-5 of an (8, n, n) uint32 grid, in place."""
    n = T.shape[0]
    g0 = grid[0].view(np.float32); g1 = grid[1].view(np.float32)
    g2 = grid[2].view(np.float32); g3 = grid[3].view(np.float32)
    for r in numba.prange(n):
        for c in range(n):
            g0[r, c] = T[r, c]; g1[r, c] = So[r, c]; g2[r, c] = W[r, c]; g3[r, c] = sed[r, c]
            grid[4, r, c] = np.uint32(F[0, r, c] & 0xFFFF) | (np.uint32(F[1, r, c] & 0xFFFF) << np.uint32(16))
            grid[5, r, c] = np.uint32(F[2, r, c] & 0xFFFF) | (np.uint32(F[3, r, c] & 0xFFFF) << np.uint32(16))


@numba.njit(inline='always')
def _interp(x, xs, cs, k):
    m = xs.shape[0]
    if x <= xs[0]:
        return cs[0, k]
    if x >= xs[m - 1]:
        return cs[m - 1, k]
    for i in range(1, m):
        if x < xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return cs[i - 1, k] + (cs[i, k] - cs[i - 1, k]) * t
    return cs[m - 1, k]


@numba.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def render_natural(ground, soft, water, lava, sed, temp, light, lava_xs, lava_cs, out):
    """dustview.render(fr, 'natural'): sand / rock shaded by the surface slope (np.gradient), water
    tinted by depth and mud, lava coloured by temperature. out: (n, n, 3) uint8."""
    n = ground.shape[0]
    L0, L1, L2 = light[0], light[1], light[2]
    for r in numba.prange(n):
        for c in range(n):
            # np.gradient: central differences inside, one-sided at the edges
            if r == 0:
                gy = (ground[1, c] + water[1, c] + lava[1, c]) - (ground[0, c] + water[0, c] + lava[0, c])
            elif r == n - 1:
                gy = (ground[r, c] + water[r, c] + lava[r, c]) - (ground[r - 1, c] + water[r - 1, c] + lava[r - 1, c])
            else:
                gy = ((ground[r + 1, c] + water[r + 1, c] + lava[r + 1, c]) - (ground[r - 1, c] + water[r - 1, c] + lava[r - 1, c])) * 0.5
            if c == 0:
                gx = (ground[r, 1] + water[r, 1] + lava[r, 1]) - (ground[r, 0] + water[r, 0] + lava[r, 0])
            elif c == n - 1:
                gx = (ground[r, c] + water[r, c] + lava[r, c]) - (ground[r, c - 1] + water[r, c - 1] + lava[r, c - 1])
            else:
                gx = ((ground[r, c + 1] + water[r, c + 1] + lava[r, c + 1]) - (ground[r, c - 1] + water[r, c - 1] + lava[r, c - 1])) * 0.5
            sh = (-gx * L0 - gy * L1 + L2) / np.sqrt(gx * gx + gy * gy + 1.0)
            sh = min(max(sh, 0.0), 1.0) * 0.75 + 0.25
            s = min(max(soft[r, c] / 1.5, 0.0), 1.0)
            c0 = (118 + (214 - 118) * s) * sh
            c1 = (108 + (188 - 108) * s) * sh
            c2 = (98 + (138 - 98) * s) * sh
            d = water[r, c]
            if d > 0:
                t = np.exp(-d / 10.0)
                w0 = 18 * (1 - t) + 70 * t
                w1 = 52 * (1 - t) + 150 * t
                w2 = 105 * (1 - t) + 185 * t
                mud = min(max(sed[r, c] / (0.02 * d + 1e-6), 0.0), 1.0) * 0.6
                w0 += (150 - w0) * mud
                w1 += (118 - w1) * mud
                w2 += (80 - w2) * mud
                a = 1 - np.exp(-d / 2.0)
                c0 += (w0 - c0) * a
                c1 += (w1 - c1) * a
                c2 += (w2 - c2) * a
            lv = lava[r, c]
            tm = temp[r, c]
            a = 1 - np.exp(-lv / 0.5)
            c0 += (_interp(tm, lava_xs, lava_cs, 0) - c0) * a
            c1 += (_interp(tm, lava_xs, lava_cs, 1) - c1) * a
            c2 += (_interp(tm, lava_xs, lava_cs, 2) - c2) * a
            out[r, c, 0] = np.uint8(min(max(c0, 0.0), 255.0))
            out[r, c, 1] = np.uint8(min(max(c1, 0.0), 255.0))
            out[r, c, 2] = np.uint8(min(max(c2, 0.0), 255.0))


@numba.njit(parallel=True, nogil=True, cache=True)
def natural_layers(ground, soft, water, lava, sed, temp, lava_xs, lava_cs, gx_out, gy_out, s_out, o_out, a_out,
                   lava_colour=True):
    """The parts of render_natural, per cell, for the textured terrain (flowtex.shade_textured): surface
    gradient (gx, gy as np.gradient of ground + water + lava), sand fraction s = soft / 1.5 clamped, and the
    water + lava layer over the ground as one colour o (n, n, 3) and opacity a: render_natural's
    c += (w - c) * aw; c += (lava - c) * al  ==  c * (1 - a) + o * a. lava_colour False: water only (the
    textured view draws lava on the terrain ramp, as the game's shader 97)."""
    n = ground.shape[0]
    for r in numba.prange(n):
        for c in range(n):
            if r == 0:
                gy = (ground[1, c] + water[1, c] + lava[1, c]) - (ground[0, c] + water[0, c] + lava[0, c])
            elif r == n - 1:
                gy = (ground[r, c] + water[r, c] + lava[r, c]) - (ground[r - 1, c] + water[r - 1, c] + lava[r - 1, c])
            else:
                gy = ((ground[r + 1, c] + water[r + 1, c] + lava[r + 1, c]) - (ground[r - 1, c] + water[r - 1, c] + lava[r - 1, c])) * 0.5
            if c == 0:
                gx = (ground[r, 1] + water[r, 1] + lava[r, 1]) - (ground[r, 0] + water[r, 0] + lava[r, 0])
            elif c == n - 1:
                gx = (ground[r, c] + water[r, c] + lava[r, c]) - (ground[r, c - 1] + water[r, c - 1] + lava[r, c - 1])
            else:
                gx = ((ground[r, c + 1] + water[r, c + 1] + lava[r, c + 1]) - (ground[r, c - 1] + water[r, c - 1] + lava[r, c - 1])) * 0.5
            gx_out[r, c] = gx
            gy_out[r, c] = gy
            s_out[r, c] = min(max(soft[r, c] / 1.5, 0.0), 1.0)
            aw = 0.0
            w0 = 0.0
            w1 = 0.0
            w2 = 0.0
            d = water[r, c]
            if d > 0:
                t = np.exp(-d / 10.0)
                w0 = 18 * (1 - t) + 70 * t
                w1 = 52 * (1 - t) + 150 * t
                w2 = 105 * (1 - t) + 185 * t
                mud = min(max(sed[r, c] / (0.02 * d + 1e-6), 0.0), 1.0) * 0.6
                w0 += (150 - w0) * mud
                w1 += (118 - w1) * mud
                w2 += (80 - w2) * mud
                aw = 1 - np.exp(-d / 2.0)
            al = 1 - np.exp(-lava[r, c] / 0.5) if lava_colour else 0.0
            a = 1 - (1 - aw) * (1 - al)
            a_out[r, c] = a
            if a > 0:
                tm = temp[r, c]
                o_out[r, c, 0] = (w0 * aw * (1 - al) + _interp(tm, lava_xs, lava_cs, 0) * al) / a
                o_out[r, c, 1] = (w1 * aw * (1 - al) + _interp(tm, lava_xs, lava_cs, 1) * al) / a
                o_out[r, c, 2] = (w2 * aw * (1 - al) + _interp(tm, lava_xs, lava_cs, 2) * al) / a
            else:
                o_out[r, c, 0] = 0.0
                o_out[r, c, 1] = 0.0
                o_out[r, c, 2] = 0.0


@numba.njit(parallel=True, nogil=True, cache=True)
def _blur_pass(a, out):
    n, m = a.shape
    q = f32(0.25)
    for r in numba.prange(n):
        rp = r + 1 if r + 1 < n else 0
        rm = r - 1 if r > 0 else n - 1
        for c in range(m):
            cp = c + 1 if c + 1 < m else 0
            cm = c - 1 if c > 0 else m - 1
            out[r, c] = q * (((a[rm, c] + a[rp, c]) + a[r, cm]) + a[r, cp])


def blur(a, passes=6):
    """flowtex._blur (passes of a wrap-around 4-neighbour average), compiled."""
    a = np.ascontiguousarray(a, np.float32)
    b = np.empty_like(a)
    for _ in range(passes):
        _blur_pass(a, b)
        a, b = b, a
    return a


@numba.njit(parallel=True, nogil=True, cache=True)
def _grass_color_opacity(c6, ramp, burn, out):
    n0, n1 = c6.shape
    for r in numba.prange(n0):
        for c in range(n1):
            a0 = f32(0); a1 = f32(0); a2 = f32(0); a3 = f32(0); burnt = f32(0)
            for k in range(5):
                dy = (0, -1, -1, 1, 1)[k]
                dx = (0, -1, 1, -1, 1)[k]
                rr = min(max(r + dy, 0), n0 - 1)              # edge cells repeat (np.pad mode='edge')
                cc = min(max(c + dx, 0), n1 - 1)
                t = c6[rr, cc] & 0xFF
                i = t & 15
                a0 += ramp[i, 0]; a1 += ramp[i, 1]; a2 += ramp[i, 2]; a3 += ramp[i, 3]
                if t >= 0x40:
                    burnt += f32(1)
            f = burnt * f32(0.2)
            av0 = a0 * f32(0.2); av1 = a1 * f32(0.2); av2 = a2 * f32(0.2); av3 = a3 * f32(0.2)
            out[r, c, 0] = av0 + (burn[0] - av0) * f
            out[r, c, 1] = av1 + (burn[1] - av1) * f
            out[r, c, 2] = av2 + (burn[2] - av2) * f
            out[r, c, 3] = av3 + (burn[3] - av3) * f


def grass_color_opacity(c6):
    """ganja.grass_color_opacity compiled (the game's GrassColorRT per cell; same result to float rounding)."""
    import ganja
    c6 = np.ascontiguousarray(c6, np.uint32)
    out = np.empty(c6.shape + (4,), np.float32)
    _grass_color_opacity(c6, np.ascontiguousarray(ganja.GRASS_RAMP, np.float32),
                         np.ascontiguousarray(ganja.BURN_GROUND, np.float32), out)
    return out
