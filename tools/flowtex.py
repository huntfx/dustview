"""Flow texture for DustView, following the game's own water rendering (as logged from the running game
with the hook's Direct3D logging; docs/technical-notes.md "Water rendering"). Needs numba.

Per cell (once per simulation frame):
  v = net outflow normalised as the game's CPU texture stores it (x: 30.7 / 127.5 per unit of outflow,
      y: 28.7 / 127.5; clamped to -1..1), times the measured edge fade (0 up to 36 cells from the map
      edge, full from 70), then blurred with 6 passes of a 4-neighbour average (fitted: corr 0.999,
      slope 1.00 against the game's WaterParams.xy)
  foam factor = 6-pass blur of (2 |v| - 1/255) x a foam edge fade (fitted: corr 0.996-0.999, slope 1.0).
      The foam edge fade depends on the level: levels with border water (3, 4, 8, 9, 11, 13) use the
      converter variant fxcache 343, which multiplies foam by the blue channel of the level's
      `colmap*.PC.tdt` (Tex_<id> archive of the level's Wor_<id> with NN.wor); the others (fxcache 105)
      use the same geometric fade as the velocity. set_level(n) picks it; without a level the
      geometric fade is used.
Per frame (fxcache 107 / 117):
  two UV-offset layers are dragged along v (look upstream, bilinear) at 22.4 cells/s per unit of v
  (512 x 0.5 x UVScrollParams.y 0.00293 per frame x 30 frames/s; GroundParamsUVScroll.x = 0 over
  water), |v| < 0.01 does not move; each layer restarts every 2.22 s (phase a triangle wave 0 -> 1 -> 0 at 0.03 per frame, 30 fps; half a cycle apart), blended so a
  layer is invisible when it restarts; no relaxation
  detail texture tiled 12 x across the map: ripple normal strip WaterSeqNM128 read as 8 frames of
  256 x 128 (shown square), foam Foam_D 512 x 512; final foam = saturate(foam texture x 1 x foam factor)
  (UVScrollParams.w = 1), blended towards white.
Not from the game: the ripple shading (a brightness term instead of the game's lighting), the ripple
frame rate (4 frames/s, not measured) .
debug_image(mode) shows the inputs: 'foam' (foam factor), 'velocity' (direction as hue, speed as
brightness), 'drag' (how far each layer's texture coordinates have been dragged).
"""
import os, sys
import numpy as np
import numba

TEX_BF = r'D:\Games\Steam\steamapps\common\From Dust\BinPC\Tex_BD00993D.lin.bf'
LIGHT = np.array([-0.55, -0.55, 0.63]) / np.linalg.norm([-0.55, -0.55, 0.63])
SPEED = 512 * 0.5 * 0.00293 * 30          # cells per second per unit of normalised velocity (22.5)
LAYER_PERIOD = 2 / (0.03 * 30)             # s: the game's layer phase (WaterTime.y) is a triangle wave 0 -> 1 -> 0 at 0.03 per frame,
                                           # each layer reset at the end where its weight is 0: a layer lives 2.22 s at 30 fps
TILES = 12.0                               # detail texture repeats across the map


def _noise(n, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    for o in range(octaves):
        k = 4 * 2 ** o
        out += np.kron(rng.standard_normal((k, k)).astype(np.float32), np.ones((n // k, n // k), np.float32)) / 2 ** o
    return out


def load_textures():
    """(ripple frames (8, 256, 256) brightness around 0, foam (512, 512) 0..1, source)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bf, tdt
        px, _ = tdt.decode(bf.read_entry(TEX_BF, 'WaterSeqNM128.PC.tdt'))
        n = px[0][..., :3].astype(np.float32) / 127.5 - 1          # (128, 2048, 3) normals
        b = n @ LIGHT.astype(np.float32)
        frames = np.stack([np.repeat(b[:, 256 * k:256 * (k + 1)], 2, axis=0) for k in range(8)])
        fp, _ = tdt.decode(bf.read_entry(TEX_BF, 'Foam_D.PC.tdt'))
        foam = fp[0][..., :3].astype(np.float32).mean(-1) / 255
        src = 'game textures (WaterSeqNM128, Foam_D)'
    except Exception as e:                                       # no game files: procedural stand-in
        frames = np.stack([_noise(256, k) for k in range(8)])
        foam = np.clip(_noise(512, 99) * 0.25 + 0.3, 0, 1)
        src = 'procedural (%s)' % e
    frames = (frames - frames.mean()) / max(frames.std(), 1e-6)
    return np.ascontiguousarray(frames, np.float32), np.ascontiguousarray(foam, np.float32), src


@numba.njit(parallel=True, fastmath=True, cache=True)
def _cell_fields(depth, flux, lava, min_depth, border, vx, vy, strength, foamk):
    """Unblurred per-cell velocity (normalised, faded), ripple strength and foam factor (faded)."""
    n = depth.shape[0]
    for r in numba.prange(n):
        for c in range(n):
            d = depth[r, c]
            vx[r, c] = np.float32(0)
            vy[r, c] = np.float32(0)
            strength[r, c] = np.float32(0)
            foamk[r, c] = np.float32(0)
            dr = np.float32(min(r, n - 1 - r))
            dc = np.float32(min(c, n - 1 - c))
            fv = min(max((dr - 36) / 34, np.float32(0)), np.float32(1)) * min(max((dc - 36) / 34, np.float32(0)), np.float32(1))
            ff = border[r, c] if border[0, 0] >= 0 else fv            # level texture (343) or geometric (105)
            water = d > min_depth and lava[r, c] <= 0
            if not (water or lava[r, c] > 0):                # dry: no flux, so no velocity / foam
                continue
            # net outflow, also over lava (logged: the upload's velocity bytes over lava = net outflow, same scale)
            nx = min(max((flux[1, r, c] - flux[0, r, c]) * np.float32(30.7 / 127.5), np.float32(-1)), np.float32(1))
            ny = min(max((flux[2, r, c] - flux[3, r, c]) * np.float32(28.7 / 127.5), np.float32(-1)), np.float32(1))
            sp = fv
            if not water:                                    # lava: GroundParamsUVScroll.x ~0.92 (dump), scroll factor
                sp *= np.float32(LAVA_SCROLL)                # lerp(0.5 x c38.y, c36.x, x) instead of 0.5 x c38.y
            vx[r, c] = nx * sp
            vy[r, c] = ny * sp
            if water:
                dw = min(d * np.float32(64.0 / 255.0), np.float32(1))
                strength[r, c] = np.float32(0.05) + np.float32(0.15) * np.sqrt(dw)
            z = np.float32(2) * np.sqrt(nx * nx + ny * ny) - np.float32(1.0 / 255.0)
            foamk[r, c] = min(max(z, np.float32(0)), np.float32(1)) * ff


@numba.njit(fastmath=True, inline='always')
def _bilinear2(D, k, y, x):
    """Bilinear sample of D[k, :, :, 0:2] at cell coordinates (y, x), clamped to the grid."""
    n = D.shape[1]
    y = min(max(y, np.float32(0)), np.float32(n - 1))
    x = min(max(x, np.float32(0)), np.float32(n - 1))
    y0 = int(y)
    x0 = int(x)
    y1 = min(y0 + 1, n - 1)
    x1 = min(x0 + 1, n - 1)
    fy = y - y0
    fx = x - x0
    a0 = D[k, y0, x0, 0] + (D[k, y0, x1, 0] - D[k, y0, x0, 0]) * fx
    b0 = D[k, y1, x0, 0] + (D[k, y1, x1, 0] - D[k, y1, x0, 0]) * fx
    a1 = D[k, y0, x0, 1] + (D[k, y0, x1, 1] - D[k, y0, x0, 1]) * fx
    b1 = D[k, y1, x0, 1] + (D[k, y1, x1, 1] - D[k, y1, x0, 1]) * fx
    return a0 + (b0 - a0) * fy, a1 + (b1 - a1) * fy


@numba.njit(parallel=True, fastmath=True, cache=True)
def _advect(Dold, vx, vy, step, reset, Dnew):
    """fxcache 107 on offsets D = uv - cell: look upstream by v x step (cells), take the old offset there
    (bilinear) and add the step; a layer with reset set starts again from 0; |v| < 0.01 does not move."""
    n = vx.shape[0]
    for r in numba.prange(n):
        for c in range(n):
            ux = vx[r, c]
            uy = vy[r, c]
            if ux * ux + uy * uy < np.float32(0.0001):
                ux = np.float32(0)
                uy = np.float32(0)
            dx = -ux * step
            dy = -uy * step
            for k in range(2):
                if reset[k]:
                    Dnew[k, r, c, 0] = np.float32(0)
                    Dnew[k, r, c, 1] = np.float32(0)
                    continue
                ox, oy = _bilinear2(Dold, k, np.float32(r) + dy, np.float32(c) + dx)
                Dnew[k, r, c, 0] = ox + dx
                Dnew[k, r, c, 1] = oy + dy


@numba.njit(parallel=True, fastmath=True, cache=True)
def _shade_uv(base, depth, lava, min_depth, s, D, strength, foamk, frame, foam, wt, kr, kf, out):
    """Every pixel: water pixels sample the ripple frame and the foam texture at (pixel + offset of each
    layer) x tiling, blend the two layers with wt; foam = saturate(foam texture x foam factor)."""
    n = depth.shape[0]
    fh, fw = frame.shape
    gh, gw = foam.shape
    for r in numba.prange(n):
        for c in range(n):
            water = depth[r, c] > min_depth and lava[r, c] <= 0
            b0 = np.float32(base[r, c, 0])
            b1 = np.float32(base[r, c, 1])
            b2 = np.float32(base[r, c, 2])
            st = strength[r, c]
            fk = foamk[r, c]
            for i in range(s):
                for j in range(s):
                    R = r * s + i
                    C = c * s + j
                    if not water:
                        out[R, C, 0] = base[r, c, 0]
                        out[R, C, 1] = base[r, c, 1]
                        out[R, C, 2] = base[r, c, 2]
                        continue
                    y = np.float32(r) + (np.float32(i) + np.float32(0.5)) / s
                    x = np.float32(c) + (np.float32(j) + np.float32(0.5)) / s
                    rip = np.float32(0)
                    fo = np.float32(0)
                    for k in range(2):
                        ox, oy = _bilinear2(D, k, y - np.float32(0.5), x - np.float32(0.5))
                        px = x + ox
                        py = y + oy
                        rip += wt[k] * frame[np.int64(np.floor(py * kr)) & (fh - 1), np.int64(np.floor(px * kr)) & (fw - 1)]
                        if fk > 0:
                            fo += wt[k] * foam[np.int64(np.floor(py * kf)) & (gh - 1), np.int64(np.floor(px * kf)) & (gw - 1)]
                    m = 1 + st * rip
                    c0 = b0 * m
                    c1 = b1 * m
                    c2 = b2 * m
                    if fk > 0:
                        fa = min(fo * fk, np.float32(1))
                        c0 += (235 - c0) * fa
                        c1 += (240 - c1) * fa
                        c2 += (245 - c2) * fa
                    out[R, C, 0] = np.uint8(min(max(c0, np.float32(0)), np.float32(255)))
                    out[R, C, 1] = np.uint8(min(max(c1, np.float32(0)), np.float32(255)))
                    out[R, C, 2] = np.uint8(min(max(c2, np.float32(0)), np.float32(255)))


@numba.njit(fastmath=True, inline='always')
def _bilin(a, y, x):
    """Bilinear sample of a 2-D field at cell-centre coordinates (y, x) (cell i's centre at i), clamped."""
    n = a.shape[0]
    y = min(max(y, np.float32(0)), np.float32(n - 1))
    x = min(max(x, np.float32(0)), np.float32(n - 1))
    y0 = int(y)
    x0 = int(x)
    y1 = min(y0 + 1, n - 1)
    x1 = min(x0 + 1, n - 1)
    fy = y - y0
    fx = x - x0
    a0 = a[y0, x0] + (a[y0, x1] - a[y0, x0]) * fx
    b0 = a[y1, x0] + (a[y1, x1] - a[y1, x0]) * fx
    return a0 + (b0 - a0) * fy


@numba.njit(fastmath=True, inline='always')
def _wrap2(t, u, v):
    """Bilinear, wrapping sample of a 2-D texture t at texel coordinates (u, v) (texel centres at +0.5)."""
    h, w = t.shape
    u = u - np.float32(0.5)
    v = v - np.float32(0.5)
    x0f = np.floor(u)
    y0f = np.floor(v)
    fx = u - x0f
    fy = v - y0f
    x0 = np.int64(x0f) % w
    y0 = np.int64(y0f) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    a0 = t[y0, x0] + (t[y0, x1] - t[y0, x0]) * fx
    b0 = t[y1, x0] + (t[y1, x1] - t[y1, x0]) * fx
    return a0 + (b0 - a0) * fy


@numba.njit(parallel=True, fastmath=True, cache=True)
def _shade_window(b0, b1, b2, wet, gy0, gx0, k, D, strength, foamk, frame, foam, wt, kr, kf, out):
    """Like _shade_uv for a window of the grid: output pixel (i, j) shows grid point
    (gy0 + (i + 0.5) / k, gx0 + (j + 0.5) / k). Terrain / water colour (b0-b2), the water fraction, ripple
    strength and foam factor are interpolated between cell centres; the ripple and foam textures are
    sampled per pixel, so zooming in shows them at full detail instead of enlarged pixels."""
    H, W = out.shape[0], out.shape[1]
    fh, fw = frame.shape
    gh, gw = foam.shape
    half = np.float32(0.5)
    for i in numba.prange(H):
        y = gy0 + (np.float32(i) + half) / k
        for j in range(W):
            x = gx0 + (np.float32(j) + half) / k
            yc = y - half
            xc = x - half
            c0 = _bilin(b0, yc, xc)
            c1 = _bilin(b1, yc, xc)
            c2 = _bilin(b2, yc, xc)
            w = _bilin(wet, yc, xc)
            if w > 0:
                st = _bilin(strength, yc, xc)
                fk = _bilin(foamk, yc, xc)
                rip = np.float32(0)
                fo = np.float32(0)
                for kk in range(2):
                    ox, oy = _bilinear2(D, kk, yc, xc)
                    px = x + ox
                    py = y + oy
                    rip += wt[kk] * _wrap2(frame, px * kr, py * kr)
                    if fk > 0:
                        fo += wt[kk] * _wrap2(foam, px * kf, py * kf)
                m = 1 + st * rip * w
                c0 *= m
                c1 *= m
                c2 *= m
                if fk > 0:
                    fa = min(fo * fk, np.float32(1)) * w
                    c0 += (235 - c0) * fa
                    c1 += (240 - c1) * fa
                    c2 += (245 - c2) * fa
            out[i, j, 0] = np.uint8(min(max(c0, np.float32(0)), np.float32(255)))
            out[i, j, 1] = np.uint8(min(max(c1, np.float32(0)), np.float32(255)))
            out[i, j, 2] = np.uint8(min(max(c2, np.float32(0)), np.float32(255)))


# ---- textured terrain (natural view): the game's ground textures, after terrain shader fxcache 96
GROUND_TILES = 16.0        # shader 96: ground UV = world x 1/768 + 0.5 (c131.z / .w, logged; map ~768 units) x 16
RAMP_V_HARD, RAMP_V_SOFT = 0.375, 0.625  # af4_TerrainRampUV c59 .z / .y as logged from the game (rock / sand)
RAMP_U = 0.5               # ramp U = GroundParams.w (logged 0.75-1.0) - |view depth| x 0.0035: camera-dependent, chosen
BUMP = 0.6                 # strength of the textures' normal perturbation against the cell slope (chosen)
SHOW_VEGETATION = True     # natural view: plants from grid component 6 (textured view: the game's grass layer; plain view: ganja.colours)
GROUND_GAIN = 1.35         # brightness (the game's light intensities are not modelled)
LAVA_V = 0.125             # af4_TerrainRampUV .w (logged): the ramp's lava row; V above it is orange -> red
LAVA_SCROLL = 1 + 0.92 * (0.0032 / (0.5 * 0.00293) - 1)   # lava's flow-texture speed vs water's (2.09): see _cell_fields
LAVA_FADE = 0.3            # s: lava mask eases towards its new value (game fades cells in / out; time constant chosen)
LAVA_MASK_BLUR = 1         # passes of the 4-neighbour blur on the lava mask (chosen by eye, see shade_textured)
LAVA_TILES = 8.0           # shader 97: flow-layer UV x 8 for LavaGround_D
LAVA_CLOCK = 1.543         # c68.x per second (0.0514 per frame at 30 fps, logged); weights c69 = 0.5 + 0.5 cos(clock - phase)
LAVA_PHASES = (np.pi / 2, -np.pi / 6, -5 * np.pi / 6)   # R, G, B of LavaGround_D (fitted, max error 0.0007)
_LAVA_TEX = None


def load_lava_texture():
    """LavaGround_D (512 x 512 DXT5, Tex_BD00993D; matched exactly against the hook's dump of shader 97's s10)
    as mips of float32 0..1 RGBA."""
    global _LAVA_TEX
    if _LAVA_TEX is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bf, tdt
        px, _ = tdt.decode(bf.read_entry(TEX_BF, 'LavaGround_D.PC.tdt'))
        a = px[0].astype(np.float32) / 255
        out = [np.ascontiguousarray(a)]
        while a.shape[0] > 1:
            a = 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])
            out.append(np.ascontiguousarray(a))
        _LAVA_TEX = out
    return _LAVA_TEX
_GROUND = None


def load_ground_textures():
    """(soft mips, hard mips, ramp, grass mips): SoftGround_2_D / HardGround_2_D / Grass_D (1024 x 1024 RGBA;
    mips built here by 2 x 2 averaging) as float32 0..1, and RampeGround_D's column at RAMP_U as (256, 3)."""
    global _GROUND
    if _GROUND is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bf, tdt

        def mips(name):
            px, _ = tdt.decode(bf.read_entry(TEX_BF, name))
            a = px[0].astype(np.float32) / 255
            out = [np.ascontiguousarray(a)]
            while a.shape[0] > 1:
                a = 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])
                out.append(np.ascontiguousarray(a))
            return out
        rpx, _ = tdt.decode(bf.read_entry(TEX_BF, 'RampeGround_D.PC.tdt'))
        col = int(RAMP_U * (rpx[0].shape[1] - 1))
        ramp = np.ascontiguousarray(rpx[0][:, col, :3].astype(np.float32))
        _GROUND = (mips('SoftGround_2_D.PC.tdt'), mips('HardGround_2_D.PC.tdt'), ramp, mips('Grass_D.PC.tdt'))
    return _GROUND


@numba.njit(fastmath=True, inline='always')
def _tex4(t, u, v, ch):
    """Bilinear, wrapping sample of channel ch of texture t (h, w, 4) at texel coordinates (u, v)."""
    h, w = t.shape[0], t.shape[1]
    u = u - np.float32(0.5)
    v = v - np.float32(0.5)
    x0 = np.int64(np.floor(u))
    y0 = np.int64(np.floor(v))
    fx = u - x0
    fy = v - y0
    xa = x0 % w
    xb = (x0 + 1) % w
    ya = y0 % h
    yb = (y0 + 1) % h
    a0 = t[ya, xa, ch] + (t[ya, xb, ch] - t[ya, xa, ch]) * fx
    b0 = t[yb, xa, ch] + (t[yb, xb, ch] - t[yb, xa, ch]) * fx
    return a0 + (b0 - a0) * fy


@numba.njit(fastmath=True, inline='always')
def _lerp_at(a, y0, x0, y1, x1, fy, fx):
    t0 = a[y0, x0] + (a[y0, x1] - a[y0, x0]) * fx
    t1 = a[y1, x0] + (a[y1, x1] - a[y1, x0]) * fx
    return t0 + (t1 - t0) * fy


@numba.njit(fastmath=True, inline='always')
def _lerp_at3(a, ch, y0, x0, y1, x1, fy, fx):
    t0 = a[y0, x0, ch] + (a[y0, x1, ch] - a[y0, x0, ch]) * fx
    t1 = a[y1, x0, ch] + (a[y1, x1, ch] - a[y1, x0, ch]) * fx
    return t0 + (t1 - t0) * fy


@numba.njit(parallel=True, fastmath=True, cache=True)
def _shade_textured(gxa, gya, sfa, oa, aa, wet, gy0, gx0, k, soft_t, hard_t, tpc, ramp, vh, vs, bump, gain, light,
                    water_tex, D, strength, foamk, frame, foam, wt, kr, kf, lmask, ltemp, lava_t, kl, cw,
                    grass_t, gco, use_grass, out):
    """Natural view of a grid window, per pixel: ground = the game's sand / rock textures blended by the sand
    fraction (shader 96: normal map lerp(hard, soft, s); colour = TerrainRamp at V = lerp(vh, vs, s) +
    (alpha x 2 - 1) x 0.25), lit with the cell slope + the textures' normals; then the water / lava layer
    (colour oa, opacity aa, from viewfast.natural_layers); then, if water_tex, the ripple and foam textures as
    in _shade_window. Everything per cell is interpolated between cell centres (positions computed once per
    pixel). tpc = texels per cell at the chosen mip level (soft_t, hard_t and grass_t have the same size).
    Grass (shader 96, if use_grass): colour = lerp(colour, Grass_D.rgb x GrassColorRT.rgb, saturate(2 x
    GrassColorRT.a + Grass_D.a - 1)) before the lighting; gco = ganja.grass_color_opacity (per cell, bilinear)."""
    H, W = out.shape[0], out.shape[1]
    n = sfa.shape[0]
    fh, fw = frame.shape
    gh, gw = foam.shape
    th, tw = soft_t.shape[0], soft_t.shape[1]
    nr = ramp.shape[0]
    half = np.float32(0.5)
    one = np.float32(1)
    two = np.float32(2)
    L0, L1, L2 = light[0], light[1], light[2]
    for i in numba.prange(H):
        y = gy0 + (np.float32(i) + half) / k
        yc = min(max(y - half, np.float32(0)), np.float32(n - 1))
        cy0 = int(yc)
        cy1 = min(cy0 + 1, n - 1)
        fy = yc - cy0
        tv = y * tpc - half
        ty0f = np.floor(tv)
        tfy = tv - ty0f
        ty0 = np.int64(ty0f) % th
        ty1 = (ty0 + 1) % th
        for j in range(W):
            x = gx0 + (np.float32(j) + half) / k
            xc = min(max(x - half, np.float32(0)), np.float32(n - 1))
            cx0 = int(xc)
            cx1 = min(cx0 + 1, n - 1)
            fx = xc - cx0
            s = _lerp_at(sfa, cy0, cx0, cy1, cx1, fy, fx)
            tu = x * tpc - half
            tx0f = np.floor(tu)
            tfx = tu - tx0f
            tx0 = np.int64(tx0f) % tw
            tx1 = (tx0 + 1) % tw
            nx = _lerp_at3(hard_t, 0, ty0, tx0, ty1, tx1, tfy, tfx)
            ny = _lerp_at3(hard_t, 1, ty0, tx0, ty1, tx1, tfy, tfx)
            al = _lerp_at3(hard_t, 3, ty0, tx0, ty1, tx1, tfy, tfx)
            if s > 0:
                nx += (_lerp_at3(soft_t, 0, ty0, tx0, ty1, tx1, tfy, tfx) - nx) * s
                ny += (_lerp_at3(soft_t, 1, ty0, tx0, ty1, tx1, tfy, tfx) - ny) * s
                al += (_lerp_at3(soft_t, 3, ty0, tx0, ty1, tx1, tfy, tfx) - al) * s
            v0 = vh + (vs - vh) * s
            lm = _lerp_at(lmask, cy0, cx0, cy1, cx1, fy, fx)
            if lm > 0:                                       # shader 97: lava (GroundParams.x = lava mask)
                nx += (half - nx) * lm                       # lava flattens the ground bumps
                ny += (half - ny) * lm
                al += (half - al) * lm
                lt = np.float32(0)                           # LavaGround_D on the two flow layers, x 8 per map
                for kk in range(2):
                    ox, oy = _bilinear2(D, kk, yc, xc)
                    qx = (x + ox) * kl
                    qy = (y + oy) * kl
                    lt += wt[kk] * (cw[0] * _tex4(lava_t, qx, qy, 0) + cw[1] * _tex4(lava_t, qx, qy, 1)
                                    + cw[2] * _tex4(lava_t, qx, qy, 2))
                tt = _lerp_at(ltemp, cy0, cx0, cy1, cx1, fy, fx)
                term = tt ** np.float32(0.15) if tt > np.float32(1e-4) else np.float32(0.251188636)
                fkl = _lerp_at(foamk, cy0, cx0, cy1, cx1, fy, fx)
                L = (lt + fkl + term + np.float32(1.1)) * lm * np.float32(0.5143)
                v0 += (np.float32(LAVA_V) - v0) * L          # not clamped: hot lava goes past 0.125 towards the orange top
            rv = v0 + (al * two - one) * np.float32(0.25)
            rp = min(max(rv * nr - half, np.float32(0)), np.float32(nr - 1))
            r0 = int(rp)
            r1 = min(r0 + 1, nr - 1)
            f = rp - r0
            c0 = ramp[r0, 0] + (ramp[r1, 0] - ramp[r0, 0]) * f
            c1 = ramp[r0, 1] + (ramp[r1, 1] - ramp[r0, 1]) * f
            c2 = ramp[r0, 2] + (ramp[r1, 2] - ramp[r0, 2]) * f
            if use_grass:
                ga = _lerp_at3(gco, 3, cy0, cx0, cy1, cx1, fy, fx)
                if ga > 0:
                    ta = _lerp_at3(grass_t, 3, ty0, tx0, ty1, tx1, tfy, tfx)
                    t = min(max(ga * two + ta - one, np.float32(0)), one)
                    if t > 0:
                        k255 = np.float32(255)
                        g0 = _lerp_at3(grass_t, 0, ty0, tx0, ty1, tx1, tfy, tfx) * _lerp_at3(gco, 0, cy0, cx0, cy1, cx1, fy, fx) * k255
                        g1 = _lerp_at3(grass_t, 1, ty0, tx0, ty1, tx1, tfy, tfx) * _lerp_at3(gco, 1, cy0, cx0, cy1, cx1, fy, fx) * k255
                        g2 = _lerp_at3(grass_t, 2, ty0, tx0, ty1, tx1, tfy, tfx) * _lerp_at3(gco, 2, cy0, cx0, cy1, cx1, fy, fx) * k255
                        c0 += (g0 - c0) * t
                        c1 += (g1 - c1) * t
                        c2 += (g2 - c2) * t
            px = -_lerp_at(gxa, cy0, cx0, cy1, cx1, fy, fx) + (nx * two - one) * bump
            py = -_lerp_at(gya, cy0, cx0, cy1, cx1, fy, fx) + (ny * two - one) * bump
            sh = (px * L0 + py * L1 + L2) / np.sqrt(px * px + py * py + one)
            g = (min(max(sh, np.float32(0)), one) * np.float32(0.75) + np.float32(0.25)) * gain
            c0 *= g
            c1 *= g
            c2 *= g
            a = _lerp_at(aa, cy0, cx0, cy1, cx1, fy, fx)
            if a > 0:
                c0 += (_lerp_at3(oa, 0, cy0, cx0, cy1, cx1, fy, fx) - c0) * a
                c1 += (_lerp_at3(oa, 1, cy0, cx0, cy1, cx1, fy, fx) - c1) * a
                c2 += (_lerp_at3(oa, 2, cy0, cx0, cy1, cx1, fy, fx) - c2) * a
            if water_tex:
                w = _lerp_at(wet, cy0, cx0, cy1, cx1, fy, fx)
                if w > 0:
                    st = _lerp_at(strength, cy0, cx0, cy1, cx1, fy, fx)
                    fk = _lerp_at(foamk, cy0, cx0, cy1, cx1, fy, fx)
                    rip = np.float32(0)
                    fo = np.float32(0)
                    for kk in range(2):
                        ox, oy = _bilinear2(D, kk, yc, xc)
                        qx = x + ox
                        qy = y + oy
                        rip += wt[kk] * _wrap2(frame, qx * kr, qy * kr)
                        if fk > 0:
                            fo += wt[kk] * _wrap2(foam, qx * kf, qy * kf)
                    m = 1 + st * rip * w
                    c0 *= m
                    c1 *= m
                    c2 *= m
                    if fk > 0:
                        fa = min(fo * fk, one) * w
                        c0 += (235 - c0) * fa
                        c1 += (240 - c1) * fa
                        c2 += (245 - c2) * fa
            out[i, j, 0] = np.uint8(min(max(c0, np.float32(0)), np.float32(255)))
            out[i, j, 1] = np.uint8(min(max(c1, np.float32(0)), np.float32(255)))
            out[i, j, 2] = np.uint8(min(max(c2, np.float32(0)), np.float32(255)))


def _blur(a, passes=6):
    """The game's WaterParams blur as fitted: passes of a 4-neighbour average (wrapping at the edges,
    where the fields are faded to 0 anyway). Compiled in viewfast.blur (same sums in the same order)."""
    try:
        import viewfast
        return viewfast.blur(a, passes)
    except ImportError:
        pass
    for _ in range(passes):
        a = 0.25 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    return a


_COLMAP_CACHE = {}
_TEX_ARCHIVE = {}
_TEX_CACHE = {}


def level_texture_archive(level):
    """The level's texture archive BinPC/Tex_<id>.lin.bf (the Tex_ twin of the Wor_<id> archive holding
    NN.wor), or None. level = story level number (1-13; no NN.wor exists for 7, nor for challenge maps)."""
    if level not in _TEX_ARCHIVE:
        out = None
        try:
            import glob
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import bf
            binpc = os.path.dirname(TEX_BF)
            for wor in glob.glob(os.path.join(binpc, 'Wor_*.lin.bf')):
                if '%02d.wor' % level in [e['name'] for e in bf.read_index(wor)]:
                    tex = os.path.join(binpc, os.path.basename(wor).replace('Wor_', 'Tex_'))
                    out = tex if os.path.exists(tex) else None
                    break
        except Exception:
            out = None
        _TEX_ARCHIVE[level] = out
    return _TEX_ARCHIVE[level]


def level_texture_names(level):
    """Names of the level's textures (*.PC.tdt, archive order, duplicates dropped); [] if none."""
    tex = level_texture_archive(level) if level else None
    if not tex:
        return []
    import bf
    out = []
    for e in bf.read_index(tex):
        if e['name'].endswith('.PC.tdt') and e['name'] not in out:
            out.append(e['name'])
    return out


def level_texture(level, name):
    """One of the level's textures as RGBA uint8 (h, w, 4) (top mip), or None. Cached."""
    key = (level, name)
    if key not in _TEX_CACHE:
        tex = level_texture_archive(level) if level else None
        out = None
        if tex:
            try:
                import bf, tdt
                px, _ = tdt.decode(bf.read_entry(tex, name))
                out = px[0]
            except Exception:
                out = None
        if len(_TEX_CACHE) > 8:
            _TEX_CACHE.clear()
        _TEX_CACHE[key] = out
    return _TEX_CACHE[key]


def level_colmap(level):
    """Blue channel (512, 512) 0..1 of the level's colmap texture (foam edge fade on levels with border
    water), or None if the level has none. level = story level number (1-13)."""
    if level in _COLMAP_CACHE:
        return _COLMAP_CACHE[level]
    out = None
    for name in level_texture_names(level):
        if name.lower().startswith('colmap'):
            px = level_texture(level, name)
            if px is not None:
                out = np.ascontiguousarray(px[..., 2].astype(np.float32) / 255)
            break
    _COLMAP_CACHE[level] = out
    return out


class FlowLayer:
    """Persistent state: the two UV-offset layers. update() advances them, shade() returns the display
    image (n*scale)^2 RGB (calls update), debug_image() shows the per-cell inputs."""

    def __init__(self, n, scale):
        self.n, self.s = n, scale
        self.frames, self.foam, self.source = load_textures()
        tile = n / TILES                                  # cells per tile (42.7)
        self.kr, self.kf = self.frames.shape[2] / tile, self.foam.shape[1] / tile
        self.D = np.zeros((2, n, n, 2), np.float32)
        self.D2 = np.empty_like(self.D)
        self.fields = tuple(np.zeros((n, n), np.float32) for _ in range(4))
        self.t_last = None
        self._fkey = None
        self.level, self.border = None, np.full((1, 1), -1, np.float32)

    def set_level(self, level):
        """Story level number (or None): selects the foam edge fade (see the module docstring)."""
        if level == self.level:
            return
        self.level = level
        b = level_colmap(level) if level else None
        self.border = b if b is not None else np.full((1, 1), -1, np.float32)
        self._fkey = None

    def _frame(self, t):
        f = t * 4.0                                       # ripple frame rate not measured (guess)
        k = int(f) % len(self.frames)
        a = np.float32(f - int(f))
        return np.ascontiguousarray(self.frames[k] * (1 - a) + self.frames[(k + 1) % len(self.frames)] * a, np.float32)

    def update(self, depth, flux, t, lava=None, min_depth=0.02):
        """Per-cell fields (once per simulation frame) and one advection step of the UV layers."""
        n = self.n
        depth = np.ascontiguousarray(depth, np.float32)
        lv = np.ascontiguousarray(lava if lava is not None else np.zeros_like(depth), np.float32)
        vx, vy, strength, foamk = self.fields
        key = (id(depth), id(flux))
        if self._fkey != key:                              # once per simulation frame
            _cell_fields(depth, np.ascontiguousarray(flux, np.float32), lv, np.float32(min_depth), self.border, vx, vy, strength, foamk)
            vx[:], vy[:], foamk[:] = _blur(vx), _blur(vy), _blur(foamk)
            self._fkey = key
        if self.t_last is None:
            self.t_last = t
            self.cycles = (int(np.floor(t / LAYER_PERIOD)), int(np.floor(t / LAYER_PERIOD + 0.5)))
        dt = min(max(t - self.t_last, 0.0), 0.1)
        self.t_last = t
        cyc = (int(np.floor(t / LAYER_PERIOD)), int(np.floor(t / LAYER_PERIOD + 0.5)))
        reset = np.array([cyc[0] != self.cycles[0], cyc[1] != self.cycles[1]])
        self.cycles = cyc
        _advect(self.D, vx, vy, np.float32(SPEED * dt), reset, self.D2)
        self.D, self.D2 = self.D2, self.D
        ph = (t / LAYER_PERIOD) % 1.0
        w0 = 1 - abs(1 - 2 * ph)                           # layer 0 invisible when it restarts (ph 0), layer 1 at ph 0.5
        self.wt = np.array([w0, 1 - w0], np.float32)
        return depth, lv

    def shade(self, base, depth, flux, t, lava=None, min_depth=0.02, sediment=None):
        """base: (n, n, 3) uint8 colours in cell space; returns (n*scale, n*scale, 3) uint8."""
        n, s = self.n, self.s
        depth, lv = self.update(depth, flux, t, lava, min_depth)
        vx, vy, strength, foamk = self.fields
        out = np.empty((n * s, n * s, 3), np.uint8)
        _shade_uv(np.ascontiguousarray(base), depth, lv, np.float32(min_depth), s, self.D, strength, foamk,
                  self._frame(t), self.foam, self.wt, np.float32(self.kr), np.float32(self.kf), out)
        return out

    def shade_window(self, base, depth, flux, t, gy0, gx0, k, size, lava=None, min_depth=0.02):
        """A zoomed view: (size, size, 3) uint8 of the grid window starting at grid point (gy0, gx0)
        (top-left, cell units) at k pixels per cell, smooth (see _shade_window). Grid orientation."""
        depth, lv = self.update(depth, flux, t, lava, min_depth)
        vx, vy, strength, foamk = self.fields
        b = np.asarray(base, np.float32)
        wet = ((depth > min_depth) & (lv <= 0)).astype(np.float32)
        out = np.empty((size, size, 3), np.uint8)
        _shade_window(np.ascontiguousarray(b[..., 0]), np.ascontiguousarray(b[..., 1]), np.ascontiguousarray(b[..., 2]),
                      wet, np.float32(gy0), np.float32(gx0), np.float32(k), self.D, strength, foamk,
                      self._frame(t), self.foam, self.wt, np.float32(self.kr), np.float32(self.kf), out)
        return out

    def shade_textured(self, fr, t, gy0, gx0, k, size, light, lava_xs, lava_cs, water_tex=True, min_depth=0.02):
        """Natural view with the game's sand / rock textures (see _shade_textured) for the grid window at
        (gy0, gx0), k pixels per cell, (size, size, 3) uint8, grid orientation. fr = dustview.Frame."""
        import viewfast
        n = self.n
        if not hasattr(self, '_layers'):
            self._layers = (np.empty((n, n), np.float32), np.empty((n, n), np.float32), np.empty((n, n), np.float32),
                            np.empty((n, n, 3), np.float32), np.empty((n, n), np.float32))
        gxa, gya, sfa, oa, aa = self._layers
        viewfast.natural_layers(fr.ground, fr.soft, fr.water, fr.lava, fr.sediment, fr.temperature,
                                lava_xs, lava_cs, gxa, gya, sfa, oa, aa, False)   # lava is drawn on the ramp below
        use_grass = bool(SHOW_VEGETATION and fr.grid is not None and fr.grid.shape[0] > 6 and fr.grid[6].any())
        if use_grass and getattr(self, '_gkey', None) is not fr.grid:      # once per simulation frame
            import ganja                                 # the game's grass ground layer (shaders 113 + 96)
            self._gco = ganja.grass_color_opacity(fr.grid[6])
            self._gkey = fr.grid
        gco = self._gco if use_grass else np.zeros((1, 1, 4), np.float32)
        if water_tex:
            depth, lv = self.update(fr.water, fr.flux, t, fr.lava, min_depth)
        else:
            depth, lv = np.asarray(fr.water, np.float32), np.asarray(fr.lava, np.float32)
        vx, vy, strength, foamk = self.fields
        wet = ((depth > min_depth) & (lv <= 0)).astype(np.float32)
        soft_m, hard_m, ramp, grass_m = load_ground_textures()
        tpc0 = soft_m[0].shape[0] * GROUND_TILES / n          # texels per cell at full resolution (32)
        level = int(min(max(np.floor(np.log2(max(tpc0 / k, 1.0))), 0), len(soft_m) - 1))   # ~1-2 texels per pixel
        # upload byte x = 255 wherever lava; the game softens it before the terrain pass (screen dump: lava edges ~2x
        # as wide as sand edges, mask rarely near 1); passes chosen by eye: 6 (as the water parameters) wipes out 1-2 cell channels
        if getattr(self, '_lkey', None) != id(fr.lava):                   # once per simulation frame
            m = (np.asarray(fr.lava) > 0).astype(np.float32)
            self._ltarget = _blur(m, LAVA_MASK_BLUR) if LAVA_MASK_BLUR else m
            self._lkey = id(fr.lava)
        # cells fade in / out instead of switching (seen in play); time constant chosen by eye, on the simulation clock
        prev, t_prev = getattr(self, '_lmask', None), getattr(self, '_lt', None)
        if prev is None or t_prev is None or not LAVA_FADE or t < t_prev or t - t_prev > 2.0:
            self._lmask = self._ltarget.copy()
        elif t > t_prev:
            a = np.float32(1 - np.exp(-(t - t_prev) / LAVA_FADE))
            self._lmask += (self._ltarget - self._lmask) * a
        self._lt = t
        lmask = self._lmask
        ltemp = np.where(lmask > 0, np.minimum(np.round(fr.temperature), 255) / 255, 0).astype(np.float32)
        lava_m = load_lava_texture()
        kl0 = lava_m[0].shape[0] * LAVA_TILES / n                       # 8 texels per cell
        llev = int(min(max(np.floor(np.log2(max(kl0 / k, 1.0))), 0), len(lava_m) - 1))
        clk = LAVA_CLOCK * t
        cw = np.array([0.5 + 0.5 * np.cos(clk - p) for p in LAVA_PHASES], np.float32)
        out = np.empty((size, size, 3), np.uint8)
        _shade_textured(gxa, gya, sfa, oa, aa, wet, np.float32(gy0), np.float32(gx0), np.float32(k),
                        soft_m[level], hard_m[level], np.float32(tpc0 / 2 ** level), ramp,
                        np.float32(RAMP_V_HARD), np.float32(RAMP_V_SOFT), np.float32(BUMP), np.float32(GROUND_GAIN),
                        np.asarray(light, np.float32), bool(water_tex), self.D, strength, foamk, self._frame(t),
                        self.foam, self.wt, np.float32(self.kr), np.float32(self.kf),
                        lmask, ltemp, lava_m[llev], np.float32(kl0 / 2 ** llev), cw,
                        grass_m[level], gco, use_grass, out)
        return out

    DEBUG_MODES = ('foam', 'velocity', 'drag')

    def debug_image(self, mode, depth):
        """(n, n, 3) uint8 view of the inputs (call update() first).
        foam: the foam factor 0..1 as black -> white (the game multiplies the foam texture by it).
        velocity: hue = direction, brightness = |v| (full at |v| = 0.5, i.e. ~11 cells/s).
        drag: how far the texture coordinates have been dragged, blended like the layers (black 0,
        white >= 60 cells); it grows during a layer's 2.22 s life and drops when the visible layer restarts."""
        vx, vy, strength, foamk = self.fields
        wet = (depth > 0.02)[..., None]
        if mode == 'foam':
            g = np.clip(foamk, 0, 1)[..., None] * np.array([255, 255, 255], np.float32)
            return np.where(wet, g, 25).astype(np.uint8)
        if mode == 'velocity':
            ang = np.arctan2(vy, vx)
            mag = np.clip(np.hypot(vx, vy) / 0.5, 0, 1)
            hue = (ang / (2 * np.pi)) % 1.0
            k = (hue[..., None] * 6 + np.array([0, 4, 2], np.float32)) % 6     # HSV -> RGB, s = 1
            rgb = (1 - np.clip(np.minimum(k, 4 - k), 0, 1)) * mag[..., None] * 255
            return np.where(wet, rgb, 25).astype(np.uint8)
        d = self.wt[0] * np.hypot(self.D[0, ..., 0], self.D[0, ..., 1]) + self.wt[1] * np.hypot(self.D[1, ..., 0], self.D[1, ..., 1])
        g = np.clip(d / 60.0, 0, 1)
        rgb = np.stack([g * 255, g * 220, g * 120], -1)
        return np.where(wet, rgb, 25).astype(np.uint8)
