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
  water), |v| < 0.01 does not move; each layer restarts every 10 s (half a cycle apart), blended so a
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
LAYER_PERIOD = 10.0                        # s
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
            if not (d > min_depth and lava[r, c] <= 0):
                continue
            nx = min(max((flux[1, r, c] - flux[0, r, c]) * np.float32(30.7 / 127.5), np.float32(-1)), np.float32(1))
            ny = min(max((flux[2, r, c] - flux[3, r, c]) * np.float32(28.7 / 127.5), np.float32(-1)), np.float32(1))
            vx[r, c] = nx * fv
            vy[r, c] = ny * fv
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


def level_colmap(level):
    """Blue channel (512, 512) 0..1 of the level's colmap texture (foam edge fade on levels with border
    water), or None if the level has none. level = story level number (1-13)."""
    if level in _COLMAP_CACHE:
        return _COLMAP_CACHE[level]
    out = None
    try:
        import glob
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bf, tdt
        binpc = os.path.dirname(TEX_BF)
        for wor in glob.glob(os.path.join(binpc, 'Wor_*.lin.bf')):
            names = [e['name'] for e in bf.read_index(wor)]
            if '%02d.wor' % level not in names:
                continue
            tex = os.path.join(binpc, os.path.basename(wor).replace('Wor_', 'Tex_'))
            if not os.path.exists(tex):
                continue
            for e in bf.read_index(tex):
                if e['name'].lower().startswith('colmap') and e['name'].endswith('.PC.tdt'):
                    px, _ = tdt.decode(bf.read_entry(tex, e['name']))
                    out = np.ascontiguousarray(px[0][..., 2].astype(np.float32) / 255)
                    break
            break
    except Exception:
        out = None
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

    DEBUG_MODES = ('foam', 'velocity', 'drag')

    def debug_image(self, mode, depth):
        """(n, n, 3) uint8 view of the inputs (call update() first).
        foam: the foam factor 0..1 as black -> white (the game multiplies the foam texture by it).
        velocity: hue = direction, brightness = |v| (full at |v| = 0.5, i.e. ~11 cells/s).
        drag: how far the texture coordinates have been dragged, blended like the layers (black 0,
        white >= 60 cells); it grows during a layer's 10 s life and drops when the visible layer restarts."""
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
