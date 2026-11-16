"""The engine's map-sized (512 x 512, grid-aligned) textures for DustView's "Engine texture" view.

Each entry turns one per-cell layer into an image plus a readout for the cell under the mouse. Groups:
- Water upload: the per-frame texture the CPU builds from the grid and uploads (bottom half of the
  512 x 1024 A8R8G8B8 texture; docs/technical-notes.md "Water rendering"): depth byte, sediment byte,
  net outflow along columns / rows. No edge fade, no blur.
- WaterParams: what shader fxcache 105 + the blur (104) make of it, as DustView computes it
  (tools/flowtex.py): velocity and foam, each with and without the edge fade (6-pass blur in both).
- UV scroll: how far the two advected UV layers have dragged the water texture.
- Edge masks: the velocity edge fade (geometric), the foam edge fade used for this level, and the
  level's colmap texture (the only map-sized texture in a level's Tex_ archive; story levels with border
  water have one).
- Grid: hard / soft ground and the raw bytes of components 6 (kRandomGanja: random + vegetation, low
  nibble indexes ErosionBlocking) and 7 (kGanjaTimeFire: vegetation timer + fire). Components 6 / 7 are
  filled only when reading the live game (and carried into the model at takeover); a model started from a
  .sav has them at 0 (the save's blocks are not mapped to them).
Images are in grid layout ([row, col]); DustView orients them like every other view.
"""
import numpy as np
import flowtex

N = 512
BG = 25                                    # dry / no-data grey

# (key, label); labels starting with '--' are group headers in the dropdown (not selectable)
ENTRIES = (
    ('--', '-- Water upload (per frame, no fade, no blur)'),
    ('up_depth', 'Depth byte: min(64 x depth, 255)'),
    ('up_sediment', 'Sediment byte: 250 x sediment / depth'),
    ('up_vx', 'Net outflow along columns (R - L), signed'),
    ('up_vy', 'Net outflow along rows (U - D), signed'),
    ('--', '-- WaterParams (shader 105 + blur)'),
    ('wp_velocity', 'Velocity, edge-faded (what moves the texture)'),
    ('wp_velocity_nofade', 'Velocity, not faded'),
    ('wp_foam', 'Foam factor, edge-faded (what the game shows)'),
    ('wp_foam_nofade', 'Foam factor, not faded'),
    ('--', '-- UV scroll'),
    ('uv_drag', 'UV drag (how far the texture has moved)'),
    ('--', '-- Edge masks'),
    ('fade_velocity', 'Velocity edge fade (0 to 36 cells, full from 70)'),
    ('fade_foam', 'Foam edge fade used on this level'),
    ('colmap', 'Level colmap texture (RGB)'),
    ('colmap_r', 'Level colmap: red'),
    ('colmap_g', 'Level colmap: green'),
    ('colmap_b', 'Level colmap: blue (= foam edge fade)'),
    ('colmap_a', 'Level colmap: alpha'),
    ('--', '-- Grid layers'),
    ('hard', 'Hard ground (rock) height'),
    ('soft', 'Soft ground (sand) thickness'),
) + tuple(('g%d_b%d' % (c, b), 'Component %d byte %d (%s, live)' % (c, b, 'vegetation' if c == 6 else 'veg. timer / fire'))
          for c in (6, 7) for b in range(4))

LABELS = dict(ENTRIES)
KEYS = [k for k, _ in ENTRIES if k != '--']
DEFAULT = 'wp_foam_nofade'


def _grey(v, lo=0.0, hi=1.0):
    g = np.clip((np.asarray(v, np.float32) - lo) / (hi - lo + 1e-12), 0, 1) * 255
    return np.repeat(g[..., None], 3, -1)


def _signed(v, lim=1.0):
    """Blue = negative, orange = positive, black = 0."""
    a = np.clip(np.asarray(v, np.float32) / lim, -1, 1)[..., None]
    return np.where(a > 0, a * np.array([255, 150, 40], np.float32), -a * np.array([60, 140, 255], np.float32))


try:
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _hue_nb(vx, vy, scale, out):
        n, m = vx.shape
        for r in numba.prange(n):
            for c in range(m):
                x = vx[r, c]
                y = vy[r, c]
                h6 = np.arctan2(y, x) * np.float32(3 / np.pi)
                mag = min(np.sqrt(x * x + y * y) * scale, np.float32(255))
                for ch in range(3):
                    off = np.float32(0) if ch == 0 else (np.float32(4) if ch == 1 else np.float32(2))
                    k = (h6 + np.float32(6) + off) % np.float32(6)
                    out[r, c, ch] = (1 - min(max(min(k, 4 - k), np.float32(0)), np.float32(1))) * mag
except ImportError:
    _hue_nb = None


def _hue(vx, vy, full=0.5):
    """Colour = direction, brightness = |v| (full at `full`)."""
    vx, vy = np.ascontiguousarray(vx, np.float32), np.ascontiguousarray(vy, np.float32)
    if _hue_nb is not None:
        out = np.empty(vx.shape + (3,), np.float32)
        _hue_nb(vx, vy, np.float32(255 / full), out)
        return out
    h6 = np.arctan2(vy, vx) * np.float32(3 / np.pi)                 # -3..3
    mag = np.minimum(np.sqrt(vx * vx + vy * vy) * np.float32(255 / full), np.float32(255))
    out = np.empty(vx.shape + (3,), np.float32)
    for ch, off in enumerate((0, 4, 2)):                             # HSV -> RGB, s = 1
        k = (h6 + np.float32(6 + off)) % np.float32(6)
        out[..., ch] = (1 - np.clip(np.minimum(k, 4 - k), 0, 1)) * mag
    return out


def velocity_fade():
    """fv in flowtex._cell_fields: 0 within 36 cells of the edge, linear to 1 at 70."""
    d = np.minimum(np.arange(N), N - 1 - np.arange(N)).astype(np.float32)
    f = np.clip((d - 36) / 34, 0, 1)
    return f[:, None] * f[None, :]


def _upload_velocity(fr):
    """Normalised net outflow (unfaded, unblurred) as the CPU upload holds it, 0 where dry or lava."""
    wet = (fr.water > 0.02) & (fr.lava <= 0)
    nx = np.clip((fr.flux[1] - fr.flux[0]) * np.float32(30.7 / 127.5), -1, 1)
    ny = np.clip((fr.flux[2] - fr.flux[3]) * np.float32(28.7 / 127.5), -1, 1)
    return np.where(wet, nx, 0).astype(np.float32), np.where(wet, ny, 0).astype(np.float32), wet


def render(key, fr, flow=None, level=None):
    """(rgb uint8 (N, N, 3), readout(r, c) -> str) for one entry. `flow` = an updated flowtex.FlowLayer
    (needed for the wp_* faded entries and uv_drag)."""
    wet = fr.water > 0.02
    if key == 'up_depth':
        b = np.minimum(64 * fr.water, 255)
        return _grey(b, 0, 255), lambda r, c: 'depth byte %.0f' % b[r, c]
    if key == 'up_sediment':
        b = np.where(wet, np.minimum(250 * fr.sediment / np.maximum(fr.water, 1e-6), 255), 0)
        return _grey(b, 0, 255), lambda r, c: 'sediment byte %.0f' % b[r, c]
    if key in ('up_vx', 'up_vy'):
        vx, vy, _ = _upload_velocity(fr)
        v = vx if key == 'up_vx' else vy
        return _signed(v), lambda r, c: 'normalised %+.3f (byte %.0f)' % (v[r, c], 127.5 + 127.5 * v[r, c] * (1 if key == 'up_vx' else -1))
    if key in ('wp_velocity_nofade', 'wp_foam_nofade'):
        vx, vy, w = _upload_velocity(fr)
        if key == 'wp_velocity_nofade':
            bx, by = flowtex._blur(vx), flowtex._blur(vy)
            return _hue(bx, by), lambda r, c: 'velocity %+.3f, %+.3f (|v| %.3f)' % (bx[r, c], by[r, c], np.hypot(bx[r, c], by[r, c]))
        f = flowtex._blur(np.where(w, np.clip(2 * np.hypot(vx, vy) - 1 / 255, 0, 1), 0).astype(np.float32))
        return _grey(f), lambda r, c: 'foam factor %.3f' % f[r, c]
    if key in ('wp_velocity', 'wp_foam', 'uv_drag'):
        if flow is None:
            return np.full((N, N, 3), BG, np.uint8), lambda r, c: ''
        vx, vy, _, foamk = flow.fields
        if key == 'wp_velocity':
            return _hue(vx, vy), lambda r, c: 'velocity %+.3f, %+.3f (|v| %.3f)' % (vx[r, c], vy[r, c], np.hypot(vx[r, c], vy[r, c]))
        if key == 'wp_foam':
            return _grey(foamk), lambda r, c: 'foam factor %.3f' % foamk[r, c]
        D, wt = flow.D, flow.wt
        d = wt[0] * np.hypot(D[0, ..., 0], D[0, ..., 1]) + wt[1] * np.hypot(D[1, ..., 0], D[1, ..., 1])
        rgb = np.stack([np.clip(d / 60, 0, 1) * 255, np.clip(d / 60, 0, 1) * 220, np.clip(d / 60, 0, 1) * 120], -1)
        return np.where(wet[..., None], rgb, BG), lambda r, c: 'dragged %.1f cells (layers %.1f / %.1f)' % (
            d[r, c], np.hypot(*D[0, r, c]), np.hypot(*D[1, r, c]))
    if key == 'fade_velocity':
        f = velocity_fade()
        return _grey(f), lambda r, c: 'velocity fade %.3f' % f[r, c]
    if key == 'fade_foam':
        b = flowtex.level_colmap(level) if level else None
        f, src = (b, 'colmap blue') if b is not None else (velocity_fade(), 'velocity fade (level has no colmap)')
        return _grey(f), lambda r, c: 'foam fade %.3f (%s)' % (f[r, c], src)
    if key.startswith('colmap'):
        px = None
        for name in flowtex.level_texture_names(level) if level else []:
            if name.lower().startswith('colmap'):
                px = flowtex.level_texture(level, name)
                break
        if px is None or px.shape[:2] != (N, N):
            return np.full((N, N, 3), BG, np.uint8), lambda r, c: 'no colmap texture for this level (story levels only)'
        read = lambda r, c: 'colmap RGBA %d %d %d %d' % tuple(px[r, c])
        if key == 'colmap':
            return px[..., :3].astype(np.float32), read
        return _grey(px[..., 'rgba'.index(key[-1])], 0, 255), read
    if key == 'hard':
        h = fr.ground - fr.soft
        lo, hi = np.percentile(h, 1), np.percentile(h, 99.5)
        return _grey(h, lo, hi), lambda r, c: 'hard ground %.2f' % h[r, c]
    if key == 'soft':
        s = fr.soft
        return _grey(s, 0, max(float(np.percentile(s, 99.5)), 1e-3)), lambda r, c: 'soft ground %.2f' % s[r, c]
    if key.startswith('g') and '_b' in key:
        comp, byte = int(key[1]), int(key[-1])
        b = ((fr.grid[comp] >> (8 * byte)) & 0xFF).astype(np.float32)
        word = fr.grid[comp]
        return _grey(b, 0, max(float(b.max()), 1)), lambda r, c: 'component %d = 0x%08X, byte %d = %d (low nibble %d)' % (
            comp, word[r, c], byte, b[r, c], int(b[r, c]) & 15)
    return np.full((N, N, 3), BG, np.uint8), lambda r, c: ''


def image(key, fr, flow=None, level=None):
    rgb, read = render(key, fr, flow, level)
    return np.clip(rgb, 0, 255).astype(np.uint8), read
