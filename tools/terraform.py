"""From Dust's scripted terraforming (`TerraForming` level-script effect): brushes, curves, the per-frame
object, and a Python copy of the game's brush job. Rules and addresses: findings/water-lava-sim.md,
"TerraForming effect".

How the game runs one TerraForming effect (dustai, every frame while the effect lasts):
  elapsed += dt;  size = int(StartSize + (EndSize - StartSize) * elapsed / T)   (T = the effect's Duration)
  strength = curve[CurveID](elapsed / T)   (curve y in units of its y range; Curve02's is 2)
  brush job (GalileoSimulatorWater_9_9.dll job 14): every cell within the brush's (rotated) square gets
  ground +/- texel / 256 * strength, texel = the brush bitmap's red channel, bilinear, at the mip level for
  the size; subtracting never goes below 0.
Angle is passed to cos / sin as is (the editor shows 0-360, the code treats it as radians).

Checked against the running game on Story 1 (findings): swing heights within 1-3 %, timing and shape match.
Usage as a module:
    tf = Terraform(effect_dict)          # an effect as sav_sources.effect() parses it (name 'TerraForming')
    for each frame: call = tf.frame(dt)   # None once finished, else a BrushCall
    apply_brush(G, call)                  # Python copy, or galileo.Resident.brush(*call.dll_args())
Command line: python terraform.py check [n]   compare apply_brush with the DLL on random cases
              python terraform.py curves      print the TerraForming curves at 11 points
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf
import sav_sources as ss

ARCHIVE = os.path.join(ss.INSTALL, 'BinPC', 'Wor_BD00993D.lin.bf')
# brush slot -> texture (LevelParameters_0.ai lists the brush texture keys in this order)
BRUSHES = ('Dust_Brush_0.tga', 'Dust_Brush_8.tga', 'Dust_Brush_Cratere1.tga', 'Dust_Brush_3.tga',
           'Dust_Brush_4.tga', 'Dust_Brush_1.tga', 'Dust_Brush_6.tga', 'Dust_Brush_7.tga')
# CurveID -> curve (SpecieTerraForming_0.ai: array of 10 file keys at 0x275A; 8 and 9 are empty)
CURVES = ('Curve01.crv', 'Curve02.crv', 'Curve02.crv', 'Curve_terra_map13.crv', 'Curve_terra_map13_caldeira.crv',
          'Curve_terra_slow.crv', 'Curve_terra_map13_door.crv', 'Curve_terra_map13_door_down.crv')
MIN, MAX = 32, 480          # the game's MinMax for the brush job (SkipBorderSize 2 x 16, 512 cells)
FRAME = 1 / 31              # one game frame = one simulation tick (the game steps the simulation once per frame;
                            # 30.6 ticks/s measured on Story 1); the brush runs once per frame
TEX_LOG2 = 6                # 64 x 64 brush bitmaps


def _entry(name):
    return bf.read_entry(ARCHIVE, name)


# ---------------------------------------------------------------- brushes
_tex_cache = {}


def brush_levels(slot):
    """The brush bitmap's mip chain as float32 arrays (64, 32, ..., 4), red channel. The TGA is stored
    bottom-up; rows are used as stored (which way the engine turns them is not known). Mips: 2 x 2 box
    filter, rounded to bytes (the engine's filter is not known)."""
    hit = _tex_cache.get(slot)
    if hit is None:
        d = _entry(BRUSHES[slot])
        w, h = struct.unpack_from('<HH', d, 12)
        a = np.frombuffer(d, np.uint8, w * h * 4, 18 + d[0]).reshape(h, w, 4)[..., 2].astype(np.float32)
        levels = []
        while True:
            levels.append(np.round(a))
            if a.shape[0] <= 4:
                break
            a = (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4
        hit = _tex_cache[slot] = levels
    return hit


def brush_texture(slot):
    """The mip chain as the DLL job reads it: 4 bytes per texel, the value in byte 0."""
    key = ('bytes', slot)
    hit = _tex_cache.get(key)
    if hit is None:
        parts = []
        for lv in brush_levels(slot):
            t = np.zeros(lv.shape + (4,), np.uint8)
            t[..., 0] = lv.astype(np.uint8)
            parts.append(t.tobytes())
        hit = _tex_cache[key] = b''.join(parts)
    return hit


# ---------------------------------------------------------------- curves
class Curve:
    """A `.crv` float curve: keys (x, y) with one slope per segment, evaluated as cubic Hermite (inferred;
    the engine's evaluator was not found). x outside the keys clamps to the end values."""
    def __init__(self, data):
        d = data[0x20:]
        n = struct.unpack_from('<I', d, 20)[0] // 5           # header: 4, 0, 0, x range, y range, 5 x keys, 0, 0.1, 0.1
        self.x_range, self.y_range = struct.unpack_from('<2f', d, 12)
        keys = [struct.unpack_from('<4f', d, 36 + 20 * i) for i in range(n)]
        self.x = np.array([k[0] for k in keys], np.float64)
        # stored y are fractions of the y range (Curve02: range 2.0; checked against the live game on Story 1,
        # where the swings are 2.1 x the unscaled curve's)
        self.y = np.array([k[1] for k in keys], np.float64) * self.y_range
        self.slope = np.array([k[3] for k in keys[:-1]], np.float64) * self.y_range   # out = next key's in-tangent

    def __call__(self, t):
        x, y = self.x, self.y
        if t <= x[0]:
            return float(y[0])
        if t >= x[-1]:
            return float(y[-1])
        i = int(np.searchsorted(x, t, 'right') - 1)
        w = x[i + 1] - x[i]
        s = (t - x[i]) / w
        m = self.slope[i] * w
        h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        return float(h00 * y[i] + h10 * m + h01 * y[i + 1] + h11 * m)


_curves = {}


def curve(curve_id):
    if curve_id not in _curves:
        _curves[curve_id] = Curve(_entry(CURVES[curve_id])) if 0 <= curve_id < len(CURVES) else (lambda t: 0.0)
    return _curves[curve_id]


# ---------------------------------------------------------------- the per-frame object
def effect_fields(e):
    """The TerraForming fields from sav_sources.effect()'s raw bytes (big-endian, 40 bytes)."""
    raw = bytes.fromhex(e['raw'])
    x, y, soft, brush, mode, op, s0, s1 = struct.unpack_from('>8I', raw, 0)
    angle, curve_id = struct.unpack_from('>fI', raw, 32)
    return dict(col=x, row=y, soft=soft, brush=brush, mode=mode, op=op, start_size=s0, end_size=s1,
                angle=angle, curve=curve_id, duration=e['duration'])


def brush_effect(mode, op, soft):
    """dustai 0x1006FEB0: kBrushParamEffect_* from BrushMode (0 relative, 1 absolute), BrushOperation
    (0 subtract, 1 add) and ApplyOnSoftGround."""
    base = (2 if op == 0 else 1) if mode == 0 else (4 if op == 0 else 3)
    return base + (4 if soft else 0)


class BrushCall:
    __slots__ = ('slot', 'col', 'row', 'size', 'effect', 'dir_x', 'dir_y', 'strength')

    def __init__(self, slot, col, row, size, effect, dir_x, dir_y, strength):
        self.slot, self.col, self.row, self.size, self.effect = slot, col, row, size, effect
        self.dir_x, self.dir_y, self.strength = dir_x, dir_y, strength

    def dll_args(self):
        """Arguments for galileo.Resident.brush."""
        return (brush_texture(self.slot), TEX_LOG2, self.col, self.row, self.size, self.effect,
                self.dir_x, self.dir_y, self.strength, MIN, MAX)


class Terraform:
    """One running TerraForming effect (dustai DustAi::TerraForming, update 0x10070CD0)."""
    def __init__(self, e):
        f = effect_fields(e) if 'raw' in e else dict(e)
        self.f = f
        self.T = float(f['duration'])
        self.elapsed = -1.0                      # first frame: start (elapsed 0), as 0x100701C0
        self.effect = brush_effect(f['mode'], f['op'], f['soft'])
        a = np.float32(f['angle'])               # radians as given (cos -> dir x, sin -> dir y)
        self.dir = (float(np.float32(np.cos(np.float64(a)))), float(np.float32(np.sin(np.float64(a)))))
        self.curve = curve(f['curve'])

    @property
    def done(self):
        return self.elapsed >= self.T

    def frame(self, dt=FRAME):
        """Advance one frame; the brush call for it, or None when the effect is over."""
        if self.elapsed < 0:
            self.elapsed = 0.0
        else:
            self.elapsed += dt
        return self.call_at(self.elapsed)

    def call_at(self, elapsed):
        """The brush call for the frame at `elapsed` seconds into the effect (None outside 0 <= elapsed < T)."""
        if not 0 <= elapsed < self.T:
            return None
        f = self.f
        frac = elapsed / self.T
        size = int(f['start_size'] + (f['end_size'] - f['start_size']) * frac)
        strength = self.curve(frac)
        if f['brush'] >= len(BRUSHES) or size <= 0:
            return None
        return BrushCall(f['brush'], f['col'], f['row'], size, self.effect, self.dir[0], self.dir[1], strength)


# ---------------------------------------------------------------- the brush job in Python
def _half_lerp(a, b, f):
    """The job's texture filter along one axis (measured on the DLL): texel a below 3/8 of the way, the
    rounded-up byte average of a and b up to 5/8, b beyond (bilinear at half-texel steps)."""
    return np.where(f < 0.375, a, np.where(f < 0.625, (a + b + 1) >> 1, b))


APPROXIMATE = True   # apply_brush is not exact: footprint, effects and amounts match the DLL, the texture
                     # sampling does not (terraform.py check: typically 2-10 % of the change per cell for
                     # smooth brushes, up to ~50 % for the noisy ones); use the DLL (Resident.brush) for exact


def apply_brush(G, call, lo=MIN, hi=MAX):
    """Python copy of the brush job (GalileoSimulatorWater_9_9 job 14) on the state dict G (dustsim layout:
    'T' ground surface, 'soft' soft ground), in place. Approximate (APPROXIMATE). Relative effects only;
    absolute ones (3, 4, 7, 8) are not modelled (skipped)."""
    eff = call.effect
    if eff not in (1, 2, 5, 6) or call.size <= 0:
        return
    levels = brush_levels(call.slot)
    L, k = 1 << TEX_LOG2, 0
    while call.size < L and L > 4:                 # the job's mip choice (StackData constructor)
        L >>= 1
        k += 1
    tex = levels[k]
    half = int(call.size / 2.0 * (abs(call.dir_x) + abs(call.dir_y)))   # the rotated square's bounding box
    r0, r1 = max(lo, call.row - half), min(hi - 1, call.row + half)
    c0, c1 = max(lo, call.col - half), min(hi - 1, call.col + half)
    if r0 > r1 or c0 > c1:
        return
    rows = np.arange(r0, r1 + 1, dtype=np.float32)[:, None] - np.float32(call.row)
    cols = np.arange(c0, c1 + 1, dtype=np.float32)[None, :] - np.float32(call.col)
    sc = np.float32(2.0 / call.size)
    dx, dy = np.float32(call.dir_x) * sc, np.float32(call.dir_y) * sc
    u = cols * dx + rows * dy                     # job 14: u along rows at angle (0, 1); |u|, |v| <= 1 inside
    v = cols * dy - rows * dx
    inside = (np.abs(u) <= 1) & (np.abs(v) <= 1)
    tu = np.clip((u + 1) * (L / 2), 0, L - 1)
    tv = np.clip((v + 1) * (L / 2), 0, L - 1)
    i0, j0 = np.floor(tu).astype(int), np.floor(tv).astype(int)
    fu, fv = tu - i0, tv - j0
    i1, j1 = np.minimum(i0 + 1, L - 1), np.minimum(j0 + 1, L - 1)
    # plain bilinear: the job's own filter works in half-texel steps from per-16-cell-block coordinates
    # (_half_lerp is right with the size-280 coordinates only; not reproduced, see APPROXIMATE)
    val = (tex[j0, i0] * (1 - fu) + tex[j0, i1] * fu) * (1 - fv) + (tex[j1, i0] * (1 - fu) + tex[j1, i1] * fu) * fv
    d = (val.astype(np.float32) / np.float32(256) * np.float32(call.strength)).astype(np.float32) * inside
    T = G['T'][r0:r1 + 1, c0:c1 + 1]
    S = G['soft'][r0:r1 + 1, c0:c1 + 1]
    if eff in (1, 5):
        T += d
        if eff == 5:
            S += d
    elif eff == 6:                                 # soft subtract: only the soft layer can go
        take = np.minimum(d, S)
        S -= take
        T -= take
    else:
        new = np.maximum(T - d, np.float32(0))
        S[...] = np.minimum(S, new)
        T[...] = new


# ---------------------------------------------------------------- checks
def _check(n=40, seed=1):
    """Compare apply_brush with the DLL (galileo.Resident.brush) on random grids and brush calls."""
    import galileo, dustsim
    rng = np.random.default_rng(seed)
    R = galileo.Resident(galileo.param_image(galileo.fluid_params(), 2))
    worst = 0.0
    try:
        for i in range(n):
            g = np.zeros((8, 512, 512), np.uint32)
            T = rng.uniform(0, 60, (512, 512)).astype(np.float32)
            So = np.minimum(rng.uniform(0, 8, (512, 512)).astype(np.float32), T)
            g[0], g[1] = T.view(np.uint32), So.view(np.uint32)
            a = float(rng.choice([0.0, 30.0, rng.uniform(0, 7)]))
            call = BrushCall(int(rng.integers(0, 8)), int(rng.integers(20, 492)), int(rng.integers(20, 492)),
                             int(rng.choice([4, 12, 33, 50, 64, 100, 120, 150, 280])), int(rng.choice([1, 2, 5, 6])),
                             float(np.float32(np.cos(a))), float(np.float32(np.sin(a))), float(rng.uniform(0, 1)))
            R.put(g, range(8))
            R.brush(*call.dll_args())
            out = R.get((0, 1))
            G = dustsim.from_grid(g)
            apply_brush(G, call)
            eT = np.abs(G['T'] - out[0].view(np.float32))
            eS = np.abs(G['soft'] - out[1].view(np.float32))
            dT = np.abs(out[0].view(np.float32) - T)
            worst = max(worst, float(eT.max()))
            print('%2d slot %d size %3d eff %d angle %5.2f: changed %5d cells, max |model - dll| ground %.4f '
                  '(of max change %.3f), soft %.4f, cells off by > 1%% of change %d'
                  % (i, call.slot, call.size, call.effect, a, int((dT > 0).sum()), eT.max(), dT.max(), eS.max(),
                     int((eT > 0.01 * max(dT.max(), 1e-6)).sum())), flush=True)
    finally:
        R.close()
    print('worst ground difference %.4f' % worst)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'curves'
    if cmd == 'check':
        _check(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == 'curves':
        for i, n in enumerate(CURVES):
            c = curve(i)
            print('%d %-34s %s' % (i, n, ' '.join('%.3f' % c(t / 10) for t in range(11))))


if __name__ == '__main__':
    main()
