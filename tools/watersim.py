"""From Dust water + erosion simulator in numpy, VERSION 0.

Status: a TEXTBOOK model of the kind the game's simulator appears to be, not yet the
game's exact rules. What is taken from the game:
  - the map data: ground surface, soft ground and water depth from a .sav (../docs/sav-format.md)
  - the grid (512 x 512) and per-cell state (4 outflow fluxes Left/Right/Up/Down, soft ground,
    sediment), as named in GalileoSimulatorWater.dll's debug strings
  - a fixed map border acting as open sea (the DLL has BorderWater/BorderHard/BorderSoft values)
What is NOT from the game yet (textbook "virtual pipes" shallow water + simple erosion,
Mei et al. 2007 style): the update equations, the constants (gravity, time step, damping,
erosion/deposition rates, sand slope), lava, temperature, vegetation, evaporation.
See ../findings/python-water-sim.md for the plan to replace these with the game's own rules.

Batch use (run with the project venv, which has numpy + pillow):
  .venv/Scripts/python tools/watersim.py <map.sav> --out data/sim/NAME [options]
Options:
  --steps N         simulation steps (default 3000)
  --every N         save a frame every N steps (default 30)
  --upscale K       run on a K-times finer grid (bilinear-upsampled map); keeps the physics
                    in world units (cell size 1/K, time step 1/K, steps x K)
  --source X,Y,R,Q  pour water at grid cell X,Y (512-grid coordinates) in a disc of radius R,
                    Q height units per time unit; default: on the highest inland point
  --no-source       no extra water, just let the map settle
Writes NAME.gif (animation), NAME_before_after.png, and prints progress.
Interactive use: tools/watersim_live.py (same simulator).
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav  # noqa: E402

DEFAULT_MAP = r'D:\Games\Steam\steamapps\common\From Dust\DUST\DustSave\Ingame\ALPHA_Map1_Erosion.sav'


# ---------------------------------------------------------------- map loading
def load_map(path):
    _, u = sav.read_sav(path)

    def layer(k):
        a = np.frombuffer(u, dtype='<f4', count=sav.GRID * sav.GRID,
                          offset=sav.GRID_START + k * sav.LAYER_BYTES).reshape(sav.GRID, sav.GRID)
        a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, 0.0, 1e4)

    ground, water, soft = layer(0), layer(1), layer(2)
    hard = np.maximum(ground - soft, 0.0)
    return hard, soft, water


def upsample(a, k):
    if k == 1:
        return a.copy()
    img = Image.fromarray(a.astype(np.float32), mode='F')
    return np.asarray(img.resize((a.shape[1] * k, a.shape[0] * k), Image.BILINEAR), dtype=np.float32)


# ---------------------------------------------------------------- simulator
class Sim:
    """Virtual-pipe shallow water over hard + soft ground, with soft-ground erosion.

    Directions: L = toward column-1, R = column+1, U = row-1, D = row+1. A flux fX at a cell
    is the water volume per time unit leaving that cell toward neighbour X. Neighbour
    shifts use slices into preallocated buffers (no wrap-around, few allocations).
    """

    def __init__(self, hard, soft, water, cell=1.0, dt=0.02, g=9.81, damping=0.998,
                 k_capacity=0.08, k_erode=0.2, k_deposit=0.3, repose=0.8, slump_rate=0.25, border=32):
        # border: the game keeps a 32-cell band around the map constant (water, ground and soft
        # identical between every map's start state and the player's saves after play)
        f32 = np.float32
        self.hard, self.soft, self.water = hard.astype(f32), soft.astype(f32), water.astype(f32)
        self.sed = np.zeros_like(self.water)                  # suspended soft ground (height units)
        self.fL, self.fR, self.fU, self.fD = (np.zeros_like(self.water) for _ in range(4))
        self.l, self.dt, self.g, self.damping = cell, dt, g, damping
        self.kc, self.ke, self.kd = k_capacity, k_erode, k_deposit
        self.repose, self.slump_rate = repose, slump_rate
        self.border = border
        self.init = (self.hard.copy(), self.soft.copy(), self.water.copy())
        self.shape = self.water.shape
        self._d = [np.zeros_like(self.water) for _ in range(4)]          # neighbour differences
        self._t = [np.zeros_like(self.water) for _ in range(8)]          # scratch
        self.source = None
        self.t = 0.0

    # -- helpers
    def ground(self):
        return self.hard + self.soft

    def _diffs(self, a):
        """a minus each neighbour (0 where the neighbour is off the map): L, R, U, D."""
        dL, dR, dU, dD = self._d
        dL[:, 0] = 0; np.subtract(a[:, 1:], a[:, :-1], out=dL[:, 1:])
        dR[:, -1] = 0; np.subtract(a[:, :-1], a[:, 1:], out=dR[:, :-1])
        dU[0, :] = 0; np.subtract(a[1:, :], a[:-1, :], out=dU[1:, :])
        dD[-1, :] = 0; np.subtract(a[:-1, :], a[1:, :], out=dD[:-1, :])
        return dL, dR, dU, dD

    @staticmethod
    def _gather(out, L, R, U, D):
        """out = what each cell receives from neighbours sending L/R/U/D toward it."""
        out.fill(0)
        out[:, 1:] += R[:, :-1]
        out[:, :-1] += L[:, 1:]
        out[1:, :] += D[:-1, :]
        out[:-1, :] += U[1:, :]
        return out

    # -- one time step
    def step(self):
        dt, l, g = self.dt, self.l, self.g
        area = l * l
        W, S, sed = self.water, self.soft, self.sed
        fL, fR, fU, fD = self.fL, self.fR, self.fU, self.fD
        surf, out, inflow, tmp, qx, qy, W_old, cap = self._t

        # 1. Flux: each cell pushes water through 4 "pipes" toward lower neighbours.
        np.add(self.hard, S, out=surf); surf += W
        k = dt * g * l                       # dt * A * g / l with pipe cross-section A = l*l
        for f, d in zip((fL, fR, fU, fD), self._diffs(surf)):
            f *= self.damping; d *= k; f += d; np.maximum(f, 0, out=f)
        fL[:, 0] = 0; fR[:, -1] = 0; fU[0, :] = 0; fD[-1, :] = 0
        np.add(fL, fR, out=out); out += fU; out += fD
        # never let a cell send out more water than it has
        np.multiply(out, dt, out=tmp); tmp += 1e-9
        np.divide(W, tmp, out=tmp); tmp *= area; np.minimum(tmp, 1.0, out=tmp)
        for f in (fL, fR, fU, fD, out):
            f *= tmp

        # 2. Sediment travels with the water (same fluxes, at the cell's concentration)
        np.maximum(W, 1e-4, out=tmp); np.divide(sed, tmp, out=tmp)          # concentration
        self._gather(cap, fL * tmp, fR * tmp, fU * tmp, fD * tmp)
        cap -= out * tmp; cap *= dt / area; sed += cap; np.maximum(sed, 0, out=sed)

        # 3. Water height from in/out flow
        W_old[...] = W
        self._gather(inflow, fL, fR, fU, fD)
        inflow -= out; inflow *= dt / area; W += inflow; np.maximum(W, 0, out=W)

        # 4. Flow speed from the net flow through each cell
        np.subtract(fR, fL, out=qx); qx[:, 1:] += fR[:, :-1]; qx[:, :-1] -= fL[:, 1:]
        np.subtract(fD, fU, out=qy); qy[1:, :] += fD[:-1, :]; qy[:-1, :] -= fU[1:, :]
        W_old += W; W_old *= 0.5; np.maximum(W_old, 1e-3, out=W_old)       # mean depth
        np.hypot(qx, qy, out=tmp); tmp *= 0.5 / l; tmp /= W_old            # speed

        # 5. Erosion / deposition of soft ground (hard ground is never eroded)
        np.minimum(W, 2.0, out=cap); cap *= tmp; cap *= self.kc               # carrying capacity
        cap -= sed                                                              # >0: can take more
        erode = np.clip(cap, 0, None); erode *= self.ke * dt * 10; np.minimum(erode, S, out=erode)
        deposit = np.clip(-cap, 0, None); deposit *= self.kd * dt * 10
        S += deposit; S -= erode; sed += erode; sed -= deposit
        dry = W < 1e-3                                                          # dry cells drop their load
        S[dry] += sed[dry]; sed[dry] = 0.0

        # 6. Soft ground slides where it is steeper than the repose slope
        self._slump()

        # 7. Extra water from a fixed source
        if self.source is not None:
            x, y, r, rate = self.source
            self.brush('water', x, y, r, rate * dt)

        # 8. Map border = open sea at its starting state
        self._reset_edges()
        self.t += dt

    def _slump(self):
        S = self.soft
        G = self.hard + S
        lim = self.repose * self.l
        ex = []
        for d in self._diffs(G):
            d -= lim; np.maximum(d, 0, out=d); ex.append(d)
        total = ex[0] + ex[1] + ex[2] + ex[3]
        move = np.minimum(S, self.slump_rate * 0.5 * total)
        frac = move / (total + 1e-9)
        S -= move
        S += self._gather(self._t[3], ex[0] * frac, ex[1] * frac, ex[2] * frac, ex[3] * frac)

    def _reset_edges(self):
        b = self.border
        h0, s0, w0 = self.init
        for sl in (np.s_[:b, :], np.s_[-b:, :], np.s_[:, :b], np.s_[:, -b:]):
            self.water[sl] = w0[sl]; self.soft[sl] = s0[sl]; self.hard[sl] = h0[sl]; self.sed[sl] = 0.0

    # -- tools
    def brush(self, layer, cx, cy, r, amount):
        """Add (amount > 0) or remove (< 0) height on 'water' or 'soft' in a soft-edged disc."""
        n0, n1 = self.shape
        x0, x1 = max(0, int(cx - r)), min(n1, int(cx + r) + 2)
        y0, y1 = max(0, int(cy - r)), min(n0, int(cy + r) + 2)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        fall = 1.0 - ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        np.clip(fall, 0, 1, out=fall)
        arr = getattr(self, layer)[y0:y1, x0:x1]
        arr += amount * fall
        np.maximum(arr, 0, out=arr)
        if layer == 'water' and amount < 0:          # removed water takes its sediment along
            s = self.sed[y0:y1, x0:x1]; s[arr <= 0] = 0

    def set_source(self, x, y, r, rate):
        self.source = (x, y, r, rate)

    def reset(self):
        h0, s0, w0 = self.init
        self.hard[...] = h0; self.soft[...] = s0; self.water[...] = w0
        for a in (self.sed, self.fL, self.fR, self.fU, self.fD):
            a.fill(0)
        self.t = 0.0


# ---------------------------------------------------------------- rendering
class Renderer:
    """Hill-shaded ground (cached, refreshed every few frames) + water coloured by depth (lookup table)."""
    LIGHT = np.array([-0.6, -0.6, 0.55]) / np.linalg.norm([-0.6, -0.6, 0.55])

    def __init__(self, refresh_every=8):
        self.refresh_every = refresh_every
        self.frame = 0
        self.base = None
        d = np.arange(1024, dtype=np.float32) / 24.0                      # depth per LUT entry
        a = 1 - np.exp(-d / 2.0)
        t = np.exp(-d / 10.0)[:, None]
        shallow = np.array([70, 150, 185], np.float32); deep = np.array([18, 52, 105], np.float32)
        self.lut_c = (deep * (1 - t) + shallow * t).astype(np.float32)
        self.lut_a = a.astype(np.float32)
        self.mud = np.array([150, 118, 80], np.float32)

    def ground_colour(self, sim):
        ground = sim.ground()
        gy, gx = np.gradient(ground / sim.l)
        L = self.LIGHT
        shade = np.clip((-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx * gx + gy * gy + 1.0), 0, 1) * 0.75 + 0.25
        rock = np.array([118, 108, 98], np.float32); sand = np.array([214, 188, 138], np.float32)
        s = np.clip(sim.soft / 1.5, 0, 1)[..., None]
        return ((rock + (sand - rock) * s) * shade[..., None]).astype(np.float32)

    def __call__(self, sim, refresh=False):
        if self.base is None or refresh or self.frame % self.refresh_every == 0:
            self.base = self.ground_colour(sim)
        self.frame += 1
        W = sim.water
        idx = np.minimum(W * 24.0, 1023).astype(np.int16)
        a = self.lut_a[idx][..., None]
        wc = self.lut_c[idx]
        m = np.clip(sim.sed / (0.02 * W + 1e-6), 0, 1)[..., None] * 0.6
        wc += (self.mud - wc) * m
        col = self.base + (wc - self.base) * a
        return Image.fromarray(col.astype(np.uint8), 'RGB')


def render(sim):
    return Renderer()(sim, refresh=True)


# ---------------------------------------------------------------- batch main
def default_source(hard, soft, water, K=1):
    """Highest dry land cell at least 64 cells (512-grid) from the map edge."""
    inner = np.zeros(hard.shape, bool); m = 64 * K
    inner[m:-m, m:-m] = True
    g = np.where(inner & (water < 0.05), hard + soft, -1.0)
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    return ix / K, iy / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sav', nargs='?', default=DEFAULT_MAP); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--every', type=int, default=30)
    ap.add_argument('--upscale', type=int, default=1)
    ap.add_argument('--source'); ap.add_argument('--no-source', action='store_true')
    ap.add_argument('--dt', type=float, default=0.02)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    hard, soft, water = load_map(a.sav)
    K = a.upscale
    hard, soft, water = upsample(hard, K), upsample(soft, K), upsample(water, K)
    sim = Sim(hard, soft, water, cell=1.0 / K, dt=a.dt / K, border=32 * K)

    if not a.no_source:
        if a.source:
            x, y, r, q = (float(v) for v in a.source.split(','))
        else:
            (x, y), r, q = default_source(hard, soft, water, K), 6.0, 6.0
        sim.set_source(x * K, y * K, r * K, q)
        print(f'source at ({x:.0f},{y:.0f}) radius {r:g}, {q:g} units/time', flush=True)

    steps, every = a.steps * K, a.every * K
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rend = Renderer(refresh_every=1)
    frames = [rend(sim)]
    vol0 = float(sim.water.sum())
    t0 = time.time()
    for i in range(1, steps + 1):
        sim.step()
        if i % every == 0:
            frames.append(rend(sim))
        if i % max(1, steps // 10) == 0:
            print(f'step {i}/{steps}  t={sim.t:.1f}  water vol x{sim.water.sum() / vol0:.3f}  '
                  f'max flux {float(sim.fR.max()):.2f}  {time.time() - t0:.1f}s', flush=True)

    disp = [f.resize((512, 512), Image.LANCZOS) if K > 1 else f for f in frames]
    disp[0].save(a.out + '.gif', save_all=True, append_images=disp[1:], duration=40, loop=0)
    pair = Image.new('RGB', (frames[0].width * 2 + 8, frames[0].height), (40, 40, 40))
    pair.paste(frames[0], (0, 0)); pair.paste(frames[-1], (frames[0].width + 8, 0))
    pair.save(a.out + '_before_after.png')
    print(f'wrote {a.out}.gif ({len(frames)} frames) and {a.out}_before_after.png '
          f'in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
