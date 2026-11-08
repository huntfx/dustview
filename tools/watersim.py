"""From Dust water + erosion simulator in numpy, VERSION 0.

Status: a TEXTBOOK model of the kind the game's simulator appears to be, not yet the
game's exact rules. What is taken from the game:
  - the map data: ground surface, soft ground and water depth from a .sav (../docs/sav-format.md)
  - the grid (512 x 512) and per-cell state (4 outflow fluxes Left/Right/Up/Down, soft ground,
    sediment), as named in GalileoSimulatorWater.dll's debug strings
  - a fixed map border acting as open sea (the DLL has BorderWater/BorderHard/BorderSoft values)
What is NOT from the game yet (textbook "virtual pipes" shallow water + simple erosion,
Mei et al. 2007 style): the update equations, the constants (gravity, time step, damping,
erosion/deposition rates), lava, temperature, vegetation, evaporation.
See ../findings/python-water-sim.md for the plan to replace these with the game's own rules.

Usage (run with the project venv, which has numpy + pillow):
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
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav  # noqa: E402


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
    """Virtual-pipe shallow water over hard + soft ground, with soft-ground erosion."""

    def __init__(self, hard, soft, water, cell=1.0, dt=0.02, g=9.81, damping=0.998,
                 k_capacity=0.08, k_erode=0.2, k_deposit=0.3, border=1):
        self.hard, self.soft, self.water = hard.copy(), soft.copy(), water.copy()
        self.sed = np.zeros_like(water)                       # suspended soft ground (height units)
        self.fL, self.fR, self.fU, self.fD = (np.zeros_like(water) for _ in range(4))
        self.l, self.dt, self.g, self.damping = cell, dt, g, damping
        self.kc, self.ke, self.kd = k_capacity, k_erode, k_deposit
        n0, n1 = water.shape
        self.edge = np.zeros(water.shape, bool)
        self.edge[:border, :] = self.edge[-border:, :] = True
        self.edge[:, :border] = self.edge[:, -border:] = True
        self.init = (hard.copy(), soft.copy(), water.copy())
        self.ys, self.xs = np.mgrid[0:n0, 0:n1].astype(np.float32)
        self.source = None
        self.t = 0.0

    def ground(self):
        return self.hard + self.soft

    def step(self):
        dt, l, g = self.dt, self.l, self.g
        W = self.water
        surf = self.hard + self.soft + W

        # 1. Flux: each cell pushes water through 4 "pipes" toward lower neighbours.
        k = dt * g * l          # dt * A * g / l with pipe cross-section A = l*l
        fL = np.maximum(0.0, self.damping * self.fL + k * (surf - np.roll(surf, 1, axis=1)))
        fR = np.maximum(0.0, self.damping * self.fR + k * (surf - np.roll(surf, -1, axis=1)))
        fU = np.maximum(0.0, self.damping * self.fU + k * (surf - np.roll(surf, 1, axis=0)))
        fD = np.maximum(0.0, self.damping * self.fD + k * (surf - np.roll(surf, -1, axis=0)))
        fL[:, 0] = 0; fR[:, -1] = 0; fU[0, :] = 0; fD[-1, :] = 0
        out = fL + fR + fU + fD
        # never let a cell send out more water than it has
        scale = np.minimum(1.0, W * l * l / (out * dt + 1e-9))
        fL *= scale; fR *= scale; fU *= scale; fD *= scale
        out *= scale

        # 2. Water height from in/out flow
        inflow = (np.roll(fR, 1, axis=1) + np.roll(fL, -1, axis=1) +
                  np.roll(fD, 1, axis=0) + np.roll(fU, -1, axis=0))
        W_new = np.maximum(0.0, W + dt * (inflow - out) / (l * l))

        # 3. Velocity field (cells per time unit) from the net flow through each cell
        depth = np.maximum(0.5 * (W + W_new), 1e-3)
        qx = 0.5 * (np.roll(fR, 1, axis=1) - fL + fR - np.roll(fL, -1, axis=1))
        qy = 0.5 * (np.roll(fD, 1, axis=0) - fU + fD - np.roll(fU, -1, axis=0))
        u = qx / (l * depth)
        v = qy / (l * depth)
        u = np.where(W_new > 1e-3, u, 0.0); v = np.where(W_new > 1e-3, v, 0.0)

        # 4. Erosion / deposition of soft ground (hard ground is never eroded)
        speed = np.sqrt(u * u + v * v)
        capacity = self.kc * speed * np.minimum(W_new, 2.0)
        diff = capacity - self.sed
        erode = np.where(diff > 0, np.minimum(self.ke * diff * dt * 10, self.soft), 0.0)
        deposit = np.where(diff < 0, self.kd * (-diff) * dt * 10, 0.0)
        self.soft += deposit - erode
        self.sed += erode - deposit
        dry = W_new < 1e-3                      # dry cells drop everything they carry
        self.soft[dry] += self.sed[dry]
        self.sed[dry] = 0.0

        # 5. Carry sediment along with the water (semi-Lagrangian, bilinear)
        self.sed = self._advect(self.sed, u, v)

        # 6. Extra water from the "pour" tool
        if self.source is not None:
            mask, rate = self.source
            W_new[mask] += rate * dt

        # 7. Map border = open sea at its starting state
        h0, s0, w0 = self.init
        W_new[self.edge] = w0[self.edge]
        self.soft[self.edge] = s0[self.edge]
        self.sed[self.edge] = 0.0

        self.water = W_new
        self.fL, self.fR, self.fU, self.fD = fL, fR, fU, fD
        self.t += dt

    def _advect(self, F, u, v):
        n0, n1 = F.shape
        X = np.clip(self.xs - u * self.dt / self.l, 0, n1 - 1.001)
        Y = np.clip(self.ys - v * self.dt / self.l, 0, n0 - 1.001)
        x0 = X.astype(np.int32); y0 = Y.astype(np.int32)
        fx = X - x0; fy = Y - y0
        return ((F[y0, x0] * (1 - fx) + F[y0, x0 + 1] * fx) * (1 - fy) +
                (F[y0 + 1, x0] * (1 - fx) + F[y0 + 1, x0 + 1] * fx) * fy).astype(np.float32)

    def set_source(self, x, y, r, rate):
        d2 = (self.xs - x) ** 2 + (self.ys - y) ** 2
        self.source = (d2 <= r * r, rate)


# ---------------------------------------------------------------- rendering
LIGHT = np.array([-0.6, -0.6, 0.55]); LIGHT /= np.linalg.norm(LIGHT)


def render(sim, zscale=1.0):
    ground = sim.ground()
    gy, gx = np.gradient(ground * zscale / sim.l)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    shade = np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) * nz, 0, 1) * 0.75 + 0.25
    rock = np.array([118, 108, 98], np.float32)
    sand = np.array([214, 188, 138], np.float32)
    s = np.clip(sim.soft / 1.5, 0, 1)[..., None]
    col = (rock * (1 - s) + sand * s) * shade[..., None]

    W = sim.water
    a = (1 - np.exp(-W / 2.0))[..., None]
    shallow = np.array([70, 150, 185], np.float32)
    deep = np.array([18, 52, 105], np.float32)
    t = np.exp(-W / 10.0)[..., None]
    wcol = deep * (1 - t) + shallow * t
    mud = np.clip(sim.sed / (0.02 * np.maximum(W, 1e-3) + 1e-6), 0, 1)[..., None] * 0.6
    wcol = wcol * (1 - mud) + np.array([150, 118, 80], np.float32) * mud
    col = col * (1 - a) + wcol * a
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), 'RGB')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sav'); ap.add_argument('--out', required=True)
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
    sim = Sim(hard, soft, water, cell=1.0 / K, dt=a.dt / K, border=K)

    if not a.no_source:
        if a.source:
            x, y, r, q = (float(v) for v in a.source.split(','))
        else:
            # highest dry land cell at least 64 cells (512-grid) from the map edge
            inner = np.zeros(hard.shape, bool); m = 64 * K
            inner[m:-m, m:-m] = True
            g = np.where(inner & (water < 0.05), hard + soft, -1.0)
            iy, ix = np.unravel_index(np.argmax(g), g.shape)
            x, y, r, q = ix / K, iy / K, 6.0, 6.0
        sim.set_source(x * K, y * K, r * K, q)
        print(f'source at ({x:.0f},{y:.0f}) radius {r:g}, {q:g} units/time', flush=True)

    steps, every = a.steps * K, a.every * K
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    frames = [render(sim)]
    before = frames[0]
    vol0 = float(sim.water.sum())
    t0 = time.time()
    for i in range(1, steps + 1):
        sim.step()
        if i % every == 0:
            frames.append(render(sim))
        if i % max(1, steps // 10) == 0:
            print(f'step {i}/{steps}  t={sim.t:.1f}  water vol x{sim.water.sum() / vol0:.3f}  '
                  f'max speed {np.nanmax(np.abs(sim.fR)):.2f}  {time.time() - t0:.1f}s', flush=True)

    disp = [f.resize((512, 512), Image.LANCZOS) if K > 1 else f for f in frames]
    disp[0].save(a.out + '.gif', save_all=True, append_images=disp[1:], duration=40, loop=0)
    pair = Image.new('RGB', (frames[0].width * 2 + 8, frames[0].height), (40, 40, 40))
    pair.paste(before, (0, 0)); pair.paste(frames[-1], (frames[0].width + 8, 0))
    pair.save(a.out + '_before_after.png')
    print(f'wrote {a.out}.gif ({len(frames)} frames) and {a.out}_before_after.png '
          f'in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
