"""Check the Python re-implementation (dustsim.py) against the game's own simulation step.

Loads a map's saved start state, runs the game's DLL for N steps (via galileo.py / simstep.exe),
and compares dustsim.step() with the game one step at a time (re-synced to the game's state each
step, so errors don't accumulate). Prints per-step max/mean differences inside the border band.

Usage (from FromDust/, with the project venv):
  .venv/Scripts/python tools/compare_sim.py ALPHA_Map9_TheGreatCaldeira 10
  .venv/Scripts/python tools/compare_sim.py --random 5      # random landscape with sand + sediment
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import galileo as g, dustsim as m, sav

SAVES = os.path.join(g.INSTALL, 'DUST', 'DustSave', 'Ingame')


def load_map(name):
    _, u = sav.read_sav(os.path.join(SAVES, name + '.sav'))
    layer = lambda k: np.frombuffer(u[0x80 + k * 0x100000:0x80 + (k + 1) * 0x100000], np.float32).reshape(512, 512).copy()
    T, W, So, C3 = layer(0), layer(1), layer(2), layer(3)      # .sav blocks: terrain, fluid, soft, temp/sediment
    border = (float(T[5, 5] + W[5, 5]), float(T[5, 5]), float(So[5, 5]))
    return T, So, W, C3, border


def random_map(seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:512, 0:512] / 512.0
    So = (rng.random((512, 512)) * 2).astype(np.float32)
    T = (18 + 8 * np.sin(7 * x) * np.cos(5 * y)).astype(np.float32) + So
    W = np.maximum(0, 24 - T + rng.random((512, 512))).astype(np.float32)
    C3 = (W * rng.random((512, 512)) * 0.2).astype(np.float32)
    return T, So, W, C3, (0.0, 0.0, 0.0)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == '--random':
        name, (T, So, W, C3, border) = 'random', random_map()
    else:
        name, (T, So, W, C3, border) = sys.argv[1], load_map(sys.argv[1])
    steps = int(sys.argv[2])
    P = g.fluid_params(border=border)
    PD = {n: float(P[i]) for i, n in enumerate(g.PARAM_NAMES)}
    grid = g.empty_grid()
    for c, a in [(g.TERRAIN, T), (g.SOFT, So), (g.WATER, W), (g.TEMPSED, C3)]:
        g.setf(grid, c, a)
    print('%s: %d lava cells, %d water cells, border (surface, hard, soft) = %s' %
          (name, (W < 0).sum(), (W > 0).sum(), tuple(round(b, 2) for b in border)))
    outs = g.run(grid, P, steps=steps, skip_border=2, dump_every_step=True)
    inn = (slice(34, 478), slice(34, 478))   # inside the border ring
    G = m.from_grid(grid)
    for k, o in enumerate(outs):
        mine, game = m.step(G, PD), m.from_grid(o)
        msg = []
        for key in ['T', 'soft', 'W', 'sed']:
            d = np.abs(mine[key] - game[key])[inn]
            msg.append('%s max %.1e mean %.1e' % (key, d.max(), d.mean()))
        fm = (mine['F'] != game['F'])[:, inn[0], inn[1]].sum()
        kind = (np.abs(mine['W'] - game['W']) > 1e-3)[inn].sum()
        print('step %2d  %s | flux mismatches %d | cells with |fluid diff| > 0.001: %d' % (k + 1, ' | '.join(msg), fm, kind))
        G = game


if __name__ == '__main__':
    main()
