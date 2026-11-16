"""Check the compiled step (dustsim_nb) against the numpy step (dustsim.step_numpy): bit for bit,
step after step, on real maps and on a random landscape with water, sand, sediment and lava mixed.
Also prints the time per step of each.

Usage (FromDust/.venv): python tools/check_nb.py [steps] [map ...]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import galileo as g, dustsim, dustsim_nb, compare_sim

DEFAULT_MAPS = ['ALPHA_Map3_Vallee', 'ALPHA_Map10_FireAndWater', 'ALPHA_Map9_TheGreatCaldeira', 'random+lava']


def state(name):
    if name.startswith('random'):
        T, So, W, C3, border = compare_sim.random_map()
        if 'lava' in name:                       # lava blobs into water and onto dry sand
            rng = np.random.default_rng(7)
            y, x = np.mgrid[0:512, 0:512]
            for _ in range(40):
                cy, cx, r = rng.integers(40, 470), rng.integers(40, 470), rng.integers(4, 20)
                m = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
                W[m] = -rng.uniform(0.5, 6)
                C3[m] = rng.uniform(5, 60)
    else:
        T, So, W, C3, border = compare_sim.load_map(name)
    P = g.fluid_params(border=border)
    PD = {n: float(P[i]) for i, n in enumerate(g.PARAM_NAMES)}
    F = np.zeros((4, 512, 512), np.int32)
    return {'T': T, 'soft': So, 'W': W, 'sed': C3, 'F': F}, PD


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    maps = sys.argv[2:] or DEFAULT_MAPS
    dustsim_nb.step(*state(maps[0]))                         # compile (or load the cache)
    ok_all = True
    for name in maps:
        G0, P = state(name)
        A, B = G0, {k: v.copy() for k, v in G0.items()}
        ta = tb = 0.0
        bad = None
        for s in range(steps):
            t0 = time.perf_counter(); A = dustsim.step_numpy(A, P); t1 = time.perf_counter()
            B = dustsim_nb.step(B, P); t2 = time.perf_counter()
            ta += t1 - t0; tb += t2 - t1
            diff = [k for k in ('T', 'soft', 'W', 'sed', 'F') if not same(A[k], B[k])]
            if diff:
                bad = (s + 1, diff, {k: int((A[k] != B[k]).sum()) for k in diff})
                break
        n = bad[0] if bad else steps
        print('%-30s lava %6d | %s | numpy %6.1f ms/step, numba %5.1f ms/step' % (
            name, int((G0['W'] < 0).sum()), 'identical for %d steps' % steps if not bad else
            'DIFFERS at step %d: %s' % (bad[0], bad[2]), 1000 * ta / n, 1000 * tb / n), flush=True)
        ok_all &= bad is None
    print('all identical' if ok_all else 'MISMATCH')


if __name__ == '__main__':
    main()
