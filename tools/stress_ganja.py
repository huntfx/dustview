"""Random-state stress test for tools/ganja_step.py against the DLL (all types, ages, fire states, water, lava, thin sand)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, galileo as g, vegsim, ganja_step as gs

def random_grid(seed):
    rng = np.random.default_rng(seed)
    N = g.N
    grid = g.empty_grid()
    # smooth-ish terrain with steps, sand 0-0.3, water 0-2 with lava patches
    T = 10 + np.cumsum(rng.normal(0, 0.6, (N, N)), axis=1) * 0.05 + rng.integers(0, 3, (N, N)) * rng.random((N, N)) * 2
    S = np.clip(rng.normal(0.15, 0.12, (N, N)), 0, None)
    W = np.where(rng.random((N, N)) < 0.5, 0, rng.random((N, N)) ** 3 * 2)
    W = np.where(rng.random((N, N)) < 0.05, -rng.random((N, N)) * 0.5, W)
    for c, a in ((g.TERRAIN, T), (g.SOFT, S), (g.WATER, W)):
        g.setf(grid, c, a.astype(np.float32))
    typ = rng.integers(0, 16, (N, N))
    typ = np.where(rng.random((N, N)) < 0.3, 0, typ)
    # patches of one type so 4-same neighbours happen
    blocks = rng.integers(0, 16, (N // 8, N // 8)).repeat(8, 0).repeat(8, 1)
    typ = np.where(rng.random((N, N)) < 0.3, blocks, typ)
    status = rng.integers(0, 4, (N, N))
    fire = rng.random((N, N)) < 0.05
    burned = rng.random((N, N)) < 0.05
    b0 = np.where(burned, 0x80, typ | (status << 4) | np.where(fire, 0x40, 0))
    b1 = rng.integers(0, 16, (N, N)) | np.where(rng.random((N, N)) < 0.05, 0x10, 0)
    grid[6] = (b0 | (b1 << 8)).astype(np.uint32)
    vt = np.where(rng.random((N, N)) < 0.5, rng.integers(0, 1200, (N, N)), rng.integers(0, 5000, (N, N)))
    ft = np.where(rng.random((N, N)) < 0.6, 0, rng.integers(0, 700, (N, N)))
    ft = np.where(burned, rng.integers(0, 250, (N, N)), ft)
    grid[7] = (vt | (ft << 16)).astype(np.uint32)
    return grid

if __name__ == '__main__':
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for vset in range(4):
        for seed in (1, 2):
            print(f'== species set {vset}, seed {seed}', flush=True)
            gs.check(random_grid(seed * 10 + vset), ticks=ticks, vset=vset, quiet=False, examples=2, margin=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
