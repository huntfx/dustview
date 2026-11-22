"""Fire rules of From Dust's vegetation / fire step, in Python, checked against the game's own DLL.

Covers the fire part only: the fire time (component 7 high half = ignition heat, then burn timer,
then burned counter) and the fire / burned bits of component 6. Growth (type, status, vegetation
time) is not modelled here. Rules: findings/vegetation-fire-layers.md, "Fire rules".

  python fire_rules.py [MapName|path.sav] [ticks]     # default ALPHA_Map6_Fire 300

Each tick is predicted from the DLL's previous output (re-synced), so a mismatch shows one rule
difference, not drift. Prints per-tick counts and examples of mismatching cells.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galileo as g
import vegsim

BORDER = 32          # the step leaves the outer 32 cells alone (skip_border 2 x 16)


def table(block, off):
    """16 per-type u16 values of one block of the 512-byte per-type table."""
    b = np.frombuffer(bytes(block), np.uint8)
    return (b[off:off + 16].astype(np.int32) | (b[off + 16:off + 32].astype(np.int32) << 8))


def predict(c6, c7, fluid, block, fire_progress=10, burned_time=246):
    """Next fire time (int32 array) and fire / burned bits (bool arrays) from one tick's state.
    c6, c7: uint32 grid components 6 / 7; fluid: float32 component 2 (negative = lava)."""
    b = (c6 & 0xFF).astype(np.int32)
    t = b & 15
    rnd = ((c6 >> 8) & 15).astype(np.int32)
    ft = (c7 >> 16).astype(np.int32)
    burned = (b & 0x80) != 0
    lava = fluid < 0
    # Step1: a plant under lava counts as burning (working copy only; its own output is recomputed)
    fire_in = ((b & 0x40) != 0) | ((t != 0) & lava)
    f = fire_in.astype(np.int32)
    n = np.zeros_like(f)
    n[1:, :] += f[:-1, :]; n[:-1, :] += f[1:, :]; n[:, 1:] += f[:, :-1]; n[:, :-1] += f[:, 1:]
    # the game sums the neighbours' 0x40 bits in a byte (paddb, Step2_ComputeViability 0x100283DD) and
    # reads 2 bits: 4 burning neighbours wrap to 0 (the cell then cools as if none were burning)
    n &= 3
    # water depth x 128 (lava counts as no water), rounded to nearest like cvtps2dq
    w = np.clip(np.rint(np.where(lava, 0, fluid).astype(np.float32) * np.float32(128)), -32768, 32767).astype(np.int32)

    thr = table(block, 0x00)[t] + 8 * rnd          # FireIgnition[type] + Ignition8n[random nibble]
    dur = table(block, 0x20)[t]
    resist = table(block, 0x160)[t]

    heating = ~burned & (ft < thr)
    new = np.where(heating, ft + 8 * n, ft + fire_progress)
    new = np.where(heating & (n == 0), (ft + 1) >> 1, new)          # pavgw(ft, 0): halves, rounding up
    new = np.where(w >= resist, 0, new)                             # water puts fire out, holds heat at 0
    new = np.where(t == 0, 0, new)                                  # bare ground holds no heat
    new = np.where(burned, ft + 1, new)                             # burned counter
    new &= 0xFFFF
    fire = ~burned & (t != 0) & (new > thr)
    to_burned = fire & (new > dur)
    burned_out = burned & (new > burned_time)
    new = np.where(to_burned | burned_out, 0, new)
    fire_out = fire & ~to_burned
    burned_next = (burned & ~burned_out) | to_burned
    return new, fire_out, burned_next


def check(grid, ticks=300, vset=0, chunk=25, examples=4):
    fluid = g.getf(grid, g.WATER) if hasattr(g, 'getf') else grid[g.WATER].view(np.float32)
    block = vegsim.ganja_block(vset)
    inner = np.zeros((g.N, g.N), bool)
    inner[BORDER:-BORDER, BORDER:-BORDER] = True
    cur = grid.copy()
    done = 0
    totals = np.zeros(4, np.int64)
    while done < ticks:
        k = min(chunk, ticks - done)
        outs = vegsim.run_dll(cur, steps=k, vset=vset, every=True)
        prev = cur
        for i, out in enumerate(outs):
            ft, fire, burned = predict(prev[6], prev[7], fluid, block)
            gft = (out[7] >> 16).astype(np.int32)
            gfire = (out[6] & 0x40) != 0
            gburn = (out[6] & 0x80) != 0
            bad = inner & ((ft != gft) | (fire != gfire) | (burned != gburn))
            # plant died of old age / no sand this tick (growth, not modelled; beats a burn-out on the same tick): bare, timer 0
            died = bad & ((prev[6] & 15) != 0) & ~gburn & ~gfire & (gft == 0)
            ndied = int(died.sum())
            bad &= ~died
            nfire, nburn = int((gfire & inner).sum()), int((gburn & inner).sum())
            totals += (int(bad.sum()), nfire, nburn, ndied)
            tick = done + i + 1
            if bad.any() or tick % 25 == 0 or tick == 1:
                print(f'tick {tick}: burning {nfire}, burned {nburn}, died (growth) {ndied}, mismatches {int(bad.sum())}', flush=True)
            for r, c in list(zip(*np.nonzero(bad)))[:examples]:
                print(f'   r{r} c{c}: in c6 {prev[6][r, c] & 0xFFF:03x} ft {prev[7][r, c] >> 16} fluid {fluid[r, c]:.3f}'
                      f' -> game c6 {out[6][r, c] & 0xFF:02x} ft {gft[r, c]} | model fire {fire[r, c]:d} burned {burned[r, c]:d} ft {ft[r, c]}')
            prev = out
        cur = outs[-1].copy()
        for c in range(6):
            cur[c] = grid[c]
        done += k
    print(f'total over {ticks} ticks: mismatching cells {totals[0]}, burning cell-ticks {totals[1]}, burned cell-ticks {totals[2]}, '
          f'plants that died while the fire rules expected otherwise (growth, not modelled) {totals[3]}')
    return totals


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'ALPHA_Map6_Fire'
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    import compare_sim
    path = arg if arg.lower().endswith('.sav') else os.path.join(compare_sim.SAVES, arg + '.sav')
    grid = vegsim.from_sav(path)
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'{name}: species set {vegsim.level_set(name)}, {ticks} ticks', flush=True)
    check(grid, ticks, vset=vegsim.level_set(name))


if __name__ == '__main__':
    main()
