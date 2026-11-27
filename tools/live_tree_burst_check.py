"""Check the modelled fire-tree burst + fire against the running game (read-only).

Waits for a fire tree's burst in the game's per-frame vegetation edit list ([[0x1017989C]+0x2C]+0x28: entries
u32 col, row, value), captures the grid at once (the game applies the edits a tick or two later), applies
dustsim.tree_fire_burst there, runs the copied vegetation / fire step (ganja_step) for as many ticks as the game
runs in the next few seconds, and compares the fire flags / fire times around the tree and on the whole map.
Ticks in which the game's grid did not change (paused) are not counted.

Usage: .venv/Scripts/python tools/live_tree_burst_check.py [seconds_after=4] [timeout=90] [species_set=0]
       (species set: vegsim.level_set, e.g. 0 for Story 6 Wildfires, 1 for Story 7 Waves)
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_generators as lg, live_check as lc, galileo as g, dustsim, ganja_step, vegsim


def main():
    after = float(sys.argv[1]) if len(sys.argv) > 1 else 4
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 90
    vset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    p = memscan.Proc(); d = lg.dustai_delta(p); base = lc.find_sim(p)
    gm = struct.unpack('<I', p.read(0x1017989C + d, 4))[0]
    fm = struct.unpack('<I', p.read(gm + 0x2C, 4))[0]
    trees = {(t['col'], t['row']): t for t in lg.fire_trees(p, d) if t['state'] == 3}
    ring = {(c + dc, r + dr): (c, r) for (c, r) in trees for dc in (-1, 0, 1) for dr in (-1, 0, 1) if dc or dr}

    def grab():
        n = struct.unpack('<I', p.read(base + 0x92C, 4))[0]
        bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
        return n, g.unpack(p.read(bufs[(n - 1) & 1], 512 * 512 * 32))

    t0 = time.time(); tree = None
    while time.time() - t0 < timeout and tree is None:
        ptr, k, cap = struct.unpack('<3I', p.read(fm + 0x2C, 12))
        if k:
            raw = p.read(ptr, 12 * k)
            for i in range(k):
                c, r, v = struct.unpack_from('<3I', raw, 12 * i)
                if (c, r) in ring:
                    tree, value = ring[(c, r)], v
                    break
        if tree is None:
            time.sleep(0.001)
    if tree is None:
        raise SystemExit('no fire tree burst seen in %.0f s (game paused?)' % timeout)
    n0, G0 = grab()
    print('burst of the tree at col %d row %d seen (value %d); grid captured at tick %d' % (*tree, value, n0), flush=True)
    block = vegsim.ganja_block(vset); prop = vegsim.propagate_table()
    c6, c7 = G0[6].copy(), G0[7].copy()
    S = dustsim.from_grid(G0)
    DELAY = 2                       # as DustView (ModelSource.BURST_DELAY): the cells are lit 2 ticks after the burst
    # follow the game tick by tick, counting only ticks whose grid changed
    last, prev, steps, t1 = n0, G0[7], 0, time.time()
    while time.time() - t1 < after:
        n, G = grab()
        if n != last and not np.array_equal(G[7], prev):
            steps += n - last
            prev = G[7]
        last = n
        time.sleep(0.01)
    G1 = G
    lit = 0
    for k in range(steps):
        if k == DELAY:
            lit = dustsim.tree_fire_burst(c6, c7, tree[0], tree[1], block)
        c6, c7 = ganja_step.step_arrays(c6, c7, S['T'], S['soft'], S['W'], block, prop)
    TC, TR = tree
    sl = (slice(TR - 3, TR + 4), slice(TC - 3, TC + 4))
    f = lambda a: (a >> 6) & 1
    b = lambda a: (a >> 7) & 1
    print('model lit %d cells; %d game ticks followed' % (lit, steps))
    print('around the tree (7 x 7): F = burning, B = burned, game | model')
    for gr6, mr6 in zip(G1[6][sl], c6[sl]):
        cell = lambda w: 'F' if w >> 6 & 1 else ('B' if w >> 7 & 1 else '.')
        print('   ' + ''.join(cell(w) for w in gr6) + '   |   ' + ''.join(cell(w) for w in mr6))
    print('fire / burned time, game:'); print(G1[7][sl] >> 16)
    print('model:'); print(c7[sl] >> 16)
    for name, fn in (('burning', f), ('burned', b)):
        a, m = fn(G1[6]), fn(c6)
        print('whole map %s: game %d, model %d, differ %d' % (name, a.sum(), m.sum(), (a != m).sum()))


if __name__ == '__main__':
    main()
