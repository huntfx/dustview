"""Watch the running game's explosive trees (DustAi::BangTree) and catch their craters (read-only).

Trees are found by a memory scan in a background thread every ~3 s (live_generators.bang_trees, ~1.7 s) and
followed every few ms by address (bang_tree_at, ~0.3 ms for all). The main loop also polls the manager's
set-height list ([[0x1017989C]+0x2C]+0x48: (col, row, height) records the game writes into the ground that frame;
other features use it every frame too, so only records inside a known tree's 40 x 40 square count) and keeps the
grids of the last few simulation frames.
Prints every tree change. When a tree's danger bit comes on: what lies at the grid sensor's hit cell and its
3 x 3 neighbourhood in that frame (fluid, plant type, fire flag (component 6 bit 6), fire time). When a tree
starts exploding: compares dustsim.bang_tree_hole on the grids from before with the game's queued values (if the
list was caught) and with the ground 0.5 s later, and saves the grids to an .npz.

Usage: .venv/Scripts/python tools/live_bang_watch.py [seconds=900]
Log:   data/live/bang_watch_<time>.log, crater data bang_watch_<time>_<n>.npz
"""
import os, sys, time, struct, threading, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_generators as lg, live_check as lc, galileo as g, dustsim

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 512 * 512 * 32


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 900
    p = memscan.Proc()
    d = lg.dustai_delta(p)
    base = lc.find_sim(p)
    gm = struct.unpack('<I', p.read(0x1017989C + d, 4))[0]
    em = struct.unpack('<I', p.read(gm + 0x2C, 4))[0]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join(HERE, '..', 'data', 'live', 'bang_watch_%s' % stamp)
    log = open(out_base + '.log', 'w', encoding='utf-8')
    t0 = time.time()

    def out(s):
        s = '%7.2f s  %s' % (time.time() - t0, s)
        print(s, flush=True)
        log.write(s + '\n'); log.flush()

    addrs = set()
    def scanner():                                   # memory scan for new trees (seeds appear all the time)
        p2 = memscan.Proc()
        while time.time() - t0 < secs:
            try:
                addrs.update(int(t['address'], 16) for t in lg.bang_trees(p2, d))
            except Exception:
                pass
            time.sleep(3)
    threading.Thread(target=scanner, daemon=True).start()

    def frame():
        return struct.unpack('<I', p.read(base + 0x92C, 4))[0]

    def grab():
        n = frame()
        bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
        raw = p.read(bufs[(n - 1) & 1], GRID)
        return n, (g.unpack(raw) if raw else None)

    def cell_info(G, c, r):
        f = G[2].view(np.float32)[r, c]
        v6, v7 = int(G[6][r, c]), int(G[7][r, c])
        return 'fluid %+.2f ground %.1f | plant %d fire %d firetime %d' % (
            f, G[0].view(np.float32)[r, c], v6 & 15, (v6 >> 6) & 1, v7 >> 16)

    def hood(G, c, r):
        v6, v7 = G[6][r - 1:r + 2, c - 1:c + 2].astype(np.int64), G[7][r - 1:r + 2, c - 1:c + 2].astype(np.int64)
        return 'fire flags 3x3 %s, fire times %s' % ((v6 >> 6 & 1).ravel().tolist(), (v7 >> 16).ravel().tolist())

    def crater(t, grids, n, caught):
        """A tree just entered the exploding state: the crater was queued on entering, so it is in the grid of
        this or the next frame. Fit bang_tree_hole on each earlier grid against the ground 0.5 s later."""
        c, r = t['col'], t['row']
        time.sleep(0.5)
        _, A = grab()
        Ta = A[0].view(np.float32)
        r0, r1, c0, c1 = max(r - 20, 33), min(r + 20, 479), max(c - 20, 33), min(c + 20, 479)
        Wa = A[2].view(np.float32)[r0:r1, c0:c1]
        lines = []
        for fn, B in grids:
            Tb = B[0].view(np.float32)
            M = Tb.copy()
            dustsim.bang_tree_hole(M, c, r)
            dry = (B[2].view(np.float32)[r0:r1, c0:c1] == 0) & (Wa == 0)
            dm = (M - Ta)[r0:r1, c0:c1]
            lowered = (Tb - Ta)[r0:r1, c0:c1]
            lines.append('frame %d: game lowered %d cells (> 0.05), max %.2f; model - game on dry cells: |diff| < 0.05 in '
                         '%d of %d, max %.3f; corners (outside the circle, model digs): game lowered %d of %d' % (
                fn, int((lowered > 0.05).sum()), float(lowered.max()), int((np.abs(dm) < 0.05)[dry].sum()), int(dry.sum()),
                float(np.abs(dm)[dry].max()) if dry.any() else -1,
                int(((lowered > 0.05) & _corner(r0, r1, c0, c1, c, r)).sum()), int(_corner(r0, r1, c0, c1, c, r).sum())))
        recs = [x for x in caught if time.time() - x[0] < 3]
        if recs:
            rec = recs[-1][2]
            cols = np.array([x[0] for x in rec]); rows = np.array([x[1] for x in rec]); vals = np.array([x[2] for x in rec], np.float32)
            for fn, B in grids:
                M = B[0].view(np.float32).copy()
                dustsim.bang_tree_hole(M, c, r)
                lines.append('queued values vs model on frame %d: max |diff| %.6f over %d records' % (
                    fn, float(np.abs(M[rows, cols] - vals).max()), len(vals)))
        sep = '\n            '
        out('CRATER %d at tree %s (%d,%d):' % (n, t['id'], c, r) + sep + sep.join(lines))
        np.savez_compressed('%s_%d.npz' % (out_base, n), col=c, row=r, frames=[f for f, _ in grids],
                            before=np.stack([B[:3] for _, B in grids]), after=A[:3])

    out('simulator 0x%X, edit manager 0x%X; log %s' % (base, em, out_base))
    grids = collections.deque(maxlen=6)              # (frame, grid) of the last frames
    n0, G = grab()
    grids.append((n0, G))
    prev, caught, blasts = {}, [], 0
    last_other = 0
    while time.time() - t0 < secs:
        try:
            # 1. set-height list
            ptr, k = struct.unpack('<2I', p.read(em + 0x4C, 8))
            if k and ptr and k < 5000:
                raw = p.read(ptr, 12 * k)
                recs = [struct.unpack_from('<IIf', raw, 12 * i) for i in range(k)] if raw else []
                trees_sq = [(t['col'], t['row']) for t in prev.values()]
                mine = [r for r in recs if any(c - 20 <= r[0] < c + 20 and w - 20 <= r[1] < w + 20 for c, w in trees_sq)]
                if len(mine) >= 50:
                    caught.append((time.time(), frame(), mine))
                    out('set-height list caught: %d records near trees (of %d)' % (len(mine), k))
                elif len(recs) - len(mine) and time.time() - last_other > 30:
                    last_other = time.time()
                    out('(set-height list: %d records elsewhere, other game features; ignored)' % (len(recs) - len(mine)))
            # 2. grids
            n = frame()
            if n != grids[-1][0]:
                n2, G = grab()
                if G is not None:
                    grids.append((n2, G))
            # 3. trees
            cur = {}
            for a in list(addrs):
                t = lg.bang_tree_at(p, a, d)
                if t is None:
                    addrs.discard(a)
                    continue
                cur[t['id']] = t
                o = prev.get(t['id'])
                key = (t['state'], t['danger'], t['hit'], t['grid_hit'] is not None)
                okey = o and (o['state'], o['danger'], o['hit'], o['grid_hit'] is not None)
                if key != okey:
                    msg = '%stree %s (%d,%d) %s danger %s hit %s timer %s grid_hit %s clock %.2f' % (
                        '' if o else 'NEW ', t['id'], t['col'], t['row'], t['state'], t['danger'], t['hit'],
                        '%.3f' % t['danger_timer'] if 'danger_timer' in t else '-', t['grid_hit'], t.get('clock', -1))
                    Gn = grids[-1][1]
                    if t['grid_hit']:
                        msg += '\n            hit cell (frame %d): %s; %s; tree cell: %s' % (
                            grids[-1][0], cell_info(Gn, *t['grid_hit']), hood(Gn, *t['grid_hit']), cell_info(Gn, t['col'], t['row']))
                    elif o is None or t['state'] != o['state']:
                        msg += ' | tree cell: ' + cell_info(Gn, t['col'], t['row'])
                    out(msg)
                if t['state'] == 'exploding' and (o is None or o['state'] != 'exploding'):
                    blasts += 1
                    crater(t, list(grids), blasts, caught)
            for i in set(prev) - set(cur):
                out('tree %s (%d,%d) gone' % (i, prev[i]['col'], prev[i]['row']))
            prev = cur
            time.sleep(0.001)
        except Exception as e:
            out('read failed: %r' % e)
            time.sleep(0.5)

    out('done')


def _corner(r0, r1, c0, c1, c, r):
    """Cells of the window farther than 20 cells from the tree (dug only by the unclamped formula)."""
    rr, cc = np.mgrid[r0:r1, c0:c1]
    return (rr - r) ** 2 + (cc - c) ** 2 > 21 ** 2


if __name__ == '__main__':
    main()
