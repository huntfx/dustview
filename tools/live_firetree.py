"""Measure what fire trees (Map 6) do to water, from the running game (read-only).

For each pair of consecutive game frames: run the game's own water step (DLL) on the first, apply that tick's
per-tick modifier list (springs, vents, drains, repels), and compare with the game's second frame. What is
left is fluid the simulation does not explain. It is summed per vegetation type of the cell, by burning /
not burning, and by distance (cells) from the nearest plant of the candidate tree types.

Usage: .venv/Scripts/python tools/live_firetree.py [pairs=60] [types=9,10]
Log:   data/live/firetree_<time>.log (also printed); last pair's arrays + dF summed over all pairs (dF_sum, n_pairs)
       in data/live/firetree_<time>.npz (tools/firetree_spots.py reads it)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, galileo as g, dustsim as m, ganja, live_generators as lg

HERE = os.path.dirname(os.path.abspath(__file__))


def dist_to(mask, maxd=12):
    """Chebyshev distance (cells) to the nearest True cell, capped at maxd."""
    d = np.full(mask.shape, maxd, np.int32)
    cur = mask.copy()
    d[cur] = 0
    for k in range(1, maxd):
        n = cur.copy()
        n[1:] |= cur[:-1]; n[:-1] |= cur[1:]; n[:, 1:] |= cur[:, :-1]; n[:, :-1] |= cur[:, 1:]
        n[1:, 1:] |= cur[:-1, :-1]; n[:-1, :-1] |= cur[1:, 1:]; n[1:, :-1] |= cur[:-1, 1:]; n[:-1, 1:] |= cur[1:, :-1]
        d[n & ~cur] = k
        cur = n
    return d


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    tree_types = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else '9,10').split(',')]
    os.makedirs(os.path.join(HERE, '..', 'data', 'live'), exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    logp = os.path.join(HERE, '..', 'data', 'live', 'firetree_%s.log' % stamp)
    log = open(logp, 'w', encoding='utf-8')
    def out(s=''):
        print(s, flush=True); log.write(s + '\n'); log.flush()

    p = memscan.Proc()
    base = lc.find_sim(p)
    vec = lg.known_modifier_list(p)
    if vec is None:
        vec = lg.find_modifier_list(p, lg.generators(p))
    out('simulator %s, modifier list %s, tree types %s, log %s' % (hex(base), vec and hex(vec), tree_types, logp))

    acc = {}           # key -> [cells x ticks, sum dF, sum of losses only]
    by_d = np.zeros((13, 3))
    near_entries = []
    done = skipped = 0
    dF_sum = np.zeros((512, 512))
    shown_types = False
    while done < pairs:
        try:
            cap = lc.capture_pair(p, base)
        except SystemExit:
            out('(paused)'); time.sleep(1); continue
        if not cap:
            continue
        entries = lg.read_modifier_list(p, vec) if vec else []
        n, A, B, prm = cap
        GA, GB = m.from_grid(A), m.from_grid(B)
        P = np.frombuffer(prm[:224], np.float32).copy()
        X = m.from_grid(g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                              param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
        m.apply_modifiers(X, entries)
        if int((GB['F'] != X['F']).any(0).sum()) > 3000:
            skipped += 1; continue                          # torn read
        dF = np.abs(GB['W']) - np.abs(X['W'])
        dF[:33] = dF[479:] = 0; dF[:, :33] = dF[:, 479:] = 0
        dF[np.abs(dF) < 1e-4] = 0
        dF_sum += dF
        veg = ganja.decode(A)
        typ, fire, burned = veg['type'], veg['fire'] == 1, veg['burned'] == 1
        trees = np.isin(typ, tree_types) & ~burned
        if not shown_types:
            shown_types = True
            vals, cnt = np.unique(typ, return_counts=True)
            out('plant cells per type: ' + ', '.join('%d:%d' % (v, c) for v, c in zip(vals, cnt)))
            out('tree cells %d, of them burning %d; burning cells overall %d, burned %d' % (
                trees.sum(), (trees & fire).sum(), fire.sum(), burned.sum()))
            ys, xs = np.nonzero(trees)
            if len(ys):
                out('tree cells span rows %d-%d cols %d-%d' % (ys.min(), ys.max(), xs.min(), xs.max()))
        wet = np.abs(GA['W']) > 1e-3
        for t in np.unique(typ):
            for fl in (False, True):
                sel = (typ == t) & (fire == fl) & wet
                if sel.any():
                    a = acc.setdefault((int(t), fl), [0, 0.0, 0.0])
                    a[0] += int(sel.sum()); a[1] += float(dF[sel].sum()); a[2] += float(np.minimum(dF[sel], 0).sum())
        d = dist_to(trees)
        for k in range(13):
            sel = (d == k) & wet
            by_d[k] += (sel.sum(), dF[sel].sum(), np.minimum(dF[sel], 0).sum())
        for e in entries:
            r, c = int(round(e['row'])), int(round(e['col']))
            if 0 <= r < 512 and 0 <= c < 512 and d[r, c] <= 3:
                near_entries.append((n, e))
        done += 1
        if done % 10 == 0:
            out('  %d pairs (tick %d), %d torn skipped; unexplained fluid near trees (d<=1) so far %+.3f' % (
                done, n, skipped, by_d[:2, 1].sum()))
    np.savez_compressed(logp[:-4] + '.npz', dF=dF, dF_sum=dF_sum, n_pairs=done, type=typ, fire=fire, burned=burned, water=GA['W'], terrain=GA['T'])

    out('\nUnexplained fluid change in WET cells over %d ticks (negative = game lost more than the step explains)' % done)
    out(' type fire | cell-ticks | total dF  | losses only | per cell-tick')
    for (t, fl), (nc, s, lo) in sorted(acc.items()):
        out(' %4d %4s | %10d | %+9.3f | %+11.3f | %+.5f' % (t, 'yes' if fl else 'no', nc, s, lo, s / nc))
    out('\nBy distance from the nearest live tree cell (types %s), wet cells:' % tree_types)
    out('  d | cell-ticks | total dF  | losses only | per cell-tick')
    for k in range(13):
        nc, s, lo = by_d[k]
        if nc:
            out(' %2s | %10d | %+9.3f | %+11.3f | %+.5f' % ('%d' % k if k < 12 else '12+', nc, s, lo, s / nc))
    out('\nModifier-list entries within 3 cells of a tree: %d' % len(near_entries))
    for n, e in near_entries[:20]:
        out('  tick %d type %d row %.1f col %.1f radius %.2f amount %+.4f temp %.1f' % (
            n, e['type'], e['row'], e['col'], e['radius'], e['amount'], e['temperature']))
    out('done')


if __name__ == '__main__':
    main()
