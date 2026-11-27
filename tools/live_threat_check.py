"""Check the decoded village water-threat sampler against the running game, tick by tick (read-only).

For every built village each game tick: read the game's own sampling angle (Village +0x2A0), reference height
(+0x2B4) and water threat timer (+0x284, reset to 0 on a hit); apply dustsim's point test to the live grid and
compare "hit" with the game's timer reset. Points: with no water repel running, 4 points at angle + i x 90 deg,
distance 28.5 world units x (1 - i/4) (1 cell = 1.5 units); with one running (a water RepulseFluid entry at the village in the per-tick list),
1 point at the repel's size (list radius x 3/2, i.e. the size before truncation, +-1). Hit: depth > 2 and
surface above the reference. Angle variants tried side by side: the value read (after this update's advance?)
and one update earlier (A - 2 pi / 31), to settle which one the game used.

Usage: .venv/Scripts/python tools/live_threat_check.py [seconds=60]
"""
import collections, math, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_generators as lg, live_check as lc, galileo as g, dustsim  # noqa: E402

VARIANTS = {'read': 0.0, 'previous': -2 * math.pi / 31.0}


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    t_wait = time.time()
    while True:                              # wait (up to 10 min) for the game and a loaded level
        try:
            p = memscan.Proc(); d = lg.dustai_delta(p); base = lc.find_sim(p)
            break
        except BaseException as e:
            if time.time() - t_wait > 600:
                raise
            print('waiting for a level: %s' % e, flush=True); time.sleep(5)
    vec = lg.known_modifier_list(p)
    vils = []
    for a in p.find(struct.pack('<2I', lg.VILLAGE_VTABLES[0] + d, lg.VILLAGE_VTABLES[1] + d)):
        r = p.read(a, 0x2C0)
        if r is None:
            continue
        col, row = struct.unpack_from('<2I', r, 0x144)
        if 0 < col < 512 and 0 < row < 512:
            vils.append((a, col, row))
    print('villages:', [(c, r) for _, c, r in vils], '| modifier list', 'found' if vec else 'NOT found', flush=True)
    R = dustsim.THREAT_RADIUS_FACTOR * dustsim.VILLAGE_RADIUS
    stats = {v: collections.Counter() for v in VARIANTS}
    per_village = {v: collections.defaultdict(collections.Counter) for v in VARIANTS}
    last, prev_timer, t0, counted = None, {}, time.time(), 0
    while time.time() - t0 < secs:
        n = struct.unpack('<I', p.read(base + 0x92C, 4))[0]
        if n == last:
            time.sleep(0.002); continue
        last = n
        reps = [e for e in (lg.read_modifier_list(p, vec, settle=0.004) if vec else []) if e['type'] == 1 and e['f5'] < 0]
        bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
        raw = p.read(bufs[(n - 1) & 1], 512 * 512 * 32)
        if raw is None:
            continue
        Gr = g.unpack(raw); T = Gr[0].view(np.float32); W = Gr[2].view(np.float32)
        counted += 1
        for a, col, row in vils:
            r = p.read(a, 0x2C0)
            wx, wy = struct.unpack_from('<2f', r, 0x178)       # world position
            ang, ref, timer = (struct.unpack_from('<f', r, o)[0] for o in (0x2A0, 0x2B4, 0x284))
            pt = prev_timer.get(a); prev_timer[a] = timer
            if pt is None or (pt != 0.0 and timer - pt > 0.2):
                continue
            game_hit = timer == 0.0
            run = [e for e in reps if abs(e['col'] - col) <= 3 and abs(e['row'] - row) <= 3 and e['amount'] < 2]
            for vname, off in VARIANTS.items():
                A = ang + off
                if run:                                     # single point at the repel's size (radius = trunc(size x 2/3))
                    size = (run[0]['radius'] + 0.5) * 1.5             # world units (radius = trunc(size / 1.5))
                    pts = [(A, size)]
                else:
                    pts = [(A + i * math.pi / 2, R * (1 - i / 4)) for i in range(4)]
                hit = False
                for aa, dist in pts:
                    c = int((wx + math.cos(aa) * dist + 384.0) * (2 / 3)); rr = int((wy + math.sin(aa) * dist + 384.0) * (2 / 3))
                    if 33 <= c < 479 and 33 <= rr < 479:
                        w = float(W[rr, c])
                        if w > dustsim.THREAT_WATER and float(T[rr, c]) + w > ref:
                            hit = True
                key = ('both' if hit and game_hit else 'none' if not hit and not game_hit
                       else 'model_only' if hit else 'game_only')
                key += '_running' if run else ''
                stats[vname][key] += 1
                per_village[vname]['(%d,%d)' % (col, row)][key] += 1
    print('ticks compared:', counted)
    for v in VARIANTS:
        print('angle %s: %s' % (v, dict(stats[v])))
        for vil, c in per_village[v].items():
            print('   %s %s' % (vil, dict(c)))


if __name__ == '__main__':
    main()
