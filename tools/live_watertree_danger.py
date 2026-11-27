"""Follow the running game's water trees (read-only): danger bit, grid sensor cell (+0x3AC), store, state; prints
every change with what lies at the sensor cell (lava / water / plant / fire flag) and its offset from the tree.

Usage: .venv/Scripts/python tools/live_watertree_danger.py [seconds=600]
Log:   data/live/watertree_lava_<time>.log
"""
import sys, os, time, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, memscan, live_generators as lg, live_check as lc, galileo as g

p = memscan.Proc(); d = lg.dustai_delta(p); base = lc.find_sim(p)
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live', 'watertree_lava_%s.log' % time.strftime('%Y%m%d_%H%M%S')), 'w')
t0 = time.time()
def out(s):
    s = '%7.2f s  %s' % (time.time() - t0, s); print(s, flush=True); log.write(s + '\n'); log.flush()
trees = lg.water_trees(p, d)
addrs = {int(t['address'], 16): t for t in trees}
out('%d water trees: %s' % (len(trees), ', '.join('%s (%d,%d) %s %.0f' % (t['id'], t['col'], t['row'], t['state'], t['stored']) for t in trees)))
def grid():
    n = struct.unpack('<I', p.read(base + 0x92C, 4))[0]
    bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
    return g.unpack(p.read(bufs[(n - 1) & 1], 512 * 512 * 32))
prev = {}
G = grid(); tg = time.time()
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600
while time.time() - t0 < secs:
    if time.time() - tg > 0.05:
        G = grid(); tg = time.time()
    for a, t in addrs.items():
        raw = p.read(a, 0x3D8)
        if raw is None or struct.unpack_from('<I', raw, 0)[0] != lg.WATER_TREE_VTABLES[0] + d:
            if prev.get(a) != 'gone':
                out('tree %s gone' % t['id']); prev[a] = 'gone'
            continue
        col, row = struct.unpack_from('<2I', raw, 0x144)
        stored = struct.unpack_from('<f', raw, 0x39C)[0]
        bits = struct.unpack_from('<I', raw, 0x3D4)[0]
        dc = struct.unpack_from('<2i', raw, 0x3AC)
        sub = struct.unpack_from('<I', raw, 0x2A0)[0]
        st = 'absorbing' if sub == a + 0x3B4 else 'blow_up' if sub == a + 0x3C8 else 'other'
        key = (st, bits & 1, dc if 33 <= dc[0] < 479 and 33 <= dc[1] < 479 else None, int(stored // 250))
        if prev.get(a) != key:
            msg = 'tree %s (%d,%d) %s danger %d sensor cell %s store %.0f bits %x' % (t['id'], col, row, st, bits & 1, key[2], stored, bits)
            if key[2]:
                c, r = key[2]
                f = G[2].view(np.float32)[r, c]
                msg += ' | cell: fluid %+.2f (%s), plant %d fire %d, offset (%+d,%+d)' % (
                    f, 'lava' if f < 0 else 'water' if f > 0 else 'dry', G[6][r, c] & 15, (G[6][r, c] >> 6) & 1, c - col, r - row)
            out(msg); prev[a] = key
    time.sleep(0.002)
