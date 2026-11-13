"""Compare the Python model and the DLL harness with the *running* game (read-only).

Captures From Dust's live simulation grid on two consecutive simulator frames (double buffer at
ISimulatorWater+0x950, frame counter +0x92C), runs dustsim.step() and the game's DLL (simstep) on
the first, and compares both with the game's own second frame.
Usage: .venv/Scripts/python tools/live_check.py [pairs]
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, galileo as g, dustsim as m

GRID = 512 * 512 * 32


def find_sim(p):
    """Locate ISimulatorWater by its ParamF array (starts 0.999, 0.032, 0.4, 0.002) and sanity fields."""
    for a in p.find(struct.pack('<4f', 0.999, 0.032, 0.4, 0.002)) + p.find(struct.pack('<4f', 0.99, 0.0032, 0.4, 0.002)):
        base = a - 0x20
        d = p.read(base + 0x920, 12)
        if d and struct.unpack('<3I', d) == (9, 9, 2):
            return base
    raise SystemExit('simulator object not found (is a level loaded?)')


def capture_pair(p, base):
    """Return (n, grid after step n, grid after step n+1, params) or None if a frame was missed.
    The frame counter changes when a step *starts* (the step runs on worker threads), so a buffer is
    only read once the following step is running: that step reads it and writes the other buffer."""
    rd = lambda off, n: p.read(base + off, n)
    frame = lambda: struct.unpack('<I', rd(0x92C, 4))[0]
    def wait_change(f, timeout=3.0):
        t0 = time.time()
        while frame() == f:
            if time.time() - t0 > timeout:
                raise SystemExit('simulation frame counter not moving (game paused?)')
            time.sleep(0.0005)
        return frame()
    n = wait_change(frame())                 # step n just started: buf[(n-1)&1] is complete
    bufs = struct.unpack('<2I', rd(0x950, 8))
    if wait_change(n) != n + 1:              # step n+1 started: result of step n complete
        return None
    a = p.read(bufs[n & 1], GRID)
    prm = rd(0x20, 0x330)
    if frame() != n + 1 or wait_change(n + 1) != n + 2:
        return None
    b = p.read(bufs[(n + 1) & 1], GRID)
    if frame() != n + 2:
        return None
    return n, g.unpack(a), g.unpack(b), prm


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    p = memscan.Proc()
    base = find_sim(p)
    print('ISimulatorWater at %s' % hex(base))
    inn = (slice(34, 478), slice(34, 478))
    done = 0
    while done < pairs:
        cap = capture_pair(p, base)
        if not cap:
            print('missed a frame, retrying'); continue
        n, A, B, prm = cap
        P = np.frombuffer(prm[:224], np.float32).copy()
        pi, pu8, ganja = prm[0xE0:0x100], prm[0x100:0x130], prm[0x130:0x330]
        PD = {k: float(P[i]) for i, k in enumerate(g.PARAM_NAMES)}
        dll = g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(pi, np.int32), param_u8=pu8, ganja=ganja)
        mine = m.step(m.from_grid(A), PD)
        game, dl = m.from_grid(B), m.from_grid(dll)
        W = game['W']
        print('state %d -> %d: %d water cells, %d lava cells, %d cells changed by the game' %
              (n, n + 1, (W > 0).sum(), (W < 0).sum(), (A[:4] != B[:4]).any(0).sum()))
        for label, X in [('DLL harness', dl), ('Python model', mine)]:
            msg = []
            for key in ['T', 'soft', 'W', 'sed']:
                d = np.abs(X[key] - game[key])[inn]
                msg.append('%s max %.1e' % (key, d.max()))
            fm = (X['F'] != game['F'])[:, inn[0], inn[1]].sum()
            big = (np.abs(X['W'] - game['W']) > 1e-3)[inn]
            print('  %-12s %s | flux mismatches %d | fluid diff > 0.001 in %d cells' % (label, ' | '.join(msg), fm, big.sum()))
            if big.any() and label == 'DLL harness':
                ys, xs = np.nonzero(big)
                print('     e.g. at', [(int(y) + 34, int(x) + 34) for y, x in list(zip(ys, xs))[:6]])
        done += 1


if __name__ == '__main__':
    main()
