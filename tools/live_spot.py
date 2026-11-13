"""Watch one spot of a running From Dust level (read-only): what the level adds there each tick
(game's next state minus the game's own step) and the ground/fluid state on it.

Usage: .venv/Scripts/python tools/live_spot.py <row> <col> [radius=3] [seconds=300] [interval=1]
Log: data/live/spot_<row>_<col>_<time>.log (also printed). Used to find when a spring stops.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, galileo as g, dustsim as m

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    y, x = int(sys.argv[1]), int(sys.argv[2])
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 300
    every = float(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(os.path.join(HERE, '..', 'data', 'live'), exist_ok=True)
    logp = os.path.join(HERE, '..', 'data', 'live', time.strftime('spot_%d_%d_%%Y%%m%%d_%%H%%M%%S.log' % (y, x)))
    log = open(logp, 'w', encoding='utf-8')
    def out(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    p = memscan.Proc()
    base = lc.find_sim(p)
    sl = (slice(y - r, y + r + 1), slice(x - r, x + r + 1))
    out('watching (%d,%d) radius %d; log %s' % (y, x, r, logp))
    out('   t(s)   tick | added total  cells>0 | terrain min/mean/max | soft mean/max | fluid mean/max | rock(T-soft) mean')
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            cap = lc.capture_pair(p, base)
        except SystemExit:                    # game paused: wait instead of quitting
            out('%7.1f  (paused)' % (time.time() - t0)); time.sleep(2); continue
        if not cap:
            continue
        n, A, B, prm = cap
        P = np.frombuffer(prm[:224], np.float32).copy()
        step = m.from_grid(g.run(A, P, steps=1, skip_border=2, param_i=np.frombuffer(prm[0xE0:0x100], np.int32),
                                 param_u8=prm[0x100:0x130], ganja=prm[0x130:0x330]))
        game = m.from_grid(B)
        add = (np.abs(game['W']) - np.abs(step['W']))[sl]
        T, So, W = game['T'][sl], game['soft'][sl], game['W'][sl]
        if np.abs(game['W'] - step['W']).sum() > 1000:      # mistimed read (half-written buffer)
            continue
        out('%7.1f %6d | %+9.3f  %4d   | %6.2f %6.2f %6.2f | %5.2f %5.2f | %5.2f %5.2f | %6.2f' % (
            time.time() - t0, n + 1, add.sum(), (add > 1e-3).sum(), T.min(), T.mean(), T.max(),
            So.mean(), So.max(), W.mean(), W.max(), (T - So).mean()))
        time.sleep(every)
    out('done')


if __name__ == '__main__':
    main()
