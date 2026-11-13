"""Watch one fluid generator's bury activity next to the ground at its centre (read-only).

Every ~1 s logs: the generator's activity (+0x21C, 1 = free, 0 = buried), its fade-in counter and
mode, and the game's hard ground, soft ground and water at the given cell. Use it while covering a
spring in play to see when and how it stops.

Usage: .venv/Scripts/python tools/live_bury.py <row> <col> [seconds=240]
The generator is picked as the live one whose rate/size match best is ambiguous, so it is chosen by
the cell given: the generator objects don't store grid cells directly, so pass --rate R to pick it.
Log: data/live/bury_<row>_<col>_<time>.log
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check as lc, dustsim as m, live_generators as lg


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    row, col = int(args[0]), int(args[1])
    secs = float(args[2]) if len(args) > 2 else 240
    rate = float(sys.argv[sys.argv.index('--rate') + 1]) if '--rate' in sys.argv else None
    p = memscan.Proc()
    gens = [g for g in lg.generators(p) if g['mode'] != 0]
    if rate is not None:
        gens = [g for g in gens if abs(g['rate'] - rate) < 1e-3]
    if len(gens) != 1:
        raise SystemExit('pick one generator with --rate; candidates: %s' % [(g['address'], g['rate']) for g in gens])
    addr = int(gens[0]['address'], 16)
    base = lc.find_sim(p)
    here = os.path.dirname(os.path.abspath(__file__))
    logp = os.path.join(here, '..', 'data', 'live', time.strftime('bury_%d_%d_%%Y%%m%%d_%%H%%M%%S.log' % (row, col)))
    log = open(logp, 'w', encoding='utf-8')

    def out(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    out('generator %s rate %g at row %d col %d; log %s' % (gens[0]['address'], gens[0]['rate'], row, col, logp))
    t0 = time.time()
    while time.time() - t0 < secs:
        raw = p.read(addr, 0x220)
        mode = struct.unpack_from('<I', raw, 0x208)[0]
        fade, act = struct.unpack_from('<2f', raw, 0x218)
        try:
            cap = lc.capture_pair(p, base)
        except SystemExit:
            out('t=%5.1fs  (paused)  activity %.3f' % (time.time() - t0, act)); time.sleep(1); continue
        if not cap:
            continue
        G = m.from_grid(cap[2])
        sl = (slice(row - 2, row + 3), slice(col - 2, col + 3))
        out('t=%5.1fs  activity %.3f fade %g mode %d | centre hard %.2f soft %.2f water %.2f | 5x5 mean hard %.2f soft %.2f' % (
            time.time() - t0, act, fade, mode, G['T'][row, col], G['soft'][row, col], G['W'][row, col],
            G['T'][sl].mean(), G['soft'][sl].mean()))
        time.sleep(1)
    out('done')


if __name__ == '__main__':
    main()
