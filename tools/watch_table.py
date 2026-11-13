"""Log where (and whether) a known source record sits in From Dust's memory over time (read-only).

Searches all readable memory every few seconds for a record's first five floats
(col, row, radius, 0, rate) and logs the address(es), so repeated level loads show whether the
source table lands at a fixed address. Default record: Map 4's first lava vent.

Usage: .venv/Scripts/python tools/watch_table.py [seconds=300] [col row radius rate]
Log: data/live/watch_table_<time>.log
"""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    col, row, rad, rate = (float(v) for v in sys.argv[2:6]) if len(sys.argv) > 5 else (257, 321, 2.5, 0.66)
    pat = struct.pack('<5f', col, row, rad, 0, rate)
    here = os.path.dirname(os.path.abspath(__file__))
    logp = os.path.join(here, '..', 'data', 'live', time.strftime('watch_table_%Y%m%d_%H%M%S.log'))
    log = open(logp, 'w', encoding='utf-8')
    def out(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    out('watching for record col %g row %g radius %g rate %g' % (col, row, rad, rate))
    t0, last = time.time(), None
    while time.time() - t0 < secs:
        try:
            p = memscan.Proc()
            hits = tuple(hex(a) for a in p.find(pat))
        except SystemExit:
            hits = ('(game not running)',)
        if hits != last:
            out('t=%6.1fs  %s' % (time.time() - t0, ', '.join(hits) if hits else '(not in memory)'))
            last = hits
        time.sleep(2)
    out('done')


if __name__ == '__main__':
    main()
