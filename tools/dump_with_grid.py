"""Pair a hook F3 texture dump with the game's grid at the same moment (read-only).

Waits for dustdump_convert_s1.bin (the first file F3 writes: the per-frame upload) to change, then reads the
simulation grid straight away (~0.1 s later) and saves it with the dump time. Run with the game running and
the hook installed, then press F3 in the game.
Usage: python dump_with_grid.py OUT.npz [timeout_s]
"""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_check, galileo
from dustview import GRID_BYTES

GAME = r'D:\Games\Steam\steamapps\common\From Dust'


def main():
    out = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    path = os.path.join(GAME, 'dustdump_convert_s1.bin')
    p = memscan.Proc()
    base = live_check.find_sim(p)
    m0 = os.path.getmtime(path) if os.path.exists(path) else 0
    print('waiting for F3...', flush=True)
    t_end = time.time() + timeout
    while time.time() < t_end:
        if os.path.exists(path) and os.path.getmtime(path) != m0:
            t_dump = os.path.getmtime(path)
            m = struct.unpack('<I', p.read(base + 0x92C, 4))[0]
            bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
            raw = p.read(bufs[(m - 1) & 1], GRID_BYTES)
            t_read = time.time()
            grid = galileo.unpack(raw)
            np.savez_compressed(out, grid=grid, tick=m - 1, t_dump=t_dump, t_read=t_read)
            print('grid of tick %d saved to %s (%.2f s after the dump file)' % (m - 1, out, t_read - t_dump), flush=True)
            return
        time.sleep(0.02)
    print('timed out', flush=True)


if __name__ == '__main__':
    main()
