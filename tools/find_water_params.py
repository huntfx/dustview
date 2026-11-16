"""Look for the water-parameters texture (the game's per-cell velocity + foam, `WaterParams`) in the
running game's memory (read-only) and compare it with the simulation grid.

Idea: the final water shader reads velocity from WaterParams.xy (stored 0..1, 0.5 = still) and foam
from .z. If the CPU builds this texture, a 512 x 512 x 4-byte buffer exists in memory where still water
gives two neighbouring bytes of 0x80. The scan finds memory runs of >= 256 pixels whose bytes 1 and 2
(or 0 and 1) are all 0x80, then reports candidate buffers.

Usage: python find_water_params.py scan          (writes data/live/water_params_candidates.txt)

Result (2026-11-16): one 2 MB block with two 512 x 512 textures; the second is the water parameters
(byte 2 / 3 = 128 + ~30 x net outflow x / -y, byte 0 = min(64 x depth, 255), byte 1 = 250 x sediment
concentration). The 0x80 signature also matched the 16-bit height texture (TextureZgZw). Details in
docs/technical-notes.md ("Water rendering").
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live')


def scan(min_run=256, chunk=1 << 24):
    p = memscan.Proc()
    hits = []
    t0 = time.time()
    total = 0
    for base, size in p.regions():
        if size < (1 << 20):                        # a 512 x 512 x 4 texture needs 1 MB
            continue
        for off in range(0, size, chunk):
            n = min(chunk, size - off)
            raw = p.read(base + off, n)
            if raw is None:
                continue
            total += n
            a = np.frombuffer(raw, np.uint8)[: (n // 4) * 4].reshape(-1, 4)
            for lanes in ((1, 2), (0, 1), (2, 3)):
                m = (a[:, lanes[0]] == 0x80) & (a[:, lanes[1]] == 0x80)
                if m.sum() < min_run:
                    continue
                # run lengths of consecutive True
                d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
                starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                long = np.flatnonzero(ends - starts >= min_run)
                if len(long):
                    hits.append((base + off, n, lanes, int((ends - starts)[long].sum()), int(len(long)),
                                 base + off + 4 * int(starts[long[0]])))
        print('scanned %.0f MB, %d candidate chunks, %.0f s' % (total / 1e6, len(hits), time.time() - t0), flush=True)
    return hits


def main():
    if sys.argv[1:2] == ['scan']:
        hits = scan()
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, 'water_params_candidates.txt')
        with open(path, 'w') as f:
            for h in hits:
                f.write('chunk %08x size %d lanes %s pixels-in-runs %d runs %d first-run %08x\n' % h)
        for h in sorted(hits, key=lambda h: -h[3])[:20]:
            print('chunk %08x size %d lanes %s pixels-in-runs %d runs %d first-run %08x' % h)
        print('wrote', path)


if __name__ == '__main__':
    main()
