"""The game's cursor limit per level: DustAi::OffmapCollider (the level's ColMap*.ai in NN.wor).

Usage:
  python offmap.py            table of every story level's tolerance and inset range

Fields (LyN reflection records, names hashed with galileo.lyn_hash; object offsets from the registration
code in dustai.win32.f.dll 0x10038250-0x10038470):
  m_segments            +0x24   u32[128] = 4 x 32 insets in cells: [0] south (bounds y, looked up by x),
                                [1] north (y, by x), [2] west (x, by y), [3] east (x, by y)
  m_verticalOffset      +0x224  f32 (1.0 everywhere)
  m_cursorTolerance     +0x228  f32, world units (1 cell = 1.5)
  m_nbIterations        +0x22C  u32 (2): how many times the clamp is applied
  m_activateLightings, m_lightingDuration / LaunchInterval / Random, m_displaySegments1-4: the
                                edge lightning effect (method around 0x10037D00, timers +0x230-0x244)
Clamp (dustai 0x10037E70, decoded 2026-11-27): position -> cell (col, row) (0x101400C0, clamped to
0..511); per axis i = (cell - 32) // 14, f = (cell - 32) / 14 - i, value = S[i-1] + (S[i] - S[i-1]) * f
(i-1 clamped to >= 0; unsigned maths, so cells < 32 would index out of range, not handled here);
  y_min(x) = -336 + 1.5 * south(x) + tol      y_max(x) = 334.5 - tol - 1.5 * north(x)
  x_min(y) = -336 + 1.5 * west(y)  + tol      x_max(y) = 334.5 - tol - 1.5 * east(y)
in world units (-336 = cell 32, 334.5 = cell 479: the inner edges of the fixed 32-cell border band).
"""
import os, struct, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf, galileo, wor_objects

CLASS = galileo.lyn_hash('DustAi::OffmapCollider')
H_SEGMENTS = galileo.lyn_hash('m_segments')
H_TOL = galileo.lyn_hash('m_cursorTolerance')
H_ITER = galileo.lyn_hash('m_nbIterations')
H_LIGHT = galileo.lyn_hash('m_activateLightings')


def limits(level):
    """{'tolerance', 'iterations', 'lightning', 'segments' (4, 32) int} or None (no OffmapCollider)."""
    path = wor_objects.level_archives()[level]
    with open(path, 'rb') as f:
        for e in bf.read_index(path):
            if not e['name'].endswith('.ai'):
                continue
            f.seek(e['offset'])
            d = f.read(e['size'])
            if struct.pack('<I', CLASS) not in d:
                continue
            field = lambda h, fmt: struct.unpack_from(fmt, d, d.index(struct.pack('<I', h)) + 12)[0]
            s = d.index(struct.pack('<I', H_SEGMENTS)) + 33        # record header, then 128 u32
            return {'tolerance': field(H_TOL, '<f'), 'iterations': field(H_ITER, '<I'),
                    'lightning': field(H_LIGHT, '<I'),
                    'segments': np.frombuffer(d, '<u4', 128, s).reshape(4, 32).astype(int)}
    return None


def _inset(seg, cell):
    """Interpolated inset (cells) for integer grid cells `cell` (array), as the game's clamp does."""
    t = (np.asarray(cell) - 32) / 14.0
    i = np.clip(np.floor(t).astype(int), 0, 31)
    f = t - i
    j = np.maximum(i - 1, 0)
    return seg[j] + (seg[i] - seg[j]) * f


def reachable(lim):
    """(512, 512) bool [row, col]: cells whose centre the cursor can reach."""
    tol, S = lim['tolerance'], lim['segments']
    cells = np.arange(512)
    w = (cells + 0.5 - 256) * 1.5                                  # world coordinate of cell centres
    y_lo = -336 + 1.5 * _inset(S[0], cells) + tol                 # per column
    y_hi = 334.5 - tol - 1.5 * _inset(S[1], cells)
    x_lo = -336 + 1.5 * _inset(S[2], cells) + tol                 # per row
    x_hi = 334.5 - tol - 1.5 * _inset(S[3], cells)
    ok_y = (w[:, None] >= y_lo[None, :]) & (w[:, None] <= y_hi[None, :])     # [row, col]
    ok_x = (w[None, :] >= x_lo[:, None]) & (w[None, :] <= x_hi[:, None])
    return ok_y & ok_x


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('level  tol(world)  lightning  insets beyond the 32-cell band (cells): south / north / west / east  reachable cells')
    for lvl in sorted(wor_objects.level_archives()):
        lim = limits(lvl)
        if not lim:
            print(f'{lvl:5s}  no OffmapCollider')
            continue
        S = lim['segments']
        rng = ' / '.join(f'{a.min()}-{a.max()}' for a in S)
        n = reachable(lim).sum()
        print(f'{lvl:5s}  {lim["tolerance"]:6.1f}      {lim["lightning"]}         {rng:40s}  {n:6d} ({100 * n / 512 ** 2:.0f} %)')


if __name__ == '__main__':
    main()
