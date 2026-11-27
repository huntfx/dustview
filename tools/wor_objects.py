"""List the placed scenery objects of a level's world archive with their grid cell.

Usage:
  python wor_objects.py <level>        level = story number (1, 2, ... 13, '5bis') or a Wor_*.lin.bf path
  python wor_objects.py all            every NN.wor archive, outside-the-grid objects only

A placed object is a 322-byte `.obj` entry (32-byte entry header + 290-byte body). Read from its end:
  end-36 .. end-24   f32 x, y, z  world position
  end-164 .. end-36  two 4x4-ish float blocks: scale*rotation rows (x, y, z rows at +0, +16, +32),
                     then the pure rotation (not used here)
World -> grid: col = 256 + x * 2/3, row = 256 + y * 2/3 (findings/water-lava-sim.md); the grid is
cells 0..511, i.e. world -384..+384 on both axes. DustView draws row 0 at the bottom by default, so
'south' = negative rows.
"""
import math, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf, galileo

BIN = os.path.join(galileo.INSTALL, 'BinPC')


def level_archives():
    """{'01': path, ...} for every NN.wor world archive."""
    out = {}
    for f in sorted(os.listdir(BIN)):
        if not f.startswith('Wor_'):
            continue
        p = os.path.join(BIN, f)
        for e in bf.read_index(p):
            n = e['name']
            if n.endswith('.wor') and n[:-4].replace('bis', '').isdigit():
                out[n[:-4]] = p
    return out


def objects(path):
    out = []
    with open(path, 'rb') as f:
        for e in bf.read_index(path):
            if not e['name'].endswith('.obj') or e['size'] != 322:
                continue
            f.seek(e['offset'])
            d = f.read(322)
            x, y, z = struct.unpack_from('<3f', d, 322 - 36)
            m = struct.unpack_from('<12f', d, 322 - 164)
            scale = tuple(math.sqrt(m[i] ** 2 + m[i + 1] ** 2 + m[i + 2] ** 2) for i in (0, 4, 8))
            col, row = 256 + x * 2 / 3, 256 + y * 2 / 3
            out.append({'name': e['name'][:-4], 'x': x, 'y': y, 'z': z, 'col': col, 'row': row,
                        'scale': scale, 'inside': 0 <= col < 512 and 0 <= row < 512})
    return out


def side(o):
    s = []
    if o['row'] < 0: s.append('south')
    if o['row'] >= 512: s.append('north')
    if o['col'] < 0: s.append('west')
    if o['col'] >= 512: s.append('east')
    return '-'.join(s) or 'inside'


def show(objs, only_outside=False):
    for o in sorted(objs, key=lambda o: o['name']):
        if only_outside and o['inside']:
            continue
        print(f"  {o['name']:26s} world ({o['x']:8.1f},{o['y']:8.1f},{o['z']:6.1f})  "
              f"col {o['col']:6.0f} row {o['row']:6.0f}  scale {o['scale'][0]:.2f},{o['scale'][1]:.2f},"
              f"{o['scale'][2]:.2f}  {side(o)}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    arg = sys.argv[1]
    if arg == 'all':
        for lvl, p in sorted(level_archives().items()):
            print(f'{lvl}.wor  ({os.path.basename(p)})')
            show(objects(p), only_outside=True)
        return
    if os.path.exists(arg):
        p = arg
    else:
        p = level_archives()[arg.zfill(2) if arg.isdigit() else arg]
    print(os.path.basename(p))
    show(objects(p))


if __name__ == '__main__':
    main()
