"""From Dust map/save (.sav) reader.

Usage:
  python sav.py <file.sav> [--out DIR] [--bin]

Prints the header, unpacks the zlib payload and, with --out, writes:
  <name>.bin           raw decompressed payload (only with --bin; 7.9 MB each)
  <name>_rock.png      layer 0 (float32, ground surface incl. soft ground)
  <name>_water.png     layer 1 (float32, water depth)
  <name>_soft.png      layer 2 (float32, soft ground thickness, part of layer 0)
  <name>_layers.json   min/max per float layer
Format notes: ../docs/sav-format.md
"""
import argparse, array, json, os, struct, sys, zlib

GRID = 512
LAYER_BYTES = GRID * GRID * 4
GRID_START = 0x80  # 0x30-0x7F: 20-value prefix block (7F fill, or BE floats on some maps)
FLOAT_LAYERS = {0: 'rock', 1: 'water', 2: 'soft'}


def read_sav(path):
    d = open(path, 'rb').read()
    ver, usize, one, csize = struct.unpack('>IIII', d[:16])
    u = zlib.decompress(d[16:16 + csize])
    if len(u) != usize:
        raise ValueError(f'size mismatch: header {usize}, got {len(u)}')
    return {'version': ver, 'usize': usize, 'field3': one, 'csize': csize}, u


def float_layer(u, k):
    a = array.array('f')
    a.frombytes(u[GRID_START + k * LAYER_BYTES: GRID_START + (k + 1) * LAYER_BYTES])
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sav')
    ap.add_argument('--out')
    ap.add_argument('--bin', action='store_true', help='also write the raw decompressed payload')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    hdr, u = read_sav(a.sav)
    print('header', hdr)
    stats = {}
    for k, name in FLOAT_LAYERS.items():
        lay = float_layer(u, k)
        v = [x for x in lay if abs(x) < 1e30]
        stats[name] = {'min': min(v), 'max': max(v)}
        print(f'layer {k} {name}: {min(v):.3f} .. {max(v):.3f}')
    if a.out:
        from PIL import Image
        os.makedirs(a.out, exist_ok=True)
        base = os.path.join(a.out, os.path.splitext(os.path.basename(a.sav))[0])
        if a.bin:
            open(base + '.bin', 'wb').write(u)
        for k, name in FLOAT_LAYERS.items():
            lay = float_layer(u, k); lo, hi = stats[name]['min'], stats[name]['max']
            img = Image.new('L', (GRID, GRID))
            img.putdata([int(255 * (x - lo) / (hi - lo + 1e-9)) if abs(x) < 1e30 else 0 for x in lay])
            img.save(f'{base}_{name}.png')
        json.dump({'header': hdr, 'layers': stats}, open(base + '_layers.json', 'w'), indent=1)
        print('written to', a.out)


if __name__ == '__main__':
    main()
