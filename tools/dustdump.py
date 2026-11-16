"""Read the texture dumps written by the hook's F3 (dustdump_<pass>_s<stage>.bin in the game folder).

File: u32 width, height, D3DFORMAT, pitch; then the rows (pitch bytes each; DXT: one row per 4 pixel rows).
Usage: python dustdump.py list | python dustdump.py png <name> <out.png>
"""
import glob, os, struct, sys
import numpy as np

GAME = r'D:\Games\Steam\steamapps\common\From Dust'
FMT = {21: ('A8R8G8B8', 4, np.uint8), 22: ('X8R8G8B8', 4, np.uint8), 34: ('G16R16', 4, np.uint16), 112: ('G16R16F', 4, np.float16),
       113: ('A16B16G16R16F', 8, np.float16), 114: ('R32F', 4, np.float32), 116: ('A32B32G32R32F', 16, np.float32),
       64: ('fmt64', 4, np.uint8)}


def load(name):
    """name like 'water_s0' -> (array (h, w, channels) in memory order, format name). For A8R8G8B8 the
    memory order is B, G, R, A, i.e. shader channels z, y, x, w."""
    d = open(os.path.join(GAME, 'dustdump_%s.bin' % name), 'rb').read()
    w, h, f, pitch = struct.unpack_from('<4I', d, 0)
    if f in (0x31545844, 0x35545844):                     # DXT1 / DXT5: decode the first mip level
        import tdt
        fmt, bpb = (0x09, 8) if f == 0x31545844 else (0x0B, 16)
        rows = (h + 3) // 4
        data = b''.join(d[16 + y * pitch:16 + y * pitch + ((w + 3) // 4) * bpb] for y in range(rows))
        return tdt.decode_level(data, w, h, fmt)[..., [2, 1, 0, 3]], 'DXT1' if fmt == 0x09 else 'DXT5'   # as B, G, R, A
    if f not in FMT:
        return None, 'format %d (%dx%d)' % (f, w, h)
    fname, bpp, dt = FMT[f]
    rows = np.frombuffer(d, np.uint8, h * pitch, 16).reshape(h, pitch)[:, :w * bpp]
    a = np.ascontiguousarray(rows).view(dt).reshape(h, w, -1)
    return a, fname


def shader_rgba(name):
    """(h, w, 4) float array in the shader's x, y, z, w order (0..1 for 8-bit formats)."""
    a, f = load(name)
    if a is None:
        return None, f
    if f in ('A8R8G8B8', 'X8R8G8B8', 'DXT1', 'DXT5'):
        return a[..., [2, 1, 0, 3]].astype(np.float32) / 255, f
    if f in ('G16R16F', 'G16R16'):
        a = a.astype(np.float32) / (65535.0 if f == 'G16R16' else 1)
        return np.concatenate([a[..., [0, 1]], np.zeros(a.shape[:2] + (2,), np.float32)], -1), f   # R=x, G=y
    return a.astype(np.float32), f


def main():
    if sys.argv[1] == 'list':
        for p in sorted(glob.glob(os.path.join(GAME, 'dustdump_*.bin'))):
            d = open(p, 'rb').read(16)
            w, h, f, pitch = struct.unpack('<4I', d)
            print('%-28s %5dx%-5d %s' % (os.path.basename(p)[9:-4], w, h, FMT.get(f, (str(f),))[0]))
    elif sys.argv[1] == 'png':
        from PIL import Image
        a, f = shader_rgba(sys.argv[2])
        ch = [a[..., k] for k in range(4)]
        ims = []
        for c in ch:
            lo, hi = np.percentile(c, [1, 99])
            ims.append(np.clip((c - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8))
        Image.fromarray(np.concatenate(ims, 1)).save(sys.argv[3])
        print(f, 'channels x y z w side by side ->', sys.argv[3])


if __name__ == '__main__':
    main()
