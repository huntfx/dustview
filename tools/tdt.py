"""From Dust / LyN `.tdt` texture decoder (DXT1 / DXT5) -> numpy RGBA / PNG.

Usage:
  python tdt.py info <file.tdt>
  python tdt.py png <file.tdt> <out.png>          first mip level of every slice, slices side by side
  python tdt.py extract <Tex_xxx.lin.bf> <Name>  <out.png>   (entry `<Name>.PC.tdt`)

Layout (worked out from the water / foam textures, 2026-11-16; all little-endian):
  0x00  32-byte big-file entry header (u32 size-after-header twice, ..., "TDT_" at +20)
  0x20  u32 2
  0x24  u8 format, u8 mip count, u16 ?          (format 0x09 = DXT1, 0x0B = DXT5)
  0x28  u8 format again, u8 slice count, u16 width, u16 height
  0x2E  pixel data: per slice, mip levels from largest to smallest (4x4 blocks, row-major); a few
        spare bytes at the end.
Checked: Foam_D (DXT1 512x512, 10 mips), WaterSeqNM128 (DXT5 2048x128 = 16 ripple normal-map frames
of 128x128 side by side, 8 mips). 06Water_NMseq_ASeq_D / 08Water_NMASeq_D (DXT5, "16 slices of
128x128", 1 mip) do not decode as plain slices (a repeating 4-frame pattern of garbage): different
layout, not worked out.
"""
import os, struct, sys
import numpy as np

FORMATS = {0x09: ('DXT1', 8), 0x0B: ('DXT5', 16)}


def header(d):
    fmt, mips = d[0x24], d[0x25]
    fmt2, slices = d[0x28], d[0x29]
    w, h = struct.unpack_from('<2H', d, 0x2A)
    return dict(format=fmt, mips=mips, format2=fmt2, slices=slices, width=w, height=h)


def level_bytes(w, h, bpb):
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb


def _rgb565(c):
    c = c.astype(np.uint32)
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], -1).astype(np.int32)


def _colour_blocks(blk, dxt1):
    """blk: (n, 8) uint8 colour part -> (n, 16, 3) uint8."""
    c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
    c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1) | (not dxt1)
    four = np.broadcast_to(np.asarray(four)[..., None], p0.shape)
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)                         # (n, 4, 3)
    idx = blk[:, 4:8].astype(np.uint32)
    bits = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
    sel = (bits[:, None] >> (2 * np.arange(16))) & 3            # (n, 16)
    return np.take_along_axis(pal, sel[..., None].astype(np.int64).repeat(3, -1), 1).astype(np.uint8)


def _alpha_blocks(blk):
    """blk: (n, 8) uint8 DXT5 alpha part -> (n, 16) uint8."""
    a0, a1 = blk[:, 0].astype(np.int32), blk[:, 1].astype(np.int32)
    k = np.arange(8)
    eight = np.stack([a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)], 1)
    six = np.stack([a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [a0 * 0, a0 * 0 + 255], 1)
    pal = np.where((a0 > a1)[:, None], eight, six)
    bits = np.zeros(len(blk), np.uint64)
    for i in range(6):
        bits |= blk[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    sel = (bits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)
    return np.take_along_axis(pal, sel.astype(np.int64), 1).astype(np.uint8)


def decode_level(buf, w, h, fmt):
    name, bpb = FORMATS[fmt]
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    blk = np.frombuffer(buf, np.uint8, bw * bh * bpb).reshape(-1, bpb)
    if name == 'DXT1':
        rgb = _colour_blocks(blk, True)
        c0 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        c1 = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        idx = blk[:, 4:8].astype(np.uint32)
        bits = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
        sel = (bits[:, None] >> (2 * np.arange(16))) & 3
        a = np.where((c0 <= c1)[:, None] & (sel == 3), 0, 255).astype(np.uint8)
    else:
        a = _alpha_blocks(blk[:, :8])
        rgb = _colour_blocks(blk[:, 8:], False)
    px = np.concatenate([rgb, a[..., None]], -1).reshape(bh, bw, 4, 4, 4)
    return px.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)[:h, :w]


def decode(d):
    """All slices' first mip level: (slices, h, w, 4) uint8 RGBA."""
    hd = header(d)
    fmt, slices, w, h, mips = hd['format2'], max(1, hd['slices']), hd['width'], hd['height'], max(1, hd['mips'])
    _, bpb = FORMATS[fmt]
    per_slice = sum(level_bytes(max(1, w >> m), max(1, h >> m), bpb) for m in range(mips))
    for start in (0x2E, 0x2C, 0x2A):                    # pick the data offset that makes the size fit
        if len(d) - start >= per_slice * slices:
            break
    out = []
    for s in range(slices):
        o = start + s * per_slice
        out.append(decode_level(d[o:o + level_bytes(w, h, bpb)], w, h, fmt))
    return np.stack(out), dict(hd, data_offset=start, per_slice=per_slice, spare=len(d) - start - per_slice * slices)


def main():
    from PIL import Image
    cmd = sys.argv[1]
    if cmd == 'extract':
        import bf
        d = bf.read_entry(sys.argv[2], sys.argv[3] + '.PC.tdt')
        if d is None:
            sys.exit('not found')
        path_out = sys.argv[4]
    else:
        d = open(sys.argv[2], 'rb').read()
        path_out = sys.argv[3] if len(sys.argv) > 3 else None
    if cmd == 'info':
        print(header(d), len(d))
        return
    px, hd = decode(d)
    print(hd)
    Image.fromarray(np.concatenate(list(px), 1)).save(path_out)
    print('wrote', path_out)


if __name__ == '__main__':
    main()
