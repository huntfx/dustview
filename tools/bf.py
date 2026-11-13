"""From Dust / LyN `.lin.bf` big-file archive reader.

Usage:
  python bf.py list <file.bf>                 list entries (name, size, offset)
  python bf.py stats <dir with .bf files>     count entries/bytes per extension across archives
  python bf.py extract <file.bf> <name> <out> extract one entry by name

Layout (from the community bftool, github.com/DrChat/FromDustHook, checked against this install):
  0x00  "ABE\\0" magic
  0x18  u32 header size (0x1440 = 5184)
  0x2C  u32 file count
  headerSize + 12: file records, 200 bytes each:
        +0   name[64]  (NUL-terminated, rest 0x7F)
        +84  i32 id    (starts at -1)
        +88  u32 size  (whole entry, incl. its 32-byte inner header)
        +108 u32 offset (from start of archive)
  Entry data is not compressed. Most entries start with a 32-byte header
  (u32 size-after-header, ..., ext[4] at +20); .bik/.tga do not.
"""
import collections, os, struct, sys

REC = 200


def read_index(path):
    with open(path, 'rb') as f:
        hdr = f.read(0x1440)
        if hdr[:4] != b'ABE\0':
            raise ValueError(f'{path}: bad magic {hdr[:4]!r}')
        hsize = struct.unpack_from('<I', hdr, 0x18)[0]
        count = struct.unpack_from('<I', hdr, 0x2C)[0]
        f.seek(hsize + 12)
        raw = f.read(count * REC)
    out = []
    for i in range(count):
        r = raw[i * REC:(i + 1) * REC]
        name = r[:64].split(b'\0')[0].decode('latin-1')
        fid, size = struct.unpack_from('<iI', r, 84)
        off = struct.unpack_from('<I', r, 108)[0]
        out.append({'name': name, 'id': fid, 'size': size, 'offset': off})
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cmd, *args = sys.argv[1:]
    if cmd == 'list':
        for e in read_index(args[0]):
            print(f"{e['offset']:10d} {e['size']:10d}  {e['name']}")
    elif cmd == 'stats':
        cnt, byt = collections.Counter(), collections.Counter()
        files = sorted(f for f in os.listdir(args[0]) if f.endswith('.bf'))
        total = 0
        for fn in files:
            idx = read_index(os.path.join(args[0], fn))
            total += len(idx)
            for e in idx:
                ext = os.path.splitext(e['name'])[1].lower() or '(none)'
                cnt[ext] += 1; byt[ext] += e['size']
        print(f'{len(files)} archives, {total} entries')
        for ext, n in cnt.most_common():
            print(f'{ext:8s} {n:6d} entries {byt[ext] / 1e6:9.1f} MB')
    elif cmd == 'extract':
        path, name, out = args
        for e in read_index(path):
            if e['name'] == name:
                with open(path, 'rb') as f:
                    f.seek(e['offset']); data = f.read(e['size'])
                open(out, 'wb').write(data)
                print(f'wrote {len(data)} bytes to {out}')
                return
        sys.exit(f'{name} not found')


if __name__ == '__main__':
    main()
