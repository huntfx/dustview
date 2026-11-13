"""Minimal reader for MSVC .pdb files (MSF 7.0): type records from the TPI stream.

Enough to list enums (names + values) and structs/classes (members + byte offsets),
which turns raw `[reg+0x30]` accesses in a disassembly into named fields.

Usage:
  python pdb.py <file.pdb> enum  <substring>      # enums whose name contains substring
  python pdb.py <file.pdb> struct <substring>     # structs/classes/unions (with fields)
  python pdb.py <file.pdb> names                  # all UDT names with sizes
  python pdb.py <file.pdb> globals <substring>    # public/global symbol names (+ section:offset)

Format notes (from the LLVM PDB docs): superblock at 0, block size at +0x20, stream
directory block map at +0x34. Stream 2 = TPI. TPI header 56 bytes, then records
(u16 length, u16 kind, payload). Type indices start at TypeIndexBegin (0x1000).
"""
import struct, sys

class MSF:
    def __init__(self, path):
        d = open(path, 'rb').read()
        assert d[:29] == b'Microsoft C/C++ MSF 7.00\r\n\x1aDS', 'not an MSF 7.0 pdb'
        self.d = d
        bs, fmb, nblocks, dirsz, _, dirmap = struct.unpack_from('<6I', d, 0x20)
        self.bs = bs
        ndirblocks = (dirsz + bs - 1) // bs
        dirblocks = struct.unpack_from('<%dI' % ndirblocks, d, dirmap * bs)
        dirdata = b''.join(d[b * bs:(b + 1) * bs] for b in dirblocks)[:dirsz]
        nstreams = struct.unpack_from('<I', dirdata, 0)[0]
        sizes = struct.unpack_from('<%dI' % nstreams, dirdata, 4)
        off = 4 + 4 * nstreams
        self.streams = []
        for sz in sizes:
            if sz == 0xFFFFFFFF:
                sz = 0
            n = (sz + bs - 1) // bs
            blocks = struct.unpack_from('<%dI' % n, dirdata, off)
            off += 4 * n
            self.streams.append((sz, blocks))

    def stream(self, i):
        sz, blocks = self.streams[i]
        return b''.join(self.d[b * self.bs:(b + 1) * self.bs] for b in blocks)[:sz]

NUMERIC = {0x8000: ('<b', 1), 0x8001: ('<h', 2), 0x8002: ('<H', 2), 0x8003: ('<i', 4),
           0x8004: ('<I', 4), 0x8009: ('<q', 8), 0x800a: ('<Q', 8)}

def numeric(b, o):
    v = struct.unpack_from('<H', b, o)[0]
    if v < 0x8000:
        return v, o + 2
    fmt, n = NUMERIC[v]
    return struct.unpack_from(fmt, b, o + 2)[0], o + 2 + n

def cstr(b, o):
    e = b.index(b'\0', o)
    return b[o:e].decode('latin1'), e + 1

def align4(o):
    # field-list sub-records are padded with LF_PAD bytes (0xF0..0xFF)
    return (o + 3) & ~3

PRIM = {0x03: 'void', 0x10: 'char', 0x20: 'uchar', 0x11: 'short', 0x21: 'ushort',
        0x12: 'long', 0x22: 'ulong', 0x74: 'int', 0x75: 'uint', 0x13: 'int64', 0x23: 'uint64',
        0x40: 'float', 0x41: 'double', 0x30: 'bool', 0x68: 'int8', 0x69: 'uint8',
        0x72: 'int16', 0x73: 'uint16', 0x70: 'char', 0x71: 'wchar'}

class Types:
    def __init__(self, msf):
        t = msf.stream(2)
        hdrsz, tbegin, tend, recbytes = struct.unpack_from('<4I', t, 4)
        self.begin = tbegin
        self.recs = {}
        o = hdrsz
        ti = tbegin
        while o < hdrsz + recbytes:
            ln, kind = struct.unpack_from('<HH', t, o)
            self.recs[ti] = (kind, t[o + 4:o + 2 + ln])
            o += 2 + ln
            ti += 1

    def name(self, ti):
        if ti < self.begin:
            base = PRIM.get(ti & 0xFF, 'prim%x' % ti)
            return base + ('*' if ti & 0xF00 else '')
        kind, p = self.recs.get(ti, (None, b''))
        if kind in (0x1504, 0x1505, 0x1506, 0x1507):
            return self.udt(ti)['name']
        if kind == 0x1002:  # LF_POINTER
            return self.name(struct.unpack_from('<I', p, 0)[0]) + '*'
        if kind == 0x1001:  # LF_MODIFIER
            return self.name(struct.unpack_from('<I', p, 0)[0])
        if kind == 0x1503:  # LF_ARRAY
            el, idx = struct.unpack_from('<II', p, 0)
            n, _ = numeric(p, 8)
            return '%s[%d bytes]' % (self.name(el), n)
        if kind == 0x1205:  # LF_BITFIELD
            ty, ln, pos = struct.unpack_from('<IBB', p, 0)
            return '%s:%d@%d' % (self.name(ty), ln, pos)
        return 'T%x(kind %x)' % (ti, kind or 0)

    def udt(self, ti):
        kind, p = self.recs[ti]
        if kind in (0x1504, 0x1505):  # class / struct
            count, prop, flist, derived, vshape = struct.unpack_from('<HHIII', p, 0)
            size, o = numeric(p, 16)
        elif kind == 0x1506:  # union
            count, prop, flist = struct.unpack_from('<HHI', p, 0)
            size, o = numeric(p, 8)
        elif kind == 0x1507:  # enum
            count, prop, utype, flist = struct.unpack_from('<HHII', p, 0)
            size, o = 0, 12
        nm, _ = cstr(p, o)
        return {'name': nm, 'size': size, 'flist': flist, 'fwd': bool(prop & 0x80), 'kind': kind}

    def fields(self, flist):
        out = []
        if flist not in self.recs:
            return out
        kind, p = self.recs[flist]
        o = 0
        while o < len(p):
            sub = struct.unpack_from('<H', p, o)[0]
            if sub == 0x1502:  # LF_ENUMERATE
                v, o2 = numeric(p, o + 4)
                nm, o2 = cstr(p, o2)
                out.append(('enum', nm, v, None))
            elif sub == 0x150d:  # LF_MEMBER
                ty = struct.unpack_from('<I', p, o + 4)[0]
                off, o2 = numeric(p, o + 8)
                nm, o2 = cstr(p, o2)
                out.append(('member', nm, off, ty))
            elif sub == 0x150e:  # LF_STMEMBER
                ty = struct.unpack_from('<I', p, o + 4)[0]
                nm, o2 = cstr(p, o + 8)
                out.append(('static', nm, None, ty))
            elif sub == 0x1400:  # LF_BCLASS
                ty = struct.unpack_from('<I', p, o + 4)[0]
                off, o2 = numeric(p, o + 8)
                out.append(('base', self.name(ty), off, ty))
            elif sub == 0x1409:  # LF_VFUNCTAB
                o2 = o + 8
                out.append(('vfptr', 'vfptr', 0, None))
            elif sub == 0x1510:  # LF_NESTTYPE
                ty = struct.unpack_from('<I', p, o + 4)[0]
                nm, o2 = cstr(p, o + 8)
            elif sub == 0x1511:  # LF_ONEMETHOD
                attr, ty = struct.unpack_from('<HI', p, o + 2)
                mprop = (attr >> 2) & 7
                o2 = o + 8 + (4 if mprop in (4, 6) else 0)
                nm, o2 = cstr(p, o2)
                out.append(('method', nm, None, ty))
            elif sub == 0x150f:  # LF_METHOD (overloaded)
                cnt, ml = struct.unpack_from('<HI', p, o + 2)
                nm, o2 = cstr(p, o + 8)
                out.append(('method', nm, None, None))
            elif sub == 0x1404:  # LF_INDEX (continuation)
                ty = struct.unpack_from('<I', p, o + 4)[0]
                out += self.fields(ty)
                o2 = o + 8
            elif sub == 0x1401 or sub == 0x1402:  # LF_VBCLASS / LF_IVBCLASS
                o2 = o + 12
                _, o2 = numeric(p, o2)
                _, o2 = numeric(p, o2)
            else:
                out.append(('?', 'unknown subleaf %x' % sub, None, None))
                break
            o = align4(o2)
        return out

    def find(self, sub, kinds):
        seen = set()
        for ti, (kind, p) in self.recs.items():
            if kind in kinds:
                u = self.udt(ti)
                if sub.lower() in u['name'].lower() and not u['fwd'] and u['name'] not in seen:
                    seen.add(u['name'])
                    yield ti, u

def main():
    path, cmd = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else ''
    sys.stdout.reconfigure(encoding='utf-8')
    msf = MSF(path)
    T = Types(msf)
    if cmd == 'enum':
        for ti, u in T.find(arg, (0x1507,)):
            print('enum %s' % u['name'])
            for k, nm, v, _ in T.fields(u['flist']):
                print('  %-60s = %d (0x%x)' % (nm, v, v & 0xFFFFFFFF))
    elif cmd == 'struct':
        for ti, u in T.find(arg, (0x1504, 0x1505, 0x1506)):
            print('%s  (size 0x%x)' % (u['name'], u['size']))
            for k, nm, off, ty in T.fields(u['flist']):
                if k in ('member', 'base'):
                    print('  +0x%04x  %-45s %s' % (off, nm, T.name(ty)))
                elif k == 'static':
                    print('  static  %-45s %s' % (nm, T.name(ty)))
            print()
    elif cmd == 'names':
        for ti, u in T.find(arg, (0x1504, 0x1505, 0x1506, 0x1507)):
            print('0x%-6x %s' % (u['size'], u['name']))

if __name__ == '__main__':
    main()
