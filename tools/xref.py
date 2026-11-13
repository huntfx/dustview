"""Find where strings in a 32-bit PE are referenced from code.

Usage:
  python xref.py <exe> <asm> <string> [<string> ...] [--ctx N]

<asm> is the output of:  dumpbin -nologo -disasm:nobytes <exe> > exe.asm
For each matching string (ASCII or UTF-16LE, substring match) prints the string's
virtual address and every disassembly line that mentions it, with N lines of context.
"""
import re, struct, sys, bisect

def sections(d):
    pe = struct.unpack_from('<I', d, 0x3c)[0]
    nsec = struct.unpack_from('<H', d, pe + 6)[0]
    optsz = struct.unpack_from('<H', d, pe + 20)[0]
    base = struct.unpack_from('<I', d, pe + 24 + 28)[0]
    out = []
    for i in range(nsec):
        o = pe + 24 + optsz + i * 40
        name = d[o:o + 8].rstrip(b'\0').decode()
        vs, va, rs, ra = struct.unpack_from('<IIII', d, o + 8)
        out.append((name, va, vs, ra, rs))
    return base, out

def off2va(base, secs, off):
    for name, va, vs, ra, rs in secs:
        if ra <= off < ra + rs:
            return base + va + off - ra
    return None

def va2off(base, secs, v):
    r = v - base
    for name, va, vs, ra, rs in secs:
        if va <= r < va + max(vs, rs):
            return r - va + ra
    return None

def main():
    args = sys.argv[1:]
    ctx = 0
    if '--ctx' in args:
        i = args.index('--ctx'); ctx = int(args[i + 1]); del args[i:i + 2]
    exe, asm, *needles = args
    sys.stdout.reconfigure(encoding='utf-8')
    d = open(exe, 'rb').read()
    base, secs = sections(d)
    lines = open(asm, encoding='latin-1').read().splitlines()
    for n in needles:
        hits = []
        for enc in ('ascii', 'utf-16le'):
            pat = n.encode(enc)
            for m in re.finditer(re.escape(pat), d):
                # walk back to string start
                s = m.start()
                step = 1 if enc == 'ascii' else 2
                while s - step >= 0 and d[s - step:s] != b'\0' * step and 32 <= d[s - step] < 127:
                    s -= step
                end = d.index(b'\0' * step, m.start())
                text = d[s:end].decode(enc, 'replace')
                hits.append((off2va(base, secs, s), text, enc))
        for va, text, enc in sorted(set(hits)):
            if va is None: continue
            key = '%Xh' % va
            pat = re.compile(r'(?<![0-9A-F])0*' + key)
            refs = [i for i, l in enumerate(lines) if key in l and pat.search(l)]
            print(f'== {text!r} ({enc}) @ {va:08X}: {len(refs)} refs')
            for i in refs:
                for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                    print(('>> ' if j == i else '   ') + lines[j])
                if ctx: print('   ...')

if __name__ == '__main__':
    main()
