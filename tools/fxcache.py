"""From Dust compiled shader cache (`Cache/fxlist_PC.fxl`): list and disassemble the D3D9 shaders.

Usage:
  python fxcache.py list [regex]        index, offset, size, vs/ps, constant/sampler names (filter by regex)
  python fxcache.py dis <index> [...]   disassembly (D3DDisassemble from Windows' D3DCompiler_47.dll)

The cache holds 373 compiled shaders (vs_3_0 / ps_3_0, "Microsoft (R) HLSL Shader Compiler 9.29"),
stored back to back with small records between them; each starts with the version token
(00 03 FE FF = vs_3_0, 00 03 FF FF = ps_3_0) followed by a CTAB comment (the constant table with names)
and ends with the end token FF FF 00 00. `FxDescription_PC.fxd` / `keyList_PC.kl` (names of effects and
their define keys) are not decoded.
"""
import ctypes, os, re, sys

CACHE = r'D:\Games\Steam\steamapps\common\From Dust\Cache\fxlist_PC.fxl'
SKIP_NAMES = ('Microsoft', 'HLSL', 'Shader', 'Compiler', 'CTAB', 'ps_', 'vs_')


def shaders(path=CACHE):
    d = open(path, 'rb').read()
    out = []
    for i, m in enumerate(re.finditer(rb'\x00\x03[\xfe\xff]\xff\xfe\xff..CTAB', d, re.S)):
        s = m.start()
        e = s
        while True:                                    # end token on a 4-byte boundary
            e = d.find(b'\xff\xff\x00\x00', e)
            if (e - s) % 4 == 0:
                break
            e += 1
        blob = d[s:e + 4]
        names = sorted(set(x.decode() for x in re.findall(rb'[A-Za-z_][A-Za-z_0-9]{2,}', blob[:4096]))
                       - set(), key=str)
        names = [n for n in names if not n.startswith(SKIP_NAMES)]
        out.append(dict(index=i, offset=s, size=len(blob), kind='ps' if blob[2] == 0xFF else 'vs', names=names, blob=blob))
    return out


def disassemble(blob):
    dll = ctypes.WinDLL('D3DCompiler_47.dll')
    f = dll.D3DDisassemble
    f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    f.restype = ctypes.c_long
    buf = ctypes.create_string_buffer(blob, len(blob))
    out = ctypes.c_void_p()
    hr = f(buf, len(blob), 0, None, ctypes.byref(out))
    if hr != 0:
        raise OSError('D3DDisassemble failed: 0x%08X' % (hr & 0xFFFFFFFF))
    # ID3DBlob: vtable[3] = GetBufferPointer, vtable[4] = GetBufferSize, vtable[2] = Release
    vt = ctypes.cast(ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vt[3])(out)
    size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vt[4])(out)
    text = ctypes.string_at(ptr, size).decode('latin-1').rstrip('\0')
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vt[2])(out)
    return text


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cmd, *args = sys.argv[1:]
    sh = shaders()
    if cmd == 'list':
        pat = re.compile(args[0], re.I) if args else None
        for s in sh:
            if pat is None or any(pat.search(n) for n in s['names']):
                print('%3d %7x %5d %s  %s' % (s['index'], s['offset'], s['size'], s['kind'], ' '.join(s['names'])))
    elif cmd == 'dis':
        for a in args:
            s = sh[int(a)]
            print('// ===== shader %d (%s, offset 0x%x, %d bytes)' % (s['index'], s['kind'], s['offset'], s['size']))
            print(disassemble(s['blob']))


if __name__ == '__main__':
    main()
