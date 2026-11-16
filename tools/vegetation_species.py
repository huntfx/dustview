"""Read the vegetation species settings from the game data (no game needed).

BinPC/Wor_BD00993D.lin.bf (_GLOBAL.wor) holds SpecieVegetationA_0.ai ... SpecieVegetationN_0.ai (one per
vegetation type: A = type 1 ... N = type 14, from m_childType 0x201000NN) and VegetationInterface_0.ai.
They are LyN reflection records: `u32 nameHash, u32 type, u32 size, value`; a struct is
`hash(struct name), 0x420, size | 0x80000000, u32 fieldCount, fields...`. Field and struct names are the
strings of AInative/bin/dll/dustai.win32.f.dll, hashed with galileo.lyn_hash. Types: 0x405 int / handle,
0x410 float. Each species file has 4 `DustAi::SpecieVegetal::s_vegetationParams` sets (list
m_vegetationParameters) and 4 `s_slowFireParams`; which set a level uses: findings/vegetation-fire-layers.md.

Usage:  .venv/Scripts/python tools/vegetation_species.py   -> prints the tables, writes
        data/vegetation/species.tsv (one row per species and set) and data/vegetation/fields.txt
        (every named field of every file, in file order)
"""
import os, re, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf, galileo as g

ARCHIVE = os.path.join(g.INSTALL, 'BinPC', 'Wor_BD00993D.lin.bf')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = None


def names():
    """hash -> name for every identifier-like string in dustai.win32.f.dll."""
    global _NAMES
    if _NAMES is None:
        dll = open(os.path.join(g.INSTALL, 'AInative', 'bin', 'dll', 'dustai.win32.f.dll'), 'rb').read()
        _NAMES = {g.lyn_hash(m.decode()): m.decode() for m in set(re.findall(rb'[A-Za-z_][A-Za-z0-9_:]{3,}', dll))}
        _NAMES.update(EXTRA_NAMES)
    return _NAMES


# names not stored as text anywhere in the install, recovered by hashing word combinations
EXTRA_NAMES = {0x50C064F5: 'm_lifeGroundType'}


def entry(name):
    d = open(ARCHIVE, 'rb').read()
    e = [x for x in bf.read_index(ARCHIVE) if x['name'] == name][0]
    return d[e['offset']:e['offset'] + e['size']]


def _value(blob, o, t, s):
    if s == 4 and t == 0x410:
        return struct.unpack_from('<f', blob, o)[0]
    if s == 4:
        return struct.unpack_from('<I', blob, o)[0]
    return blob[o:o + s].hex()


def structs(blob, struct_name):
    """Every serialized `struct_name` record in blob, as {field: value} dicts in file order."""
    nm = names()
    key = struct.pack('<I', g.lyn_hash(struct_name))
    out = []
    for o in [i for i in range(len(blob) - 16) if blob[i:i + 4] == key]:
        _, typ, size, cnt = struct.unpack_from('<IIII', blob, o)
        if typ != 0x420:
            continue
        o += 16
        rec = {}
        for _ in range(cnt):
            h, t, s = struct.unpack_from('<III', blob, o)
            if s & 0x80000000:                     # nested struct / array: u32 count, then s & 0x7FFFFFFF bytes
                rec[nm.get(h, '%08x' % h).replace('m_', '', 1)] = blob[o + 12:o + 16 + (s & 0x7FFFFFFF)].hex()
                o += 16 + (s & 0x7FFFFFFF)
            else:
                rec[nm.get(h, '%08x' % h).replace('m_', '', 1)] = _value(blob, o + 12, t, s)
                o += 12 + s
        out.append(rec)
    return out


def named_fields(blob):
    """(offset, name, value) for every 4-byte field whose hash is a known name (scan; for overview)."""
    nm = names()
    out = []
    for o in range(len(blob) - 16):
        h, t, s = struct.unpack_from('<III', blob, o)
        if h in nm and s == 4 and t in (0x405, 0x410, 0x428, 0x429, 0x433):
            out.append((o, nm[h], _value(blob, o + 12, t, 4)))
    return out


def species():
    """{type number: {'letter', 'veg': [4 sets], 'slow_fire': [4 sets], 'extra': {...}}}"""
    out = {}
    for k in range(14):
        letter = chr(ord('A') + k)
        blob = entry('SpecieVegetation%s_0.ai' % letter)
        extra = {n.replace('m_', '', 1): v for _, n, v in named_fields(blob)
                 if n in ('m_childType', 'm_spreadingVegetationType', 'm_isVillageVegetation', 'm_absorptionResistanceTime',
                          'm_humansBlockingFactor', 'm_snappedVerticalOffset', 'm_nbVegetationTypes', 'm_erosionBlocking',
                          'm_biomass', 'm_constantUpdate')}
        typ = extra.get('childType', 0) & 0xFF
        out[typ] = dict(letter=letter, veg=structs(blob, 'DustAi::SpecieVegetal::s_vegetationParams'),
                        slow_fire=structs(blob, 'DustAi::SpecieVegetal::s_slowFireParams'), extra=extra)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sp = species()
    os.makedirs(os.path.join(HERE, 'data', 'vegetation'), exist_ok=True)
    keys = []
    for v in sp.values():
        for s in v['veg']:
            keys += [k for k in s if k not in keys]
    xkeys = []
    for v in sp.values():
        xkeys += [k for k in v['extra'] if k not in xkeys]
    rows = ['\t'.join(['type', 'letter', 'set', 'slowFire_ignition', 'slowFire_duration'] + keys + xkeys)]
    for typ in sorted(sp):
        v = sp[typ]
        for i, s in enumerate(v['veg']):
            sf = v['slow_fire'][i] if i < len(v['slow_fire']) else {}
            fmt = lambda x: ('0x%08X' % x if isinstance(x, int) and x > 0xFFFFFF else ('%g' % x if isinstance(x, float) else str(x)))
            rows.append('\t'.join([str(typ), v['letter'], str(i), fmt(sf.get('fireIgnition', '')), fmt(sf.get('fireDuration', ''))]
                                  + [fmt(s.get(k, '')) for k in keys] + [fmt(v['extra'].get(k, '')) for k in xkeys]))
    open(os.path.join(HERE, 'data', 'vegetation', 'species.tsv'), 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
    with open(os.path.join(HERE, 'data', 'vegetation', 'fields.txt'), 'w', encoding='utf-8') as f:
        for name in ['VegetationInterface_0.ai'] + ['SpecieVegetation%s_0.ai' % chr(ord('A') + k) for k in range(14)]:
            f.write('== %s\n' % name)
            for o, n, val in named_fields(entry(name)):
                if n not in ('m_type', 'm_soundBankID', 'm_active'):
                    f.write('  +%05x %-40s %s\n' % (o, n, val))
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
