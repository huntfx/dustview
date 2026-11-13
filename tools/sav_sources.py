"""List a From Dust map's water springs / lava vents and its level script, straight from the .sav.

No game needed. Everything is parsed by structure: nested blocks of
  u32 size (counting the size word itself), 12 34 56 78, content, 87 65 43 21
with big-endian numbers and UTF-16LE strings (u32 length in characters first).

Fluid generators (springs, lava vents, drains): a manager block tagged 0x10020002 holds a list of
entries tagged 0x20020002. Each entry has a handle block (0, class, index, serial; the class number
is assigned per map: 41 on Maps 9/10/11, 59 on Map 3), a world
position block (x, y, z) and ends with 11 settings words:
  u32 ?, u32 0, u32 height mode (0 = stored height that only follows erosion down, buriable;
  1 = surface + offset; 2 = rides on the surface, can never be buried), f32 stored height (z),
  u32 Mode, u32 size (diameter in cells),
  u32 temperature (editor units = 8 x simulator units; 0 = water), u32 sediment %,
  f32 rate (m_quantityPerFrame: fluid per cell per tick; negative = drain),
  f32 f10 (100 or 0; unknown: 100 on Map 2's buriable spring, 0 on Map 11's spring that could not be
  buried), u32 flag11 (0/1; unknown). Neither is the runtime bury threshold (5 for every generator, from
  the class default in FluidGenerator_0.ai) nor m_geiser (a reference); see tools/live_generators.py.
World -> grid: col = 256 + x * 2/3, row = 256 + y * 2/3 (truncated).
Mode: 0 off, 1 in game, 2 in game + sandbox, 3 sphere power only.

Level script (dustai GameSceneManager): a rule list = f32 _Timer, f32 Duration, u32 _IsPlaying,
u32 Loops, block of rules. A rule = u32 ObjectId, f32 Start, f32 Duration, u32 _State,
f32 _VerificationTime, block of conditions, block of effects, u32 TimelineIcon,
u32 DisplayInTimeline, f32 TimelineEffectDuration. Conditions and effects start with their class
name. Effects: f32 Duration, f32 _Timer, u32 UseShakeCam, then class fields. FluidGenerator: u32 Type
(0 = spawn a new generator at PosX/PosY, 1 = change the target generator), u32 ID length, UTF-16
target ID (padded to 4 bytes), u32 PosX, u32 PosY (FFFFFFFF = keep), u32 Temperature (editor),
u32 SedimentsQuantity %, f32 Quantity (rate per tick), u32 Radius (really the diameter). A Sequence effect is
itself a rule list with its own clock (Duration = loop length when Loops = 1).
Script target IDs (e.g. "13C000A4") are handles: serial << 19 | index << 10 | class << 2; for
FluidGenerator targets, class + index pick the generator (index = its slot in the list).
Timing rules (from dustai, see findings/water-lava-sim.md): a rule arms once its list's clock
passes Start, fires as soon as all its conditions hold while clock < Start + Duration (Duration < 0
= no deadline), and fires only once. A FluidGenerator effect switches the target on (mode 1) with
the given settings; if its Duration >= 0 it switches the target off again after Duration seconds.

Usage: python sav_sources.py <map name or .sav path> [--raw]   (--raw also dumps unknown effects)
"""
import os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav

INSTALL = r'D:\Games\Steam\steamapps\common\From Dust'
BEG = bytes.fromhex('12345678')
END = bytes.fromhex('87654321')
MODES = {0: 'off', 1: 'in game', 2: 'in game + sandbox', 3: 'sphere power only'}
HEIGHT_MODES = {0: 'fixed, buriable', 1: 'surface + offset', 2: 'on surface, never buried'}
TYPETEST = {-2: '<=', -1: '<', 0: '==', 1: '>', 2: '>='}


def u32(u, o): return struct.unpack_from('>I', u, o)[0]
def i32(u, o): return struct.unpack_from('>i', u, o)[0]
def f32(u, o): return struct.unpack_from('>f', u, o)[0]


def block(u, p):
    """Block starting at p (its size word). Returns (content_start, content_end, next_offset)."""
    s = u32(u, p)
    if s < 12 or u[p + 4:p + 8] != BEG or u[p + s - 4:p + s] != END:
        raise ValueError('no block at 0x%x' % p)
    return p + 8, p + s - 4, p + s


def counted(u, p):
    """Block holding u32 count + count blocks. Returns ([(content_start, content_end)], next_offset)."""
    c, e, nxt = block(u, p)
    items, q = [], c + 4
    for _ in range(u32(u, c)):
        ic, ie, q = block(u, q)
        items.append((ic, ie))
    return items, nxt


def utf16(u, p):
    """u32 length (characters) + UTF-16LE text, padded to a multiple of 4 bytes (odd lengths: 2 bytes)."""
    n = u32(u, p)
    return u[p + 4:p + 4 + 2 * n].decode('utf-16-le'), p + 4 + 2 * n + (2 * n) % 4


def handle_id(cls, index, serial):
    return '%X' % (serial << 19 | index << 10 | cls << 2)


def handle_index(ident):
    v = int(ident, 16)
    return (v >> 2) & 0xFF, (v >> 10) & 0x1FF


# ---------------------------------------------------------------- generators
MANAGER = re.compile(re.escape(struct.pack('>2I', 0x10020002, 28) + BEG + bytes(4)) + b'....' +
                     re.escape(struct.pack('>I', 511)), re.S)


def generators(u):
    """Return the map's fluid generators in list order (script IDs refer to them by class + index)."""
    for m in MANAGER.finditer(u):
        try:
            _, _, p = block(u, m.start() + 4)     # manager handle block
            c, e, _ = block(u, p)                  # u32 capacity?, u32 1, then the counted list
            items = counted(u, c + 8)[0]
        except (ValueError, struct.error):
            continue
        if items and all(b - a == 252 and u32(u, a + 8) == 0x20020002 for a, b in items):
            return _generator_entries(u, items)
    return []


def _generator_entries(u, items):
    out = []
    for oc, oe in items:
        ic, ie, _ = block(u, oc)
        if u32(u, ic) != 0x20020002:
            raise ValueError('unexpected generator entry at 0x%x' % ic)
        hc, _, q = block(u, ic + 4)
        _, cls, index, serial = struct.unpack_from('>4I', u, hc)
        pc, _, _ = block(u, q)
        x, y, z = struct.unpack_from('>3f', u, pc)
        s = ie - 44
        w0, w1, w2 = struct.unpack_from('>3I', u, s)
        height = f32(u, s + 12)
        mode, size, temp, sed = struct.unpack_from('>4I', u, s + 16)
        rate, f10 = struct.unpack_from('>2f', u, s + 32)
        flag11 = u32(u, s + 40)
        out.append(dict(index=index, cls=cls, id=handle_id(cls, index, serial), world=(x, y, z),
                        col=int(256 + x * 2 / 3), row=int(256 + y * 2 / 3), height=height,
                        mode=mode, size=size, radius=size / 2, temperature_editor=temp, temperature=temp / 8,
                        sediment=sed / 100, rate=rate, f10=f10, flag11=flag11,
                        kind='drain' if rate < 0 else 'lava' if temp else 'water', height_mode=w2,
                        words=(w0, w1, w2), offset=s))
    return out


def sources(u):
    """Backwards-compatible name: the generator list."""
    return generators(u)


# ---------------------------------------------------------------- script
def find_root(u):
    """The level's top-level rule list: the first rule-list block in the file tail whose rules parse."""
    p = u.find(struct.pack('>2I', 0x10020002, 28) + BEG)   # script comes after the generators
    p = max(p, len(u) // 2)
    while True:
        p = u.find(BEG, p + 1)
        if p < 0:
            return None
        try:
            c, e, _ = block(u, p - 4)
            lst, nxt = rulelist(u, c)
            if nxt == e and lst['rules']:
                return lst
        except (ValueError, struct.error, UnicodeDecodeError):
            pass


def rulelist(u, p):
    timer, duration = struct.unpack_from('>2f', u, p)
    playing, loops = struct.unpack_from('>2I', u, p + 8)
    if playing > 1 or loops > 1:
        raise ValueError('not a rule list')
    items, nxt = counted(u, p + 16)
    return dict(timer=timer, duration=duration, playing=playing, loops=loops,
                rules=[rule(u, a, b) for a, b in items]), nxt


def rule(u, a, b):
    rid = u32(u, a)
    start, duration = struct.unpack_from('>2f', u, a + 4)
    state = u32(u, a + 12)
    vtime = f32(u, a + 16)
    conds, q = counted(u, a + 20)
    effs, q = counted(u, q)
    icon, display = struct.unpack_from('>2I', u, q)
    tl = f32(u, q + 8)
    if q + 12 != b:
        raise ValueError('rule size mismatch at 0x%x' % a)
    return dict(id='%08X' % rid, start=start, duration=duration, state=state, verification_time=vtime,
                conditions=[condition(u, x, y) for x, y in conds], effects=[effect(u, x, y) for x, y in effs],
                timeline_icon=icon, display_in_timeline=display, timeline_duration=tl, offset=a)


def condition(u, a, b):
    name, q = utf16(u, a)
    d = dict(name=name, raw=u[q:b].hex())
    if name == 'CreatedVillagesCount' and b - q >= 16:
        d.update(villages=u32(u, q + 8), test=TYPETEST.get(i32(u, q + 12), i32(u, q + 12)))
    elif name == 'RandomTime' and b - q >= 28:
        d.update(min_time=f32(u, q + 20), max_time=f32(u, q + 24))
    return d


def effect(u, a, b):
    name, q = utf16(u, a)
    if name == 'Sequence':
        lst, nxt = rulelist(u, q)
        return dict(name=name, **lst)
    d = dict(name=name, duration=f32(u, q), timer=f32(u, q + 4), raw=u[q + 12:b].hex())
    if name == 'FluidGenerator':
        n = u32(u, q + 16)
        ident = u[q + 20:q + 20 + 2 * n].decode('utf-16-le')
        r = b - 24                      # settings are the last 6 words (odd-length IDs are padded)
        col, row, temp, sed = struct.unpack_from('>4I', u, r)
        d.update(target=ident, type=u32(u, q + 12), col=None if col == 0xFFFFFFFF else col, row=None if row == 0xFFFFFFFF else row,
                 temperature_editor=temp, temperature=temp / 8, sediment=sed / 100, rate=f32(u, r + 16),
                 size=u32(u, r + 20), radius=u32(u, r + 20) / 2)
    elif name in ('Tsunami', 'RandomTsunami') and b - q >= 28:
        d.update(template=u32(u, q + 12), direction=u32(u, q + 16), amplitude=f32(u, q + 20), village_protection=u32(u, q + 24))
    elif name == 'Wave' and b - q >= 32:
        d.update(template=u32(u, q + 12), direction=u32(u, q + 16), amplitude=f32(u, q + 20), cycle=u32(u, q + 24), cycle_duration=f32(u, q + 28))
    elif name == 'Tide' and b - q >= 28:
        d.update(amplitude=f32(u, q + 12), high_duration=f32(u, q + 16), low_duration=f32(u, q + 20), transition_duration=f32(u, q + 24))
    elif b - q >= 20 and u32(u, q + 16) == 8:            # effects that target one object by ID
        d['target'] = u[q + 20:q + 36].decode('utf-16-le', 'replace')
    return d


def events(u):
    """All FluidGenerator effects in the script, flattened (with the rule context they sit in)."""
    root = find_root(u)
    out = []

    def walk(lst, path):
        for r in lst['rules']:
            for e in r['effects']:
                if e['name'] == 'Sequence':
                    walk(e, path + [r])
                elif e['name'] == 'FluidGenerator':
                    out.append(dict(e, rule=r, path=path))
    if root:
        walk(root, [])
    return out


# ---------------------------------------------------------------- printing
def fmt_t(t):
    return '-' if t < 0 else ('%gs' % t)


def describe_conditions(r):
    out = []
    for c in r['conditions']:
        if 'villages' in c:
            out.append('villages built %s %d' % (c['test'], c['villages']))
        elif 'min_time' in c:
            out.append('random wait %g-%gs' % (c['min_time'], c['max_time']))
        else:
            out.append(c['name'])
    return out


def describe_effect(e, gens):
    if e['name'] == 'FluidGenerator':
        cls, idx = handle_index(e['target'])
        g = next((x for x in gens if x['index'] == idx and x['cls'] == cls), None)
        if e['type'] == 0:
            who = '(new, spawned by the script)'
        else:
            who = ('#%d (%s at col %d row %d)' % (idx, g['kind'], g['col'], g['row'])) if g else e['target'] + ' (not in list)'
        where = '' if e['col'] is None and e['row'] is None else ' move to col %s row %s,' % (
            '-' if e['col'] is None else e['col'], '-' if e['row'] is None else e['row'])
        what = ('lava %g' % e['temperature']) if e['temperature_editor'] else 'water'
        if e['sediment']:
            what += ' sediment %g' % e['sediment']
        until = 'for %gs, then off' % e['duration'] if e['duration'] >= 0 else 'permanently'
        return 'generator %s ON:%s rate %g radius %g %s, %s' % (who, where, e['rate'], e['radius'], what, until)
    EDGES = {2: 'col 36 (low x)', 4: 'row 476 (high y)', 6: 'col 476 (high x)', 8: 'row 36 (low y)'}   # dustai jump tables
    if e['name'] in ('Tsunami', 'RandomTsunami'):
        return '%s from the %s edge, amplitude %g, for %gs%s' % (e['name'], EDGES.get(e['direction'], e['direction']), e['amplitude'], e['duration'],
                                                                ', villages protected' if e['village_protection'] else '')
    if e['name'] == 'Wave':
        return 'Wave on the %s edge, amplitude %g, cycle %gs%s' % (EDGES.get(e['direction'], e['direction']), e['amplitude'], e['cycle_duration'],
                                                                  '' if e['duration'] < 0 else ' (for %gs)' % e['duration'])
    if e['name'] == 'Tide':
        return 'Tide: sea level drops by %g over %gs, stays low %gs, rises over %gs, stays high %gs, repeating' % (
            e['amplitude'], e['transition_duration'], e['low_duration'], e['transition_duration'], e['high_duration'])
    s = e['name']
    if 'target' in e:
        s += ' ' + e['target']
    d = e.get('duration', -1)
    if d == 0 or 1e-3 <= d < 1e5:          # other classes don't all start with Duration; skip junk values
        s += ' (duration %gs)' % d
    return s


def print_rulelist(lst, gens, indent, raw):
    for r in sorted(lst['rules'], key=lambda r: r['start']):
        when = 'at %gs' % r['start'] if r['start'] > 0 else 'from start'
        window = '' if r['duration'] < 0 else ' (only until %gs)' % (r['start'] + r['duration'])
        conds = describe_conditions(r)
        cond = (' once ' + ' and '.join(conds)) if conds else ''
        tl = ' [timeline icon %d, %gs]' % (r['timeline_icon'], r['timeline_duration']) if r['display_in_timeline'] else ''
        print('%s- rule %s %s%s%s:%s' % (indent, r['id'], when, cond, window, tl))
        for e in r['effects']:
            if e['name'] == 'Sequence':
                loop = 'loops every %gs' % e['duration'] if e['loops'] else (
                    'runs %gs' % e['duration'] if e['duration'] >= 0 else 'runs until its rules are done')
                print('%s    Sequence (own clock from when it starts; %s):' % (indent, loop))
                print_rulelist(e, gens, indent + '      ', raw)
            else:
                print('%s    %s%s' % (indent, describe_effect(e, gens), ('  raw ' + e['raw']) if raw and e['name'] != 'FluidGenerator' else ''))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    raw = '--raw' in sys.argv
    arg = args[0]
    path = arg if arg.lower().endswith('.sav') else os.path.join(INSTALL, 'DUST', 'DustSave', 'Ingame', arg + '.sav')
    _, u = sav.read_sav(path)
    gens = generators(u)
    print('%d fluid generators:' % len(gens))
    for g in gens:
        print('  #%-2d %-5s col %3d row %3d radius %4.1f rate %7.4f sediment %.2f temperature %6.2f | mode %d (%s) | height mode %d (%s) | f10 %g flag11 %d | id %s | words %s' % (
            g['index'], g['kind'], g['col'], g['row'], g['radius'], g['rate'], g['sediment'], g['temperature'],
            g['mode'], MODES.get(g['mode'], '?'), g['height_mode'], HEIGHT_MODES.get(g['height_mode'], '?'), g['f10'], g['flag11'], g['id'], g['words']))
    root = find_root(u)
    if not root:
        print('no level script found')
        return
    print('level script (%d rules; times are seconds of game time):' % len(root['rules']))
    print_rulelist(root, gens, '  ', raw)


if __name__ == '__main__':
    main()
