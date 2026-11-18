"""List the fluid generators (springs, lava vents, drains) of the running From Dust level (read-only).

Finds dustai's load address (it is relocated: e.g. +0x1660000 from its preferred base 0x10000000),
then every object whose vtable is the generator class (preferred VA 0x1015A4D8), and prints its
runtime fields (offsets from dustai's field registration at 0x10066080 and the update code):
  +0x1E8 m_size (diameter), +0x1EC m_quantity, +0x1F0 m_temperature (editor), +0x1F4 m_sediment %,
  +0x1F8 m_quantityPerFrame (rate), +0x1FC m_buryTresholhd, +0x200 m_geiser (a reference, not a bool),
  +0x208 Mode (0 off, 1 in game, 2 in game + sandbox, 3 sphere power only), +0x20C switched on,
  +0x1D8 height mode (base entity: 0 stored height +0x1DC, only lowered by erosion; 1 surface + offset;
  2 follows the surface, so never buried), +0x214 fade-in length (ticks), +0x218 fade-in counter, +0x21C activity (1 = free, 0 = buried).
The object with mode 0 and rate 0.667 / size 5 that is in no map's list is the class default.

Usage: .venv/Scripts/python tools/live_generators.py
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, xref

DLL = r'D:\Games\Steam\steamapps\common\From Dust\AInative\bin\dll\dustai.win32.f.dll'
GENERATOR_VTABLE = 0x1015A4D8
MODES = {0: 'off', 1: 'in game', 2: 'in game + sandbox', 3: 'sphere power only'}


def dustai_delta(p):
    """Relocation of dustai in the running game (actual address - preferred address)."""
    d = open(DLL, 'rb').read()
    base, secs = xref.sections(d)
    marker = b'DustAi::GameSceneManager\0'
    va = xref.off2va(base, secs, d.find(marker))
    for h in p.find(marker):
        delta = h - va
        if p.find(struct.pack('<I', GENERATOR_VTABLE + delta)):
            return delta
    raise SystemExit('dustai not found in the game process')


def generators(p, delta=None):
    delta = dustai_delta(p) if delta is None else delta
    out = []
    for a in p.find(struct.pack('<I', GENERATOR_VTABLE + delta)):
        g = read_generator(p, a)
        if g is not None:
            out.append(g)
    return out


def read_generator(p, a):
    """Runtime fields of the generator object at address a (None if it does not look like one)."""
    raw = p.read(a, 0x220)
    if raw is None or len(raw) < 0x220:
        return None
    size, qty, temp, sed = struct.unpack_from('<4i', raw, 0x1E8)
    rate, bury = struct.unpack_from('<2f', raw, 0x1F8)
    geiser, _, mode, on = struct.unpack_from('<4I', raw, 0x200)
    fade_len, fade, activity = struct.unpack_from('<3f', raw, 0x214)
    height_mode = struct.unpack_from('<I', raw, 0x1D8)[0]
    stored_z = struct.unpack_from('<f', raw, 0x1DC)[0]
    if not (0 <= size <= 128 and mode <= 3 and on <= 1 and abs(rate) < 1000):
        return None                                 # stray match, not an object
    return (dict(address=hex(a), size=size, radius=size / 2, quantity=qty, temperature_editor=temp,
                    temperature=temp / 8, sediment=sed / 100, rate=rate, bury_threshold=bury,
                    geiser=hex(geiser), mode=mode, on=on, fade=fade, fade_length=fade_len, activity=activity,
                    height_mode=height_mode, stored_z=stored_z))


# ---------------------------------------------------------------- the simulator's per-tick modifier list
# Every tick the game clears and refills a list of 36-byte entries that the simulation applies:
#   u32 type (0 = fluid source, 1 = other, probably terrain), f32 col, f32 row, f32 radius, f32 ?,
#   f32 amount (fluid per cell this tick, after burying / fade-in / Jellify; negative = drain),
#   f32 ? (+-1 on type 1), f32 sediment fraction, f32 temperature (internal; > 0 = lava)
# held by a vector {u32 data pointer, u32 count, u32 capacity}. Found 2026-11-13 on Map 10: the count
# goes 0 -> full once per tick and stays full for most of it.
ENTRY = 36


def _valid_entry(raw, o):
    t = struct.unpack_from('<I', raw, o)[0]
    col, row, rad = struct.unpack_from('<3f', raw, o + 4)
    return t in (0, 1) and 0 <= col < 512 and 0 <= row < 512 and col == int(col) and row == int(row) and 0 < rad < 64


_HINT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'live', 'modifier_list_address.txt')


def valid_modifier_list(p, vec):
    """Does `vec` look like the per-tick modifier vector {ptr, count, capacity}? Count <= capacity <= 4096 and
    the entries in use (or, with count 0 as while paused, the first one in the buffer) are valid."""
    try:
        hdr = p.read(vec, 12)
        if hdr is None:
            return False
        ptr, n, cap = struct.unpack('<3I', hdr)
        if not (0 < cap <= 4096 and n <= cap and ptr):
            return False
        raw = p.read(ptr, ENTRY * max(n, 1))
        return raw is not None and all(_valid_entry(raw, ENTRY * k) for k in range(max(n, 1)))
    except Exception:
        return False


def known_modifier_list(p):
    """The address remembered from an earlier search (it stayed at 0x7ED70124 across levels and game restarts
    on 2026-11-18), if it still checks out; else None."""
    try:
        for line in open(_HINT_FILE, encoding='utf-8'):
            vec = int(line.split()[0], 16)
            if valid_modifier_list(p, vec):
                return vec
    except (OSError, ValueError, IndexError):
        pass
    return None


def remember_modifier_list(vec):
    try:
        os.makedirs(os.path.dirname(_HINT_FILE), exist_ok=True)
        open(_HINT_FILE, 'w', encoding='utf-8').write('%08X  per-tick modifier list vector (found by find_modifier_list)\n' % vec)
    except OSError:
        pass


def find_modifier_list(p, gens):
    """Address of the vector {ptr, count, capacity} (or None). Locates one entry by a generator running at
    full strength (radius, 0, rate), walks back to the array start, then finds the pointer to it."""
    for g in gens:
        if g['mode'] == 0 or g['activity'] < 1:
            continue
        for a in p.find(struct.pack('<3f', g['radius'], 0.0, g['rate'])):
            start = a - 12                                   # entry: type, col, row, radius, 0, rate
            raw = p.read(start - ENTRY * 64, ENTRY * 65)
            if raw is None or not _valid_entry(raw, ENTRY * 64):
                continue
            k = 64
            while k > 0 and _valid_entry(raw, ENTRY * (k - 1)):
                k -= 1
            first = start - ENTRY * (64 - k)
            for v in p.find(struct.pack('<I', first)):
                n, cap = struct.unpack('<2I', p.read(v + 4, 8))
                if n <= cap <= 4096 and cap > 0:
                    return v
    return None


def read_modifier_list(p, vec, settle=0.03, last_tick=False):
    """Current entries of the modifier list as dicts. Waits (up to `settle` s) for the refill to finish:
    reads until the count has stopped changing and is not 0.
    last_tick: if the count is 0 (the game empties the list after each tick, so it stays 0 while the game is
    paused), read the entries still in the buffer instead: the last tick's list (contiguous valid entries up
    to the capacity; a longer older list could leave extra stale entries behind it). Use only when paused."""
    import time
    t0, last = time.time(), None
    ptr, n, cap = struct.unpack('<3I', p.read(vec, 12))
    while time.time() - t0 < settle:
        ptr, n, cap = struct.unpack('<3I', p.read(vec, 12))
        if n and n == last:
            break
        last = n
        time.sleep(0.001)
    if not n and last_tick and 0 < cap <= 4096:
        raw = p.read(ptr, ENTRY * cap)
        while raw is not None and n < cap and _valid_entry(raw, ENTRY * n):
            n += 1
    if not n or n > 4096:
        return []
    raw = p.read(ptr, ENTRY * n)
    if raw is None:
        return []
    out = []
    for k in range(n):
        t = struct.unpack_from('<I', raw, ENTRY * k)[0]
        f = struct.unpack_from('<8f', raw, ENTRY * k + 4)
        out.append(dict(type=t, col=f[0], row=f[1], radius=f[2], f3=f[3], amount=f[4], f5=f[5], sediment=f[6], temperature=f[7]))
    return out


# ---------------------------------------------------------------- totems (DustAi::Statue / DustAi::Village)
STATUE_VTABLES = (0x1015B470, 0x1015B460)     # +0, +4 of a Statue object (preferred VAs)
VILLAGE_VTABLES = (0x1015BEE0, 0x1015BED0)    # +0, +4 of a Village object


def totems(p, delta=None):
    """Live totems: statue objects (+0x54 handle ID, +0x144/+0x148 grid cell, +0x1B0 power index) and
    their Village objects (+0x238 statue handle; +0x2C4 = 1 on the one totem without a built village on
    Map 12, so built = (+0x2C4 == 0): inferred from one level). The class-default object is skipped."""
    delta = dustai_delta(p) if delta is None else delta
    out = {}
    for a in p.find(struct.pack('<2I', STATUE_VTABLES[0] + delta, STATUE_VTABLES[1] + delta)):
        raw = p.read(a, 0x228)
        if raw is None:
            continue
        hid = struct.unpack_from('<I', raw, 0x54)[0]
        col, row = struct.unpack_from('<2I', raw, 0x144)
        if hid in (0xFFFFFFFE, 0xFFFFFFFF) or not (0 < col < 512 and 0 < row < 512):
            continue
        out[hid] = dict(id='%X' % hid, col=col, row=row, power=struct.unpack_from('<I', raw, 0x1B0)[0], built=False)
    for a in p.find(struct.pack('<2I', VILLAGE_VTABLES[0] + delta, VILLAGE_VTABLES[1] + delta)):
        raw = p.read(a, 0x2C8)
        if raw is None:
            continue
        sid = struct.unpack_from('<I', raw, 0x238)[0]
        if sid in out:
            out[sid]['built'] = struct.unpack_from('<I', raw, 0x2C4)[0] == 0
    return list(out.values())



# ---------------------------------------------------------------- village repel songs
SONG_VTABLES = {'water': 0x1016C1CC, 'lava': 0x1016C214}     # dustai objects for entities 0x20400009 / 0x2040000A


def songs(p, delta=None):
    """Live repel-song objects: kind ('water' / 'lava'), grid cell (+0x7C / +0x80), size ([+0x74]+0x1EC)
    and the RepulseFluid entry the game pushes for it every tick (water radius trunc(size x 2/3),
    lava size x 2/3; mode -1 water / +1 lava; amount 1). Works without the per-tick modifier list (which
    needs a running generator to be found)."""
    delta = dustai_delta(p) if delta is None else delta
    out = []
    for kind, vt in SONG_VTABLES.items():
        for a in p.find(struct.pack('<I', vt + delta)):
            raw = p.read(a, 0x8C)
            if raw is None:
                continue
            col, row = struct.unpack_from('<2I', raw, 0x7C)
            if not (0 < col < 512 and 0 < row < 512):
                continue                                      # code bytes that mention the vtable
            sz = p.read(struct.unpack_from('<I', raw, 0x74)[0] + 0x1EC, 4)
            size = struct.unpack('<f', sz)[0] if sz else float('nan')
            if not (1 < size < 200):
                continue
            radius = float(int(size * 2 / 3)) if kind == 'water' else size * 2 / 3
            out.append(dict(kind=kind, col=col, row=row, size=size, type=1, radius=radius, amount=1.0,
                            f5=-1.0 if kind == 'water' else 1.0, f3=0.0, sediment=0.0, temperature=0.0))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = memscan.Proc()
    gens = generators(p)
    print('%d generator objects' % len(gens))
    for g in gens:
        print('  %s radius %4.1f rate %7.4f sediment %.2f temperature %6.2f | mode %d (%s) on %d | fade-in %g/%g | activity %.3f | bury threshold %g | height mode %d stored z %.2f | geiser %s' % (
            g['address'], g['radius'], g['rate'], g['sediment'], g['temperature'], g['mode'], MODES.get(g['mode'], '?'),
            g['on'], g['fade'], g['fade_length'], g['activity'], g['bury_threshold'], g['height_mode'], g['stored_z'], g['geiser']))


if __name__ == '__main__':
    main()
