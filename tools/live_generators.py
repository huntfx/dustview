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


def read_modifier_list(p, vec, settle=0.03):
    """Current entries of the modifier list as dicts. Waits (up to `settle` s) for the refill to finish:
    reads until the count has stopped changing and is not 0."""
    import time
    t0, last, n = time.time(), None, 0
    while time.time() - t0 < settle:
        ptr, n, cap = struct.unpack('<3I', p.read(vec, 12))
        if n and n == last:
            break
        last = n
        time.sleep(0.001)
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
