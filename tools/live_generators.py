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
        raw = p.read(a, 0x220)
        size, qty, temp, sed = struct.unpack_from('<4i', raw, 0x1E8)
        rate, bury = struct.unpack_from('<2f', raw, 0x1F8)
        geiser, _, mode, on = struct.unpack_from('<4I', raw, 0x200)
        fade_len, fade, activity = struct.unpack_from('<3f', raw, 0x214)
        height_mode = struct.unpack_from('<I', raw, 0x1D8)[0]
        stored_z = struct.unpack_from('<f', raw, 0x1DC)[0]
        if not (0 <= size <= 128 and mode <= 3 and on <= 1 and abs(rate) < 1000):
            continue                                    # stray match, not an object
        out.append(dict(address=hex(a), size=size, radius=size / 2, quantity=qty, temperature_editor=temp,
                        temperature=temp / 8, sediment=sed / 100, rate=rate, bury_threshold=bury,
                        geiser=hex(geiser), mode=mode, on=on, fade=fade, fade_length=fade_len, activity=activity,
                        height_mode=height_mode, stored_z=stored_z))
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
