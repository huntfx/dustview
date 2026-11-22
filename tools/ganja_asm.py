"""Annotate the disassembly of the vegetation / fire step (SimulatorGanjaIterator::Step2 etc.).

Usage:
  dumpbin -nologo -disasm:nobytes SSE41/GalileoSimulatorWater_9_9.dll > w99.asm   (scratchpad)
  python ganja_asm.py w99.asm Step2 [start_addr] [end_addr]

Replaces ConstParam / per-type table offsets with names. Inside Step2, `edx` = the iterator
(`this`), ConstParam = this+0x2A0 (held in `esi`), the per-type table = ConstParam+0x1A0 =
this+0x440 (layout: findings/vegetation-fire-layers.md, "Per-type table"). Each 16 x u16 block
is read as two 16-byte halves (low / high type bytes after a pshufb by type).
"""
import re
import sys

CONST = {  # offsets from ConstParam (esi)
    0x60: 'DefaultGrowingTime', 0x70: 'FireSpeedProgressBase', 0x80: 'BurnedTime',
    0x90: 'SoftGroundQuantityMin',
    0x3A0: 'M_TypeMask0F', 0x3B0: 'M_Fire40', 0x3C0: 'M_Burned80', 0x3D0: 'M_TypeBurned8F',
    0x3E0: 'M_Flowering30', 0x3F0: 'M_Young10', 0x400: 'M_FireNeigh07', 0x410: 'M_Viability01',
    0x420: 'M_Random0F', 0x430: 'M_Accretion7FFF', 0x440: 'PropagateTable',
}
TYPE_BLOCKS = {
    0x00: 'FireIgnition', 0x20: 'FireDuration', 0x40: 'RandomGrowingTime', 0x60: 'EvolveDuration',
    0x80: 'AdultDuration', 0xA0: 'NextType', 0xC0: 'FlowerStart', 0xE0: 'FlowerStop',
    0x100: 'LifeWaterMin', 0x120: 'LifeWaterMax', 0x140: 'FireAnim', 0x160: 'LifeFireResistWH',
    0x180: 'SpreadWaterMin', 0x1A0: 'SpreadWaterMax', 0x1C0: 'SpreadDeltaMax', 0x1E0: 'Ignition8n',
}


def name_this(off):
    """Name an offset from the iterator (edx)."""
    if off >= 0x440 and off < 0x640:
        rel = off - 0x440
        blk = rel & ~0x1F
        return f'T.{TYPE_BLOCKS[blk]}{"_hi" if rel & 0x10 else "_lo"}'
    if off >= 0x2A0:
        return 'C.' + CONST.get(off - 0x2A0, hex(off - 0x2A0))
    return None


def annotate(line):
    def rep_esi(m):
        off = int(m.group(1), 16)
        return f'[{"C." + CONST[off] if off in CONST else m.group(0)[1:-1]}]'

    def rep_edx(m):
        n = name_this(int(m.group(1), 16))
        return f'[{n}]' if n else m.group(0)

    line = re.sub(r'\[esi\+([0-9A-F]+)h\]', rep_esi, line)
    line = re.sub(r'\[edx\+([0-9A-F]+)h\]', rep_edx, line)
    line = line.replace('xmmword ptr ', '')
    return line


def main():
    path, func = sys.argv[1], sys.argv[2]
    start = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4], 16) if len(sys.argv) > 4 else 1 << 40
    inside = False
    for raw in open(path, encoding='latin-1'):
        if raw and not raw.startswith(' '):
            inside = f'?{func}@SimulatorGanjaIterator' in raw
            continue
        if not inside:
            continue
        m = re.match(r'\s+([0-9A-F]{8}):', raw)
        if m and start <= int(m.group(1), 16) < end:
            print(annotate(raw.rstrip()[2:]))


if __name__ == '__main__':
    main()
