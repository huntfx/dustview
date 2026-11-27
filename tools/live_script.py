"""Read the level script's running effects from the running game (read-only).

DustManager [0x1017989C + delta] -> +0x28 sequence manager -> +0x18 array of running script actions,
+0x1C count (delta = dustai's relocation, live_generators.dustai_delta). Each action: vtable, +0x08 class
name, +0x0C Duration, +0x10 _Timer (seconds it has run); class fields from each action's serializer
(vtable slot 10): see findings/water-lava-sim.md "Reading running script effects from the live game".
Used by DustView's compare mode to give the model the same TerraForming brushes and edge waves as the game.

Usage: python live_script.py   (prints the running actions once a second)
"""
import os, struct, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_generators

DUSTMANAGER = 0x1017989C
VTABLES = {0x10160EB8: 'TerraForming', 0x1015F468: 'Wave', 0x10160FD0: 'Tsunami', 0x1015F288: 'Tide'}


class Reader:
    def __init__(self, p, delta=None):
        self.p = p
        self.delta = live_generators.dustai_delta(p) if delta is None else delta
        self.vt = {v + self.delta: n for v, n in VTABLES.items()}

    def _u(self, a):
        return struct.unpack('<I', self.p.read(a, 4))[0]

    def running(self):
        """The running actions this moment: dicts with name, address, duration, timer and the class fields
        (TerraForming in terraform.effect_fields' names, Wave / Tsunami / Tide in sav_sources.effect's)."""
        p = self.p
        G = self._u(DUSTMANAGER + self.delta)
        seq = self._u(G + 0x28)
        arr, n = struct.unpack('<2I', p.read(seq + 0x18, 8))
        if not 0 <= n <= 256:
            return []
        ptrs = struct.unpack('<%dI' % n, p.read(arr, 4 * n)) if n else ()
        out = []
        for a in ptrs:
            raw = p.read(a, 0x58)
            if not raw or len(raw) < 0x58:
                continue
            name = self.vt.get(struct.unpack_from('<I', raw, 0)[0])
            if name is None:
                continue
            dur, timer = struct.unpack_from('<2f', raw, 0x0C)
            d = dict(name=name, address=a, duration=dur, timer=timer)
            u = lambda o: struct.unpack_from('<I', raw, o)[0]
            f = lambda o: struct.unpack_from('<f', raw, o)[0]
            if name == 'TerraForming':
                d.update(col=u(0x1C), row=u(0x20), soft=u(0x24), brush=u(0x28), start_size=u(0x30), end_size=u(0x34),
                         angle=f(0x3C), mode=u(0x48), op=u(0x4C), curve=u(0x54))
            elif name == 'Wave':
                d.update(template=u(0x1C), amplitude=f(0x20), direction=u(0x24), cycle=u(0x28), cycle_duration=f(0x2C))
            elif name == 'Tsunami':
                d.update(template=u(0x1C), amplitude=f(0x20), direction=u(0x24), village_protection=u(0x28))
            elif name == 'Tide':
                d.update(amplitude=f(0x1C), high_duration=f(0x20), low_duration=f(0x24), transition_duration=f(0x28))
            out.append(d)
        return out


def main():
    import memscan
    sys.stdout.reconfigure(encoding='utf-8')
    r = Reader(memscan.Proc())
    while True:
        for d in r.running():
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != 'address'})
        print('-')
        time.sleep(1)


if __name__ == '__main__':
    main()
