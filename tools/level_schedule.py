"""Run a From Dust level's script over time: which springs / lava vents / drains are on, with what
settings, and the tide's border sea level, tick by tick. Uses the decoded rules (findings/water-lava-sim.md,
"Where per-map sources and their timing are defined" and "Tides, waves and tsunamis").

What is modelled:
- rule lists (the level's top list and every Sequence, with their own clocks and looping); a rule arms
  once its clock passes Start, fires when all its conditions hold while clock < Start + Duration
  (Duration < 0: no deadline), once per run of its list;
- conditions: CreatedVillagesCount (needs the player's village count over time: `villages`),
  RandomTime (random wait MinTime..MaxTime, seeded); every other condition is False unless given in
  `conditions` (name -> function(t) or bool);
- effects: FluidGenerator (Type 1 changes generator #index, Type 0 spawns a new one; switches it on
  with the settings, moves it if a cell is given; after Duration >= 0 seconds switches it off), Tide
  (the border sea level cycle), Sequence; Wave / Tsunami / everything else are logged as events only;
- generators: mode 0 off, 1/2 on, 3 sphere-power only (off here); a generator switched on fades in over
  100 ticks; stored height and height mode are carried for the bury rule (dustsim.bury_activity); a
  script move of a height-mode-0 generator resets its stored height to the ground there (if `ground`
  is passed to tick()).

Usage as a module:
    sch = LevelSchedule('ALPHA_Map10_FireAndWater', villages=lambda t: 1 if t > 120 else 0)
    for each tick: sch.tick(dt=1/31, ground=G['T']); sources = sch.sources(); border = sch.border_level
Command line (prints a timeline of state changes):
    python level_schedule.py ALPHA_Map11_Void [seconds=600] [--villages "0:0,200:1,900:3"]
"""
import os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav, sav_sources as ss

INGAME = os.path.join(ss.INSTALL, 'DUST', 'DustSave', 'Ingame')
TYPETEST = {'<=': lambda a, b: a <= b, '<': lambda a, b: a < b, '==': lambda a, b: a == b,
            '>': lambda a, b: a > b, '>=': lambda a, b: a >= b}
FADE_TICKS = 100


class Generator:
    def __init__(self, g):
        self.index = g['index']
        self.cls = g.get('cls')
        self.col, self.row = g['col'], g['row']
        self.size, self.rate = g['size'], g['rate']
        self.temperature, self.sediment = g['temperature'], g['sediment']
        self.mode = g['mode']
        self.z = g.get('height', 0.0)
        self.height_mode = g.get('height_mode', 2)
        self.fade = FADE_TICKS if self.mode in (1, 2) else 0

    def on(self):
        return self.mode in (1, 2)

    def as_source(self):
        f = min(1.0, self.fade / FADE_TICKS)
        return dict(col=self.col, row=self.row, radius=self.size / 2, rate=self.rate * f, sediment=self.sediment,
                    temperature=self.temperature, z=self.z, height_mode=self.height_mode, bury_threshold=5.0,
                    index=self.index)


class RuleList:
    def __init__(self, data, playing=False):
        self.data = data
        self.duration, self.loops = data['duration'], bool(data['loops'])
        self.t = 0.0
        self.playing = playing
        self.rules = [dict(r=r, state=0, conds=None) for r in data['rules']]

    def restart(self):
        self.t = 0.0
        self.playing = True
        for x in self.rules:
            x['state'], x['conds'] = 0, None


class LevelSchedule:
    def __init__(self, level, villages=None, conditions=None, seed=1, border_level=None):
        path = level if level.lower().endswith('.sav') else os.path.join(INGAME, level + '.sav')
        _, u = sav.read_sav(path)
        self.gens = {g['index']: Generator(g) for g in ss.generators(u)}
        self.spawned = []                                 # generators spawned by Type-0 effects
        self.root = RuleList(ss.find_root(u), playing=True)
        self.villages = villages or (lambda t: 0)
        self.conditions = conditions or {}
        self.rng = random.Random(seed)
        self.time = 0.0
        self.active = []                                  # running effects: dict(effect, timer, target)
        self.events = []                                  # (time, text)
        if border_level is None:                          # the map's border sea level (ground + water at (5, 5))
            import numpy as np
            lay = lambda k: np.frombuffer(u[0x80 + k * 0x100000:0x80 + (k + 1) * 0x100000], np.float32).reshape(512, 512)
            border_level = float(lay(0)[5, 5] + lay(1)[5, 5])
        self.border_level = border_level
        self.tide = None
        self.lists = [self.root]

    # ---------------------------------------------------------------- conditions
    def _cond_start(self, c):
        st = dict(c=c, t=0.0, done=False)
        if c['name'] == 'RandomTime':
            lo, hi = c.get('min_time', 0.0), c.get('max_time', 0.0)
            st['wait'] = self.rng.uniform(lo, hi) if hi > lo else lo
        return st

    def _cond_true(self, st, dt, t):
        c = st['c']
        name = c['name']
        if name == 'CreatedVillagesCount':
            return TYPETEST.get(c.get('test'), lambda a, b: False)(self.villages(t), c.get('villages', 0))
        if name == 'RandomTime':
            st['t'] += dt
            return st['t'] > st['wait']
        v = self.conditions.get(name, False)
        return bool(v(t) if callable(v) else v)

    # ---------------------------------------------------------------- effects
    def _log(self, text):
        self.events.append((self.time, text))

    def _target(self, e):
        if e.get('type') == 0:                            # spawn a new generator
            g = Generator(dict(index=-1 - len(self.spawned), col=e['col'] or 0, row=e['row'] or 0, size=e['size'],
                               rate=0.0, temperature=0.0, sediment=0.0, mode=0, height_mode=2))
            self.spawned.append(g)
            return g
        cls, idx = ss.handle_index(e['target'])
        return self.gens.get(idx)

    def _start_effect(self, e, ground):
        name = e['name']
        if name == 'Sequence':
            lst = RuleList(e)
            lst.restart()
            self.lists.append(lst)
            self._log('Sequence starts (%s)' % ('loops every %gs' % e['duration'] if e['loops'] else 'once'))
            return
        if name == 'FluidGenerator':
            g = self._target(e)
            if g is None:
                self._log('FluidGenerator target %s not in the level' % e['target'])
                return
            was_on = g.on()
            g.mode = 1
            g.temperature, g.sediment, g.rate, g.size = e['temperature'], e['sediment'], e['rate'], e['size']
            if e['col'] is not None and e['row'] is not None and 33 <= e['col'] < 479 and 33 <= e['row'] < 479:
                g.col, g.row = e['col'], e['row']
                if g.height_mode == 0 and ground is not None:
                    g.z = float(ground[g.row, g.col])
            if not was_on:
                g.fade = 0
            self._log('generator #%d ON at (%d,%d): rate %g radius %g %s%s' % (
                g.index, g.col, g.row, g.rate, g.size / 2, ('lava %g' % g.temperature) if g.temperature else 'water',
                '' if e['duration'] < 0 else ', off after %gs' % e['duration']))
            if e['duration'] >= 0:
                self.active.append(dict(e=e, timer=0.0, gen=g))
            return
        if name == 'Tide':
            h0 = self.border_level if self.border_level is not None else 0.0
            self.tide = dict(h0=h0, amp=e['amplitude'], high=e['high_duration'], low=e['low_duration'],
                             trans=e['transition_duration'], state=1, timer=0.0,
                             speed=-e['amplitude'] / e['transition_duration'])
            self._log('Tide starts: -%g over %gs, low %gs, rise %gs, high %gs' % (e['amplitude'], e['transition_duration'], e['low_duration'],
                                                                                  e['transition_duration'], e['high_duration']))
            return
        self._log('%s (not simulated)%s' % (ss.describe_effect(e, []), ''))

    def _tick_effects(self, dt):
        keep = []
        for a in self.active:
            a['timer'] += dt
            if a['timer'] >= a['e']['duration']:
                a['gen'].mode = 0
                self._log('generator #%d OFF (effect ended)' % a['gen'].index)
            else:
                keep.append(a)
        self.active = keep

    def _tick_tide(self, dt):
        td = self.tide
        if td is None:
            return
        lvl = self.border_level
        if td['state'] == 1:
            lvl += td['speed'] * dt
            low = max(0.0, td['h0'] - td['amp'])
            if low > lvl:
                lvl, td['state'], td['timer'] = low, 2, 0.0
        elif td['state'] == 2:
            td['timer'] += dt
            if td['timer'] > td['low']:
                td['timer'], td['speed'], td['state'] = 0.0, td['amp'] / td['trans'], 3
        elif td['state'] == 3:
            lvl += td['speed'] * dt
            if lvl > td['h0']:
                lvl, td['state'], td['timer'] = td['h0'], 4, 0.0
        elif td['state'] == 4:
            td['timer'] += dt
            if td['timer'] > td['high']:
                td['timer'], td['speed'], td['state'] = 0.0, -td['amp'] / td['trans'], 1
        self.border_level = max(lvl, 0.0)

    # ---------------------------------------------------------------- main loop
    def _tick_list(self, lst, dt, ground):
        if not lst.playing:
            return
        if lst.duration >= 0 and lst.t >= lst.duration:
            if lst.loops:
                lst.restart()
            else:
                lst.playing = False
                return
        t = lst.t
        for x in lst.rules:
            r = x['r']
            if x['state'] == 0:
                if t > r['start']:
                    x['state'] = 1
                    x['conds'] = [self._cond_start(c) for c in r['conditions']]
            elif x['state'] == 1:
                if t < r['start'] + r['duration'] or r['duration'] < 0:
                    if all(self._cond_true(st, dt, self.time) for st in x['conds']):
                        x['state'] = 3
                        for e in r['effects']:
                            self._start_effect(e, ground)
                else:
                    x['state'] = 2
        lst.t += dt

    def tick(self, dt=1 / 31, ground=None):
        for lst in list(self.lists):
            self._tick_list(lst, dt, ground)
        self.lists = [l for l in self.lists if l.playing or l is self.root]
        self._tick_effects(dt)
        self._tick_tide(dt)
        for g in list(self.gens.values()) + self.spawned:
            if g.on() and g.fade < FADE_TICKS:
                g.fade += 1
        self.time += dt

    def write_back(self, sources):
        """Copy what the model changed (stored height lowered by erosion) back into the generators."""
        by = {g.index: g for g in list(self.gens.values()) + self.spawned}
        for s in sources:
            g = by.get(s.get('index'))
            if g is not None and 'z' in s:
                g.z = s['z']

    def sources(self):
        return [g.as_source() for g in list(self.gens.values()) + self.spawned if g.on() and g.rate != 0]


def parse_villages(spec):
    pts = sorted((float(a), int(b)) for a, b in (p.split(':') for p in spec.split(',')))
    return lambda t: max([v for s, v in pts if s <= t] or [0])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    level, secs = args[0], float(args[1]) if len(args) > 1 else 600
    vil = parse_villages(sys.argv[sys.argv.index('--villages') + 1]) if '--villages' in sys.argv else (lambda t: 0)
    sch = LevelSchedule(level, villages=vil)
    n = int(secs * 31)
    last_tide = None
    for k in range(n):
        sch.tick()
        if sch.tide is not None and sch.tide['state'] != last_tide:
            last_tide = sch.tide['state']
            sch._log('tide state %d, border level %.2f' % (last_tide, sch.border_level))
    for t, text in sch.events:
        print('%7.1fs  %s' % (t, text))
    print('at %gs: %d sources on' % (secs, len(sch.sources())))
    for s in sch.sources():
        print('   #%d (%d,%d) rate %+.3f radius %g %s' % (s['index'], s['col'], s['row'], s['rate'], s['radius'], 'lava %g' % s['temperature'] if s['temperature'] else 'water'))


if __name__ == '__main__':
    main()
