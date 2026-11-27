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


STATUE_MANAGER = re.compile(re.escape(struct.pack('>2I', 0x10200007, 28) + BEG + bytes(4)) + b'....' +
                            re.escape(struct.pack('>I', 511)), re.S)


def statues(u):
    """The level's totems (dustai DustAi::Statue; manager tag 0x10200007, entries 0x20200007, 552 bytes):
    handle (class, index, serial: the ID the script's statue conditions use, e.g. "110804AC"), world
    position -> grid cell, and the words after the 5 slot blocks: u32 1026, u32 `power`, u32 ?, f32
    height (meaning of `power` inferred: it differs per statue, and each totem grants a power)."""
    m = STATUE_MANAGER.search(u)
    if not m:
        return []
    _, _, p = block(u, m.start() + 4)
    items = None
    for k in range(p, p + 0x800, 4):                        # the list follows a run of per-manager words
        try:
            c, e, _ = block(u, k)
            cand = counted(u, c + 8)[0]
        except (ValueError, struct.error):
            continue
        if cand and all(u32(u, block(u, a)[0]) == 0x20200007 for a, b in cand):
            items = cand
            break
    out = []
    for oc, oe in items or []:
        ic, ie, _ = block(u, oc)
        hc, _, q = block(u, ic + 4)
        _, cls, index, serial = struct.unpack_from('>4I', u, hc)
        pc, _, q = block(u, q)
        x, y, z = struct.unpack_from('>3f', u, pc)
        for _ in range(4):                                  # rotation/scale-like blocks and two handles
            _, _, q = block(u, q)
        q += 4                                              # u32 0
        _, _, q = block(u, q)                               # the 5 slots
        _, _, q = block(u, q)                               # [0]
        w1026, power, w3 = struct.unpack_from('>3I', u, q)
        height = f32(u, q + 12)
        out.append(dict(index=index, cls=cls, id=handle_id(cls, index, serial), world=(x, y, z),
                        col=int(256 + x * 2 / 3), row=int(256 + y * 2 / 3), height=height, power=power,
                        w1026=w1026, w3=w3))
    return out



# ---------------------------------------------------------------- villages and their repel songs (progress saves)
# Level files have none of these; a player's progress save (Documents\Dust\DustSave\profile_NN\ingame# MAP_xx.SAV, xx = story level - 1) has one village entry per built totem and one entry per learned song.
VILLAGE_TAG, REPEL_WATER_TAG, REPEL_LAVA_TAG = 0x20800001, 0x20400009, 0x2040000A
PROGRESS_DIR = os.path.expanduser(r'~\Documents\Dust\DustSave')


FIRE_TREE_TAG = b'\x28\x00\x00\x02'


def fire_trees(u):
    """Fire trees (Map 6, planted by the player) in a save, decoded 2026-11-27 from profile_01 MAP_05.SAV and
    checked against the same trees in the running game after loading it. Entity type 0x28000002, preceded by
    the record sizes (0x104 / 0xF8 for state 3, 0x174 / 0x168 for state 1). From the type word: +0x10 handle
    (class 53, index, serial), +0x28 world position x, y, z (grid: col = 256 + 2x/3, row = 256 + 2y/3), +0x40
    facing, +0xD4 state (3 = cycling, live_generators.fire_trees; 1 = other, its record carries what looks like
    a destination and a movement vector; such trees were seen to move a few cells in the game), +0xD8 the
    tree's clock (s), state 3: +0xE4 when the next active phase starts. Idle / active times, size and rate are
    not in the record (class settings: 40 s, 20 s, 10, 5 in the running game)."""
    out = []
    i = u.find(FIRE_TREE_TAG)
    while i >= 0:
        sizes = (u32(u, i - 0x10), u32(u, i - 8)) if i >= 0x10 else None
        if sizes in ((0x104, 0xF8), (0x174, 0x168)):
            x, y, z = f32(u, i + 0x28), f32(u, i + 0x2C), f32(u, i + 0x30)
            state = u32(u, i + 0xD4)
            t = dict(id=handle_id(u32(u, i + 0x10), u32(u, i + 0x14), u32(u, i + 0x18)),
                     col=int(256 + 2 * x / 3), row=int(256 + 2 * y / 3), x=x, y=y, z=z, state=state,
                     clock=f32(u, i + 0xD8), idle=40.0, active_time=20.0, size=10.0, rate=5.0)
            t['next'] = f32(u, i + 0xE4) if sizes[0] == 0x104 else 25.0
            out.append(t)
        i = u.find(FIRE_TREE_TAG, i + 1)
    return out


WATER_TREE_TAG = b'\x28\x00\x00\x03'
# class settings (live, Story 7 Waves, 2026-11-27; not in the save records), names as live_generators.WATER_TREE_FIELDS
WATER_TREE_SETTINGS = dict(max_quantity=1500.0, generation_speed=5.0, absorption_radius=10.0, absorption_cell_min=0.5,
                           absorption_speed=40.0, absorption_factor=0.9, rejection_radius=5.0, rejection_speed=200.0,
                           quantity_to_blow_up=2000.0, blow_up_rejection_speed=300.0, danger_radius=15.0,
                           # DustTree / DustEntity settings (BinPC/Wor_15104E37.lin.bf FiguierBarrique_0.ai, 2026-11-27)
                           time_to_activate_seed=10.0, time_to_grow_tree=15.0, explosion_duration=5.0, nb_seeds=1,
                           nb_seeds_random=0, diffusion_radius_min=3.0, diffusion_radius_max=4.0,
                           water_threshold_max=20.0)


def water_trees(u):
    """Water trees (DustAi::Figuier) in a save or level file, decoded 2026-11-27 from profile_01 MAP_06.SAV
    against the same 25 trees in the running game (same handles, cells, stores). Entity type 0x28000003, preceded
    by the record sizes (0x108 / 0xFC rooted, 0x17C / 0x170 seed). Same header as fire_trees: +0x10 handle,
    +0x28 world position, +0xD4 state (3 = rooted, absorbing; 1 = seed), +0xD8 clock (s); rooted: +0xE8 the
    water stored in the tree (live +0x39C). Class settings are not saved (WATER_TREE_SETTINGS, the same values
    as FiguierBarrique_0.ai and the running game)."""
    out = []
    i = u.find(WATER_TREE_TAG)
    while i >= 0:
        sizes = (u32(u, i - 0x10), u32(u, i - 8)) if i >= 0x10 else None
        if sizes in ((0x108, 0xFC), (0x17C, 0x170)):
            x, y, z = f32(u, i + 0x28), f32(u, i + 0x2C), f32(u, i + 0x30)
            state = u32(u, i + 0xD4)
            t = dict(id=handle_id(u32(u, i + 0x10), u32(u, i + 0x14), u32(u, i + 0x18)),
                     col=int(256 + 2 * x / 3), row=int(256 + 2 * y / 3), x=x, y=y, z=z,
                     state='absorbing' if state == 3 else 'seed' if state == 1 else str(state),
                     clock=f32(u, i + 0xD8), stored=f32(u, i + 0xE8) if sizes[0] == 0x108 else 0.0, danger=False)
            t.update(WATER_TREE_SETTINGS)
            out.append(t)
        i = u.find(WATER_TREE_TAG, i + 1)
    return out


BANG_TREE_TAG = b'\x28\x00\x00\x04'
# Class settings of the explosive tree (DustAi::BangTree), BinPC/Wor_15104E22.lin.bf Bombarde_0.ai (read 2026-11-27;
# not in the save records). Radii in world units (1.5 per cell).
BANG_TREE_SETTINGS = dict(hole_radius=30.0, hole_depth=20.0, grid_perception_radius=10.0,
                          entities_perception_radius=20.0, explosion_delay=0.2, explosion_duration=1.0,
                          time_to_activate_seed=10.0, time_to_grow_tree=10.0, nb_seeds=1, nb_seeds_random=0,
                          diffusion_radius_min=3.0, diffusion_radius_max=4.0, water_threshold_max=5.0,
                          lava_threshold_max=5.0, seed_ejection_strength=20.0, seed_ejection_angle=20.0,
                          time_to_degenerate_tree=10.0)
BANG_TREE_STATES = {1: 'seed', 2: 'growing', 3: 'adult', 4: 'ending', 6: 'exploding'}


def bang_trees(u):
    """Explosive trees (DustAi::BangTree, entity type 0x28000004) in a save or level file, decoded 2026-11-27
    (findings/water-lava-sim.md "Explosive trees"). Same header as the other trees (+0x10 handle, +0x28 world
    position, +0xD4 state, +0xD8 the state's clock). Record sizes: 0x104 / 0xF8 adult (state 3: +0xDC life time,
    -1 = none, +0xE0 the danger timer, +0xE8 the BangTree bits, bit 0 = danger), 0x178 / 0x16C seed (state 1),
    0xFC / 0xF0 states 2 (growing) and 4 (end of life). Class settings: BANG_TREE_SETTINGS."""
    out = []
    i = u.find(BANG_TREE_TAG)
    while i >= 0:
        sizes = (u32(u, i - 0x10), u32(u, i - 8)) if i >= 0x10 else None
        if sizes in ((0x104, 0xF8), (0x178, 0x16C), (0xFC, 0xF0)):
            x, y, z = f32(u, i + 0x28), f32(u, i + 0x2C), f32(u, i + 0x30)
            state = u32(u, i + 0xD4)
            t = dict(id=handle_id(u32(u, i + 0x10), u32(u, i + 0x14), u32(u, i + 0x18)),
                     col=int(256 + 2 * x / 3), row=int(256 + 2 * y / 3), x=x, y=y, z=z,
                     state=BANG_TREE_STATES.get(state, str(state)), clock=f32(u, i + 0xD8), danger_timer=0.0,
                     danger=False)
            if sizes[0] == 0x104:
                t.update(life_time=f32(u, i + 0xDC), danger_timer=f32(u, i + 0xE0),
                         danger=bool(u32(u, i + 0xE8) & 1))
            t.update(BANG_TREE_SETTINGS)
            out.append(t)
        i = u.find(BANG_TREE_TAG, i + 1)
    return out


def _entry_start(u, tag):
    """Offsets just after each entry tag (the handle block follows)."""
    return [m.start() + 4 for m in re.finditer(re.escape(struct.pack('>I', tag)), u)]


def villages(u):
    """Built villages (DustAi::Village, manager 0x10800001, entries 0x20800001): handle index, grid cell,
    building and human counts, and the index of the statue (totem) they belong to. The class-default
    entry (statue index 511, no buildings) is skipped."""
    out = []
    for p in _entry_start(u, VILLAGE_TAG):
        try:
            hc, _, p = block(u, p)
            index = u32(u, hc + 8)
            pc, _, p = block(u, p)
            x, y, _z = struct.unpack_from('>3f', u, pc)
            for _ in range(4):                                  # direction, ?, two handles
                _, _, p = block(u, p)
            p += 4                                              # u32 0
            _, _, p = block(u, p)                               # 5 slots
            _, _, p = block(u, p)                               # [0]
            p += 12                                             # 3 words (18, 0, 0x41980000)
            bc, _, p = block(u, p)
            buildings = u32(u, bc)
            p += 8                                              # f32 timer, u32 80
            sc, _, p = block(u, p)
            statue = u32(u, sc + 8)
            _, _, p = block(u, p)                               # [0]
            mc, _, p = block(u, p)
            humans = u32(u, mc)
        except (ValueError, struct.error):
            continue
        if statue == 511:
            continue
        out.append(dict(index=index, col=int(256 + x * 2 / 3), row=int(256 + y * 2 / 3), statue=statue,
                        buildings=buildings, humans=humans))
    return out


def repel_songs(u):
    """Learned repel songs (entities 0x20400009 = repels water, 0x2040000A = repels lava): grid cell,
    f32 size (40 water / 35 lava seen) and the index of the village holding it. In the game each one adds
    a RepulseFluid modifier at its cell every tick (dustai 0x10134AC0 / 0x10135210): water radius =
    trunc(size x 2/3) (26), lava radius = size x 2/3 (23.33). Class-default entries are skipped."""
    out = []
    for tag, kind in ((REPEL_WATER_TAG, 'water'), (REPEL_LAVA_TAG, 'lava')):
        for p in _entry_start(u, tag):
            try:
                _, _, p = block(u, p)
                pc, _, p = block(u, p)
                x, y, _z = struct.unpack_from('>3f', u, pc)
                for _ in range(4):
                    _, _, p = block(u, p)
                p += 4
                _, _, p = block(u, p)                           # [0]
                _, _, p = block(u, p)                           # list of cells
                size = f32(u, p + 12)
                vc, _, p = block(u, p + 24)
                village = u32(u, vc + 8)
            except (ValueError, struct.error):
                continue
            if village == 511:
                continue
            out.append(dict(kind=kind, col=int(256 + x * 2 / 3), row=int(256 + y * 2 / 3), size=size, village=village))
    return out


def song_radius(kind, size):
    return float(int(size * 2 / 3)) if kind == 'water' else size * 2 / 3


def progress_totems(u):
    """Per totem of a progress save: statue index -> dict(built, water, lava) with the repel radii (0 = none)."""
    vil = {v['index']: v for v in villages(u)}
    out = {t['index']: dict(built=False, water=0.0, lava=0.0) for t in statues(u)}
    for v in vil.values():
        out.setdefault(v['statue'], dict(built=False, water=0.0, lava=0.0))['built'] = True
    for s in repel_songs(u):
        v = vil.get(s['village'])
        if v is not None:
            out[v['statue']][s['kind']] = song_radius(s['kind'], s['size'])
    return out


def find_progress_save(level_statues, root=PROGRESS_DIR):
    """The player's progress save for the level whose totems are `level_statues` (matched by totem
    positions), newest first if several profiles match; None if there is none."""
    import glob
    key = sorted((t['col'], t['row']) for t in level_statues)
    if not key:
        return None
    hits = []
    for path in glob.glob(os.path.join(root, 'profile_*', 'ingame', 'MAP_*.SAV')):
        try:
            _, u = sav.read_sav(path)
        except Exception:
            continue
        if sorted((t['col'], t['row']) for t in statues(u)) == key:
            hits.append((os.path.getmtime(path), path))
    return max(hits)[1] if hits else None

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
    elif name == 'CheckVegetationNumberOnce' and b - q >= 16:
        d.update(cells=u32(u, q + 8), keep_evaluated=u32(u, q + 12))
    elif name in ('Population', 'PopulationUnder') and b - q >= 12:
        d.update(population=u32(u, q + 8))
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
        elif 'cells' in c:
            out.append('vegetation covers >= %d cells' % c['cells'])
        elif c['name'] == 'Population' and 'population' in c:
            out.append('population >= %d' % c['population'])
        elif c['name'] == 'PopulationUnder' and 'population' in c:
            out.append('population < %d' % c['population'])
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
    sts = statues(u)
    print('%d totems (statues):' % len(sts))
    for t in sts:
        print('  #%-2d col %3d row %3d | power %d | id %s | words %d %d height %.2f' % (t['index'], t['col'], t['row'], t['power'], t['id'], t['w1026'], t['w3'], t['height']))
    root = find_root(u)
    if not root:
        print('no level script found')
        return
    print('level script (%d rules; times are seconds of game time):' % len(root['rules']))
    print_rulelist(root, gens, '  ', raw)


if __name__ == '__main__':
    main()
