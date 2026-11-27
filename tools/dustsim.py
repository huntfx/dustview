"""Re-implementation of From Dust's water/sand/lava step, checked against the game's own DLL.

step(G, P) does one full tick; from_grid() converts a galileo.py grid; P is a dict name -> value
(e.g. {n: float(v) for n, v in zip(galileo.PARAM_NAMES, galileo.fluid_params())}).
Check against the game with tools/compare_sim.py.

Built rule by rule from GalileoSimulatorWater_9_9.dll (SimulatorWaterIterator::Step1/Step2/Step4)
and from experiments with tools/galileo.py (which runs the real DLL). Findings and the
evidence for every rule: findings/water-lava-sim.md.

Grid convention: arrays [y, x], 512 x 512, same components as galileo.py.
Fluxes are kept as integers in 1/256 units, exactly like the game (u16 pairs).
"""
import numpy as np

L, R, U, D = range(4)   # flux directions (kFluxLeft/Right/Up/Down)


def shift(a, dy, dx, fill=0):
    """out[y, x] = a[y + dy, x + dx] (neighbour value), `fill` outside the grid."""
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[yd, xd] = a[ys, xs]
    return out

# neighbour offsets per direction: left = x-1, right = x+1, up = y+1, down = y-1 (game convention,
# found by matching the DLL: kFluxUp is the y+1 neighbour in the grid layout)
NB = {L: (0, -1), R: (0, 1), U: (1, 0), D: (-1, 0)}
OPP = {L: R, R: L, U: D, D: U}
DEBUG = False


def ramp(x, x0, x1, y0, y1):
    """Linear from y0 at x0 to y1 at x1, clamped to the end values (ConstParam ...ThresholdScaleMinMax)."""
    t = np.clip((x - x0) / (x1 - x0), 0, 1)
    return (y0 + t * (y1 - y0)).astype(np.float32)


def step(G, P):
    """One full simulation step: the compiled version (dustsim_nb, numba) when available and DEBUG is
    off, otherwise step_numpy. Both give bit-identical results (checked with tools/check_nb.py)."""
    if _nb is not None and not DEBUG:
        return _nb.step(G, P)
    return step_numpy(G, P)


def step_numpy(G, P):
    """One full simulation step (water, sand, sediment and lava). G: dict with
    T (terrain surface), soft (soft ground), W (fluid height; negative = lava),
    sed (sediment for water / temperature for lava), F (int32[4] stored outflows, 1/256 units).
    Returns a new dict in the same layout. Order follows the DLL (Step1, then flow + transport)."""
    f32 = np.float32
    T, So, F = G['T'].copy(), G['soft'].copy(), G['F']
    lava = G['W'] < 0
    H = np.abs(G['W']).astype(f32)                     # fluid depth, either kind
    Sd = np.where(lava, 0, G['sed']).astype(f32)       # sediment (water cells)
    Tm = np.where(lava, G['sed'], 0).astype(f32)       # temperature (lava cells)
    water = ~lava
    So = np.where(lava, f32(0), So).astype(f32)         # lava turns the sand under it into rock (T unchanged; DLL, 2026-11-18)

    # ---------------- Step1: water cells -- erosion and deposit
    sumf = F.sum(0).astype(f32) / f32(256)
    wet = water & (H > f32(1 / 255))
    v = np.where(wet, sumf / np.sqrt(np.where(wet, H, 1)), 0).astype(f32)
    e = np.clip(f32(P['SoftGroundErosionFactor']) * (v - f32(P['SoftGroundErosionThreshold'])),
                0, f32(P['SoftGroundErosionMax']))
    e = np.where(wet & (H <= f32(P['SoftGroundErosionWaterHeightMax'])), e, 0)
    e = np.minimum(e, So).astype(f32)
    So -= e; T -= e; Sd += e; H += e
    d = np.clip(f32(P['SedimentDepotFactor']) * (f32(P['SedimentDepotThreshold']) - v) * Sd,
                0, f32(P['SedimentDepotMax']))
    d = np.maximum(d, Sd - H)            # more sediment than fluid (not reachable in play): dump the excess
    d = np.where(water, np.minimum(d, Sd), 0).astype(f32)
    So += d; T += d; Sd -= d; H = np.where(water, np.maximum(H - d, 0), H).astype(f32)

    # ---------------- Step1: lava cells -- cooling and solidifying
    k = f32(P['TemperatureDampingFactor']) + f32(P['WaterHeightTemperatureDampingFactor']) *         np.maximum(0, f32(P['WaterHeightTemperatureDampingThreshold']) - H)
    Tm = np.where(lava & (Tm >= f32(P['TemperatureDampingThreshold'])), np.maximum(Tm - k, 0), Tm).astype(f32)
    solid = lava & (Tm <= f32(P['WaterTemperatureMax']))
    T = np.where(solid, T + H, T).astype(f32)
    H = np.where(solid, 0, H).astype(f32)
    Tm = np.where(solid, 0, Tm).astype(f32)
    lava &= ~solid

    # ---------------- flow (each cell with its own material's parameters)
    ratio = np.where(H > 0, Sd / np.where(H > 0, H, 1), 0).astype(f32)
    damp = np.where(lava,
                    ramp(Tm, P['LavaFluxDampingThresholdMin'], P['LavaFluxDampingThresholdMax'],
                         P['LavaFluxDampingMin'], P['LavaFluxDampingMax']),
                    ramp(ratio, P['SedimentRatioThreshold0'], P['SedimentRatioThreshold1'],
                         P['WaterFluxDamping'], P['WaterFluxDampingMin'])).astype(f32)
    fluid = np.where(lava,
                     ramp(Tm, P['LavaFluidityThreshold0'], P['LavaFluidityThreshold1'],
                          P['LavaFluidityValue0'], P['LavaFluidityValue1']),
                     ramp(ratio, P['SedimentRatioThreshold0'], P['SedimentRatioThreshold1'],
                          P['WaterFluidity'], P['WaterFluidityMin'])).astype(f32)
    S = T + H
    f = np.zeros((4,) + T.shape, f32)
    for kk, (dy, dx) in NB.items():
        f[kk] = np.maximum(0, damp * (F[kk].astype(f32) / f32(256)) + fluid * (S - shift(S, dy, dx)))
    f = np.minimum(f, np.where(lava, f32(P['LavaFluxMask']), f32(P['WaterFluxMask'])))   # per-direction cap
    f *= ~(lava & (f.sum(0) < f32(P['LavaFluxThreshold'])))      # lava yield: too little push, no flow
    tot = f.sum(0)
    f *= np.where(tot > H, H / np.where(tot > 0, tot, 1), 1).astype(f32)
    Fq = np.rint(f * 256).astype(np.int32)
    f_flow = f                                       # (debug: flows before rounding)

    # ---------------- lateral erosion (sand slumping into mud), water/dry cells
    thr = [P['LateralErosionSlopeThresholdLeft'], P['LateralErosionSlopeThresholdRight'],
           P['LateralErosionSlopeThresholdUp'], P['LateralErosionSlopeThresholdDown']]
    k_lat = f32(P['ErosionSumFluxSumFactor'])
    drop = np.max([k_lat * (T - shift(T, *NB[kk]) - f32(thr[kk])) for kk in NB], axis=0)
    share = np.where(H > 0, f.sum(0) / np.where(H > 0, H, 1), 1)
    share = np.where(share > f32(P['ErosionSumFluxSumThreshold']), share, 0)
    lat = np.where(lava, 0, np.minimum(np.clip(drop, 0, None) * share, So)).astype(f32)
    So -= lat; T -= lat
    ratio = np.where(H + lat > 0, (Sd + lat) / np.where(H + lat > 0, H + lat, 1), 0).astype(f32)
    H = H + lat; Sd = Sd + lat

    # ---------------- transport
    out = f.sum(0)
    rem = H - out                                    # what stays in the cell
    # float leftovers of a fully drained lava cell (~1e-7) count as empty: the DLL (approximate
    # reciprocals, other summation order) ends at exactly 0 there and keeps no lava below 1/256
    rem = np.where(lava & (np.abs(rem) < f32(1e-6)), f32(0), rem).astype(f32)
    # a drained water cell's float leftover can come out slightly negative (-1.2e-7 seen: 1.56 out as
    # 399/256); negative fluid means lava, so a spring then 'hardened' it and made 8 x its rate of water
    # (Map 4, 2026-11-18). The DLL ends at 0 there.
    rem = np.where(~lava & (rem < 0), f32(0), rem).astype(f32)
    fin = Fq.astype(f32) / f32(256)                  # what receivers get (rounded)
    def gather(a):                                   # sum over the 4 neighbours of a[opposite dir]
        return sum(shift(a[OPP[kk]], *NB[kk]) for kk in NB)
    lava_in = gather(np.where(lava, fin, 0))
    water_in = gather(np.where(lava, 0, fin))
    heat_in = gather(np.where(lava, fin * Tm, 0))
    sq = np.rint(fin * np.where(lava, 0, ratio) * f32(4096))   # sediment, 1/4096 units
    sed_in = gather(sq).astype(f32) / f32(4096)
    Sn = (np.where(lava, 0, Sd - out * ratio) + sed_in).astype(f32)
    # Mixing, row by row as the DLL does it (found 2026-11-13 against the DLL on flowing lava; this is
    # also the old "drained lava" quirk). The inflow from the row below (y - 1, through that cell's Up
    # outflow) is classified by that cell's NEW state from this step: if it ended as lava, the inflow is
    # lava at its new temperature; otherwise (drained, water) it counts as water. The other three
    # neighbours count by their old state, lava at its cooled temperature. Lava + water: the water boils
    # away but dilutes the temperature; diluted to WaterTemperatureMax or below, the lava hardens at once
    # and the cell stays water. Ground += the lava volume if the cell was water; if it was lava: lava volume +
    # the lava that flowed in this step (counted a second time) + the water volume, and this also happens
    # when all its lava left and water came in (2026-11-18, against the DLL with the running game's
    # parameters, WaterEnergy 8, on the Fire and Water map; earlier checks used 0 and missed it).
    noD = [kk for kk in NB if kk != D]
    lin0 = sum(shift(np.where(lava, fin, 0)[OPP[kk]], *NB[kk]) for kk in noD).astype(f32)   # lava inflow (3 sides)
    lv0 = (np.where(lava, rem, 0) + lin0).astype(f32)
    wv0 = (np.where(lava, 0, rem) + sum(shift(np.where(lava, 0, fin)[OPP[kk]], *NB[kk]) for kk in noD)).astype(f32)
    num0 = (np.where(lava, rem * Tm, 0) + sum(shift(np.where(lava, fin * Tm, 0)[OPP[kk]], *NB[kk]) for kk in noD)).astype(f32)
    fD = shift(fin[U], *NB[D])                                 # inflow from row y - 1 (any fluid)
    Tmax = f32(P['WaterTemperatureMax'])
    C3 = np.empty_like(Sn); Wn = np.empty_like(Sn)
    lava_vol = np.empty_like(Sn); water_vol = np.empty_like(Sn)
    below_lava = np.zeros(Sn.shape[1], bool); below_C3 = np.zeros(Sn.shape[1], f32)
    rows = range(Sn.shape[0]) if (lv0 > 0).any() else ()      # no lava anywhere: vectorised below
    if not rows:
        lava_vol, water_vol = lv0, wv0 + fD
        C3, Wn = Sn.copy(), water_vol.astype(f32)
    for y in rows:
        f = fD[y]
        lv = lv0[y] + np.where(below_lava, f, 0)
        lin = lin0[y] + np.where(below_lava, f, 0)
        wv = wv0[y] + np.where(below_lava, 0, f)
        bl = lv > 0
        tn = np.where(bl, (num0[y] + np.where(below_lava, f * below_C3, 0)) / np.where(bl, lv + wv, 1), 0).astype(f32)
        fr = bl & (tn <= Tmax)
        old = lava[y]                            # the cell was lava at the start of the step
        fr_old = old & ~bl & (wv > 0)            # its lava all left and water came in: hardens too (DLL)
        if fr.any() or fr_old.any():             # was lava: ground += lava + lava inflow again + water;
            T[y] = np.where(fr | fr_old, T[y] + np.where(old, (lv + lin) + wv, lv), T[y])   # was water: += lava
            bl = bl & ~fr
        C3[y] = np.where(bl, tn, Sn[y])
        Wn[y] = np.where(bl, -lv, wv)
        lava_vol[y], water_vol[y] = lv, wv
        below_lava, below_C3 = bl, C3[y]
    T = T.astype(f32)
    out_d = {'T': T, 'soft': So, 'W': Wn.astype(f32), 'sed': C3.astype(f32), 'F': Fq}
    if DEBUG:
        out_d['_dbg'] = dict(lava_vol=lava_vol, water_vol=water_vol, rem=rem, lava_in=lava_in, water_in=water_in, heat_in=heat_in, Tm=Tm, lava=lava, f=f_flow, Fq=Fq)
    return out_d


try:                                   # compiled step (needs numba, e.g. the FromDust/.venv)
    import dustsim_nb as _nb
except ImportError:
    _nb = None


def from_grid(grid):
    """galileo.py grid (8, 512, 512) uint32 -> dict for step()."""
    gf = lambda c: grid[c].view(np.float32).copy()
    lr, ud = grid[4], grid[5]
    F = np.stack([lr & 0xFFFF, lr >> 16, ud & 0xFFFF, ud >> 16]).astype(np.int32)
    return {'T': gf(0), 'soft': gf(1), 'W': gf(2), 'sed': gf(3), 'F': F}





# ---------------------------------------------------------------- outside the step: border and sources
SKIP = 32           # SkipBorderSize 2 (x 16 cells): rows/cols 0-31 and 480-511 never change
RING = (SKIP, 511 - SKIP)


def set_border_ring(G, P):
    """Reset the outermost processed ring (row/col 32 and 479) to the map's border values. The game
    does this before each step (live: the ring reads just below the border level after a step)."""
    ground = np.float32(P['BorderHardgroundSurfaceHeight'])
    soft = np.float32(P['BorderSoftgroundSurfaceHeight'])
    water = max(np.float32(P['BorderWaterSurfaceHeight']) - ground, np.float32(0))
    a, b = RING
    for sl in ((a, slice(a, b + 1)), (b, slice(a, b + 1)), (slice(a, b + 1), a), (slice(a, b + 1), b)):
        G['T'][sl], G['soft'][sl], G['W'][sl], G['sed'][sl] = ground, soft, water, 0


_DISC_CACHE = {}


def disc(col, row, radius):
    """Cells of a source: dx^2 + dy^2 <= r^2 around the centre cell (as measured in the game).
    Cached (the returned arrays are shared: don't modify them)."""
    key = (int(col), int(row), float(radius))
    hit = _DISC_CACHE.get(key)
    if hit is None:
        if len(_DISC_CACHE) > 4096:
            _DISC_CACHE.clear()
        hit = _DISC_CACHE[key] = _disc(*key)
    return hit


def _disc(col, row, radius):
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = dx * dx + dy * dy <= radius * radius
    ys, xs = (dy[keep] + int(row)), (dx[keep] + int(col))
    ok = (ys >= 0) & (ys < 512) & (xs >= 0) & (xs < 512)
    return ys[ok], xs[ok]


def bury_activity(G, s):
    """The game's bury rule (dustai 0x10065DE0), for sources that carry their height ('z') and height
    mode: mode 0 keeps a stored height that is lowered when the surface drops below it, and the source
    fades out linearly as the ground over it rises 0..threshold (5) above that height; mode 2 rides on the
    surface and is never buried. Without 'z' the given 'activity' (default 1) is used as is.
    Checked live on Map 2 (ground 0.4 below / 6.4 above the stored height -> activity 1 / 0). Which
    height lowers the stored height (here ground + fluid) is inferred, not traced."""
    if 'z' not in s:
        return s.get('activity', 1.0)
    if s.get('height_mode', 2) != 0:
        return 1.0
    r, c = int(s['row']), int(s['col'])
    ground = float(G['T'][r, c])
    surface = ground + abs(float(G['W'][r, c]))
    if 0 < surface < s['z']:
        s['z'] = surface
    T = s.get('bury_threshold', 5.0)
    return 1.0 - min(max(ground - s['z'], 0.0), T) / T if T > 0 else 1.0


WATER_ENERGY = 8.0          # ParamF[46] WaterEnergy as the running game has it (galileo.fluid_params gives 0)
WATER_TEMPERATURE_MAX = 12.5


def fluid_modifier(G, ys, xs, rate, sediment=0.0, temperature=0.0, P=None):
    """kModifier_Fluid on the cells ys, xs, as GalileoSimulatorWater.dll does it (AddWater / RemoveWater
    / AddLava / RemoveLava, chosen by the sign of `rate` and temperature > 0; checked with
    galileo.modify on random grids). E = WaterEnergy (8), Tmax = WaterTemperatureMax (12.5).
      AddWater:    water cell: depth += rate, sediment = min(sediment + rate x fraction, old depth); dry cell: depth = rate (no
                   sediment); lava cell (depth d, temperature T0): T = T0 - E x rate / d, and if that is
                   <= Tmax the lava hardens (ground += d) and the cell becomes water of depth
                   d + max(0, E x rate - T0 x d) (as the DLL does: no division by E), sediment 0.
      RemoveWater: water cells: depth = max(depth - |rate|, 0); sediment = max(sediment - |rate| x the source's
                   sediment setting, 0) (a drain with sediment 0.1 also takes mud; checked against the DLL
                   2026-11-18); lava untouched.
      AddLava:     lava / dry cell: lava += rate, temperature = T; water cell (depth w): h = w x E / T
                   hardens from the new lava: if h <= rate: ground += h, lava = rate - h at T;
                   else ground += rate and the water loses rate x T / E (sediment kept); outflows of
                   every water cell hit become 0.
      RemoveLava:  lava cells: lava = max(lava - |rate|, 0); emptied cells become dry (temperature 0)."""
    f32 = np.float32
    E = f32(P['WaterEnergy'] if P is not None and P.get('WaterEnergy') else WATER_ENERGY)
    Tmax = f32(P['WaterTemperatureMax'] if P is not None and P.get('WaterTemperatureMax') else WATER_TEMPERATURE_MAX)
    W, Sd, T, F = G['W'], G['sed'], G['T'], G['F']
    w = W[ys, xs].astype(f32)
    s0 = Sd[ys, xs].astype(f32)
    rate, sediment, temperature = f32(rate), f32(sediment), f32(temperature)
    if temperature <= 0:
        if rate > 0:                                                    # AddWater
            lava = w < 0
            d = -w
            cooled = s0 - E * rate / np.where(lava, d, f32(1))
            harden = lava & (cooled <= Tmax)
            newW = np.where(w > 0, w + rate, np.where(w == 0, rate,
                   np.where(harden, d + np.maximum(f32(0), E * rate - s0 * d), w)))
            newS = np.where(w > 0, np.minimum(s0 + rate * sediment, w), np.where(w == 0, s0, np.where(harden, f32(0), cooled)))
            T[ys, xs] = np.where(harden, T[ys, xs] + d, T[ys, xs])
            W[ys, xs], Sd[ys, xs] = newW.astype(f32), newS.astype(f32)
        else:                                                           # RemoveWater
            W[ys, xs] = np.where(w > 0, np.maximum(w + rate, f32(0)), w).astype(f32)
            if sediment:                     # the drain's sediment setting removes sediment too (DLL, 2026-11-18)
                Sd[ys, xs] = np.where(w > 0, np.maximum(s0 + rate * sediment, f32(0)), s0).astype(f32)
    else:
        if rate > 0:                                                    # AddLava
            water = w > 0
            h = w * E / temperature
            some = water & (h <= rate)
            alls = water & (h > rate)
            T[ys, xs] = np.where(some, T[ys, xs] + h, np.where(alls, T[ys, xs] + rate, T[ys, xs]))
            newW = np.where(alls, w - rate * temperature / E, np.where(some, -(rate - h), -(np.maximum(-w, f32(0)) + rate)))
            newS = np.where(alls, s0, temperature)
            W[ys, xs], Sd[ys, xs] = newW.astype(f32), newS.astype(f32)
            if water.any():
                F[:, ys[water], xs[water]] = 0
        else:                                                           # RemoveLava
            lava = w < 0
            left = np.maximum(-w + rate, f32(0))
            empty = lava & (left == 0)
            W[ys, xs] = np.where(empty, f32(0), np.where(lava, -left, w)).astype(f32)
            Sd[ys, xs] = np.where(empty, f32(0), s0).astype(f32)


def ground_modifier(G, ys, xs, amount, soft=False):
    """kModifier_HardGround / kModifier_SoftGround on the cells ys, xs (as the DLL; checked with
    galileo.modify): HardGround moves the ground surface by `amount` but the hard part (ground - soft)
    cannot go below 0; SoftGround adds `amount` to the soft layer and the surface, removing at most the
    soft layer there. Water, sediment / temperature and outflows are untouched."""
    f32 = np.float32
    T, So = G['T'], G['soft']
    a = f32(amount)
    if soft:
        s0 = So[ys, xs]
        d = np.maximum(a, -s0)
        So[ys, xs] = (s0 + d).astype(f32)
        T[ys, xs] = (T[ys, xs] + d).astype(f32)
    else:
        T[ys, xs] = np.maximum(T[ys, xs] + a, So[ys, xs]).astype(f32)


def apply_sources(G, sources, P=None):
    """Apply springs, lava vents and drains (dicts with col, row, radius, rate per cell per tick,
    sediment fraction, temperature; optional height data for the bury rule) with the game's
    modifier rules (fluid_modifier) on each source's disc (dx^2 + dy^2 <= r^2).
    Every source's bury activity is worked out first, on G as given, then the sources are applied (as
    modifier_calls does; the game's per-tick list already holds the final amounts, so its AI decides them
    before any modifier runs. Inferred, not traced; until 2026-11-21 this worked them out one after the
    other, which differed where a source sits inside another one's disc)."""
    for s in sources:
        s['activity'] = bury_activity(G, s)
    for s in sources:
        rate = s['rate'] * s['activity']
        if rate == 0:
            continue
        ys, xs = s.get('_cells') or disc(s['col'], s['row'], s['radius'])
        fluid_modifier(G, ys, xs, rate, s.get('sediment', 0.0), s.get('temperature', 0.0), P)


def apply_modifiers(G, entries, P=None):
    """Apply the game's per-tick modifier list in its order: type 0 = fluid source (dict with 'amount'
    or 'rate'), type 1 = RepulseFluid (col, row, radius, amount, f5 = mode), 2 / 3 = HardGround /
    SoftGround (col, row, radius, amount), 4-7 = RemoveWater / AddWater / RemoveLava / AddLava (typed_fluid_args)."""
    for e in entries:
        t = e.get('type', 0)
        if t == 0:
            apply_sources(G, [dict(e, rate=e.get('amount', e.get('rate', 0)))] if 'amount' in e else [e], P)
        elif t == 1:
            repulse_fluid(G, e['col'], e['row'], e['radius'], e['amount'], e['f5'])
        elif t in (2, 3):                                   # HardGround, SoftGround
            ys, xs = disc(e['col'], e['row'], e['radius'])
            ground_modifier(G, ys, xs, e['amount'], soft=(t == 3))
        elif 4 <= t <= 7:                                   # RemoveWater, AddWater, RemoveLava, AddLava
            ys, xs = disc(e['col'], e['row'], e['radius'])
            fluid_modifier(G, ys, xs, *typed_fluid_args(e), P=P)


def fire_tree_entries(trees, dt=1 / 31):
    """Advance fire trees (dicts from live_generators.fire_trees: col, row, state, idle, active_time, size, rate,
    clock, next) by dt seconds and return this tick's per-tick-list entries, as dustai 0x10083540 does per frame:
    clock += dt; before `next`: idle; for `active_time` s from `next`: push RemoveWater (type 4) on the disc of
    radius `size` with amount -rate x dt; then set the 8 neighbouring cells' plants alight (0x10082DD0 ->
    0x10141550: sets 'burst' on the tree for this tick, see tree_fire_burst) and `next` = clock + idle. Only state 3 trees cycle.
    dt = 1/31: the game's amount is rate x frame time (0.166 in a 1/30 s frame) and its clock runs on the
    same frame time; the model's tick stands for 1/31 s like the level script."""
    out = []
    for t in trees:
        t['active'] = t['burst'] = False
        if t.get('state') != 3:
            continue
        t['clock'] += dt
        if t['next'] > t['clock']:
            continue
        if t['clock'] - t['next'] < t['active_time']:
            t['active'] = True
            out.append(dict(type=4, col=float(t['col']), row=float(t['row']), radius=float(t['size']), f3=0.0,
                            amount=-t['rate'] * dt, f5=0.0, sediment=0.0, temperature=0.0))
        else:
            t['burst'] = True                    # tree_fire_burst lights its 8 neighbours this tick
            t['next'] = t['clock'] + t['idle']
    return out


WATER_TREE_BLOW_UP_TIME = 5.0   # s in the blow-up state (DustTree +0x288, read live 2026-11-27)


def _tree_burning(f):
    """dustai 0x10082FF0: a fire tree counts as danger while in state 3 with its clock past `next`."""
    return f.get('state') == 3 and f['clock'] > f['next']


def water_tree_step(G, trees, fire_trees=(), dt=1 / 31, rng=None, grid_danger=None):
    """One frame of every water tree (DustAi::Figuier; dicts from live_generators.water_trees or
    sav_sources.water_trees), decoded 2026-11-27 (findings/water-lava-sim.md "Water trees"). Absorption is applied
    to G in place (dustai writes it straight into the grid, 0x100DA3C0); returns this frame's pour entries
    (type 0 fluid modifiers, like a spring) for apply_modifiers.
    Absorbing state (0x10082720): danger (a burning fire tree within danger_radius, or grid_danger(G, t): the game's
    grid sensor, seen live to fire on burning plants and lava in a +-trunc(danger_radius x 2/3) = +-10 square;
    pass bang_fire_danger(c6, 'danger_radius')) -> pour rejection_speed x dt; else store += generation_speed x dt (capped at
    max_quantity), absorb, and switch to blow_up once store > quantity_to_blow_up. Blow-up state: pour
    blow_up_rejection_speed x dt for explosion_duration (5 s), then the tree is removed from `trees` and its seeds
    added (0x1007F4F0: nb_seeds + random 0..nb_seeds_random, each at a random angle and a random distance
    diffusion_radius_min..max world units = x 2/3 cells; seen live: one seed 2-3 cells away).
    Seed (state 1, 0x10081060): roots once its clock reaches time_to_activate_seed (10 s) and its cell passes the
    game's cell test (0x1007EE60: no lava, |fluid| <= water_threshold_max). APPROXIMATE: the game also needs the
    seed to have come to rest and buried itself (drift physics and the shared bury rule not modelled, seeds stay
    put) and has exclusion / entity checks not read, so a seed that stays a seed in the game (e.g. Waves 155840FC)
    roots here. Growing (state 2, 0x10080990): time_to_grow_tree (15 s), then absorbing with an empty store.
    Trees are mutated in place; rng (numpy Generator) picks the seed positions."""
    out = []
    W, sed = G['W'], G['sed']
    rng = rng if rng is not None else np.random.default_rng()
    born, dead = [], []
    for t in trees:
        t['poured'] = t['absorbed'] = 0.0
        st = t.get('state')
        if st == 'seed':
            t['clock'] = t.get('clock', 0.0) + dt
            if t['clock'] >= t.get('time_to_activate_seed', 10.0) and _water_tree_cell_ok(W, t):
                t['state'], t['clock'] = 'growing', 0.0
            continue
        if st == 'growing':
            t['clock'] = t.get('clock', 0.0) + dt
            if t['clock'] >= t.get('time_to_grow_tree', 15.0):
                t['state'], t['stored'], t['danger'] = 'absorbing', 0.0, False
            continue
        if st == 'blow_up':
            t['blow_clock'] = t.get('blow_clock', 0.0) + dt
            out += _water_tree_pour(t, t['blow_up_rejection_speed'], dt)
            if t['blow_clock'] >= t.get('explosion_duration', WATER_TREE_BLOW_UP_TIME):
                dead.append(t)
                born += _water_tree_seeds(t, rng)
            continue
        if st != 'absorbing':
            continue
        r2 = t['danger_radius'] ** 2
        t['danger'] = any(_tree_burning(f) and (f['col'] - t['col']) ** 2 + (f['row'] - t['row']) ** 2 <= r2
                          for f in fire_trees) or bool(grid_danger and grid_danger(G, t))
        if t['danger']:
            out += _water_tree_pour(t, t['rejection_speed'], dt)
            continue
        room = max(t['max_quantity'] - t['stored'], 0.0)
        t['stored'] += min(t['generation_speed'] * dt, room)
        t['absorbed'] = _water_tree_absorb(W, sed, t)
        t['stored'] += t['absorbed']
        if t['stored'] > t['quantity_to_blow_up']:
            t['state'], t['blow_clock'] = 'blow_up', 0.0
    for t in dead:
        trees.remove(t)
    trees.extend(born)
    return out


def _water_tree_cell_ok(W, t):
    """The rooting cell test (dustai 0x1007EE60) as it applies to water trees: no lava on the cell and |fluid| <=
    water_threshold_max (20); height limits 0..255 and ground type 'any' always pass."""
    f = float(W[int(t['row']), int(t['col'])])
    return f >= 0 and abs(f) <= t.get('water_threshold_max', 20.0)


def _water_tree_seeds(t, rng):
    """Seeds a blown-up tree leaves (dustai 0x1007F4F0 with argument 0): nb_seeds + random 0..nb_seeds_random, each
    at a random angle and distance diffusion_radius_min..max world units (2/3 cell per unit) from the tree."""
    n = int(t.get('nb_seeds', 1)) + int(rng.integers(0, int(t.get('nb_seeds_random', 0)) + 1))
    out = []
    for k in range(n):
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(t.get('diffusion_radius_min', 3.0), t.get('diffusion_radius_max', 4.0)) * 2 / 3
        c = int(min(max(t['col'] + d * np.cos(a), 33), 478))
        r = int(min(max(t['row'] + d * np.sin(a), 33), 478))
        s = {k2: v for k2, v in t.items() if k2 not in ('address', 'blow_clock', 'danger_cell', 'bits', 'state_clock')}
        s.update(id='%s.s%d' % (t['id'], k), col=c, row=r, state='seed', clock=0.0, stored=0.0, danger=False,
                 poured=0.0, absorbed=0.0)
        out.append(s)
    return out


def _water_tree_absorb(W, sed, t):
    """dustai 0x100DA3C0 for one tree, on W / sed in place; returns the amount credited to the tree."""
    R = float(t['absorption_radius'])
    r = int(R * 2 / 3)
    c0, r0 = int(t['col']), int(t['row'])
    rows = np.arange(max(r0 - r, 33), min(r0 + r, 478) + 1)
    cols = np.arange(max(c0 - r, 33), min(c0 + r, 478) + 1)
    if not len(rows) or not len(cols):
        return 0.0
    w = W[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    s = sed[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    d2 = ((rows[:, None] - r0) ** 2 + (cols[None, :] - c0) ** 2).astype(np.float32)
    weight = np.float32(1) - np.minimum(d2 * np.float32(1 / (R * R)), np.float32(1))
    cap = np.float32(t['absorption_speed'] / (2 * r) ** 2)
    wet = w > 0
    take = np.minimum(np.maximum(w - np.float32(t['absorption_cell_min']), 0), cap) * weight
    take = np.where(wet, take, np.float32(0)).astype(np.float32)
    frac = np.where(wet, take / np.where(wet, w, 1), 0).astype(np.float32)
    w -= take
    s *= np.float32(1) - frac
    return float(take.sum(dtype=np.float64)) * t['absorption_factor']


def _water_tree_pour(t, speed, dt):
    """dustai 0x10081930: pour min(speed x dt, store) as a type-0 fluid modifier of radius rejection_radius,
    spread as amount / (pi x trunc(2R/3)^2) per cell."""
    amount = min(speed * dt, t['stored'])
    t['stored'] -= amount
    t['poured'] = amount
    R = float(t['rejection_radius'])
    area = np.pi * int(R * 2 / 3) ** 2
    if amount <= 0 or area <= 0:
        return []
    return [dict(type=0, col=float(t['col']), row=float(t['row']), radius=R, f3=0.0, amount=amount / area,
                 f5=0.0, sediment=0.0, temperature=0.0)]


TREE_FIRE_BASE = 16 + 1    # value added to FireIgnition: [[[0x1017989C]+0x20]+0x5C] (16 on Map 6, constant) + 1


def tree_fire_burst(c6, c7, col, row, block):
    """A fire tree's burst at the end of its burning phase (dustai 0x10082DD0 -> 0x10141550), on the vegetation
    components c6 / c7 in place: each of the 8 neighbouring cells (inside cells 33..478) holding a plant of type t
    is queued as a vegetation edit (col, row, FireIgnition[t] + 17), and the game applies that as fire flag set
    (component 6 bit 6) and fire time = that value (component 7 bits 16-31), seen live 2026-11-27 on Map 6 (type 1:
    57, then the normal fire step; types 2 / 6, FireIgnition 4800: 4817, burned at once). block = the 512-byte
    per-type vegetation table (vegsim.ganja_block; FireIgnition low bytes at +0x00, high bytes at +0x10).
    Returns the number of cells lit."""
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            c, r = int(col) + dc, int(row) + dr
            if (dc or dr) and 33 <= c < 479 and 33 <= r < 479:
                t = int(c6[r, c]) & 15
                if t:
                    v = (block[t] | block[0x10 + t] << 8) + TREE_FIRE_BASE
                    c6[r, c] = np.uint32(int(c6[r, c]) | 0x40)
                    c7[r, c] = np.uint32((int(c7[r, c]) & 0xFFFF) | (min(v, 0xFFFF) << 16))
                    n += 1
    return n


# ---------------------------------------------------------------- explosive trees (DustAi::BangTree)
def bang_tree_hole(T, col, row, hole_radius=30.0, hole_depth=20.0):
    """The crater an explosive tree blasts (dustai 0x1007D870, read 2026-11-27), on the ground surface T in place;
    returns the number of cells written. r = trunc(hole_radius x 2/3) (20 cells); every cell of the 2r x 2r square
    col - r .. col + r - 1, row - r .. row + r - 1 inside 33..478 gets, from the ground before the blast
    (c = the tree's cell, g = the cell; world distance d = 1.5 x cell distance):
        depth  = ((hole_radius - d) / hole_radius)^2 x hole_depth      (not clamped at d > hole_radius: the square's
                                                                        corners are dug too, up to ~3.4 deep)
        target = max(c - depth, 0)
        new    = g if target > g else max(target, g - depth), clamped to 0..255
    i.e. a bowl hanging from the tree's own ground height; no cell drops more than `depth`. The value is queued
    (0x10140E80 -> manager list +0x48) and written as the ground height (component 0) by 0x100DC897; the sand
    thickness (component 1) is left as it was, so the material is simply removed. Water depth is untouched."""
    f = np.float32
    R, D = f(hole_radius), f(hole_depth)
    inv = f(1) / R
    r = int(float(R) * (2 / 3))
    c0, r0 = int(col), int(row)
    cols = np.arange(c0 - r, c0 + r)
    rows = np.arange(r0 - r, r0 + r)
    cols = cols[(cols >= 33) & (cols < 479)]
    rows = rows[(rows >= 33) & (rows < 479)]
    if not len(cols) or not len(rows) or not (0 <= r0 < T.shape[0] and 0 <= c0 < T.shape[1]):
        return 0
    wx = lambda v: np.asarray(v, f) * f(1.5) + f(0.75) + f(-384)
    dx = (wx(c0) - wx(cols))[None, :]
    dy = (wx(r0) - wx(rows))[:, None]
    dist = np.sqrt(dx * dx + dy * dy).astype(f)
    depth = (((R - dist) * inv) ** 2 * D).astype(f)
    c = f(T[r0, c0])
    target = np.maximum(c - depth, f(0))
    g = T[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    new = np.where(target > g, g, np.maximum(target, g - depth))
    g[...] = np.clip(new, 0, 255).astype(f)
    return g.size


def bang_fire_danger(c6, radius_key='grid_perception_radius'):
    """grid_danger for bang_tree_step: True when a burning plant cell (vegetation component 6, fire flag bit 6; c6
    may be None) or a lava cell (fluid < 0, any depth) lies in the square of +-trunc(grid_perception_radius x 2/3)
    = +-6 cells around the tree. The game's grid sensor (kind 0x0F) was seen live to report burning plant cells
    (Waves) and lava cells down to 0.03 deep (Story 10, 2026-11-27), at offsets up to 6 cells on both axes at once
    (e.g. (+6, -6)), so a square, not a disc. The water tree's grid sensor is the same kind (radius_key
    'danger_radius': +-10 cells; seen live on Story 10 to fire on burning plants and lava, offsets up to (+10, +10))."""
    def danger(G, t):
        r = int(t.get(radius_key, 10.0) * 2 / 3)
        c0, r0 = int(t['col']), int(t['row'])
        y0, y1, x0, x1 = max(r0 - r, 0), min(r0 + r + 1, 512), max(c0 - r, 0), min(c0 + r + 1, 512)
        if (G['W'][y0:y1, x0:x1] < 0).any():
            return True
        return c6 is not None and bool(((c6[y0:y1, x0:x1] & 0x40) != 0).any())
    return danger


def _tree_cell_ok(W, t):
    """The rooting cell test (dustai 0x1007EE60): water cells need |fluid| <= water_threshold_max, lava cells
    |fluid| <= lava_threshold_max (both minimums 0); height 0..255 and ground type 'any' always pass."""
    v = float(W[int(t['row']), int(t['col'])])
    lim = t.get('lava_threshold_max', 0.0) if v < 0 else t.get('water_threshold_max', 20.0)
    return abs(v) <= lim and (v >= 0 or lim > 0)


def bang_tree_step(G, trees, fire_trees=(), water_trees=(), dt=1 / 31, rng=None, grid_danger=None):
    """One frame of every explosive tree (dicts from sav_sources.bang_trees / live_generators.bang_trees), decoded
    2026-11-27 (findings/water-lava-sim.md "Explosive trees"). Changes G['T'] in place (craters) and the tree lists
    (removals, seeds); returns the blasts of this frame as (col, row) tuples.
    Adult (state 3, the generic tree update 0x10080A50): danger = a burning fire tree within
    entities_perception_radius (20 world units; 0x1007DB10), or the tree was hit by another blast (+0x35C bit 2), or
    grid_danger(G, tree) (the game's grid sensor, kind 0x0F within grid_perception_radius: what it detects is NOT
    decoded; None = not modelled). While in danger the danger timer grows by dt (it is never reset); past
    explosion_delay (0.2 s) the tree switches to exploding. Entering the exploding state (0x10080590 -> 0x1007E090):
    one seed thrown (0x1007F4F0(1): ejection physics not modelled, the seed is placed 2-2.7 cells away like a
    water tree's seed), the crater (bang_tree_hole), and every entity within entities_perception_radius is hit
    (0x1007DC70): seeds are removed, rooted trees of any kind get the hit flag (fire / water trees have
    explosion_delay 0, so they go a frame later: a fire tree is removed after 1 s, a water tree after 5 s with its
    store, nothing poured; other entity kinds would be removed too, not modelled). The exploding tree is removed
    after explosion_duration (1 s). Seeds root after time_to_activate_seed if the cell test passes (water and lava
    <= 5; seed drift, rest and bury checks not modelled), grow for time_to_grow_tree (10 s), then are adults with
    no life time (-1 in every save). State 'ending' (4) trees are kept as they are (not modelled)."""
    rng = rng if rng is not None else np.random.default_rng()
    T = G['T']
    blasts, born, gone = [], [], []
    for t in trees:
        st = t.get('state')
        t['clock'] = t.get('clock', 0.0) + dt
        if st == 'seed':
            if t['clock'] >= t.get('time_to_activate_seed', 10.0) and _tree_cell_ok(G['W'], t):
                t['state'], t['clock'] = 'growing', 0.0
        elif st == 'growing':                     # a blast's hit flag stays: the tree goes off once grown (seen live)
            if t['clock'] >= t.get('time_to_grow_tree', 10.0):
                t.update(state='adult', clock=0.0, danger_timer=0.0, danger=False)
        elif st == 'exploding':
            if t['clock'] >= t.get('explosion_duration', 1.0):
                gone.append(t)
        elif st == 'ending':                      # withering (state 4): removed after time_to_degenerate_tree,
            if t['clock'] >= t.get('time_to_degenerate_tree', 10.0):   # leaving a seed (generate seeds when dying)
                gone.append(t)
                born += _water_tree_seeds(dict(t, nb_seeds_random=0), rng)
        elif st == 'adult':
            reach2 = (t['entities_perception_radius'] / 1.5) ** 2
            t['danger'] = bool(t.get('hit')) or any(
                _tree_burning(f) and (f['col'] - t['col']) ** 2 + (f['row'] - t['row']) ** 2 <= reach2
                for f in fire_trees) or bool(grid_danger and grid_danger(G, t))
            if t['danger']:
                t['danger_timer'] = t.get('danger_timer', 0.0) + dt
            if t.get('danger_timer', 0.0) > t['explosion_delay']:
                t['state'], t['clock'] = 'exploding', 0.0
                blasts.append(t)
            elif not _tree_cell_ok(G['W'], t):   # cell test fails (water / lava deeper than 5): withers
                t['state'], t['clock'] = 'ending', 0.0   # (0x10080A50 -> state 4 +0x328; seen live 2026-11-27)
    for t in blasts:                                   # the game runs each tree's blast inside its own update;
        bang_tree_hole(T, t['col'], t['row'], t['hole_radius'], t['hole_depth'])   # same-frame order = list order
        born += _water_tree_seeds(dict(t, nb_seeds_random=0), rng)
        reach2 = (t['entities_perception_radius'] / 1.5) ** 2
        for lst, kind in ((trees, 'bang'), (fire_trees, 'fire'), (water_trees, 'water')):
            for o in lst:
                if o is t or (o['col'] - t['col']) ** 2 + (o['row'] - t['row']) ** 2 > reach2:
                    continue
                if o.get('state') in ('seed', 1):
                    gone.append(o)
                elif kind == 'bang':
                    o['hit'] = True
                elif o.get('state') not in ('exploding',):
                    o['state'], o['explode_clock'] = 'exploding', 0.0   # delay 0: next frame, then removed
    for lst, life in ((fire_trees, 1.0), (water_trees, 5.0)):
        for o in list(lst):
            if o.get('state') == 'exploding':
                o['explode_clock'] = o.get('explode_clock', 0.0) + dt
                if o['explode_clock'] >= o.get('explosion_duration', life):
                    lst.remove(o)
    for o in gone:
        for lst in (trees, fire_trees, water_trees):
            if o in lst:
                lst.remove(o)
    for s in born:
        s.update(state='seed', clock=0.0, danger_timer=0.0, danger=False, hit=False)
    trees.extend(born)
    return [(t['col'], t['row']) for t in blasts]


def typed_fluid_args(e):
    """(rate, sediment, temperature) for fluid_modifier from a list entry of type 4-7, which names the
    modifier directly instead of picking it by sign and temperature as type 0 does. Fire trees (Map 6) push
    type 4 = RemoveWater with amount -0.166, radius 10 (seen live 2026-11-27); 5-7 not seen yet, signs
    assumed the same as type 0's (the DLL's Remove* modifiers take a negative rate)."""
    t, a = e['type'], abs(float(e.get('amount', 0.0)))
    sed = float(e.get('sediment', 0.0))
    if t == 4:
        return -a, sed, 0.0
    if t == 5:
        return a, sed, 0.0
    if t == 6:
        return -a, 0.0, max(float(e.get('temperature', 0.0)), 1.0)
    return a, 0.0, max(float(e.get('temperature', 0.0)), 1.0)


def modifier_calls(G, sources=(), repulsors=()):
    """The game's own modifier calls for springs / vents / drains and RepulseFluid entries, for
    galileo.Resident.modify: [(kind, center, param)] in the order apply_sources + repulse_fluid use.
    Param = (signed rate, +-1, sediment, temperature) for the fluid modifiers (checked 2026-11-21 against
    fluid_modifier on a random grid: identical but for last-bit lava temperatures), (amount, mode, sediment,
    0) for RepulseFluid. Sets each source's 'activity' (bury rule) on G as given, before any modifier
    runs (as apply_sources)."""
    calls = []
    for s in sources:
        act = bury_activity(G, s)
        s['activity'] = act
        rate = float(s['rate'] * act)
        if rate == 0:
            continue
        temp = float(s.get('temperature', 0.0))
        kind = ('AddLava' if rate > 0 else 'RemoveLava') if temp > 0 else ('AddWater' if rate > 0 else 'RemoveWater')
        calls.append((kind, (s['col'], s['row'], s['radius'], 0.0),
                      (rate, 1.0 if rate > 0 else -1.0, float(s.get('sediment', 0.0)), temp)))
    for e in repulsors:
        calls.append(('RepulseFluid', (e['col'], e['row'], e['radius'], 0.0),
                      (e['amount'], e['f5'], float(e.get('sediment', 0.0)), 0.0)))
    return calls


# ---------------------------------------------------------------- village repels (findings/water-lava-sim.md
# "Village repel push, exact" and "Village threat check, decoded")
REPEL = {   # per song: base size (m_radius, level 0), m_scaleFactor; peak: +6 radius at 1 + 2.2 (water only)
    'water': dict(size=40.0, scale=0.2, peak=True),
    'lava': dict(size=35.0, scale=0.05, peak=False),
}
REPEL_DURATION = 10.0        # s, m_minimalApplicationTime (episode length measured live: exactly 310 ticks)
VILLAGE_RADIUS = 19.0        # Village +0x1C8 (live, all Map 10 villages)
THREAT_RADIUS_FACTOR = 1.5   # m_threatDetectionRadiusFactor (Village_0.ai)
THREAT_WATER = 2.0           # m_threatWaterHeightThreshold: water depth
THREAT_LAVA = 1.5            # m_threatLavaHeightThreshold: lava depth
THREAT_MEMORY = 3.0          # s: 2 pi / (2 pi rad/s) x 3 (dustai 0x1014B210); a threat counts this long after the last hit
SWEEP_RATE = 2 * np.pi       # rad/s: the sampling angle (Village +0x2A0) turns once per second
WORLD_PER_CELL = 1.5         # world units per grid cell (dustai 0x101400C0: cell = trunc((world + 384) x 2/3), live)
REF_RADIUS = 10              # model only: reference height = lowest ground within this many cells (the game uses
                             # the lowest of the village's buildings, +0x2B4; live 0-2.2 below this estimate)
REPEL_BAR_TICKS = 61.8       # music bar measured live (~2.0 s): peak pushes on every bar, all villages in step


class VillageRepels:
    """Village repels as the game runs them (dustai, decoded 2026-11-27; findings "Village threat check, decoded").
    Threat sampling per village every update (Village update 0x10075650 -> 0x10074520 water, 0x100747D0 lava / fire):
      - while no repel of that song runs: 4 points at angles A + i x 90 deg and distances R x (1 - i/4), i = 0..3,
        R = 1.5 x 19 = 28.5 world units = 19 cells (the points sweep circles of 19, 14.25, 9.5 and 4.75 cells);
      - while it runs: 1 point at angle A and distance = the repel's current size in world units (40-48 water,
        35-36.75 lava; / 1.5 for cells);
      - A turns 2 pi per second. Points outside cells 33..478 are skipped.
      Water hit: depth > 2 and water surface (ground + depth) above the village's reference height. Lava hit:
      lava depth > 1.5. Fire hit: a burning plant (grid component 6 bit 0x40; needs `fire`).
      A song's threat flag is on while its last hit is less than 3 s old (water song: water hits; lava song: lava
      or fire hits).
    A song starts when its flag is on (the village knows it) and runs 10 s; at the end it restarts while the flag is
    still on. While on: radius = size x 2/3 (water truncated), amount 1; the size jumps to size x (1 + scale) once
    per episode at the first half bar and shrinks back in 1 s; water also pushes one tick of radius + 6 at amount 3.2
    on every bar. Not modelled: the villager's walk before the first push, the game's music clock (bars run from
    model tick 0), the exact reference height (the game uses the village's buildings)."""

    def __init__(self, rate=31.0):
        self.rate, self.state, self.threat = rate, {}, {}

    @staticmethod
    def _cell(col, row, ang, dist):
        # village at its cell centre; points in world units, converted to a cell by truncation as the game does
        c = int(np.floor(col + 0.5 + np.cos(ang) * dist / WORLD_PER_CELL))
        r = int(np.floor(row + 0.5 + np.sin(ang) * dist / WORLD_PER_CELL))
        return (c, r) if 33 <= c < 479 and 33 <= r < 479 else None

    @staticmethod
    def reference_height(G, col, row):
        T = G['T']
        r0, c0 = max(row - REF_RADIUS, 0), max(col - REF_RADIUS, 0)
        sub = T[r0:row + REF_RADIUS + 1, c0:col + REF_RADIUS + 1]
        yy, xx = np.ogrid[r0:r0 + sub.shape[0], c0:c0 + sub.shape[1]]
        return float(sub[(yy - row) ** 2 + (xx - col) ** 2 <= REF_RADIUS ** 2].min())

    def _hits(self, G, col, row, kind, ang, active_size, fire):
        W, T = G['W'], G['T']
        if active_size is None:
            R = THREAT_RADIUS_FACTOR * VILLAGE_RADIUS
            pts = [self._cell(col, row, ang + i * 0.5 * np.pi, R * (1 - 0.25 * i)) for i in range(4)]
        else:
            pts = [self._cell(col, row, ang, active_size)]
        ref = self.reference_height(G, col, row) if kind == 'water' else 0.0
        for pt in pts:
            if pt is None:
                continue
            c, r = pt
            w = float(W[r, c])
            if kind == 'water':
                if w > THREAT_WATER and float(T[r, c]) + w > ref:
                    return True
            elif -w > THREAT_LAVA or (fire is not None and int(fire[r, c]) & 0x40):
                return True
        return False

    def entries(self, G, reps, tick, fire=None):
        """This tick's RepulseFluid entries: entries with a 'totem' key are village songs and are gated by
        the rule above; others pass unchanged. fire: grid component 6 (uint32) for the lava song's fire test."""
        out, bar, dt = [], REPEL_BAR_TICKS, 1.0 / self.rate
        ang = SWEEP_RATE * tick * dt
        for e in reps:
            tid = e.get('totem')
            if tid is None:
                out.append(e)
                continue
            kind = 'water' if e['f5'] < 0 else 'lava'
            key, cfg = (tid, kind), REPEL[kind]
            st = self.state.get(key)
            running = st is not None and tick < st['end']
            col, row = int(round(e['col'])), int(round(e['row']))
            hit = self._hits(G, col, row, kind, ang, st['size'] if running else None, fire)
            age = 0.0 if hit else self.threat.get(key, 1e9) + dt
            self.threat[key] = age
            if not running:
                if age >= THREAT_MEMORY:
                    self.state.pop(key, None)
                    continue
                st = self.state[key] = dict(end=tick + REPEL_DURATION * self.rate, jump=None, size=cfg['size'])
            base = cfg['size']
            if st['jump'] is None and int(tick / bar - 0.5) != int((tick - 1) / bar - 0.5):
                st['jump'] = tick                                   # first half bar of the episode
            size = base
            if st['jump'] is not None:
                size = max(base, base * (1 + cfg['scale']) - base * cfg['scale'] * (tick - st['jump']) / self.rate)
            st['size'] = size
            radius = float(int(size * 2 / 3)) if kind == 'water' else size * 2 / 3
            amount = 1.0
            if cfg['peak'] and int(tick / bar) != int((tick - 1) / bar):
                radius, amount = float(int(size * 2 / 3) + 6), 3.2
            out.append(dict(e, radius=radius, amount=amount))
        return out


def tick(G, P, sources=(), repulsors=(), step_fn=None):
    """One full game tick as far as modelled: border ring reset, step(), sources, then RepulseFluid
    modifiers (village outposts; entries with col, row, radius, amount, f5). The 32-cell band
    outside the ring is kept unchanged. Not modelled: tides/waves, script changes, player powers.
    step_fn(G, P) replaces step() (e.g. the game's own DLL step; it must return fresh arrays)."""
    a, b = RING
    set_border_ring(G, P)
    out = (step_fn or step)(G, P)                      # fresh arrays: copy the old band into them
    band = ((slice(0, a), slice(None)), (slice(b + 1, None), slice(None)),
            (slice(a, b + 1), slice(0, a)), (slice(a, b + 1), slice(b + 1, None)))
    for k in ('T', 'soft', 'W', 'sed'):
        for sl in band:
            out[k][sl] = G[k][sl]
    for ys, xs in band:
        out['F'][:, ys, xs] = G['F'][:, ys, xs]
    apply_sources(out, sources, P)
    for e in repulsors:
        repulse_fluid(out, e['col'], e['row'], e['radius'], e['amount'], e['f5'])
    return out


def repulse_fluid(G, col, row, radius, amount, mode, skip=SKIP):
    """kModifier_RepulseFluid (SimulatorModifierTemplate::RepulseFluid in GalileoSimulatorWater.dll):
    pushes fluid out of a disc by rewriting the stored outflows. Village outposts use it (mode -1 =
    water; mode > 0 = lava). For each cell of the chosen fluid (water mode: depth >= 0, so dry cells too) with d <= radius:
      s = 1 - d / radius
      outflows = old outflows * (1 - s) + 256 * amount * s * (outward unit vector split into L/R/U/D)
      if their sum exceeds the depth x 256, all four are scaled down to the depth; then rounded.
    Nothing else changes. Matched against the DLL with tools/simstep/modstep.exe (galileo.modify)."""
    f32 = np.float32
    W, F = G['W'], G['F']
    r0, r1 = max(int(row - radius - 1), skip), min(int(row + radius + 1), 511 - skip)
    c0, c1 = max(int(col - radius - 1), skip), min(int(col + radius + 1), 511 - skip)
    ys, xs = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    dx, dy = (xs - f32(col)).astype(f32), (ys - f32(row)).astype(f32)
    d = np.sqrt(dx * dx + dy * dy).astype(f32)
    w = W[r0:r1 + 1, c0:c1 + 1]
    target = (w < 0) if mode > 0 else (w >= 0)          # water mode includes dry cells (their flows -> 0)
    inside = target & (d <= f32(radius))                # centre cell: old flows x 0, no push -> 0
    if not inside.any():
        return
    inv_r = f32(1) / f32(radius)                        # the DLL uses a refined rsqrt(r^2): last-bit
    s = (f32(1) - d * inv_r).astype(f32)                # differences can flip an exact .5 when rounding
    push = f32(256) * f32(amount) * s
    dd = np.where(d > 0, d, 1)
    comp = np.stack([np.maximum(-dx, 0), np.maximum(dx, 0), np.maximum(dy, 0), np.maximum(-dy, 0)]) / dd * push   # L R U D
    old = F[:, r0:r1 + 1, c0:c1 + 1].astype(f32)
    new = (old * (f32(1) - s) + comp).astype(f32)
    cap = np.abs(w) * f32(256)
    tot = new.sum(0)
    new = np.where(tot > cap, new * (cap / np.where(tot > 0, tot, 1)), new)
    region = F[:, r0:r1 + 1, c0:c1 + 1]
    region[:, inside] = np.rint(new[:, inside]).astype(region.dtype)


EDGE_LINE = (36, 476)      # the wave manager writes the line 36 cells in from each edge, cells 36..475


def edge_cells(side):
    """(ys, xs) of the 440 cells the wave manager walks on edge `side`, in buffer order i = 36..475
    (dustai 0x10111351..): 0 = column 36 (y = i), 1 = row 476 (x = i), 2 = column 476 (y = 512 - i),
    3 = row 36 (x = 512 - i)."""
    i = np.arange(36, 476)
    if side == 0:
        return i, np.full_like(i, 36)
    if side == 1:
        return np.full_like(i, 476), i
    if side == 2:
        return 512 - i, np.full_like(i, 476)
    return np.full_like(i, 36), 512 - i


def apply_edges(G, border_level, profiles):
    """Waves / tsunamis (dustai wave manager 0x10111250): for every edge with an active wave, set the fluid
    of its line cells to max(border level - ground, 0) + profile[i]. profiles: {side: 512 floats}."""
    W, T = G['W'], G['T']
    for side, prof in profiles.items():
        ys, xs = edge_cells(side)
        W[ys, xs] = (np.maximum(np.float32(border_level) - T[ys, xs], 0) + prof[36:476]).astype(np.float32)
