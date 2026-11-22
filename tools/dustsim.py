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
    SoftGround (col, row, radius, amount). Other types are skipped."""
    for e in entries:
        t = e.get('type', 0)
        if t == 0:
            apply_sources(G, [dict(e, rate=e.get('amount', e.get('rate', 0)))] if 'amount' in e else [e], P)
        elif t == 1:
            repulse_fluid(G, e['col'], e['row'], e['radius'], e['amount'], e['f5'])
        elif t in (2, 3):                                   # HardGround, SoftGround
            ys, xs = disc(e['col'], e['row'], e['radius'])
            ground_modifier(G, ys, xs, e['amount'], soft=(t == 3))


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
