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
    fin = Fq.astype(f32) / f32(256)                  # what receivers get (rounded)
    def gather(a):                                   # sum over the 4 neighbours of a[opposite dir]
        return sum(shift(a[OPP[kk]], *NB[kk]) for kk in NB)
    lava_in = gather(np.where(lava, fin, 0))
    water_in = gather(np.where(lava, 0, fin))
    heat_in = gather(np.where(lava, fin * Tm, 0))
    sq = np.rint(fin * np.where(lava, 0, ratio) * f32(4096))   # sediment, 1/4096 units
    sed_in = gather(sq).astype(f32) / f32(4096)
    lava_vol = np.where(lava, rem, 0) + lava_in
    water_vol = np.where(lava, 0, rem) + water_in
    becomes_lava = lava_vol > 0
    # lava + water: the water boils away but dilutes the lava's temperature
    Tn = np.where(becomes_lava, (np.where(lava, rem * Tm, 0) + heat_in) /
                  np.where(becomes_lava, lava_vol + water_vol, 1), 0)
    Sn = np.where(lava, 0, Sd - out * ratio) + sed_in
    Wn = np.where(becomes_lava, -lava_vol, water_vol)
    C3 = np.where(becomes_lava, Tn, Sn)
    out_d = {'T': T, 'soft': So, 'W': Wn.astype(f32), 'sed': C3.astype(f32), 'F': Fq}
    if DEBUG:
        out_d['_dbg'] = dict(lava_vol=lava_vol, water_vol=water_vol, rem=rem, lava_in=lava_in, water_in=water_in, heat_in=heat_in, Tm=Tm, lava=lava)
    return out_d


def from_grid(grid):
    """galileo.py grid (8, 512, 512) uint32 -> dict for step()."""
    gf = lambda c: grid[c].view(np.float32).copy()
    lr, ud = grid[4], grid[5]
    F = np.stack([lr & 0xFFFF, lr >> 16, ud & 0xFFFF, ud >> 16]).astype(np.int32)
    return {'T': gf(0), 'soft': gf(1), 'W': gf(2), 'sed': gf(3), 'F': F}



