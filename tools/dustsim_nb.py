"""Compiled (numba) version of dustsim.step_numpy: same rules, same float32 operations in the same
order, so the results are bit-identical (tools/check_nb.py compares the two; tools/compare_sim.py
compares with the game's DLL). Imported by dustsim.step when numba is available.

No fastmath: every float32 operation must round exactly as numpy does. Constants are passed as a
float32 array (a Python float literal would turn the arithmetic into float64 under numba).
Passes: A = Step1 per cell (erosion / deposit, lava cooling / solidifying); B = outflows, lateral
erosion; C = what each cell receives from its neighbours; D = the row-by-row lava/water mixing
(each row depends on the row below, so it runs in order).
"""
import numba
import numpy as np

f32 = np.float32

_SCALARS = ['SoftGroundErosionFactor', 'SoftGroundErosionThreshold', 'SoftGroundErosionMax',
            'SoftGroundErosionWaterHeightMax', 'SedimentDepotFactor', 'SedimentDepotThreshold',
            'SedimentDepotMax', 'TemperatureDampingFactor', 'WaterHeightTemperatureDampingFactor',
            'WaterHeightTemperatureDampingThreshold', 'TemperatureDampingThreshold', 'WaterTemperatureMax',
            'LavaFluxMask', 'WaterFluxMask', 'LavaFluxThreshold',
            'LateralErosionSlopeThresholdLeft', 'LateralErosionSlopeThresholdRight',
            'LateralErosionSlopeThresholdUp', 'LateralErosionSlopeThresholdDown',
            'ErosionSumFluxSumFactor', 'ErosionSumFluxSumThreshold']
_RAMPS = [('LavaFluxDampingThresholdMin', 'LavaFluxDampingThresholdMax', 'LavaFluxDampingMin', 'LavaFluxDampingMax'),
          ('SedimentRatioThreshold0', 'SedimentRatioThreshold1', 'WaterFluxDamping', 'WaterFluxDampingMin'),
          ('LavaFluidityThreshold0', 'LavaFluidityThreshold1', 'LavaFluidityValue0', 'LavaFluidityValue1'),
          ('SedimentRatioThreshold0', 'SedimentRatioThreshold1', 'WaterFluidity', 'WaterFluidityMin')]
# indices into the constant array
(SEF, SET, SEM, SEWHM, SDF, SDT, SDM, TDF, WHTDF, WHTDT, TDT, WTM, LFM, WFM, LFT,
 THR_L, THR_R, THR_U, THR_D, KLAT, ESFST) = range(21)
RAMP0 = 21      # 4 ramps x (x0, x1 - x0, y0, y1 - y0): lava damping, water damping, lava fluidity, water fluidity
WET = RAMP0 + 16
NCONST = WET + 1


def constants(P):
    """Parameter dict -> float32 constant array, rounded exactly as dustsim.step_numpy's numpy
    expressions round them (ramp: (x1 - x0) and (y1 - y0) in double, then float32)."""
    c = np.zeros(NCONST, f32)
    for i, n in enumerate(_SCALARS):
        c[i] = P[n]
    for j, (a, b, ya, yb) in enumerate(_RAMPS):
        x0, x1, y0, y1 = float(P[a]), float(P[b]), float(P[ya]), float(P[yb])
        c[RAMP0 + 4 * j:RAMP0 + 4 * j + 4] = (x0, x1 - x0, y0, y1 - y0)
    c[WET] = 1 / 255
    return c


@numba.njit(inline='always')
def _ramp(x, c, i):
    t = (x - c[i]) / c[i + 1]
    t = min(max(t, f32(0)), f32(1))
    return c[i + 2] + t * c[i + 3]


@numba.njit(parallel=True, cache=True)
def _step(T0, So0, W0, sed0, F, c):
    N, M = T0.shape
    z, one = f32(0), f32(1)
    T1 = np.empty_like(T0); So1 = np.empty_like(T0); H1 = np.empty_like(T0)
    Sd1 = np.empty_like(T0); Tm1 = np.empty_like(T0); lava1 = np.empty((N, M), np.bool_)

    # ---------------- A: Step1 (water: erosion and deposit; lava: cooling and solidifying)
    for y in numba.prange(N):
        for x in range(M):
            w = W0[y, x]
            lava = w < z
            water = not lava
            H = abs(w)
            Sd = z if lava else sed0[y, x]
            Tm = sed0[y, x] if lava else z
            So = So0[y, x]
            T = T0[y, x]
            sumf = f32(F[0, y, x] + F[1, y, x] + F[2, y, x] + F[3, y, x]) / f32(256)
            wet = water and H > c[WET]
            v = sumf / np.sqrt(H) if wet else z
            e = min(max(c[SEF] * (v - c[SET]), z), c[SEM])
            if not (wet and H <= c[SEWHM]):
                e = z
            e = min(e, So)
            So = So - e; T = T - e; Sd = Sd + e; H = H + e
            d = min(max(c[SDF] * (c[SDT] - v) * Sd, z), c[SDM])
            d = max(d, Sd - H)
            d = min(d, Sd) if water else z
            So = So + d; T = T + d; Sd = Sd - d
            if water:
                H = max(H - d, z)
            k = c[TDF] + c[WHTDF] * max(z, c[WHTDT] - H)
            if lava and Tm >= c[TDT]:
                Tm = max(Tm - k, z)
            if lava and Tm <= c[WTM]:                           # solidifies
                T = T + H; H = z; Tm = z; lava = False
            T1[y, x] = T; So1[y, x] = So; H1[y, x] = H; Sd1[y, x] = Sd; Tm1[y, x] = Tm; lava1[y, x] = lava

    # ---------------- B: outflows (each cell with its own material's parameters), lateral erosion
    Fq = np.empty((4, N, M), np.int32)
    T2 = np.empty_like(T0); So2 = np.empty_like(T0); Sd2 = np.empty_like(T0)
    ratio2 = np.empty_like(T0); outs = np.empty_like(T0); rem = np.empty_like(T0)
    for y in numba.prange(N):
        for x in range(M):
            H = H1[y, x]; T = T1[y, x]; Sd = Sd1[y, x]; lava = lava1[y, x]
            ratio = Sd / H if H > z else z
            if lava:
                damp = _ramp(Tm1[y, x], c, RAMP0)
                fluid = _ramp(Tm1[y, x], c, RAMP0 + 8)
                cap = c[LFM]
            else:
                damp = _ramp(ratio, c, RAMP0 + 4)
                fluid = _ramp(ratio, c, RAMP0 + 12)
                cap = c[WFM]
            S = T + H
            # neighbours L (x-1), R (x+1), U (y+1), D (y-1); 0 outside the grid
            SL = T1[y, x - 1] + H1[y, x - 1] if x > 0 else z
            SR = T1[y, x + 1] + H1[y, x + 1] if x < M - 1 else z
            SU = T1[y + 1, x] + H1[y + 1, x] if y < N - 1 else z
            SD = T1[y - 1, x] + H1[y - 1, x] if y > 0 else z
            f0 = min(max(z, damp * (f32(F[0, y, x]) / f32(256)) + fluid * (S - SL)), cap)
            f1 = min(max(z, damp * (f32(F[1, y, x]) / f32(256)) + fluid * (S - SR)), cap)
            f2 = min(max(z, damp * (f32(F[2, y, x]) / f32(256)) + fluid * (S - SU)), cap)
            f3 = min(max(z, damp * (f32(F[3, y, x]) / f32(256)) + fluid * (S - SD)), cap)
            if lava and ((f0 + f1) + f2) + f3 < c[LFT]:         # lava yield: too little push, no flow
                f0 = z; f1 = z; f2 = z; f3 = z
            tot = ((f0 + f1) + f2) + f3
            if tot > H:
                s = H / tot
                f0 = f0 * s; f1 = f1 * s; f2 = f2 * s; f3 = f3 * s
            Fq[0, y, x] = np.int32(np.rint(f0 * f32(256)))
            Fq[1, y, x] = np.int32(np.rint(f1 * f32(256)))
            Fq[2, y, x] = np.int32(np.rint(f2 * f32(256)))
            Fq[3, y, x] = np.int32(np.rint(f3 * f32(256)))
            # lateral erosion (sand slumping into mud), water / dry cells
            TL = T1[y, x - 1] if x > 0 else z
            TR = T1[y, x + 1] if x < M - 1 else z
            TU = T1[y + 1, x] if y < N - 1 else z
            TD = T1[y - 1, x] if y > 0 else z
            drop = c[KLAT] * ((T - TL) - c[THR_L])
            drop = max(drop, c[KLAT] * ((T - TR) - c[THR_R]))
            drop = max(drop, c[KLAT] * ((T - TU) - c[THR_U]))
            drop = max(drop, c[KLAT] * ((T - TD) - c[THR_D]))
            fs = ((f0 + f1) + f2) + f3
            share = fs / H if H > z else one
            if not share > c[ESFST]:
                share = z
            So = So1[y, x]
            lat = z if lava else min(max(drop, z) * share, So)
            So = So - lat; T = T - lat
            HL = H + lat
            ratio = (Sd + lat) / HL if HL > z else z
            Sd = Sd + lat
            r = HL - fs                                          # what stays in the cell
            if lava and abs(r) < f32(1e-6):
                r = z
            T2[y, x] = T; So2[y, x] = So; Sd2[y, x] = Sd; ratio2[y, x] = ratio; outs[y, x] = fs; rem[y, x] = r

    # ---------------- C: what each cell receives (from neighbour n, through n's outflow towards it)
    Sn = np.empty_like(T0); lv0 = np.empty_like(T0); wv0 = np.empty_like(T0)
    num0 = np.empty_like(T0); fD = np.empty_like(T0)
    for yp in numba.prange(N):
        y = np.int64(yp)                 # (prange's index is unsigned: y - 1 would become float)
        for x in range(M):
            sed_in = z
            lsum = z; wsum = z; hsum = z
            for kk in range(4):          # L, R, U, D: neighbour and its direction towards this cell
                if kk == 0:
                    ny, nx, dr = y, x - 1, 1
                elif kk == 1:
                    ny, nx, dr = y, x + 1, 0
                elif kk == 2:
                    ny, nx, dr = y + 1, x, 3
                else:
                    ny, nx, dr = y - 1, x, 2
                if ny < 0 or ny >= N or nx < 0 or nx >= M:
                    fin = z; nl = False; ntm = z; nr = z
                else:
                    fin = f32(Fq[dr, ny, nx]) / f32(256)
                    nl = lava1[ny, nx]; ntm = Tm1[ny, nx]; nr = ratio2[ny, nx]
                sq = np.rint(fin * (z if nl else nr) * f32(4096))
                sed_in = sed_in + sq
                if kk < 3:
                    lsum = lsum + (fin if nl else z)
                    wsum = wsum + (z if nl else fin)
                    hsum = hsum + (fin * ntm if nl else z)
                else:
                    fD[y, x] = fin if (y > 0) else z     # inflow from row y - 1 (any fluid)
            lava = lava1[y, x]
            r = rem[y, x]
            Sn[y, x] = (z if lava else Sd2[y, x] - outs[y, x] * ratio2[y, x]) + sed_in / f32(4096)
            lv0[y, x] = (r if lava else z) + lsum
            wv0[y, x] = (z if lava else r) + wsum
            num0[y, x] = (r * Tm1[y, x] if lava else z) + hsum

    # ---------------- D: mixing, row by row (inflow from row y - 1 classified by its NEW state)
    Tmax = c[WTM]
    Wn = np.empty_like(T0); C3 = np.empty_like(T0)
    below_lava = np.zeros(M, np.bool_); below_C3 = np.zeros(M, f32)
    for y in range(N):
        for x in range(M):
            f = fD[y, x]
            bL = below_lava[x]
            lv = lv0[y, x] + (f if bL else z)
            wv = wv0[y, x] + (z if bL else f)
            bl = lv > z
            tn = (num0[y, x] + (f * below_C3[x] if bL else z)) / (lv + wv) if bl else z
            if bl and tn <= Tmax:                       # diluted below WaterTemperatureMax: hardens
                T2[y, x] = T2[y, x] + lv
                bl = False
            C3[y, x] = tn if bl else Sn[y, x]
            Wn[y, x] = -lv if bl else wv
            below_lava[x] = bl
            below_C3[x] = C3[y, x]
    return T2, So2, Wn, C3, Fq


_CONST_CACHE = {}


def step(G, P):
    """Same as dustsim.step_numpy (bit-identical), compiled."""
    key = tuple(sorted((k, float(v)) for k, v in P.items() if isinstance(v, (int, float, np.floating))))
    c = _CONST_CACHE.get(key)
    if c is None:
        c = _CONST_CACHE[key] = constants(P)
    a = lambda k: np.ascontiguousarray(G[k], f32)
    T, So, W, sed, F = _step(a('T'), a('soft'), a('W'), a('sed'), np.ascontiguousarray(G['F'], np.int32), c)
    return {'T': T, 'soft': So, 'W': W, 'sed': sed, 'F': F}
