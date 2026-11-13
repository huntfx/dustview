# How From Dust computes water, sand and lava

Status: **answered.** The complete per-step update is known and re-implemented in Python ([../tools/dustsim.py](../tools/dustsim.py)). It was checked against the game's own simulation DLL, run outside the game on the same inputs: water, sand, sediment and lava match to float rounding (fluxes identical in all but a handful of cells per step). Open edge cases are listed under TODO.

## What the game does each tick
One tick = one call of the simulation step (30 per second at the default frame cap). There is no time step: every rate below is "per tick".

- **The grid:** 512 × 512 cells. Each cell holds terrain height, how much of that is soft ground (sand), a **fluid height**, one extra value, and four outflows (to the left, right, up and down neighbours).
- **Water and lava are the same fluid.** A negative fluid height means lava. The extra value is the suspended sand for water and the temperature for lava.
- **Flow (water):** each cell pushes fluid to its four direct neighbours: `new outflow = 0.999 × last tick's outflow + 0.032 × (my surface − neighbour's surface)`, never negative, and scaled down so a cell never sends more than it holds. There are no diagonal pipes. Water reaches a diagonal neighbour in two ticks via a side neighbour, so a puddle on flat ground spreads as a slightly diamond-shaped blob.
- **Sand is carried inside the water column.** Flow speed = total outflow / √depth. Fast (> 0.4), shallow (≤ 4) water picks sand up, `0.002 × (speed − 0.4)`, at most 0.01 per tick. Slow water (< 0.6) drops `0.2 × (0.6 − speed)` of its sand per tick, i.e. 12% when still. Muddy water flows worse: from 50% sand upwards, damping drops towards 0.7 and fluidity towards 0.03.
- **Steep sand slumps:** sand more than 1 unit higher than a neighbour turns `0.05 × (drop − 1)` per tick into mud (a column of pure sand) that then flows and settles. This happens only when the cell is dry or at least 10% of its water drains away that tick; under standing water, slopes hold.
- **Lava flow:** same rule with fluidity 0.01 at ≤ 12.5° rising to 0.02 at ≥ 125°, damping 0.95 → 0.99 over the same range, and a **yield threshold**: if the total push is under 0.05, the lava doesn't move at all, so thin or cool lava stays put.
- **Lava cooling:** while at or above 12.5°, temperature drops `0.0375 + 1.0 × max(0, 1 − depth)` per tick. Deep lava loses 0.0375 per tick (from 255° that's ~6,500 ticks ≈ 3.6 min); lava half a unit deep loses 0.54 per tick. At or below 12.5° it turns into rock of its own depth.
- **Lava meets water:** water flowing into lava boils away and cools the lava as if mixed in at 0°. Lava flowing into water boils that water off too but is diluted by it (`temperature = lava heat / (lava + water)`), so a little lava entering a lake hardens almost at once.
- **Border:** the outer 32 cells never change; the ring just inside is reset to the map's border values every tick (sea level, or a drain).
- **Outside the step** (seen in the running game, not part of this DLL code): waterfall sources are topped up each tick, a sea edge is driven up and down in a ~12 s wave/tide cycle, and the player's powers edit the grid. **Jellify** doesn't stop the simulation: it swaps the water parameters map-wide for 60 s (the power's timer) (fluidity 0.032 → 0.0032, damping 0.999 → 0.99, flow capped at 0.005 per side per tick), so water almost stops but still creeps.
- **Rendering** is separate from all this: the simulation only knows the 512 × 512 heights. The smooth look comes from the display jobs in the same DLL (`SetupDisplayListTemplate` builds the terrain/water meshes with LOD, `UpdateTextureTemplate` uses bilinear fetches, normals use `NormalScaleGround/Water` = 2). Judged from function and parameter names; that code was not read.

## Where things live
| What | Where |
|---|---|
| Simulation code | `SSE41/GalileoSimulatorWater_9_9.dll` (also SSE3/SSSE3 builds), `SimulatorWaterIterator::Step1` (per-cell: erosion, deposit, cooling), `Step2_split` (flow, lateral erosion, transport), `Step4` (store), driven by `Run12345_*`. PDB shipped, so function names, struct layouts and enums are readable ([../tools/pdb.py](../tools/pdb.py)) |
| Parameter preprocessing | `SimulatorWaterIterator::ConstParam::ConstParam(SimulatorParam*)` (`0x100049C5` in the SSE41 worker) |
| Parameter values | `BinPC/Wor_BD00993D.lin.bf`, entry `SpecieFluid_0.ai`: serialized `DustAi::SpecieFluid` with `m_fluidParameters[2]`, `m_slowWaterParameters[2]`, `m_slowLavaParameters[2]` |
| Field names/offsets of the stored structs | `AInative/bin/dll/dustai.win32.f.dll`, reflection registration `0x100F9DE0`–`0x100FA5E9` |
| Copy into the simulator (with unit scaling) | dustai `0x100FBA20`: one call per value to the setter `0x10049A40(index, value)` (queues index/value pairs for a deferred write) |

## Grid cell (32 bytes, 8 components)
From the PDB enum `ISimulatorWaterConst::tGridDataOffset`:

| # | Component | Notes |
|---|---|---|
| 0 | `kTerrainSurfaceHeight` | top of the ground (rock + soft ground) |
| 1 | `kSoftGroundQuantity` | sand thickness (part of component 0) |
| 2 | `kWaterHeight` | fluid depth; **negative = lava** |
| 3 | `kTemperatureSedimentQuantity` | suspended sand (water) or temperature (lava) |
| 4 | `kOutFluxLeftRight` | two u16 outflows, 1/256 units (low = Left, high = Right) |
| 5 | `kOutFluxUpDown` | two u16 outflows (low = Up, high = Down) |
| 6 | `kRandomGanja` | random + vegetation ("ganja"); low nibble indexes `ErosionBlocking` |
| 7 | `kGanjaTimeFire` | vegetation timer + fire |

Directions: Left = x−1, Right = x+1, **Up = y+1, Down = y−1** (found by matching the DLL). Memory layout: float index = `(y << 12) + (x >> 4)·128 + comp·16 + (x & 15)`, i.e. each row is 32 groups of 16 cells, each group 8 components × 16 values (manager accessors `ISimulatorWater::fGetFluidHeight` `0x10001F10` etc.). `fGetTemperature` returns component 3 only if component 2 < 0, `fGetSedimentQuantity` only if ≥ 0.

## One simulation step, in order
`S = terrain + |fluid|` is the surface height. Values are from the start of the step unless noted. Constants are the values the game actually uses (see the parameter section).

**Step 1, water cells:**
1. Speed `v = (sum of the 4 stored outflows) / sqrt(depth)` (0 where depth ≤ 1/255).
2. Erosion `e = clamp(0.002·(v − 0.4), 0, 0.01)` where depth ≤ 4, at most the soft ground present (blocked on cells whose vegetation byte maps to an `ErosionBlocking` mask). Sand goes from the ground into the column: terrain −e, soft −e, depth +e, sediment +e.
3. Deposit `d = clamp(0.2·(0.6 − v)·sediment, 0, 0.5)`, using the sediment after step 2: terrain +d, soft +d, depth −d, sediment −d. (If sediment exceeds the fluid, the excess drops at once; not reachable in play.)

**Step 1, lava cells:**
4. Cooling, while temperature ≥ 12.5: `temperature −= 0.0375 + 1.0·max(0, 1 − depth)` (values confirmed in the running game).
5. Lava at ≤ 12.5° solidifies: terrain += depth, fluid = 0.

**Step 2, flow:**
6. For each direction: `f = max(0, damping·f_old + fluidity·(S − S_neighbour))`, then `f = min(f, cap)`.
   - Water: damping 0.999 → 0.7 and fluidity 0.032 → 0.03, ramping linearly as the sediment ratio `sediment/depth` goes from 0.5 to 1.0. Cap `WaterFluxMask` (10000, i.e. none).
   - Lava: fluidity 0.01 → 0.02 and damping 0.95 → 0.99, ramping as temperature goes from 12.5° to 125°. Cap `LavaFluxMask`. **Yield:** if the four outflows add up to less than 0.05, the lava doesn't flow at all.
7. If the outflows add up to more than the cell's depth, all four are scaled down to exactly empty it.
8. Stored outflow = `round(f·256)` (u16).

**Step 2, sand slumping ("lateral erosion"), non-lava cells:**
9. `lat = 0.05·max over neighbours of (terrain − neighbour terrain − 1)`, clamped to ≥ 0 and to the soft ground present. It is multiplied by the share of the cell's fluid leaving this step (`sum(f)/depth`; a dry cell counts as 1) and dropped entirely if that share is ≤ 0.1. The amount leaves the ground (terrain −lat, soft −lat) and is added to the fluid column as pure sediment (depth +lat, sediment +lat), mixed in before the outflow leaves.

**Step 2, transport:**
10. Water: sender loses `sum(f)`, receiver gets the stored (rounded) flux. Sediment leaves at the concentration `(sediment + lat)/(depth + lat)`. The receiver gets `round(stored flux × concentration × 4096)/4096`; the sender loses the unrounded amount.
11. Lava carries its temperature. Per receiving cell: `lava = own remaining lava + lava inflow`, `water = own remaining water + water inflow`.
    - No lava → it's a water cell as above.
    - Any lava → it becomes (or stays) lava with **depth = lava only** (the water boils away) and **temperature = lava heat / (lava + water)**. Water entering lava therefore cools it as if mixed in at 0°.
    - **Exception (found live 2026-11-13 against the DLL):** if that mixed temperature is **≤ `WaterTemperatureMax` (12.5)**, the lava hardens at once: the terrain rises by the lava volume and the cell stays **water** (own + incoming water). Typical at a lava front running into shallow water. Before this rule `dustsim` turned ~230 such cells per tick into cold lava; after it, Python and DLL differ in 4–5 fluid cells per tick on a busy Map 4 frame.

Border: the outermost processed ring (row/column 32 and 479) is overwritten with the map's border values every step, and the 32 cells outside it are never touched. See [Border](#border).

## Water (evidence)
- Single columns on flat ground (depth 0.1, 1, 5, 20, three steps each): e.g. depth 5, step 2: `0.999·41/256 + 0.032·(4.36 − 0.16016) = 0.29439` per side, centre 4.36 → 3.18244 in the game. The outflow cap shows at step 5 (centre 0.0315 empties exactly) and on a raised pillar (0.5 of water, 4 × 0.125 out).
- Random 512 × 512 landscape, no sand, 10 steps compared one at a time: outflows identical in every cell (one rounding tie in 1.8 M cell-steps), depth within 2·10⁻⁶.
- **Quirk:** the sender subtracts the unrounded flux, the receiver adds the rounded one, so the game's water volume drifts slightly (a lone 0.1 column became 0.1028 after one step).

## Sand and sediment (evidence)
- **Sediment is part of the column:** in still water (depth 0.5–4, sediment 0.01–3) the game moves exactly `0.12 × sediment` from the column to the ground each step, lowering both depth and sediment. Picking up and dropping never changes the surface height, so they don't push water around by themselves.
- Uniform-flux sweep (depth 1/4/6, sediment 0/0.1/0.4, outflows 0–128): erosion `0.002·(v−0.4)`, none above depth 4, deposit `0.2·(0.6−v)·sediment` computed after erosion (v = 0.5, sediment 0.1: `0.02 × 0.1002 − 0.0002 = 0.001804`, game 0.001805).
- 60-row concentration sweep: received sediment is `round(flux × concentration × 4096)/4096` in every row.
- Sand pillars (height 0.1–5) and cliffs: a dry pillar of 3 loses exactly 0.1 (`0.05 × (3 − 1)`), next to it a 1.5 cell loses 0.025. Under water the amount scales with the outflow share: own 0.1 / neighbours 2.9 → share 0.256 → 0.0256; own 2 / neighbours 0 → 0.032. The 0.1 share threshold separates every slumping cell from every non-slumping one in a random landscape (non-slumping shares 0.077–0.0999, slumping ≥ 0.128).
- Random 512 × 512 landscapes with sand and sediment ratios up to 0.9: mean error ~10⁻⁷, 1–5 flux rounding ties per step in 178k cells.

## Lava (evidence)
Lava tests used temperatures up to 255, the range seen in real saves (below).
- Cooling and solidifying: columns at several temperatures and depths cool by `rate + factor·max(0, 1 − depth)` per step and turn to terrain of their own depth once at or below the threshold (with the stored, unscaled values: 101° / depth 0.5 → 100.2 → rock).
- Fluidity ramp and yield: 5-deep columns flow at 0.0100 (101°), 0.0111 (200°), 0.015 (550°), 0.02 (≥ 1000°) per unit height difference (unscaled thresholds). An 8 × 10 temperature × depth sweep shows flow starts exactly where the four outflows reach 0.05 (e.g. 4 × 0.0125).
- Temperature mixing: receiving cell `(0.0547 × 198.455 + 0.1055 × 199.4)/0.1602 = 199.077`, game 199.0772.
- Water into lava: lava 0.5 at 199.9° receiving 12/256 of water stays 0.5 deep at `199.9 × 0.5/(0.5 + 0.046875) = 182.77`, game 182.7657.
- Lava into water: a 1.0-deep water cell receiving 20/256 of lava at 249.96° becomes lava 0.078 deep at `0.078125 × 249.96/1.078 = 18.11`, game 18.1131.
- Random lava landscape, 5 steps: outflows identical in every cell, depth within 10⁻⁶, temperature within ~10⁻³.
- Real maps, start states from `DUST/DustSave/Ingame/*.sav`, 10 steps each, compared one at a time ([../tools/compare_sim.py](../tools/compare_sim.py)):

  | Map | Lava / water cells | Flux mismatches per step | Cells with fluid differing by > 0.001 |
  |---|---|---|---|
  | `ALPHA_Map1_Erosion` | 0 / 227,923 | 0–2 | 0–1 |
  | `ALPHA_Map9_TheGreatCaldeira` | 5,066 / 146,942 | 0–2 | 0–4 |
  | `ALPHA_Map10_FireAndWater` | 3,705 / 113,849 | 0–3 | 0–4 |

  The remaining differences are rounding ties and the drained-lava edge case under Quirks.

### Temperature range in real data
Lava temperatures in all shipped maps and the player's 13 saves are between ~12.5 and **255.0** (component 3 / `.sav` block 3; e.g. Map 9 median 200, Map 4 max 125, many maps max exactly 255). That fits the runtime parameters below: solidify at 12.5, full fluidity at 125, temperatures stored as 8.8 fixed point (max 255.99, `mv_255`).

**The built-in editor's lava temperature is in the stored units, i.e. 8× the simulator's.** The editor's maximum is **2040** (reported from play) = 255 × 8, the simulator's ceiling. Observed in play: lava placed at 100 turned to rock almost instantly (100/8 = 12.5, the solidify point); lava placed at 700 (87.5 internally) stayed hot after flattening out. Deep lava at 87.5 needs ~2,000 ticks ≈ 67 s to reach 12.5. A live read during that session showed ~50 lava cells between 75 and 100°.

## Parameters
`SimulatorParam` (`0x360` bytes) holds `mav_ParamF` at `+0x10`: 56 floats indexed by the PDB enum `kSimulatorParamF_*`. In the manager object (`ISimulatorWater`) the same array sits at `+0x20`. Other fields the step reads: `+0x340` log2 width (9), `+0x348` border skip, `+0x350` input grid, `+0x354` output grid (`PointerManager` constructor `0x100048C9`).

### Stored values and what the simulator gets
Serialized record format in `SpecieFluid_0.ai`: `u32 nameHash, u32 type, u32 size, value`; a struct is type `0x420` with size `0x80000000 | bytes` followed by its member records; floats are type `0x410`; arrays `0x1020`. Name hash (dustai `0x100342C0`, reimplemented and checked against the class hash `0x8C751F57`; also in [../tools/galileo.py](../tools/galileo.py)):
```python
def lyn_hash(s, bits=32):
    v = 0
    for c in s.encode():
        x = ((v * 61) ^ ((c - 32) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        v = ((x >> bits) ^ x) & ((1 << bits) - 1)
    return v
```
The copy routine (dustai `0x100FBA20`) names each Galileo index explicitly, which confirms the name mapping, and **multiplies every temperature value by 0.125**. "Runtime" below is what the simulator receives with `m_fluidParameters[0]` ([../tools/galileo.py](../tools/galileo.py) `fluid_params()`).

| Stored field (`m_simulatorParam_…`) | Stored | Galileo index | Runtime |
|---|---|---|---|
| WaterFluxDamping | 0.999 | 0 | 0.999 |
| WaterFluidityValue1 | 0.032 | 1 `WaterFluidity` | 0.032 |
| SoftGroundErosionThreshold / Factor / Max | 0.4 / 0.002 / 0.01 | 2 / 3 / 18 | same |
| SedimentTransportThreshold / Factor | 0 / 1 | 4 / 5 | same (no visible effect at these values) |
| SedimentDepotThreshold / Factor / Max | 0.6 / 0.2 / 0.5 | 6 / 7 / 19 | same |
| SedimentRatioThreshold0 / 1 | 0.5 / 1.0 | 8 / 9 | same |
| WaterFluxDampingMin / WaterFluidityMin | 0.7 / 0.03 | 10 / 11 | same |
| LavaFluidityThreshold0 / 1 | 100 / 1000 | 12 / 14 | **12.5 / 125** |
| LavaFluidityValue0 / 1 | 0.01 / 0.02 | 13 / 15 | same |
| LavaFluxThreshold (yield) | 0.05 | 16 | 0.05 |
| WaterTemperatureMax (solidify at or below) | 100 | 17 | **12.5** |
| TemperatureDampingThreshold (cool while at or above) | 100 | 24 | **12.5** |
| TemperatureDampingFactor (cooling per step) | 0.3 | 25 | **0.0375** |
| WaterFluxScale / LavaFluxScale | 1 / 1 | 26 / 27 | same (not tested at other values) |
| LavaFluxDampingMin / Max | 0.95 / 0.99 | 28 / 29 | same |
| SoftGroundErosionWaterHeightMax | 4.0 | 30 | 4.0 |
| LateralErosionFluxSumThreshold / Factor | 0.1 / 0.05 | 32 / 33 (`ErosionSumFluxSum…`) | same |
| LavaHeightTemperatureDampingThreshold | 1.0 | 34 (`WaterHeight…`) | 1.0 |
| LavaHeightTemperatureDampingFactor | 1.0 | 35 (`WaterHeight…`) | **1.0** (read from the running game; the copy routine doesn't write index 35, it stores value ÷ `[obj+0xA8]` at `0x100FBFE5` for some other use) |
| LateralErosionSlopeThreshold Left/Right/Up/Down | 1.0 | 36–39 | same |
| LateralErosionWaterHeightThreshold | −1.0 | 40 | same (role not isolated) |
| LavaFluxDampingThresholdMin / Max | 100 / 1000 | 43 / 47 | **12.5 / 125** |
| WaterFluxMask / LavaFluxMask (per-direction outflow cap) | 10000 ([0]) / 1000 ([1]) | 44 / 45 | same |
| GroundWaterMax | 0.1 | none | – |
| (hash `0xF271D3DB`, type `0x433`) | 0 | – | – |

Other indices: 31 border water surface (set per map and from scripts, setter call at dustai `0x10049B4C`), 46 `WaterEnergy`, 48/49 border hard/soft ground; 20–23 and 41–55 are display, pathfinding and outline values not used by the step.

### "Slow" materials
`m_slowWaterParameters` and `m_slowLavaParameters` are full alternative sets, fed through the same routine when selected (dustai indexes them by a material number; when the game selects them is not established):

| | Normal | Slow |
|---|---|---|
| Water fluidity / min | 0.032 / 0.03 | 0.0032 / 0.003 |
| Water flux damping | 0.999 | 0.99 |
| `WaterFluxMask` (max outflow per side per step) | 10000 | **0.005** |
| Lava fluidity 0 / 1 | 0.01 / 0.02 | 0.001 / 0.002 |
| Lava flux damping thresholds (stored) | 100 / 1000 | 0 / 0 |

The flux mask is a plain cap (oracle test: mask 1 limits a 40-deep column to 1.0 per side; 0.1 → 0.1; 0.005 → 1/256 per side). Slow water is almost a gel.

**The Jellify power uses exactly this set, for the whole map** (read from the running game while Jellify was active): `WaterFluxDamping` 0.99, `WaterFluidity` 0.0032, `WaterFluidityMin` 0.003, `WaterFluxMask` 0.005; all other parameters unchanged, lava included. The simulation keeps running (31 ticks/s), so water can still creep at most 0.005 per side per tick (~0.15 per second), and erosion, deposit and sand slumping carry on. The values switched back to normal at frame 2085 of the level, after being in force at frame 711, matching the power's in-game timer of **60 s**: a second, fully logged cast (`tools/memscan.py watch`) switched at tick 351 and back at tick 2211, exactly 1,860 ticks = 60.0 s. On a map with flowing lava, **no lava parameter changed** during Jellify (the power is called "Jellify Water", and the lava visibly kept flowing). When the slow *lava* set is used was not seen.

### How the parameters are combined (`ConstParam`)
Read from the constructor; `P[k]` = parameter k. `ramp` = linear, clamped to its two end values.

| Precomputed vector | Formula | Meaning |
|---|---|---|
| `mv_WaterFluxDampingThresholdScaleMinMax` | `[P9 + P10·(P9−P8)/(P0−P10), (P0−P10)/(P9−P8), P10, P0]` | damping ramp over sediment ratio: `P0` at ratio ≤ `P8`, `P10` at ≥ `P9` |
| `mv_WaterFluidityThresholdScaleMinMax` | same with `P1`, `P11` | fluidity ramp over sediment ratio |
| `mv_LavaFluidityThresholdScaleMinMax` | `[P12 − P13·(P14−P12)/(P15−P13), (P15−P13)/(P14−P12), P13, P15]` | lava fluidity ramp over temperature |
| `mv_LavaFluxDampingRange` | `(P29−P28)/(P47−P43)` | lava damping ramp over temperature |
| `mv_SoftGroundErosionThresholdFactor`, `…Factor` | `−P3·P2`, `P3` | erosion = `P3·(v − P2)` |
| `mv_SedimentTransportThreshold` | `max(P4, 1/65536)` | |
| `mv_SedimentTransportFactor` | `P5` | |
| `mv_SedimentTransportEpsilon` | `15/16` | constant |
| `mv_SedimentDepotThresholdFactor`, `…Factor` | `P7·P6`, `P7` | deposit = `P7·(P6 − v)·sediment` |
| `mv_LavaFluxThreshold`, `mv_WaterTemperatureMax`, `mv_SoftGroundErosionMax`, `mv_SedimentDepotMax` | `P16`, `P17`, `P18`, `P19` | |
| `mv_SedimentRatioMax`, `mv_SedimentRatioDenom` | `0.5`, `2` | constants |
| `mv_TemperatureDampingThreshold/Factor` | `P24`, `P25` | |
| `mv_WaterFluxScale`, `mv_LavaFluxScale` | `P26`, `P27` | |
| `mv_WaterFluxMask`, `mv_LavaFluxMask` | `P44`, `P45` | per-direction outflow cap |
| `mv_LavaFluxDampingMin/Max` | `P28`, `P29` | |
| `mv_SoftGroundErosionWaterHeightMax` | `P30` | |
| `mv_LateralErosionParams` | `[−P33·Smax, P33, P40, P32]`, `Smax = max(P36..P39)` | |
| `mv_LateralErosionSlopeThresholds` | `Smax − [P36, P37, P38, P39]` | |
| `mv_WaterHeightTemperatureDampingThreshold/Factor` | `P34`, `P35` | |
| `mv_BorderWaterSurfaceHeight`, `…TerrainSurfaceHeight`, `…SoftGroundQuantity` | `P31`, `P48`, `P49` | border band |
| `mv_255`, `mv_Rcp255` | 255, 1/255 | |
| `mv_ErosionBlocking` | `ParamU8` bytes 16–31 | 16-entry lookup for vegetation |

## Check against the running game
[../tools/memscan.py](../tools/memscan.py) reads the game's memory (read-only). [../tools/live_check.py](../tools/live_check.py) captures two consecutive simulation states and compares them with the DLL harness and the Python model.
- Simulator object (`ISimulatorWater`) found via its parameter array; in the session checked it was at `0x4F933A0` (heap, so it moves between runs). Live fields: width/height bits 9/9, **border skip 2**, frame counter `+0x92C`, cursor position `+0x940`, grid buffers `+0x950`. The frame counter stops while the game is paused.
- **Live parameter array confirms the ×1/8 temperature scaling** (12.5, 125, 0.0375), flux masks 10000 (element [0]), index 35 = 1.0, border water 39 (that map's sea level), plus display/path values: 20/21 normal scale 2, 22 display scale 1.5, 41/42 path max slope/water 100, 46 `WaterEnergy` 8, 50 minimum visible water 0.5, 52–55 outline thresholds 0.1 / 10 / 2 / 2.8.
- **Capture timing:** the frame counter increments when a step *starts*, and the step runs on worker threads. A buffer is only complete once the following step has started, so read state *n* when the counter shows *n+1*. Reading earlier gave half-written grids.
- Result (3 state pairs on a level with ~197k water and ~1.3k lava cells): the DLL harness reproduces the game's terrain, soft ground and **all outflows exactly**; the Python model matches with 1–2 flux rounding ties. About 600 cells per tick differ in fluid height, all in fixed spots: small discs (5×5 to 11×11) that the game keeps topped up (water sources: the game's level there is unchanged tick to tick while the bare step lowers it), and the column x = 476 along the right edge (see below).
- Speed of the running game: 31 simulation ticks per second (124 per 4 s), i.e. the 30 fps cap.

### Things the game does outside the step (live, one level with border sea level 39)
- **Waterfall sources:** small discs near one edge (5×5 to 11×11 cells, e.g. rows 37–47 / cols 234–244) where the game adds a fixed amount per cell each tick (0.14, 0.23, 0.53, 0.66 per cell in the four found), keeping their level steady. Five waterfalls on that edge were seen in play; four were caught by the comparison.
- **Sea edge / tides:** along the right edge (x ≈ 470–479) the game rewrites column 476 every tick (the bare step would drain it from ~47 to ~4), and the ring column 479 holds 73–86 instead of the border value 39. The sea level near that edge oscillates by about ±0.9 with a period of ~373 ticks (≈ 12 s): the column-470 mean was 46.356 / 46.368 / 46.368 at frames 4010 / 4383 / 4756. Parameter 31 (border water level) stayed at 39.0 the whole time, so the tide is not done through the border parameter. Which code writes these columns, and its formula, is not established.
- Speed: the DLL step takes ~4.3 ms (single thread, 200 steps in 0.86 s); the Python model ~75 ms.

### Sources, vents and waves (second level, 4 minutes of logging)
Logged with [../tools/live_sources.py](../tools/live_sources.py) (every ~3 s: game's next state minus the game's own step on the previous state; log `data/live/sources_20261113_140605.log`). Level with ~236k water and ~1.6k lava cells, border sea level 29.99.
- **Water source:** a 21-cell round patch (radius 2.5) centred at (191, 365), adding exactly **+0.4 per cell per tick** of water that is **30% sediment** (a muddy spring, source-table field 6 = 0.3), constant for the whole 4 minutes. The game's hint text says a spring **stops if too much earth covers it**, but in a test it **did not**: with sand piled on it up to ~110 deep (mean ~68 over the spring area, on rock at height 7), it kept adding exactly +8.4 per tick (21 × 0.4) in all 112 samples (`tools/live_spot.py`, log `data/live/spot_191_365_20261113_141626.log`). The spring stayed on in play too. On Map 2 a spring **could** be buried (threshold 5). This Map 11 spring could not because the script placed it at the ground height of that moment (z ≈ 74), well above today's ground; see "Where per-map sources and their timing are defined" below.
- **Lava vents (the volcano):** three 3 × 3 patches close together, centred at (199, 213), (202, 223), (211, 215). Each adds exactly **+0.1 lava per cell per tick** and **sets** the cell temperature to a fixed value: **150.0** for one vent, **112.5** for the other two (1200 and 900 in editor/stored units). Constant for all 4 minutes: no phase change was seen, so the volcano's phases are slower than that or event-driven.
- **Sea-edge wave:** the right-edge column (x = 476, 440 cells) was driven only between ~123 s and ~162 s (per-cell additions rising from +24 to +99, falling back, ending at −43), then left alone. So on this level the edge drive is an intermittent ~40 s wave event, not the continuous ~12 s cycle seen on the first level.
- **Moving terrain dips:** 2–11 small spots (up to 5 cells, plus-shaped), each lowering terrain by 0.02 per cell per tick and moving between samples. **Identified later: the worm-like creatures** (see TODO list).
- **Jellify Water and lava vents** (explained below: it is the power's shockwave): in play, casting Jellify Water made a vent jump up, causing an extra cascade of lava that cooled into a much higher triangle of rock. Measured tick by tick (`tools/live_ticks.py`, log `data/live/ticks_20261113_142108.log`, 3 casts): each time the water parameters switched (ticks 22282, 23223, 23800), the **terrain in the vent area rose** 6–20 ticks later (first cast: about +300 summed over 6 ticks), lava was pushed out (lava cells 199 → 735 within ~4 s) and then cooled to rock (lava volume 1214 → 970). **No lava parameter changed** (parameters polled every ~2 ms). So the game edits the terrain around the vent in reaction to the cast; it is not part of the simulation rules. **Explanation (third level, below): casting Jellify Water sends out a shockwave** from where it is cast; the casts had been aimed at the vent.
- About 1 sample in 15 was a mistimed read (half-written buffer, 150k "differences"); those were skipped.

### Third level: more sources, Jellify shockwave, a scripted volcano
Logs `data/live/sources_20261113_142819.log`, `data/live/ticks_20261113_142840.log`.
- **Border = drain at 0:** border water surface, hard and soft ground all 0, so the edge removes any water that reaches it; there is no simulated sea. The visible "sea" on this level is hidden under a cloud in game; nothing below 0 (minimum terrain 0).
- **Sources:** two lava vents, both setting their lava to **255°**: a 3 × 3 at (346, 154) adding +0.45 per cell per tick and a 21-cell round patch at (427, 236) adding +0.3. Six water springs: five 21-cell patches adding +0.56, +0.6, +0.7, +0.8, +0.8 per cell per tick and one 9 × 9 adding +0.235. Several small spots *lose* water each tick (−0.002 to −1.6 per cell); not identified (drains, or evaporation near lava?).
- **Jellify Water = shockwave:** three casts (ticks 1341, 1922, 2471; the first was cancelled after 91 ticks, which the game allows for a faster recharge). Terrain bumps followed 25 and 9 ticks after casts 2 and 3 (the third pushed lava deeper); the player saw a shockwave in game. Water parameters switch as before; lava parameters never change.
- **Scripted volcano spawn** (a countdown event shown in game, like the tidal-wave events; the third cast was aimed at it): growth from tick ~2795 (area terrain +19 per tick, peak rising 0.01 per tick to +1.9); ~tick 3000–3100 a sudden drop of ~5,400 terrain units in total as lava appears (crater); re-growth ~3100–3400; then **eruption** from ~3400: lava volume in the area 3,600 → 10,300 in ~580 ticks (≈ +11.5 per tick), lava cells 700 → 2,500, terrain slowly sinking. These are the volcano "phases" seen in play. The vent position/size/temperature of the new volcano was not captured (game paused before the scan).

### Source table in the running game (found 2026-11-13, Map 3 "Vallée")
**Corrected layout (Map 10, later the same day):** the table is the simulator's **per-tick modifier list**, a vector `{u32 data pointer, u32 count, u32 capacity}` (one session: at `0x7ED70124`). Every tick the game clears it (count 0) and refills it; it stays full for most of the tick. Each 36-byte entry is `u32 type, f32 col, f32 row, f32 radius, f32 ?, f32 amount, f32 ?, f32 sediment, f32 temperature` — one word earlier than the table below assumed (its "field 8" is the next entry's type). **Type 0 = fluid source**, with the amount actually applied this tick (rate × bury activity × fade-in × Jellify factor, e.g. a drain at activity 0.18 shows −0.19 of its −1.03; script switches show at once; buried sources are absent). Volcano lava vents are in it too (not generators). **Type 1 = another modifier** (amount 1, the ? after it ±1, radius 23–26 on Map 10): probably the scripted volcano's terrain edits; not decoded. Stale copies of earlier entries sit elsewhere in memory, which is why searching for record values found leftovers. Readers: `live_generators.find_modifier_list()` / `read_modifier_list()`; DustView reads it every frame.

Original notes:
[../tools/find_sources.py](../tools/find_sources.py) measures a level's sources live and searches memory for their values. On Map 3 the four springs' rates (0.14, 0.66, 0.23, 0.53) sit in **one array of 36-byte records** (heap, e.g. `0xB2A1D4` in that session; stored in reverse order). Each record is 9 little-endian floats:

| Field | Value (example) | Meaning |
|---|---|---|
| 0 | 67 | centre column (grid) |
| 1 | 94 | centre row (grid) |
| 2 | 2.5 | radius in cells: the disc is the cells with dx² + dy² ≤ r² (2.5 → 21 cells, 4 → 49, 5 → 81) |
| 3 | 0 | ? (0 in all 8 sources seen) |
| 4 | 0.53 | amount of fluid added per cell per tick |
| 5 | 0 | ? (0 in all 8 sources seen) |
| 6 | 0 / 0.3 | **sediment fraction** of the added water (the 3-vent level's spring 0.3: it adds 0.4 water + up to 0.12 sediment per cell per tick, measured ratio 0.297) |
| 7 | 0 / 150 / 112.5 | **lava temperature** (internal units); non-zero = lava vent, which sets the cell temperature to this value |
| 8 | 0 | ? (0 in all 8 sources seen) |

Map 3 (four clear springs, field 6 = 0) and the 3-vent level (probably Map 11 "Void", see below; records at `0xB2A1CC`–`0xB2A238` in that session):

| Col | Row | Radius | Rate | Sediment | Temperature | Source |
|---|---|---|---|---|---|---|
| 365 | 191 | 2.5 | 0.4 | 0.3 | – | muddy spring |
| 213 | 199 | 1.5 | 0.1 | – | 150 | lava vent |
| 223 | 202 | 1.5 | 0.1 | – | 112.5 | lava vent |
| 215 | 211 | 1.5 | 0.1 | – | 112.5 | lava vent |

Map 4 ("Muraille", border water 28.92; table saved as `data/live/source_table_20261113_145847.json`): lava vents at (col 257, row 321) rate 0.66 and (320, 367) rate 0.05, both temperature 125; clear springs at (143, 448) rate 0.74 and (311, 456) rate 1.56; all radius 2.5.

Map 2 ("Archipel", border water 39.99): one muddy spring at col 167, row 234, radius 3.5 (37 cells), rate 0.66, sediment 0.42.

All eight records match the live measurements exactly (centres, radius from the cell count, rates, temperatures). Radius 1.5 → 9 cells (3 × 3). Nothing found pointing at the array yet; the words just before it point into the exe's memory-pool area (`0x0092B300`), so it is a pooled allocation.

### Where per-map sources and their timing are defined (decoded 2026-11-13)
Live counterpart: [../tools/live_generators.py](../tools/live_generators.py) lists the running level's generator objects with their runtime fields (mode, fade-in, activity, bury threshold). Everything below is read from each map's `.sav` by [../tools/sav_sources.py](../tools/sav_sources.py) (`python tools/sav_sources.py ALPHA_Map11_Void` prints the generator list and the whole level script as a timeline). Field names come from dustai's serialisation code (it passes each field's name to a reader interface; the vtable slot used gives the type), behaviour from the dustai code that runs them. Addresses are in [../docs/technical-notes.md](../docs/technical-notes.md).

**Fluid generators** (springs, lava vents and drains are all the same class):
- A manager block tagged `0x10020002` holds the list; each entry (block of 264 bytes) is tagged `0x20020002`, then a handle block `0, class, index, serial`, a world-position block (x, y, z), and ends with 11 big-endian settings words: u32 ?, u32 0, u32 ?, f32 height (= z), **u32 Mode**, **u32 size** (diameter in cells), **u32 temperature** (editor units; 0 = water), **u32 sediment %**, **f32 rate** (`m_quantityPerFrame`, fluid per cell per tick; negative = drain), f32 `100` or `0` (unknown), u32 `0`/`1` (unknown). The last two are **not** the bury threshold and `m_geiser` (an earlier reading): in the running game every generator's bury threshold is 5 and `m_geiser` is a reference, whatever these two say (checked on Map 2 with [../tools/live_generators.py](../tools/live_generators.py)).
- **Mode** (enum strings in dustai): 0 off ("Unactivated"), 1 in game only, 2 in game and sandbox, 3 sphere power only. Map 3's springs are mode 2; Map 11's three vents start at mode 0 until its script turns them on; Map 5 has three mode-3 sources.
- **World → grid:** `col = 256 + x · 2/3`, `row = 256 + y · 2/3`, truncated (one cell = 1.5 world units, origin at the map centre). Checked: every generator of Maps 2, 3 and 4 lands exactly on the live source table's centre; Map 10's on its live sources.
- **Per-tick amount** (generator update at dustai `0x10066646`–`0x100667B3`): `rate × ramp × factor × activity`, then handed to the simulation with temperature, sediment and size.
  - `ramp` counts up by 1 per tick to 100 and resets to 0 every time the generator is switched on: **a generator fades in linearly over 100 ticks (~3 s)**.
  - `factor` is 1, except for sources colder than a threshold from the active fluid parameter set (i.e. **water springs, not lava vents**) while **Jellify Water** is active, when it is **`m_slowWaterSpeedFactor` = 0.5**: springs run at **half rate** during Jellify. The factor is read from the sphere-power settings (`[DustManager+0x54]+0x2C`; field names from dustai's registration at `0x1012B3AA`, which also has `m_slowWaterTimer` 60 and `m_slowWaterCooldown` 60 at +0x24/+0x28); values from `Wor_BD00993D.lin.bf/SpherePowerManager_0.ai`. That the gating flag (bit 4, set at `0x100F9790`) is Jellify's active state is inferred from the field it selects; not yet checked live.
  - `activity` is the **bury rule**: `d = (ground height at the source) − (source's own z)`, `activity = 1 − clamp(d, 0, T) / T`, buried when `d ≥ T`. **`T` (`m_buryTresholhd`) is 5 for every generator** (class default in `Wor_BD00993D.lin.bf/FluidGenerator_0.ai`; read live on Map 2). So a spring fades out linearly as the ground over it rises from 0 to 5 units above it, and stops at 5. **Confirmed on Map 2 ("The Ritual")**: its second spring (#1) starts ~6.4 units under the ground, so it is switched on and faded in but has activity 0 and never flows (the player sees only one spring); the player then buried the first spring and its activity read 0 too. **Map 11's spring did not stop under ~110 sand because of its height:** when a script moves a generator, the grid → world conversion (`0x10012EE0`) sets its **z to the ground height at that cell at the moment of the move** (plus a constant). Map 11's spring was placed at 400 s; in memory its position is (164.25, −96.75, **z ≈ 74**) (centre of cell 365, 191), while the ground there has since been worn down to ~43 (live, 2026-11-13). So the spring sits ~30 above the ground and only fades out once the ground at its centre cell passes 74, stopping at 79; the earlier pile averaged ~68. Spring activity read 1.0 throughout with threshold 5. (With `T = 0` the code would give NaN and leave activity at 1; no generator was seen with `T = 0`.)
- Counts (generators / of which drains): Map 5 115 / 31, Map 10 19 / 8, Map 12 17 / 0, Map 9 11 / 0, Map 4 9 / 5, Maps 3, 3_Demo, 11 4, Map 2 2, challenge maps 0–2; none on Maps 1, 6, 7, 8, 13 (Maps 1, 6, 13 checked in play: no lava). The earlier reader missed drains and rate-0 sources (it filtered on rate > 0); Map 10 has 19 generators, not 8. Map 9's volcano vent is an ordinary lava generator (col 169, row 184, radius 1.5, rate 1.04, 255°).

**Level script** (dustai `GameSceneManager`; strings in the `.sav` are UTF-16LE with a u32 length in characters, numbers big-endian, every record a size-prefixed `12 34 56 78` … `87 65 43 21` block):
- **Rule list:** f32 `_Timer` (its clock), f32 `Duration`, u32 `_IsPlaying`, u32 `Loops`, then the rules. The level's top-level list has `Duration −1`.
- **Rule:** u32 `ObjectId`, **f32 `Start`**, **f32 `Duration`**, u32 `_State`, f32 `_VerificationTime`, a list of conditions, a list of effects, u32 `TimelineIcon`, u32 `DisplayInTimeline`, f32 `TimelineEffectDuration`. The last three put a marker on the in-game timeline (the countdowns seen before tidal waves and the volcano spawn).
- **When a rule fires** (`0x10099310`, called every frame with its list's clock `t`): state 0 → 1 once `t > Start`; in state 1 it **fires as soon as all its conditions hold** (immediately if it has none) while `t < Start + Duration`, or with no deadline if `Duration < 0`; otherwise it expires (state 2). A rule fires once (state 3) per run of its list.
- **Conditions** start with their class name, then u32 ID, u32 ?, then fields. `CreatedVillagesCount`: u32 `VillageCount`, i32 `TypeTest` (−2 ≤, −1 <, **0 ==**, 1 >, 2 ≥; checked against the jump table). `RandomTime`: picks a random wait between f32 `MinTime` and `MaxTime` on first check, then holds once that time has passed. About 30 other condition classes exist (`Population`, `VillageDestroyed`, `NoFireOnMap`, …).
- **Effects** start with their class name, then f32 **`Duration`**, f32 `_Timer`, u32 `UseShakeCam`, then class fields. Effects with `Duration ≥ 0` stop after that many seconds (`0x10096370`: running while `_Timer ≥ 0` and (`Duration < 0` or `_Timer < Duration`)). Some classes (`Cinematic`, `VolcanoChangeQuantity`, …) don't follow this header; the reader only labels plausible durations.
- **`FluidGenerator` effect:** u32 `Type`, u32 ID length, UTF-16 target ID (odd lengths padded to 4 bytes; the first Map 10 event's 7-character ID `D984CA4` was the earlier "parse glitch"), u32 `PosX`, u32 `PosY` (grid cells, `FFFFFFFF` = keep), u32 `Temperature` (editor), u32 `SedimentsQuantity` %, f32 `Quantity` (rate), u32 `Radius` (really the diameter: 4 → radius 2). **Start:** switch the target to mode 1 with these settings, moving it if PosX/PosY are given (only inside cells 33–478). **Stop** (when its `Duration` runs out): switch the target to mode 0 (off). `Type 1` = change the targeted generator, `Type 0` = spawn a new generator at PosX/PosY (Map 3 Demo's lava, the Defi_045 water jets).
- **Target IDs are handles** `serial << 19 | index << 10 | class << 2`: the index is the generator's slot in the list, the class number is assigned per map (41 on Maps 2/4/5/9–12, 59 on Map 3, 43 / 33 on two challenge maps; the same handle format names volcanoes, e.g. `25E000AC`). Every `Type 1` target on every map resolves to a generator this way; on Map 12 the ten script-moved lava sources sit 0–15 cells from the generator with the same index.
- **`Sequence` effect** = a nested rule list with its own clock starting when the effect starts. `Duration` is its length and `Loops = 1` restarts it (clock back to 0, all rules re-armed) each time it ends. This is how repeating events (tsunamis, Map 10's volcano cycles) are built.

**Map 11 ("Void", in game "Emergence") timeline** (seconds of game time from the level start):
- Start: four water generators, three off (mode 0) and one (#1, col 356 row 181) on at rate 0.05 (all four have the save value `f10 = 0`).
- 330 s: #0, #2, #3 are moved to (213, 199), (215, 211), (223, 202) and become **lava vents: rate 0.3, radius 2, 112.5°**, permanently.
- 400 s: #1 is moved to (365, 191): **muddy spring, rate 0.4, 30 % sediment, radius 2.5**, permanently.
- 1401 s: vent (223, 202) to 150°; in the same second, **if exactly 3 villages exist**, all three vents drop to **rate 0.1, radius 1.5** with (213, 199) at 150°. The live table measured on this map (0.1 / 1.5, 150 / 112.5 / 112.5, spring 0.4 / 0.3) is the result of these two rules.
- Once the first village is built: a 400 s looping sequence starts, with a **`Tsunami` at 360 s into each loop** and timeline markers at 120 s and 360 s (the tidal-wave countdown). Volcano phase changes at 1, 40, 120, 300, 400, 880 and 1400 s.

**Map 10 ("Fire and Water") timeline:** 19 generators: springs #0, #1, #2, #3, #14 and lava vent #13 (rate 0.45, 255°) run from the start, as do eight drains (rates −0.68 to −25, radii 2.5–27.5). Script:
- From 60 s, a 400 s looping cycle: springs #0/#1/#3 go to rate 0 with lava vent #9 at 0.5 for 240 s; 100 s later springs 0.8/0.8/0.7; then 0 again with vent #4 at 0.2 for 10 s; then 0.8/0.8/0.7 again. Effect durations overlap, so a later effect's Stop can switch off a source that another effect had just turned on.
- From 60 s, a 300 s loop: at 150 s into each loop, for 100 s: lava #18 (236, 427) 0.3 at 255°, spring #17 (293, 427) 0.6, drains #16 −0.6 and #15 −20 (radius 5). Together with the cycle above this matches the live rates seen on that level (springs 0.8 / 0.8 / 0.7 / 0.6, vent 0.3, drains).
- At 20 s, a 40 s sequence touches #19 (lava at 250°, rate 0, 5 s) together with a volcano phase change: probably part of the spawned volcano.

**Other notes**
- Class templates are in the shared archive `Wor_BD00993D.lin.bf` (`FluidGenerator.obj`, `FluidGeneratorGeiser.obj`, `SpecieFluidGenerator.obj`, `Volcano.obj`, `SpecieVolcano.obj`, `VolcanoBomb.obj`, …; `LavaGeneratorFumarole.obj` / `VolcanoFumarole.obj` in `Wor_BD00AFCD.lin.bf`).
- Other effect classes relevant to fluids, not decoded yet: `Volcano`, `RandomVolcano`, `VolcanoChange*` (phase, phase duration, quantity, temperature, crater radius/depth), `Tide`, `Tsunami`, `RandomTsunami`, `Wave`, `GlobalRain`, `Caldeira`, `Crack`, `RandomLavaGenerator`, `RandomWaterGenerator`, `TerraForming`, `CoastErosion`.
- **Map identity:** the scripted-volcano level is **Map 10**, the 3-vent level with the tidal wave is **Map 11** ("Emergence"), and story level 2 **"The Ritual" is `ALPHA_Map2_Archipel`** (confirmed live: same border sea level 39.99 and spring).
- The grid edit itself probably goes through the simulator's modifiers (`SimulatorModifierTemplate::AddLava`, `RemoveLava`, `RepulseFluid` in `GalileoSimulatorWater.dll`); the generator passes temperature, amount, sediment and size to dustai `0x100F9680`. Not traced further.

### The Python model with sources (`dustsim.tick`, checked live 2026-11-13 on Map 4)
**Momentum:** the stored outflows are the water's momentum (`f = 0.999·f_old + 0.032·ΔS`), so waves keep moving after the slope that drove them is gone. Jellify Water's flux cap (0.005 per direction) clamps those outflows within a tick, so the momentum is wiped, not paused; when Jellify ends the water restarts from height differences alone (seen in play as a frozen wave dropping).

`dustsim.tick(G, P, sources)` = border ring reset, `step()`, then `apply_sources()` (each source's disc of cells `dx² + dy² ≤ r²`). Rules: a spring adds `rate` water and `rate × sediment` sediment per cell; a lava vent adds `rate` lava and sets the cell temperature; a drain lowers the fluid by up to `|rate|` (lava drains on Map 4 remove up to 0.43 where there is lava, nothing elsewhere). Compared with the game's next state, one tick at a time, with the 9 live sources (springs 1.56 and 0.74, vents 0.66 and 0.05, five lava drains −0.43): fluid differs by more than 0.001 in **~51 of 262,144 cells**, flux in 0–1 cells. Known gaps:
- the small vent (rate 0.05) gains 0.05–0.33 per cell in the game: probably sources are applied *before* the step there (lava yield threshold), not checked;
- lava drains (temperature 125) remove only lava (a cell with 0.004 water inside one kept it) and do **not** set the temperature (drain cells match the game to ~0.01); the model now drains only the source's own kind of fluid (water drains leaving lava alone is assumed). The remaining temperature differences (~180 cells) sit at lava/water edges: the step gap below;
- ~~lava/water edges~~: fixed (lava diluted to ≤ 12.5° hardens and the cell stays water, see the step rules). After the fix, one tick of `tick()` against the game: terrain 3–5, soft 3–6, fluid ~25 cells (mostly the 0.05 vent), flux 2–3; lava temperature differs from the DLL in ~200 of ~1,240 lava cells, 8 by more than 0.1° (thin lava: probably which depth the extra cooling `1.0 · max(0, 1 − h)` uses);
- player powers (picking up / dropping water, sand, lava) and level-script changes are not part of the model.
- **Torn reads:** even with the frame-counter checks (`capture_pair`), about 1 pair in 25 read from a busy viewer process is torn (a buffer copied while being written): it shows as ~12,000–18,000 cells differing from the DLL step, spread over all flowing water, with ~12,000 outflow mismatches instead of the usual ~300. DustView's view 7 skips pairs with more than 3,000 outflow mismatches; after that, Map 10 checks leave a median of 5 unexplained cells (max 81 in 213 checks). The steady ~300 outflow mismatches sit under the village outpost's water repulsion (a type-1 modifier) and similar outside edits.
- **Modifier types (PDB enum `Galileo::ISimulatorWaterConst::tModifier`):** 0 `kModifier_Fluid` (springs, vents, drains: the type-0 entries), 1 `kModifier_RepulseFluid`, 2 `HardGround`, 3 `SoftGround`, 4 `RemoveWater`, 5 `AddWater`, 6 `RemoveLava`, 7 `AddLava`. An entry is the type plus two 4-float vectors, `Center = (col, row, radius, ?)` and `Param = (amount, ±1, sediment, temperature)`; the simulator copies them into `SimulatorModifierTemplateParam` (0x50 bytes: two int vectors with line/cell ranges and grid pointers, `mvf32_Center`, `mvf32_Param`, `mvf32_WaterTemperatureMax_Energy_u_u`). Code: `SSE41/GalileoSimulatorWater.dll` (with PDB), `SimulatorModifierTemplate::RepulseFluid` at `0x1000BEE0` (~1,270 lines of SSE4.1; reads and writes all six grid components: ground, soft, fluid, sediment/temperature, both outflow pairs), `AddWater`, `RemoveWater`, `AddLava`, `RemoveLava`, `HardGround`, `SoftGround`; dispatcher `SimulatorWater::Modify(tModifier, __m128, __m128)`.
- **`RepulseFluid` decoded (village outposts), 2026-11-13.** Run on test grids with the game's own code (`galileo.modify`, harness `tools/simstep/modstep.exe`) and re-implemented as `dustsim.repulse_fluid()`; it matches the DLL in all but 1 of 6,710 changed cells over 12 random cases (one exact-.5 rounding: the DLL takes `1/radius` from a refined `rsqrtps`). The rule, for Center `(col, row, radius)` and Param `(amount, mode)`:
  - `mode > 0` acts on lava cells; `mode ≤ 0` on water cells **including dry ones** (depth ≥ 0). The outposts on Map 10 had −1 (repel water) and two other entries +1 (repel lava).
  - For each such cell with distance `d ≤ radius`: `s = 1 − d/radius`; the four stored outflows become `old × (1 − s) + 256 × amount × s × (outward unit vector split into L/R/U/D)`; if their sum exceeds `depth × 256` all four are scaled down to it; then rounded to nearest. The centre cell's outflows become 0; at `d = radius` only the depth cap applies.
  - Nothing else in the cell changes: the water is moved by the next step through these outflows (so a frozen/slowed state still holds them back), which is why the live check showed ~300 outflow mismatches and no fluid differences under an outpost.
  - `WaterEnergy` (ParamF[46]) and ParamF[17] are passed in but do not change the result.
  - Applied between steps on the current state, in the order of the game's per-tick list (`dustsim.apply_modifiers`); `dustsim.tick(G, P, sources, repulsors)` applies them after the sources.
- **Village outposts repel water:** the type-1 entry `mod +1/−1` (radius 26) on Map 10 sits on the village outpost, which in play pushes water away; so type-1 entries include `RepulseFluid` modifiers.
- **Burying in the model:** `dustsim.bury_activity()` applies the bury rule every tick for sources that carry their stored height and height mode (live sources and `--model` runs do). Seen on Map 2: the model switched the visible spring off because the viewer had paired its table record with the *other* spring (identical settings, buried since the start, activity 0); the model also froze activities at takeover. Now generators with identical settings are paired with their table records by stored height vs the ground at the record's cell, and the activity is recomputed every tick. Buried generators are absent from the live source table, so a spring buried at takeover (Map 2's second spring) is not in the model and cannot be uncovered there.

## Quirks
- **Volume drift:** water, sediment and lava are subtracted unrounded and added rounded (outflows to 1/256, sediment to 1/4096).
- **16-column temperature wrap:** lava crossing between column 16k−1 and 16k gets its temperature reduced modulo 256. If the wrapped value is ≤ the solidify threshold, it even arrives as water. The DLL hands block-edge values over as `temperature × 256` in 16 bits (driver `Run12345_Middle`). Real temperatures never exceed 255, so this never shows in play. It's only relevant when feeding the DLL custom values.
- **Drained-lava edge case:** in a few cells per 10 steps at a lava front (5 of 93 candidate cells on Map 9), a lava cell whose own lava fully drained and which received lava ends up with that lava turned into terrain **and** the same amount of water. The water boils away the next step. The exact trigger was not isolated (see TODO).
- **Lava films** thinner than about 2·10⁻⁵ disappear (seen once in a random test).

## Method: running the game's own step outside the game
The worker DLL has **no imports** and exports a job table: `sCodeAddrJob(0)` = `CallbackSimulator(SimulatorParam const*, __m128i* scratch, ulong)`, scratch size `sStackSizeJob(0)` = `0x12600` (= `SimulatorWaterIteratorContainer<9>`). One call = one full 512 × 512 step. [../tools/simstep/simstep.cpp](../tools/simstep/simstep.cpp) (32-bit, built by `build.bat`) loads the DLL and runs N steps on a grid file. [../tools/galileo.py](../tools/galileo.py) builds the grid and `SimulatorParam` image and reads the results. [../tools/compare_sim.py](../tools/compare_sim.py) checks [../tools/dustsim.py](../tools/dustsim.py) against it on a map or a random landscape. Rules were found by combining the disassembly (structure, constants) with controlled single-cell experiments (exact values).

### Border
`SkipBorderSize` is in units of 16 cells per side. With skip 2, rows/columns 0–31 and 480–511 are not touched, and the outermost processed ring (row/column 32 and 479) is overwritten with the border values every step. **The reset happens before the step** (live, Map 11: after a step the ring holds 26.2 against border level 28.92, because the ring itself flows during the step; the band outside stays at 28.92). Modelled in `dustsim.tick()`: with it, the model's border band matches the game exactly (5,868 differing cells per tick → 0, Map 4). That gives the constant 32(–33)-cell band seen in every save ([../docs/sav-format.md](../docs/sav-format.md#the-32-cell-border-band-is-constant)), so the game runs with skip 2.

## What this means for the earlier questions
- **Frame rate:** every rate is per tick with no time step, so the sim runs faster at higher tick rates. Halving fluidity would *not* simply give the same water at 60 ticks/s, because damping (0.999 per tick), erosion, deposit, cooling and the yield threshold are all per tick as well. A faithful 60 Hz version would need each of them re-derived (inference, not tested).
- **Resolution:** the rules are local and per cell, so the Python model runs at any grid size. But constants like the slump threshold (1 height unit per cell) and fluidity are tied to the cell size, so a 1024 grid needs them re-scaled to behave the same (inference).
- [python-water-sim.md](python-water-sim.md)'s textbook model is superseded by [../tools/dustsim.py](../tools/dustsim.py).

## TODO (skipped, worth revisiting)
- [x] ~~`WaterHeightTemperatureDampingFactor` (index 35) at runtime~~: **1.0**, read from the running game (the earlier guess of 0.125 was wrong).
- [ ] Which code writes the simulator's index 35, and what the value stored at `[0x10184F38]+0xAC` (`LavaHeightTemperatureDampingFactor ÷ [+0xA8]`) is used for.
- [x] ~~Find the source objects in memory~~: runtime array of 36-byte records (centre col/row, radius, rate) found on Map 3; see "Source table in the running game".
- [x] ~~Model the village outposts (`RepulseFluid`)~~: decoded and matched against the DLL (see above). Not yet confirmed against the running game (view 7 should lose the ~300 outflow mismatches at outposts).
- [ ] The other modifiers can be checked the same way with `galileo.modify`: `AddWater`, `RemoveWater`, `AddLava`, `RemoveLava` (what `kModifier_Fluid` dispatches to by amount sign and a temperature threshold at simulator `+0x16D4`; our `apply_sources` rules come from live measurements), `HardGround`, `SoftGround` (powers, terrain events). `AddWater` is called without a push in `Modify`, so its calling convention needs checking first.
- [ ] Modifier list (ex "source table"): field after radius (0 on fluid entries), the ±1 field and what type-1 entries do (terrain raise/lower by the volcano script?); the owner object of the vector.
- [x] ~~Sources on disk, positions~~: solved (position record before the settings record; col = 256 + 2x/3, row = 256 + 2y/3).
- [x] ~~Sources on disk: script sequencing/triggers, event IDs, the Map 10 ID glitch, the `0/1/2` word and the `100/0`/flag fields~~: decoded (Mode, bury threshold, `m_geiser`; rule/effect timing; handle IDs). Still unknown: the two other leading words (`2`, `0/2`) and what `m_geiser` does; Maps 7 and 8 have no generators in the save and none were seen in play.
- [x] ~~Decode the `.sav` script/event records~~: rule lists, rules, conditions, effects and `FluidGenerator` fully; other effect classes (volcano, tide, tsunami, wave, rain) only by name. Still to do: field layouts of `Tsunami`, `Tide`, `Wave`, `Volcano*`, `RandomTsunami`; the condition base's second word; how the level's top-level list is started (its saved `_IsPlaying` is 0).
- [ ] **Per-map sources for the Python model:** generators and their script changes are now fully readable (`sav_sources.py`). Still missing: a small scheduler that turns the timeline into per-tick generator states (player-driven conditions such as village counts as inputs, plus the 100-tick fade-in and the effect Stop semantics), volcano-driven lava, and scripted events (volcano spawns, tidal waves).
- [ ] (Lower priority: scripted countdown events such as volcano spawns and tidal waves are left out of the Python model for now; static springs and vents come first.) Volcano phases: one scripted spawn seen (growth → crater drop → re-growth → eruption at ≈ +11.5 lava per tick); position, vent size, temperature and phase durations of that volcano not captured; find them in the data (`VolcanoPhase`, `VolcanoPhaseDuration`, `VolcanoChange*`).
- [x] ~~Small spots losing water each tick on the third level~~: drains, i.e. sources with negative rates set by Map 10's script events.
- [ ] Jellify Water shockwave: a terrain edit 6–25 ticks after each cast (5 of 6 casts on two levels), not a parameter change. Shape/strength of the shockwave and which code applies it (needs a cell-level capture around a cast).
- [x] ~~Moving terrain dips~~: **the small worm-like creatures** (identified in play from DustView's markers, Map 4, 2026-11-13). Each is a 5-cell plus shape lowering the ground by 0.02 per cell per tick where it is (e.g. rows 275/284, cols 128/241 on Map 4), moving slowly. Which AI class does it was not found in dustai (no worm/dig names; `m_diggingUpSpeed` belongs to buriable entities). For a model they are an outside terrain edit, like powers.
- [x] ~~The ~8,700-cell terrain change in one tick~~: a volcano eruption (shown as yellow, raised-terrain markers in DustView and seen in play, Map 4). Volcano terrain edits are outside the simulation step; their rules (`Volcano*` effects) are not decoded yet.
- [x] ~~Spring blocking on Map 11~~: the spring's z (≈ 74) was fixed when the script moved it; the ground at its centre must pass ~74–79. A live test (pile sand on the centre cell while `tools/live_bury.py 191 365 --rate 0.4` logs activity) would confirm the fade-out band; the constant added to z in `0x10012EE0` (`[0x10177044]`) was not read.
- [ ] Generator `factor` during Jellify Water: code says springs × 0.5 (`m_slowWaterSpeedFactor`), lava vents unchanged. Confirm live by measuring a spring's added amount per tick before and during a cast (`live_spot.py` on a spring).
- [x] ~~`m_geiser`~~: a reference to a visual effect; for water sources with it set, the generator's simulation link creates that effect at the source (`0x100CC170`). Same value on both Map 2 springs live; it does not decide whether a source runs. The save's last two settings words (`f10` 100/0, `flag11` 0/1) remain unknown.
- [ ] Sea-edge driver: which code writes columns 476/479 and its formula (continuous ~12 s cycle, ±0.9, on one level; ~40 s wave event on another). Probably a level script or modifier in dustai; Map 8 "HighTides" worth checking. Also the moving 0.02/tick terrain dips.
- [x] ~~Confirm Jellify uses the slow-water set~~: yes, map-wide, for 60 s (in-game timer; matches the measured window).
- [ ] When the slow *lava* set (`m_slowLavaParameters`) is used: not by Jellify Water (checked on a lava map). Possibly another power, a level script, or unused.
- [ ] When the game uses `m_fluidParameters[1]` (flux caps 1000), i.e. what drives the element index in the copy routine `0x100FBA20`.
- [ ] Lava temperature in the Python step: ~200 of ~1,240 lava cells per tick differ from the DLL (median 0.005°, 8 cells > 0.1°), mostly thin lava; check which depth the cooling term uses and the order of cooling vs mixing.
- [ ] Drained-lava edge case (Quirks): trigger not isolated; probably the float sign of a zero remainder. Affects a few front cells per 10 steps.
- [ ] Vegetation `ErosionBlocking` (component 6 low nibble → `ParamU8` 16–31) not tested; `ParamU8` values the game passes not read.
- [ ] Not exercised: `SedimentTransportThreshold/Factor` at non-default values, `WaterFluxScale`/`LavaFluxScale` ≠ 1, `LateralErosionWaterHeightThreshold`, and components 6/7 (vegetation, fire), which another job (`SimulatorGanjaIterator`) updates.
- [ ] What the game does outside this step: modifiers (adding water/lava/sand from powers and sources), the border water level changing over time (Map 8 "HighTides"), and how many ticks run per frame in practice.
- [ ] `GroundWaterMax` (0.1) has no Galileo counterpart; the record with hash `0xF271D3DB` (type `0x433`) is unnamed.
