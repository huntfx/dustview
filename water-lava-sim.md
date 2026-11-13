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

**The built-in editor's lava temperature is in the stored units, i.e. 8× the simulator's.** Observed in play: lava placed at 100 turned to rock almost instantly (100/8 = 12.5, the solidify point); lava placed at 700 (87.5 internally) stayed hot after flattening out. Deep lava at 87.5 needs ~2,000 ticks ≈ 67 s to reach 12.5. A live read during that session showed ~50 lava cells between 75 and 100°.

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

## Quirks
- **Volume drift:** water, sediment and lava are subtracted unrounded and added rounded (outflows to 1/256, sediment to 1/4096).
- **16-column temperature wrap:** lava crossing between column 16k−1 and 16k gets its temperature reduced modulo 256. If the wrapped value is ≤ the solidify threshold, it even arrives as water. The DLL hands block-edge values over as `temperature × 256` in 16 bits (driver `Run12345_Middle`). Real temperatures never exceed 255, so this never shows in play. It's only relevant when feeding the DLL custom values.
- **Drained-lava edge case:** in a few cells per 10 steps at a lava front (5 of 93 candidate cells on Map 9), a lava cell whose own lava fully drained and which received lava ends up with that lava turned into terrain **and** the same amount of water. The water boils away the next step. The exact trigger was not isolated (see TODO).
- **Lava films** thinner than about 2·10⁻⁵ disappear (seen once in a random test).

## Method: running the game's own step outside the game
The worker DLL has **no imports** and exports a job table: `sCodeAddrJob(0)` = `CallbackSimulator(SimulatorParam const*, __m128i* scratch, ulong)`, scratch size `sStackSizeJob(0)` = `0x12600` (= `SimulatorWaterIteratorContainer<9>`). One call = one full 512 × 512 step. [../tools/simstep/simstep.cpp](../tools/simstep/simstep.cpp) (32-bit, built by `build.bat`) loads the DLL and runs N steps on a grid file. [../tools/galileo.py](../tools/galileo.py) builds the grid and `SimulatorParam` image and reads the results. [../tools/compare_sim.py](../tools/compare_sim.py) checks [../tools/dustsim.py](../tools/dustsim.py) against it on a map or a random landscape. Rules were found by combining the disassembly (structure, constants) with controlled single-cell experiments (exact values).

### Border
`SkipBorderSize` is in units of 16 cells per side. With skip 2, rows/columns 0–31 and 480–511 are not touched, and the outermost processed ring (row/column 32 and 479) is overwritten with the border values every step. That gives the constant 32(–33)-cell band seen in every save ([../docs/sav-format.md](../docs/sav-format.md#the-32-cell-border-band-is-constant)), so the game runs with skip 2.

## What this means for the earlier questions
- **Frame rate:** every rate is per tick with no time step, so the sim runs faster at higher tick rates. Halving fluidity would *not* simply give the same water at 60 ticks/s, because damping (0.999 per tick), erosion, deposit, cooling and the yield threshold are all per tick as well. A faithful 60 Hz version would need each of them re-derived (inference, not tested).
- **Resolution:** the rules are local and per cell, so the Python model runs at any grid size. But constants like the slump threshold (1 height unit per cell) and fluidity are tied to the cell size, so a 1024 grid needs them re-scaled to behave the same (inference).
- [python-water-sim.md](python-water-sim.md)'s textbook model is superseded by [../tools/dustsim.py](../tools/dustsim.py).

## TODO (skipped, worth revisiting)
- [x] ~~`WaterHeightTemperatureDampingFactor` (index 35) at runtime~~: **1.0**, read from the running game (the earlier guess of 0.125 was wrong).
- [ ] Which code writes the simulator's index 35, and what the value stored at `[0x10184F38]+0xAC` (`LavaHeightTemperatureDampingFactor ÷ [+0xA8]`) is used for.
- [ ] Sea-edge driver: which code writes columns 476/479 each tick and with what formula (~12 s period, ±0.9 near the edge). Probably an AI/level script or modifier in dustai; Map 8 is called "HighTides", worth checking there too.
- [ ] Waterfall sources: which object/modifier adds the fixed amounts, and whether they also set sediment/temperature (lava sources).
- [x] ~~Confirm Jellify uses the slow-water set~~: yes, map-wide, for 60 s (in-game timer; matches the measured window).
- [ ] When the slow *lava* set (`m_slowLavaParameters`) is used: not by Jellify Water (checked on a lava map). Possibly another power, a level script, or unused.
- [ ] When the game uses `m_fluidParameters[1]` (flux caps 1000), i.e. what drives the element index in the copy routine `0x100FBA20`.
- [ ] Drained-lava edge case (Quirks): trigger not isolated; probably the float sign of a zero remainder. Affects a few front cells per 10 steps.
- [ ] Vegetation `ErosionBlocking` (component 6 low nibble → `ParamU8` 16–31) not tested; `ParamU8` values the game passes not read.
- [ ] Not exercised: `SedimentTransportThreshold/Factor` at non-default values, `WaterFluxScale`/`LavaFluxScale` ≠ 1, `LateralErosionWaterHeightThreshold`, and components 6/7 (vegetation, fire), which another job (`SimulatorGanjaIterator`) updates.
- [ ] What the game does outside this step: modifiers (adding water/lava/sand from powers and sources), the border water level changing over time (Map 8 "HighTides"), and how many ticks run per frame in practice.
- [ ] `GroundWaterMax` (0.1) has no Galileo counterpart; the record with hash `0xF271D3DB` (type `0x433`) is unnamed.
