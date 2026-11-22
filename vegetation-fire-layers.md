# Vegetation and fire layers (grid components 6 and 7)

The simulation grid has 8 components per cell (findings/water-lava-sim.md, `tools/galileo.py`). Components 0-5 (ground, soft ground, fluid, sediment/temperature, outflow) are decoded and modelled. Components 6 and 7 hold the vegetation ("ganja" in the engine's naming) and fire state, updated by a separate job (`SimulatorGanjaIterator`, `SimulatorGanjaTemplate<9,9>` in `GalileoSimulatorWater_9_9.dll`). PDB enum `ISimulatorWaterConst::tGridDataOffset`: 6 = `kRandomGanja`, 7 = `kGanjaTimeFire`; the component list (`kComponent_*`) splits them into `Ganja`, `GanjaRandom`, `GanjaTime` and `GanjaFire`.

Decoder: [../tools/ganja.py](../tools/ganja.py) (`decode(grid)`, `type_table(ganja_block)`; run it with the game running for field counts and the per-type table). Viewer: DustView, View tab, *Engine texture*, entries under "Vegetation / fire" (live game only; a model started from a `.sav` has both components at 0).

## Bit layout (decoded from the code)
Source: the per-cell debug readout in `SSE41/GalileoSimulatorWater.dll` (code `0x10013AF0`–`0x10013FB0`, strings `0x100186EC`…: `Ganja = %u [s=%u][t=%u][Die =%u]`, `Fire = %u [t=%u]`, `Burned = %u [t=%u]`, `Random = (%u,%u)`, `TimeAnim(%u,%u) Index(%u) Static(%u)`) and `u32_GetGanjaFire` (`0x10002390`, reads the high u16 of component 7). The masks set in `SimulatorGanjaIterator::ConstParam` (`GalileoSimulatorWater_9_9.dll` `0x10005656`) agree: TypeMask `0x0F`, StatusYoung `0x10`, StatusFlowering `0x30`, FireMask `0x40`, BurnedMask `0x80`, TypeBurnedMask `0x8F`, FireNeighboursMask `0x07`, ViabilityMask `0x01`, RandomMask `0x0F`, AccretionMask `0x7FFF7FFF`.

**Component 6** (`kRandomGanja`, u32 per cell)
| Bits | Byte | Field (debug name) | Notes |
|---|---|---|---|
| 0-3 | b0 | vegetation **type** (`Ganja`) | 0 = none; indexes the per-type table below and `ErosionBlocking` |
| 4-5 | b0 | **status** (`s`) | 1 = young (`0x10`), 3 = flowering (`0x30`), 2 = unnamed in the code (probably adult: the table has Evolve → Adult → Flowering durations) |
| 6 | b0 | **fire** | burning now |
| 7 | b0 | **burned** | burnt |
| 8-11 | b1 | **random** nibble (`Random`, second value) | `u32_GetGanjaRandom8` builds an 8-bit random value from these nibbles |
| 12 | b1 | **die** (`Die`) | |
| 13 | b1 | **static** (`Static`) | |
| 14-15 | b1 | **index** (`Index`) | |
| 16-23 | b2 | **TimeAnim** first value | |
| 24-31 | b3 | **TimeAnim** second value | |

**Component 7** (`kGanjaTimeFire`)
| Bits | Field |
|---|---|
| 0-15 | vegetation **time** (the `t` of `Ganja`; growing / age counter) |
| 16-31 | **fire time** (the `t` of `Fire` and of `Burned`) |

The grid in memory is tiled in 16-cell strips (cell offset `((x >> 4) << 7) + (y << (shift + 3)) + (x & 15)`, component c at `+c * 0x40`), which `galileo.unpack` already undoes.

## Per-type table
16 vegetation types; each parameter is 16 values of 16 bits, low bytes at `+off`, high bytes at `+off + 0x10` inside the 512-byte "ganja" parameter block (`ISimulatorWater + 0x150`, i.e. live_check's `prm[0x130:0x330]`). Names from the debug readout (code `0x10013E28`…, indexed by the cell's type). Values read live on **Challenge 4: The Big Wave** (`Defi_006_BigWave`, identified by hard ground, 100 % match), 2026-11-16; units are probably simulation ticks (not checked).

| Parameter (offset) | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FireIgnition (0x00) | 17 | 40 | 4800 | 40 | 40 | 40 | 4800 | 40 | 40 | 8 | 8 | 40 | 4800 | 40 | 40 | 37 |
| FireDuration (0x20) | 640 | 600 | 600 | 600 | 600 | 600 | 600 | 600 | 600 | 500 | 500 | 600 | 600 | 600 | 600 | 640 |
| RandomGrowingTime (0x40) | 64 | 10 | 10 | 10 | 0 | 10 | 10 | 10 | 0 | 10 | 10 | 10 | 10 | 10 | 0 | 16 |
| EvolveDuration (0x60) | 64 | 12 | 40 | 8 | 0 | 12 | 8 | 8 | 0 | 50 | 50 | 8 | 8 | 12 | 0 | 64 |
| AdultDuration (0x80) | 256 | 1000 | 900 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 200 | 1000 | 1000 | 1000 | 1000 | 256 |
| FloweringStartTime (0xC0) | 5 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 150 | 0 | 0 | 0 | 0 | 5 |
| FloweringStopTime (0xE0) | 245 | 1000 | 900 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 500 | 180 | 1000 | 1000 | 1000 | 1000 | 245 |
| FireAnimation (0x140) | 640 | 120 | 400 | 80 | 0 | 120 | 80 | 80 | 0 | 500 | 500 | 80 | 80 | 120 | 0 | 640 |

Unnamed blocks (no debug string found): 0xA0 (bit 15 set on types 0, 2, 3, 4, 9, 15: flags?), 0x120 (15 / 128 / 256 / 896), 0x160 (16, or 1 on types 9-10, 32767 on 0 and 15), 0x180, 0x1A0, 0x1C0, and 0x1E0 = 0, 8, 16 … 120 (not per type: the lookup n -> 8 n, see "Parameters" below). Types 9 and 10 ignite almost at once (8) and flower (150 to 500 / 180): candidates for the fire trees seen on Map 6; types 2, 6, 12 hardly burn (4800). The table may differ per level (only one level read).

Live field counts on the same frame (262,144 cells): type 0 on 218,760, 1 on 36,965 (all flowering), 10 on 3,204 (mostly status 2), 11 on 3,186 (flowering), 9 on 29; random nibble 1-15 each on ~12,400 cells (uniform) and 0 on 75,599; fire / burned / fire time all 0 (no fire on this map); vegetation time 0-1036.

**Type map on The Big Wave** (same frame, DustView *Vegetation type*): type 1 is the general green cover of the islands; **type 11 forms a disc around each of the 3 totems** (totems at row/col 237/167, 301/233, 144/282 all sit on type 11: village ground); type 10 is one patch between the islands (rows 203-272, cols 252-322), with type 9 on 29 cells inside it (rows 210-220, cols 285-294). Everything else is type 0.

## Where the plant settings come from (game files)
The per-type values are data: `BinPC/Wor_BD00993D.lin.bf` (`_GLOBAL.wor`) holds `SpecieVegetationA_0.ai` … `SpecieVegetationN_0.ai`, LyN reflection records. Decoder: [../tools/vegetation_species.py](../tools/vegetation_species.py) → [../data/vegetation/species.tsv](../data/vegetation/species.tsv) (one row per type and set) and [../data/vegetation/fields.txt](../data/vegetation/fields.txt) (every named field of every file). **Letter = type number** (`m_childType` = `0x201000NN`: A = type 1 … N = type 14, all 14 checked). Types 0 and 15 have no species file (their table values, e.g. AdultDuration 256, come from elsewhere; not traced).

Each species file holds a list `m_vegetationParameters` of **4 `DustAi::SpecieVegetal::s_vegetationParams` sets** (23 fields) and 4 `s_slowFireParams` (fireIgnition, fireDuration), plus per-species fields (`m_spreadingVegetationType`, `m_isVillageVegetation` = 1 on types 11-14, `m_erosionBlocking` = 1 on types 9-10, `m_biomass`, `m_absorptionResistanceTime`, `m_humansBlockingFactor`). Fields of a set: `fireIgnition`, `fireDuration`, `randomGrowingTime`, `evolveDuration`, `adultDuration`, `floweringStartTime` / `StopTime`, `nextVegetationType`, `lifeCycleType`, `lifeGroundType` (name recovered by hashing word combinations; not stored as text anywhere in the install), `spreadWaterHeightMin` / `Max`, `spreadDeltaHeightMax`, `lifeWaterHeightMin` / `Max`, `lifeFireResistanceWaterHeight`, `accretionQuantity`, and 6 fields whose names were not recovered (hashes `248d3f06` = 149 or 0, `248d4138` = 1.2 or 0, `3cfbdbca` = 1 or 0, `3cfbdde4` = 150 or 0, `ea491c5f` = 1 or 0, `7d5eb88d` = a 0x49-byte array).

**The live per-type table is built from one set** (checked against both live tables, all 14 types, all 15 blocks; offsets in the 512-byte ganja block, values floored):
| Block | = |
|---|---|
| 0x00 | 8 x `fireIgnition` |
| 0x20 | `fireDuration` (types 9-10: 500 = the slow-fire duration, not the set's 200: open, fire) |
| 0x40 / 0x60 / 0x80 | `randomGrowingTime` / `evolveDuration` / `adultDuration` |
| 0xA0 | `nextVegetationType` & 0xFF, + 0x8000 when `lifeCycleType` = 0 |
| 0xC0 / 0xE0 | `floweringStartTime` / `StopTime` |
| 0x100 / 0x120 | 128 x `lifeWaterHeightMin` / `Max` |
| 0x140 | 10 x `evolveDuration` (labelled "FireAnimation" in the DLL's readout) |
| 0x160 | `lifeFireResistanceWaterHeight` |
| 0x180 / 0x1A0 | 128 x `spreadWaterHeightMin` / `Max` |
| 0x1C0 | 128 x `spreadDeltaHeightMax` |
| 0x1E0 | 8 x index (`m_speedIgnitionBase` x n lookup, not per type; see "Parameters") |

**Which set:** The Big Wave = set 0 (set 2 differs on type 13); Waves = set 1 or 3 (they differ only in `lifeGroundType` and the unnamed fields, which are not in the table). Sets 0 / 2 are "dry" (cover spreads only where water is at most 0.5 deep, dies above 1), sets 1 / 3 "wet" (spreads in 0.5-9.5 of water, lives up to 10); types 9 / 10 differ only in their life span (adult 1000 / 200 vs 4000 / 2000). How a level picks its set is not found.

Growth conditions in plain terms (set 0, most types): a plant spreads to a cell whose water depth is between `spreadWaterHeightMin` and `Max` (0-0.5) and whose height differs by at most `spreadDeltaHeightMax` (2) from its neighbour, and dies where water is deeper than `lifeWaterHeightMax` (1). How these are applied per tick (which neighbours, probabilities, the random nibble, `RandomGrowingTime`) is in `Step2` / `Step2_ComputeViability` of `SimulatorGanjaIterator` (not read).

## A cell's life cycle, measured during a live fire (Story 7: Waves)
Logged with `tools/ganja.py watch 120` on **Story 7: Waves** (`ALPHA_Map7_TheSandSeas`, hard ground 79 % match; play had changed the terrain) while a fire spread, 2026-11-16: [../data/live/ganja_watch_20261116_172448.log](../data/live/ganja_watch_20261116_172448.log) (counts once a second plus the full fields of 4 cells, rows/cols 408/157, 409/155, 409/156, 410/153, that burned twice in 2 minutes). Frames are simulation ticks (31 per second).

1. **Burning:** fire bit set, type and status unchanged (type 1, flowering). The fire time (component 7 high half) rises by exactly **10 per tick** (+310 over 31 ticks on every tracked cell) up to the type's **FireDuration** (600 for type 1: about 60 ticks, 2 s). TimeAnim b follows it (about fire time - 45, held at 255).
2. **Burned:** at FireDuration the cell becomes **type 0, status 0, burned bit set**; the fire time restarts at 0 and counts **1 per tick**; TimeAnim b = burned time - ~22. The highest burned time seen on any cell was 246, after which:
3. **Bare, ready to regrow:** type 0, **status 1**, burned bit clear, all timers 0 (about 246 ticks = 8 s after the fire went out; `BurnedTime` = 246: the cell stays burned while the counter is <= 246, see "Fire rules").
4. **Regrowth:** within ~40-60 ticks the cell is type 1 again (status 1 for a few ticks, then 3, since type 1's Evolve / Flowering times are 0 on this map), vegetation time from 0 at 1 per tick, **TimeAnim a rising ~2 per tick from 0 to 255** (the plant growing into view); at 255 the **static** bit is set (fully grown, index 1).
5. **Renewal:** at vegetation time ~870-890 the index goes 1 → 2, static clears and TimeAnim a ramps 0 → 255 again; at **1000 (= AdultDuration)** the vegetation time wraps to 0 and the index returns to 1. The plant replaces itself every AdultDuration ticks without leaving its type.

So the raw-byte trail seen on C7 b2 (white, black, fading back to white) is this: the low byte wrapping quickly while burning (10 per tick), the reset to 0 when the fire turns into the burned state, then the slow climb (1 per tick) through the burned phase.

Counts over the 2 minutes: burning cells 160 → ~1,070, burned 116 → ~4,900; the fire moved through types 1, 5, 13 and then 11 (village ground). Burned cells are all type 0 (the type is cleared when the fire ends). The burned count also falls at times, as cells finish the burned phase. Seen in the per-type counts on this map: the die bit on ~1,500 cells (not on the tracked cells; meaning still open).

**The per-type table differs per level.** Waves vs The Big Wave: types 1-8 and 11-14 have RandomGrowingTime 0 (10 on Big Wave), EvolveDuration 0 for most (8-40 on Big Wave), FireAnimation 0 for most; types 9 / 10 live and flower much longer (AdultDuration 4000 / 2000, flowering 3850-3900 / 1850-1900, vs 1000 / 200 and 150-500 / 150-180); the unnamed blocks 0x120 (1280 vs 128), 0x160 (255 on types 9-10), 0x180 (64 vs 0) and 0x1A0 (1216 vs 64) differ too. FireIgnition and FireDuration are the same on both.

## Observed in play (DustView, raw bytes) and what they are
Observations made before the layout was decoded, with the decoded fields they correspond to. Note: the raw-byte entries were scaled to each frame's maximum, so a map where a byte has one value everywhere showed pure white (e.g. "C6 b0 pure white" on a challenge map without fire); the viewer now uses fixed ranges and decoded fields.

| Byte | Observed | Decoded |
|---|---|---|
| C6 b0 | Solid cover where foliage is, lighter at the village, pure white where it burned | type + status + fire (0x40) + **burned (0x80)**: burnt cells have the top bit, so they are brightest |
| C6 b1 | Static on a map without fire; grows outwards within the green on another map | **random nibble** (the static) + die / static / index bits (change as vegetation spreads) |
| C6 b2 | Crushed black-and-white image, grows outwards | **TimeAnim** first value |
| C6 b3 | Resembles C7 b2 | **TimeAnim** second value |
| C7 b0 | Patterns over what can burn, rings growing outwards | low byte of the **vegetation time** (wraps every 256, hence the fine rings) |
| C7 b1 | Like b0, more zoomed out | high byte of the vegetation time (same counter, 256 × slower) |
| C7 b2 | Fire with a trail: white, then black, then back to white | low byte of the **fire time**: wraps fast while burning (10 per tick), resets to 0 when the cell turns burned, then climbs 1 per tick through the burned phase (measured, see the life cycle above) |
| C7 b3 | The active fire | high byte of the fire time |

## Running the game's own vegetation step (2026-11-21)
The vegetation / fire step is job 4 of `GalileoSimulatorWater_9_9.dll` (`sCodeAddrJob(4)` = `CallbackSimulatorGanja`, same `SimulatorParam` as the water job). It reads ground, sand and water (components 0-3) and writes components 6 / 7. Ways to run it:
- one-off: `SIMSTEP_JOB=4` (simstep.exe), `galileo.run(..., job=4)`, wrapper [../tools/vegsim.py](../tools/vegsim.py) (`from_sav`, `plant`, `run_dll`);
- every tick: `simstep --serve` keeps the DLL loaded and shares the grid through named shared memory (`galileo.Resident`, ~5 ms per tick including packing, bit-identical to the file harness). DustView's model mode uses it (Model tab: *Plants grow and burn*).

**Checked against the live game** (Story 9: The Tears of Stone, `ALPHA_Map9_TheGreatCaldeira`, identified by hard ground, 76 % match; 2026-11-21): from a live tick's grid the harness step reproduces the game's next tick **exactly in component 7** (all ~79,000 changing cells, 3 ticks) and in component 6 on all but 48-86 cells per tick (of ~22,000 changing). The parameter image built by `vegsim` (no live data needed) equals the live one byte for byte in the per-type block and ParamI, and gives the same result as the live image. The leftover cells are all **new growth the game did not make**: the step started a plant (0x11) or marked bare ground ready (0x10) where the game kept 0. They are not concentrated at the totems (median 62 cells from the nearest one, vs 162 for bare cells in general). Running the step after the water step instead made it worse (75-86 cells, and component 7 no longer exact), so the game runs it on the same start state as the water step. Probably something outside the step blocks growth there (see TODO).

### Parameters (all now known)
From the `ConstParam` constructor (`0x10005656`) and the live read:

| Where | Value | Meaning |
|---|---|---|
| ParamI[0] (`SimulatorParam+0xF0`) | 10 | `SoftGroundQuantityMin`: sand x 128 must be above this (= `m_ganjaSoftGroundQuantityMin` 0.08 x 128) |
| ParamI[4] | 1 | `DefaultGrowingTime`: vegetation time added per tick |
| ParamI[5] | 10 | `FireSpeedProgressBase` (= `m_speedProgressBase`) |
| ParamI[7] | 246 | `BurnedTime` (= `m_cellBurnedTime`) |
| ParamI[1], [2], [3], [6] | 2, 10, 1, 0 | not used by this step as far as read; names unknown |
| `SimulatorParam+0x130` (16 bytes) | 0,1,2,3,4,1,2,3,4,9,9,1,2,1,4,0 | `PropagateTable`: type t spreads into neighbours as this type = the species' `m_spreadingVegetationType` (5 -> 1, 10 -> 9, 11 / 13 -> 1, ...); 15 -> 0 |
| `SimulatorParam+0x140` (512 bytes) | species set (table above) | live Story 9 = **set 0** exactly for types 1-14 |

Each ParamI int32 holds its 16-bit value twice (`0x000A000A`): the step reads them as 16-bit lanes, so a value in the low half only reaches half the cells (an early harness run showed timers advancing in some cells only, for this reason). Types 0 and 15 (no species file) in the per-type block, live, the same on The Big Wave and Story 9: FireIgnition 17 / 37, FireDuration 640, RandomGrowingTime 64 / 16, EvolveDuration 64, AdultDuration 256, NextType 0x800F, Flowering 5-245, LifeWaterMax 15, block 0x140 640, LifeFireResist / SpreadWaterMax 32767, SpreadDeltaMax 15 (`vegsim.LIVE_T0` / `LIVE_T15`). **Block 0x1E0 is not "8 x type"**: the code indexes it by small counts (burning neighbours, the random nibble), so it is the lookup `n -> 8 n` (= `m_speedIgnitionBase` 8 x n); its values happen to equal 8 x index.

### Growth rules (read from the code, behaviour measured with the harness)
`Step2_ComputeViability` returns one byte per cell:
- low nibble: the type that would **spread into** this cell: the highest type among the 4 neighbours, counted only where the neighbour is **flowering** (status 0x30), the height step between the two cells is below the neighbour type's `SpreadDeltaMax`, this cell's water depth is inside that type's `SpreadWaterMin`-`Max`, and this cell has more sand than `SoftGroundQuantityMin`. `Step2` maps it through the `PropagateTable`.
- bits 4-5: number of burning 4-neighbours **modulo 4** (summed as 0x40 per neighbour in a byte, so 4 wraps to 0; see "Fire rules"); bit 7: viability of the cell's own plant = (water depth inside `LifeWaterMin`-`Max` and sand x 128 > `SoftGroundQuantityMin`) **or all 4 neighbours have the cell's own type** (t != 0; code `0x10027DB3`-`0x10027E11`, `0x10027FEC`). Confirmed in the harness: a type-1 plant with no sand, or under lava, lives on if its 4 neighbours are type 1, whatever their status, but not if they are type 9. A plant that is not viable starts its dying phase (vegetation time raised to `RandomGrowingTime` + r + `EvolveDuration` + `AdultDuration`, status 0) and becomes bare ground (0x10) once the time passes that + `EvolveDuration` (type 1: alive up to 1,034).

Status is recomputed from the vegetation time every tick (young below `EvolveDuration`, flowering between `FlowerStart` and `FlowerStop`), so a hand-set status on a plant with time 0 becomes young after one step.

Measured on a flat sand plain (water layer fixed, one flowering type-1 patch of 4 x 4 cells, 600 ticks): set 0 spreads over dry ground and water up to 0.5 deep (16 -> 290 cells, about one cell outwards every 30-60 ticks) and the plants die in 1.0 of water; set 1 does not spread on dry ground at all and spreads fast in 1.0 of water (16 -> 946). The Big Wave from its level file is stable over 620 ticks (cover already fills the islands; saplings 9 mature into trees 10: 27 -> 16 / 3,248 -> 3,258). Map 2 (`ALPHA_Map2_Archipel`) from its level file grows only 1,571 -> 1,599 cells in 1,500 ticks with set 0 (3,503 with set 1), while the player's save of that level has 26,569: most of the spread in play must come from outside the step (TODO).

### Starting plants from the level files
The `.sav` files hold the vegetation layers (`docs/sav-format.md`, reader `sav.vegetation_components`): the plant byte, the random nibble, vegetation time and fire time, loaded by the interface DLL's `FillGanja*` functions. DustView loads them with the map, so model runs start with the level's plants (e.g. Map 6 *Wildfires* starts with 3,218 burned cells and burning fronts). Plants are drawn in DustView's natural view with its own colours per type (`ganja.colours`; the game draws 3D models), under the water layer; burning cells orange, burned dark grey (View tab: *Plants and fire*).

## Grass on the ground (how the game draws it; in DustView since 2026-11-21)
Besides the 3D plant models, the game paints a grass layer onto the terrain from the plant byte (component 6 byte 0), in two shader passes (`Cache/fxlist_PC.fxl`, `tools/fxcache.py dis 113` / `96`):
1. **Shader 113** (`Technique_ComputeGrassColorAndOpacity`, vertex shader 112) renders the 512 x 512 target `engine/GrassColorRT` (created at exe `0x00454279`; one texel per cell). Per cell it reads the plant byte of the cell and its **4 diagonal neighbours** (from the 512 x 1024 ground upload, byte z), looks up `samp_GrassColorRamp` at U = type / 16 for each, **averages the 5** and pulls the result towards `g_BurnGroundColor` by the share of the 5 bytes that are >= 0x40 (burning or burned): `out = avg + (burn - avg) x burnt / 5`.
2. **Shader 96 / 97** (terrain): `Grass_D` (`Tex_BD00993D.lin.bf`, 1024 x 1024 DXT5, tiled 16 x per map like the sand / rock textures; sampler `g_SamplerGrassGroundNormal`) and GrassColorRT (bilinear, map UV) give `colour = lerp(terrain colour, Grass_D.rgb x GCO.rgb, saturate(2 x GCO.a + Grass_D.a - 1))` before the lighting. The texture's alpha makes thin cover patchy instead of a flat tint.

The ramp and burn colour are not in any data file found: the exe fills the 16 x 1 ramp texture (`[renderer+0x3088]`, code `0x004541F0`-`0x00454229`, swapping bytes 0 and 2 for the texture format) from **17 u32 colours** at `[[renderer] + 0x150]` (the 17th is the burn colour, set as constant c128 x 1/255 at `0x004543E9`). The renderer is the exe global `0x00921754` (the per-frame caller `0x00829C30` stores the object passed in at `[renderer]` before calling `0x00455A40`, which calls the grass pass `0x00454140`). Read live on Story 9 (2026-11-21), as R, G, B, opacity:

| Types | Colour | Opacity |
|---|---|---|
| 0 | (255, 255, 255) | 0 (no grass) |
| 1, 3, 5, 7 | (130, 130, 130): `Grass_D` at about half brightness | 163 |
| 11 / 13 (village ground) | (130, 130, 130) | 103 / 140 |
| 2, 6, 12 | (30, 16, 12) dark brown | 92 |
| 4, 8, 14 | (92, 184, 188) pale cyan | 92 |
| 9, 10 | (66, 55, 55) | 133 |
| 15 | (255, 255, 255) | 127 |
| burn colour | (48, 46, 81) | 107 |

DustView (textured natural view): `ganja.grass_color_opacity` = shader 113 with these colours (`ganja.GRASS_RAMP_U32`, `BURN_GROUND_U32`), `flowtex._shade_textured` = shader 96's blend; the untextured natural view keeps DustView's own plant colours (`ganja.colours`). Compared with a game screenshot (2026-11-21) the grass is greener / less yellow in DustView: the game's colour grading (post-process) is not modelled (game-wide TODO). Fire itself (flames) is drawn by particles in the game, not by the ground layer: burning cells show only the burn tint.

## Fire rules (read from the code, checked against the DLL, 2026-11-22)
Read from `Step1`, `Step2` (`0x10028C0A`, fire part `0x1002962F`-`0x1002A3EB`) and `Step2_ComputeViability` of `SimulatorGanjaIterator` (`GalileoSimulatorWater_9_9.dll`; annotated listing: [../tools/ganja_asm.py](../tools/ganja_asm.py)), each rule then confirmed with single-cell experiments in the harness. Written out in Python in [../tools/fire_rules.py](../tools/fire_rules.py), which predicts each tick's fire time and fire / burned bits from the previous tick and compares them with the game's DLL: **0 mismatching cells** over 300 ticks of Map 6 *Wildfires* from its level file (134,843 burning and 945,378 burned cell-ticks), and over 300 ticks of the same map with a lava band and a band of 16 water depths (0.05-0.2) painted over its plants (193,623 burning cell-ticks). Plants that die of old age on a tick are left to the growth rules (see the last point).

One 16-bit timer per cell (component 7 high half, "fire time") does three jobs: ignition heat, burn timer, burned counter. Tables: `FireIgnition` (block 0x00, = 8 x `fireIgnition`), `FireDuration` (0x20), `FireAnim` (0x140), `LifeFireResistWH` (0x160) of the per-type table; `FireSpeedProgressBase` = 10, `BurnedTime` = 246 (ParamI). Order within a tick, per cell (t = type, r = random nibble, ft = fire time):

1. **Lava (`Step1`, `0x100285A9`)**: where fluid < 0, the step's working copy gets water = 0 and sand = 0 for this cell, and a cell with a plant (t != 0) gets the fire bit in that copy only. So a plant under lava counts as a **burning neighbour** for the cells around it and dies (no sand, see viability below) unless all its 4 neighbours are plants of its own type. Its own saved state does not show fire, and it can catch fire normally from burning neighbours, since its water counts as 0. Lava without a plant does nothing, and lava next to a plant does not heat it.
2. **Burning neighbours** n = the number of the 4 direct neighbours (not diagonals) with the fire bit in the working copy, i.e. as left by the previous tick plus lava plants. The game adds the neighbours' 0x40 bits in one byte (`paddb`, `0x100283DD`) and keeps 2 bits (`psrld 6` at `0x100283F8`), so **4 burning neighbours wrap to n = 0**: a plant surrounded by fire on all 4 sides gets no heat from them and cools instead. That's a quirk of the game, confirmed in the harness (3 burning neighbours: +24 per tick; 4: heat halves).
3. **Heat / burn timer** (not burned): if ft < `FireIgnition[t]` + 8 r (heating), ft += 8 n, and with n = 0 ft = (ft + 1) >> 1 instead (`pavgw` with 0: halves rounding **up**, so 1 never decays to 0). Otherwise (alight) ft += `FireSpeedProgressBase` (10).
4. **Water puts fire out**: if round(water depth x 128) >= `LifeFireResistWH[t]` (water = 0 under lava; rounding to nearest, `cvtps2dq`), ft = 0. This is both extinguishing and fire-proofing: heat can't build up in that water. Set 0 values: 16 (depth 0.125) for most types, **1** for types 9 / 10 (any water from 1/256 deep), 32767 for types 0 / 15.
5. **Bare ground** (t = 0, not burned): ft = 0. Type 0 never ignites.
6. **Fire bit** = not burned, t != 0 and ft > `FireIgnition[t]` + 8 r. A plant therefore needs heat >= the threshold at the start of a tick to switch to burning (+10), and is alight from the first tick its timer is above the threshold. Minimum igniting heat measured for random nibble 0-15: type 1 40, 48 … 160; type 9 8, 16 … 128.
7. **Burn-out**: when a burning cell's ft > `FireDuration[t]` (600 for most types, 500 for 9 / 10), the cell becomes **0x80** (type 0, status 0, burned) with ft = 0 and vegetation time 0.
8. **Burned phase**: a burned cell takes no heat; ft += 1 per tick; when ft > `BurnedTime` (246) the cell becomes 0x10 (bare, ready for new plants) with timers 0. So ash lasts 247 ticks (~8 s).
9. **Look of the dying plant** (`FireAnim`, growth side): while burning, once ft > `FireDuration[t]` − `FireAnim[t]` the plant's status is cleared (byte 0x41) and its vegetation time is raised to at least `RandomGrowingTime` + r + `EvolveDuration` + `AdultDuration` (the start of its dying phase; 1,022 for type 1 with r = 0). `FireAnim` = 10 x `evolveDuration`, i.e. exactly `EvolveDuration` ticks at 10 per tick, so the dying phase and the fire end together. A plant whose life ends first (old age or no sand) turns into bare ground (0x10, or straight into a new young plant) with ft = 0 and leaves **no ash**. This beat the burn-out on 81 cells in the Map 6 run.

Seen in the harness while testing: the step leaves the **outer 32 cells** of the map alone (`skip_border` 2 = 2 blocks of 16 cells, the value the live check matched with); cells there neither grow nor burn.

## TODO (skipped, worth revisiting)
- [ ] Villages (observed in play, 2026-11-21, screenshot next to DustView): a village lays out bunting around itself, which changes the ground texture and which plants grow there; in the game that ground looks sandy / brown with palms where DustView's grass layer shows plain green cover. Deliberately not simulated (on request: probably a whole set of village rules). DustView differs from the game around built villages for this reason. The brown spiky trees do match (types 2 / 6 / 12, dark-brown grass colour). 3D plant models: out of scope (on request).
- [ ] Grass colours: read on one level only (Story 9); whether they differ per level (they sit in a renderer settings object whose source is not found) is not checked. The words after the 17 colours (`0101070f 0101070f 0f0f0f0f 00000808 ...`) are not decoded. Texture format of the 16 x 1 ramp (LyN format code 0x16) assumed to keep alpha (the shader needs it: type 0 has opacity 0).
- [x] ~~Fire rules: finish reading `Step2` and check every rule against the DLL~~ (done 2026-11-22: "Fire rules", `tools/fire_rules.py` exact on Map 6; annotator kept as `tools/ganja_asm.py`).
- [ ] Fire rules not exercised by a real level: lava under plants was only tested on a painted Map 6 and in single-cell tests (Map 10's level file has no plants touching lava, so no fire there); species sets other than 0 not run (set 1 / 3 may have other `LifeFireResistWH`). Fire trees / phoenix plants (Map 6) spreading fire and evaporating water are **not** in this step (game-wide TODO): the step only spreads fire to 4-neighbours by heat, so whatever makes those trees special must be outside it (AI / interface DLL).
- [ ] Not checked in the fire checker: vegetation time and status while burning (rule 9 is from single-cell runs only), and that `FireSpeedProgressBase` / `BurnedTime` other than 10 / 246 behave the same (only the live values were run).
- [ ] The 48-86 cells per tick where the game blocks new growth that the step makes (live check above): find what does it (people / creatures walking, `m_humansBlockingFactor`? paths? a second writer of component 6 in the AI or the interface DLL). Not modelled in DustView.
- [ ] Where most new vegetation in play comes from (Map 2: 1,571 plants in the level file, 26,569 in the player's save; the step alone adds few): probably villages planting village ground (types 11-14, `m_isVillageVegetation`, which spreads as type 1); the role of `VegetationInterface`'s `m_activateRandom` 1 / `m_randomGridRange` 16 is unknown. Vegetation does **not** start by itself where water meets sand (from play, 2026-11-21; an earlier note said it did). Not modelled.
- [ ] How a level picks its species set: known Big Wave 0, Story 7 1 (or 3), Story 9 0; DustView uses `vegsim.LEVEL_SETS`, default 0 (a guess for other levels).
- [ ] DustView's model feeds the vegetation step its own water, but the model's water step ignores the plants' erosion blocking (`mv_ErosionBlocking`, types 9-10; not checked in `dustsim`).
- [ ] A Python re-implementation of the vegetation step (DustView runs the game's DLL instead, which is exact; a port would only be needed without the 32-bit helper). The fire part is done and exact (`tools/fire_rules.py`); the growth part (spread, status, vegetation time, life cycle / `NextType`, viability incl. the 4-same-neighbours rule) is read in outline and partly measured, not instruction by instruction. `fire_rules.check` is the pattern to follow: predict each tick from the DLL's previous tick and list mismatching cells.
- [ ] How a level picks its vegetation set (0-3), types 0 / 15 values, the 6 unnamed set fields, the fire-duration source for types 9-10. Species display names (`Pyrix_Tree`, `FiguierBarrique`, `Bombarde` in the same archive) not tied to letters (not needed for the simulation).
- [ ] Meaning of status 2 and the `die` bit; why the index 1 → 2 switch happens at ~870-890 (static, index and TimeAnim are now explained by the measured life cycle, but only for type 1 on one map). Rules from the code: read `Step1` / `Step2` / `Step2_ComputeViability` of `SimulatorGanjaIterator` (`GalileoSimulatorWater_9_9.dll`, around `0x10011000`+; disassembly via dumpbin) for the growth, flowering and fire-spread rules (fire neighbours mask 7, the 16-byte propagate table `ConstParam+0x440`, copied from `SimulatorParam+0x130`; how `SimulatorParam` lines up with live_check's `prm` is not checked; `FireSpeedProgressBase`, `BurnedTime`, `DefaultGrowingTime`, `SoftGroundQuantityMin`).
- [ ] The per-type times are in ticks (measured for FireDuration and AdultDuration on type 1; fire progress 10 per tick = `FireSpeedProgressBase` and the burned phase = `BurnedTime` 246, both now read and checked). The table differs per level (two levels read). Unnamed table blocks still open: 0xA0 is `NextType` (life cycle, not read in detail); 0x160 is `LifeFireResistWH` (used, see "Fire rules"); 0x180-0x1C0 are the spread limits.
- [ ] Speed of the vegetation step in DustView's model (measured 2026-11-21, Map 9 save, ~38,000 plant cells): ~7.3 ms per tick on top of the ~10 ms water tick; ~4.9 ms is the game's DLL itself (numba cannot help), ~2.3 ms packing the grid into / out of the helper's shared memory (`galileo.Resident.put` / `get`, interleaved layout). A full 8-layer grid copy per tick (3 ms) was removed the same day (results bit-identical over 300 ticks). Not done, on purpose: running the DLL's vegetation step at the same time as the water tick (it is a separate process) would hide its ~5 ms, but the plants would then see the water from before that tick's water step, i.e. vegetation first, then water. Do this only once the game's own job order per tick is known (read where the game queues job 4 vs the fluid jobs in `GalileoSimulatorWater_9_9.dll` / the exe) and it turns out to be vegetation first, or the DLL can be given the post-tick water some other way; the simulation must keep matching the game.
- [ ] Which type is which plant (fire tree, water tree, grass, …): only The Big Wave read (1 = cover, 11 = village ground, 10 / 9 = one patch, unidentified); look at the type map on a story level (fire trees on Map 6) next to the game.
- [x] ~~Load components 6 / 7 from `.sav` files~~ (done 2026-11-21: `sav.vegetation_components`, DustView model mode). Fire trees evaporating water is still not in `dustsim` (findings/water-lava-sim.md TODO).
