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

Unnamed blocks (no debug string found): 0xA0 (bit 15 set on types 0, 2, 3, 4, 9, 15: flags?), 0x120 (15 / 128 / 256 / 896), 0x160 (16, or 1 on types 9-10, 32767 on 0 and 15), 0x180, 0x1A0, 0x1C0, and 0x1E0 = 8 × type (0, 8, 16 … 120: probably an offset into a per-type table elsewhere). Types 9 and 10 ignite almost at once (8) and flower (150 to 500 / 180): candidates for the fire trees seen on Map 6; types 2, 6, 12 hardly burn (4800). The table may differ per level (only one level read).

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
| 0x1E0 | 8 x type |

**Which set:** The Big Wave = set 0 (set 2 differs on type 13); Waves = set 1 or 3 (they differ only in `lifeGroundType` and the unnamed fields, which are not in the table). Sets 0 / 2 are "dry" (cover spreads only where water is at most 0.5 deep, dies above 1), sets 1 / 3 "wet" (spreads in 0.5-9.5 of water, lives up to 10); types 9 / 10 differ only in their life span (adult 1000 / 200 vs 4000 / 2000). How a level picks its set is not found.

Growth conditions in plain terms (set 0, most types): a plant spreads to a cell whose water depth is between `spreadWaterHeightMin` and `Max` (0-0.5) and whose height differs by at most `spreadDeltaHeightMax` (2) from its neighbour, and dies where water is deeper than `lifeWaterHeightMax` (1). How these are applied per tick (which neighbours, probabilities, the random nibble, `RandomGrowingTime`) is in `Step2` / `Step2_ComputeViability` of `SimulatorGanjaIterator` (not read).

## A cell's life cycle, measured during a live fire (Story 7: Waves)
Logged with `tools/ganja.py watch 120` on **Story 7: Waves** (`ALPHA_Map7_TheSandSeas`, hard ground 79 % match; play had changed the terrain) while a fire spread, 2026-11-16: [../data/live/ganja_watch_20261116_172448.log](../data/live/ganja_watch_20261116_172448.log) (counts once a second plus the full fields of 4 cells, rows/cols 408/157, 409/155, 409/156, 410/153, that burned twice in 2 minutes). Frames are simulation ticks (31 per second).

1. **Burning:** fire bit set, type and status unchanged (type 1, flowering). The fire time (component 7 high half) rises by exactly **10 per tick** (+310 over 31 ticks on every tracked cell) up to the type's **FireDuration** (600 for type 1: about 60 ticks, 2 s). TimeAnim b follows it (about fire time - 45, held at 255).
2. **Burned:** at FireDuration the cell becomes **type 0, status 0, burned bit set**; the fire time restarts at 0 and counts **1 per tick**; TimeAnim b = burned time - ~22. The highest burned time seen on any cell was 246, after which:
3. **Bare, ready to regrow:** type 0, **status 1**, burned bit clear, all timers 0 (about 246 ticks = 8 s after the fire went out; `BurnedTime` parameter, value not read).
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

## TODO (skipped, worth revisiting)
- [ ] Growth rules (next step for modelling vegetation): read `Step2_ComputeViability` (395 lines) and the growth part of `Step2` (1,374 lines) in `GalileoSimulatorWater_9_9.dll` (SSE41; disassembly with `dumpbin -disasm:nobytes`, PDB names load automatically), then check against the live game with `ganja.py` the way the water step was checked (the simstep harness could run the game's own vegetation job too: `CallbackSimulatorGanja`).
- [ ] How a level picks its vegetation set (0-3), types 0 / 15 values, the 6 unnamed set fields, the fire-duration source for types 9-10. Species display names (`Pyrix_Tree`, `FiguierBarrique`, `Bombarde` in the same archive) not tied to letters (not needed for the simulation).
- [ ] Meaning of status 2 and the `die` bit; why the index 1 → 2 switch happens at ~870-890 (static, index and TimeAnim are now explained by the measured life cycle, but only for type 1 on one map). Rules from the code: read `Step1` / `Step2` / `Step2_ComputeViability` of `SimulatorGanjaIterator` (`GalileoSimulatorWater_9_9.dll`, around `0x10011000`+; disassembly via dumpbin) for the growth, flowering and fire-spread rules (fire neighbours mask 7, the 16-byte propagate table `ConstParam+0x440`, copied from `SimulatorParam+0x130`; how `SimulatorParam` lines up with live_check's `prm` is not checked; `FireSpeedProgressBase`, `BurnedTime`, `DefaultGrowingTime`, `SoftGroundQuantityMin`).
- [ ] The per-type times are in ticks (measured for FireDuration and AdultDuration on type 1); fire progress is 10 per tick (probably `FireSpeedProgressBase`, value not read), the burned phase ~246 ticks (`BurnedTime`, not read). The table differs per level (two levels read). The unnamed table blocks (0xA0, 0x120, 0x160-0x1E0).
- [ ] Which type is which plant (fire tree, water tree, grass, …): only The Big Wave read (1 = cover, 11 = village ground, 10 / 9 = one patch, unidentified); look at the type map on a story level (fire trees on Map 6) next to the game.
- [ ] Load components 6 / 7 from `.sav` files (save blocks 3-7 not mapped; docs/sav-format.md) so the model and the viewer have them without the live game. Not modelled in `dustsim` (fire trees evaporating water: findings/water-lava-sim.md TODO).
