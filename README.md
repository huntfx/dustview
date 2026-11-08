# From Dust Water Simulator

Status: **version 0, textbook model on real map data.** It runs From Dust maps loaded from their `.sav` files, but the update rules and constants are a standard "virtual pipes" shallow-water + erosion model, **not yet the game's own**. The game-accurate version needs the steps under "Plan" below.

Tool: [../tools/watersim.py](../tools/watersim.py). Run it with the project venv (`FromDust/.venv`, numpy 2.5.4 + pillow, created 2026-11-08):
```
.venv/Scripts/python tools/watersim.py "<install>/DUST/DustSave/Ingame/ALPHA_Map1_Erosion.sav" --out data/sim/map1_erosion
```
Options: `--steps`, `--every` (frame interval), `--upscale K` (finer grid, same physics in world units), `--source X,Y,R,Q` / `--no-source`. Output: an animated GIF plus a before/after PNG.

## What comes from the game
| Thing | Source |
|---|---|
| Ground surface, soft ground, water depth (512 × 512) | `.sav` blocks 0/2/1 ([../docs/sav-format.md](../docs/sav-format.md)); hard ground = block 0 − block 2 (checked on 4 maps) |
| Per-cell state: 4 outflow fluxes, soft ground, sediment | Debug strings in `GalileoSimulatorWater.dll` (`FluxLeft/Right/Up/Down`, `SoftGroundQuantity`, `SedimentQuantity`, `0x16C3C`–`0x16CC8`) and `SaveWaterOutFlux*` functions |
| Fixed border = open sea | Debug strings `BorderWater`, `BorderHard`, `BorderSoft` (`0x16F38`–`0x16F5C`); the `.sav` outer ring is constant (Map1: ground 30.37, water 9.60) |
| Grid 512 × 512 | `<9,9>` templates; `.sav` layout |

## What is textbook (to be replaced)
- Flux update `f = max(0, damping·f + dt·g·l·Δsurface)`, outflow scaling so a cell never sends more than it holds, water update from net flux (Mei et al. 2007 style).
- Constants: g = 9.81, dt = 0.02, damping 0.998, cell size 1. Erosion capacity = 0.08 × speed × min(depth, 2), erode 0.2 / deposit 0.3 per unit time. All made up to look plausible.
- Not modelled: lava and temperature, vegetation, fire, evaporation, tides/waves at the border, the game's sub-stepping.

## Observations so far
- Map starting states are **not at rest**: on `ALPHA_Map1_Erosion` the open-sea surface varies from 34.3 to 45.5 (border 39.97), so the saved map catches the sea mid-wave. The sim sloshes and settles.
- Speed: ~40 ms per step at 512 × 512 (numpy, one thread), ~2 min for 3000 steps with rendering.

## Plan to make it game-accurate
1. **Read the water step** in `SSE41/GalileoSimulatorWater_9_9.dll` (with PDB names): `SimulatorWaterIterator::Step1` (835 lines), `Step2_split` (1505), `Step4` (1249), driven by `Run12345_*`; constants prepared in `ConstParam` (362 lines, mostly broadcasts and `1/(max−min)` ramp factors from a `SimulatorParam` block).
2. **Capture the parameter values** from memory while a map runs (the `SimulatorParam` block passed to `SimulatorWaterTemplate<9,9>::Run`), since they aren't named in the files.
3. **Validate**: load a map's start state, let the game run N frames, save, and compare the game's grids with the sim's after N steps.
4. Then `--upscale 2` becomes a real test of "finer water, same behaviour".

## TODO (skipped, worth revisiting)
- [ ] Steps 1–3 above.
- [ ] Decide what the 32-cell border shelf is in-game (outside the playable area?) and whether the game drives tides there.
- [ ] Speed: numba or a C extension if bigger grids get too slow.
