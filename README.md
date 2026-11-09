# From Dust Water Simulator

Status: **version 0, textbook model on real map data.** It runs From Dust maps loaded from their `.sav` files, but the update rules and constants are a standard "virtual pipes" shallow-water + erosion model, **not yet the game's own**. The game-accurate version needs the steps under "Plan" below.

Tool: [../tools/watersim.py](../tools/watersim.py). Run it with the project venv (`FromDust/.venv`, numpy 2.5.4 + pillow, created 2026-11-08):
```
.venv/Scripts/python tools/watersim.py "<install>/DUST/DustSave/Ingame/ALPHA_Map1_Erosion.sav" --out data/sim/map1_erosion
```
Options: `--steps`, `--every` (frame interval), `--upscale K` (finer grid, same physics in world units), `--source X,Y,R,Q` / `--no-source`. Output: an animated GIF plus a before/after PNG.

### Interactive sandbox
[../tools/watersim_live.py](../tools/watersim_live.py) opens a Tkinter window (Tk 8.6, part of the standard Python install) and runs the same simulator live:
```
.venv/Scripts/python tools/watersim_live.py            # Map 1 (Erosion)
.venv/Scripts/python tools/watersim_live.py Atlantis   # any map by part of its name
```
Left mouse adds, right mouse removes, at the cursor (soft-edged brush shown as a ring); wheel = brush size; **1** water / **2** sand; Space pause; **+/−** steps per frame; **R** reset; **S** screenshot to `data/sim/`; Esc quit. Self-test (paints water on the highest inland point, 150 frames): 14.7 fps at 3 steps/frame, displayed at 1024 × 1024.

Speed-up (2026-11-09): neighbour shifts use slices into preallocated arrays instead of `np.roll` (no copies, no wrap-around at map edges), and sediment now travels with the water fluxes at the cell's concentration instead of semi-Lagrangian advection. Step 56 → 11 ms, render 51 → 18 ms (ground shading cached, refreshed every 6–8 frames; water colour from a depth lookup table). Without the border reset, water and soft ground + sediment are conserved to 6 decimal places over 300 steps. The batch images above were made with the earlier, slower version.

## First result: `ALPHA_Map1_Erosion`, water poured on the upland at (443, 204)
3000 steps (60 time units), 141–157 s. Left: the map's saved start state; right: after the run. The poured water floods the sandy basin in the upper right and reshapes its sand.

![](../data/sim/map1_erosion_before_after.png)

Animation (101 frames): ![](../data/sim/map1_erosion.gif)

The first run without sand slumping left horizontal stripes in eroded sand (a known artefact of simple erosion rules); adding a repose-slope "sand slides" step removed them. A little streaking remains at the top edge of the large sand plateau.

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
- Sand slumping: soft ground steeper than slope 0.8 per cell sheds 25% × half its excess to lower neighbours each step. In the game sand visibly flows; whether it uses a rule like this is unknown.
- Not modelled: lava and temperature, vegetation, fire, evaporation, tides/waves at the border, the game's sub-stepping.

## Observations so far
- Map starting states are **not at rest**: on `ALPHA_Map1_Erosion` the open-sea surface varies from 34.3 to 45.5 (border 39.97), so the saved map catches the sea mid-wave. The sim sloshes and settles.
- Speed: ~11 ms per step at 512 × 512 (numpy, one thread) after the 2026-11-09 speed-up; it was ~40–56 ms before.

## The game can dump and load its own simulation state
`SimulatorWater::SaveGridData(uint, const char *file)` (`GalileoSimulatorWater.dll` `0x10007C30`, virtual) opens `file` via the Galileo kernel (`IKernel` vtable `+0x18`), writes `1 << (log2W + log2H + 5)` bytes from the current half of the double-buffered grid (`this+0x950 + (this[+0x92C] & 1) * 4`, the `mapav_GridData[0/1]` of the debug strings), and closes it (`+0x1C`). With `this+0x920 = this+0x924 = 9` (the first two `sCreate` sizes are log2 width/height) that is **8 MB = 32 bytes per cell**, the full internal state. `LoadGridData` (`0x10007CC0`) is the mirror image. Both are reachable from an in-game hook (the `SimulatorWater` object hangs off `GalileoInterface` +12, per the FromDustHook headers), which gives exact before/after snapshots of single simulation steps. The 32-byte cell layout is not decoded yet; the setup code's `Rcp2exp16` constant suggests some fields are 16-bit fixed point.

## Plan to make it game-accurate
1. **Read the water step** in `SSE41/GalileoSimulatorWater_9_9.dll` (with PDB names): `SimulatorWaterIterator::Step1` (835 lines), `Step2_split` (1505), `Step4` (1249), driven by `Run12345_*`; constants prepared in `ConstParam` (362 lines, mostly broadcasts and `1/(max−min)` ramp factors from a `SimulatorParam` block).
2. **Capture the parameter values** from memory while a map runs (the `SimulatorParam` block passed to `SimulatorWaterTemplate<9,9>::Run`), since they aren't named in the files.
3. **Validate**: load a map's start state, let the game run N frames, save, and compare the game's grids with the sim's after N steps.
4. Then `--upscale 2` becomes a real test of "finer water, same behaviour".

## TODO (skipped, worth revisiting)
- [ ] Steps 1–3 above.
- [ ] Decide what the 32-cell border shelf is in-game (outside the playable area?) and whether the game drives tides there.
- [ ] Speed: numba or a C extension if bigger grids get too slow.
