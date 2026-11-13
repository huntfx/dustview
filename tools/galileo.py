"""Drive From Dust's own water/lava step (GalileoSimulatorWater_9_9.dll) from Python.

The 32-bit harness tools/simstep/build/simstep.exe loads the game's worker DLL and runs
N steps on a grid file; this module builds the grid and parameter images and reads results.

Grid layout (from ISimulatorWater::fGetFluidHeight etc. in GalileoSimulatorWater.dll):
  float index = (y << 12) + (x >> 4) * 128 + comp * 16 + (x & 15)
  i.e. each row = 32 groups of 16 cells; each group = 8 components x 16 values (512 bytes).
Components (PDB enum tGridDataOffset):
  0 TerrainSurfaceHeight  1 SoftGroundQuantity  2 WaterHeight (negative = lava)
  3 TemperatureSedimentQuantity (sediment if water, temperature if lava)
  4 OutFluxLeftRight  5 OutFluxUpDown  (u16 pairs)  6 RandomGanja  7 GanjaTimeFire

Usage as a library:
  import galileo as g
  P = g.fluid_params()                  # the 56 ParamF values (from SpecieFluid_0.ai)
  grid = g.empty_grid()                 # (8, 512, 512) uint32 array
  g.setf(grid, g.WATER, some_float_array)
  out = g.run(grid, P, steps=10)
"""
import os, struct, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL = r'D:\Games\Steam\steamapps\common\From Dust'
WORKER = os.path.join(INSTALL, 'SSE41', 'GalileoSimulatorWater_9_9.dll')
SIMSTEP = os.path.join(HERE, 'simstep', 'build', 'simstep.exe')
MODSTEP = os.path.join(HERE, 'simstep', 'build', 'modstep.exe')
MODIFIER_DLL = os.path.join(INSTALL, 'SSE41', 'GalileoSimulatorWater.dll')
# SimulatorModifierTemplate::* RVAs in MODIFIER_DLL (from its PDB), by ISimulatorWaterConst::tModifier name
MODIFIER_RVA = {'SoftGround': 0xA8F0, 'HardGround': 0xB420, 'RepulseFluid': 0xBEE0, 'AddLava': 0xDC60,
                'RemoveLava': 0xEED0, 'AddWater': 0xF800, 'RemoveWater': 0x10B40}
N = 512

TERRAIN, SOFT, WATER, TEMPSED, FLUX_LR, FLUX_UD, RANDGANJA, GANJATIMEFIRE = range(8)

# kSimulatorParamF_* (PDB enum ISimulatorParamConst::<unnamed-tag>)
PARAM_NAMES = [
    'WaterFluxDamping', 'WaterFluidity', 'SoftGroundErosionThreshold', 'SoftGroundErosionFactor',
    'SedimentTransportThreshold', 'SedimentTransportFactor', 'SedimentDepotThreshold', 'SedimentDepotFactor',
    'SedimentRatioThreshold0', 'SedimentRatioThreshold1', 'WaterFluxDampingMin', 'WaterFluidityMin',
    'LavaFluidityThreshold0', 'LavaFluidityValue0', 'LavaFluidityThreshold1', 'LavaFluidityValue1',
    'LavaFluxThreshold', 'WaterTemperatureMax', 'SoftGroundErosionMax', 'SedimentDepotMax',
    'NormalScaleGround', 'NormalScaleWater', 'DisplayScale', 'SoftGroundDisplayMask',
    'TemperatureDampingThreshold', 'TemperatureDampingFactor', 'WaterFluxScale', 'LavaFluxScale',
    'LavaFluxDampingMin', 'LavaFluxDampingMax', 'SoftGroundErosionWaterHeightMax', 'BorderWaterSurfaceHeight',
    'ErosionSumFluxSumThreshold', 'ErosionSumFluxSumFactor', 'WaterHeightTemperatureDampingThreshold',
    'WaterHeightTemperatureDampingFactor', 'LateralErosionSlopeThresholdLeft', 'LateralErosionSlopeThresholdRight',
    'LateralErosionSlopeThresholdUp', 'LateralErosionSlopeThresholdDown', 'LateralErosionWaterHeightThreshold',
    'PathMaxSlope', 'PathMaxWaterHeight', 'LavaFluxDampingThresholdMin', 'WaterFluxMask', 'LavaFluxMask',
    'WaterEnergy', 'LavaFluxDampingThresholdMax', 'BorderHardgroundSurfaceHeight', 'BorderSoftgroundSurfaceHeight',
    'GetMinimumVisibleWater', 'Parameter51', 'LayersOutlineWaterHeightThreshold', 'LayersOutlineWaterFluxThreshold',
    'LayersOutlineZobi', 'LayersOutlineWaterSlopeThreshold']
PI = {n: i for i, n in enumerate(PARAM_NAMES)}

# dustai field name -> Galileo name where they differ (matched by name; see findings/water-lava-sim.md)
RENAME = {'WaterFluidityValue1': 'WaterFluidity',
          'LateralErosionFluxSumThreshold': 'ErosionSumFluxSumThreshold',
          'LateralErosionFactor': 'ErosionSumFluxSumFactor',
          'LavaHeightTemperatureDampingThreshold': 'WaterHeightTemperatureDampingThreshold',
          'LavaHeightTemperatureDampingFactor': 'WaterHeightTemperatureDampingFactor'}


def lyn_hash(s, bits=32):
    """LyN reflection name hash (dustai.win32.f.dll 0x100342C0)."""
    v = 0
    for c in s.encode():
        x = ((v * 61) ^ ((c - 32) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        v = ((x >> bits) ^ x) & ((1 << bits) - 1)
    return v


def stored_records(struct_name):
    """Decode every serialized `struct_name` record in Wor_BD00993D.lin.bf/SpecieFluid_0.ai.
    Returns a list of {field: value} dicts (field names without the m_simulatorParam_ prefix)."""
    sys.path.insert(0, HERE)
    import bf, re
    path = os.path.join(INSTALL, 'BinPC', 'Wor_BD00993D.lin.bf')
    d = open(path, 'rb').read()
    ent = [e for e in bf.read_index(path) if e['name'] == 'SpecieFluid_0.ai'][0]
    blob = d[ent['offset']:ent['offset'] + ent['size']]
    dll = open(os.path.join(INSTALL, 'AInative', 'bin', 'dll', 'dustai.win32.f.dll'), 'rb').read()
    names = {lyn_hash(m.group().decode()): m.group().decode() for m in re.finditer(rb'm_[A-Za-z0-9_]+', dll)}
    key = struct.pack('<I', lyn_hash('DustAi::SpecieFluid::' + struct_name))
    out = []
    for o in [i for i in range(len(blob) - 4) if blob[i:i + 4] == key]:
        _, typ, size, cnt = struct.unpack_from('<IIII', blob, o)
        o += 16
        rec = {}
        for _ in range(cnt):
            h, t, s = struct.unpack_from('<III', blob, o)
            nm = names.get(h, hex(h))
            if t == 0x410:
                rec[nm.split('_', 2)[-1]] = struct.unpack_from('<f', blob, o + 12)[0]
            o += 12 + s
        out.append(rec)
    return out


def stored_fluid_params(element=0):
    """DustAi::SpecieFluid::s_fluidParameters[element] (m_fluidParameters)."""
    return stored_records('s_fluidParameters')[element]


# Indices the game multiplies by 0.125 on the way in (dustai.win32.f.dll 0x100FBA20, setter 0x10049A40).
# Checked against the running game's parameter array (tools/memscan.py params); index 35 is 1.0 there.
TEMPERATURE_SCALED = (12, 14, 17, 24, 25, 43, 47)


def fluid_params(element=0, border=(0.0, 0.0, 0.0), material='normal', raw=False):
    """56-float ParamF array as the game feeds it to the simulator.
    material 'normal' = m_fluidParameters[element]; 'slow' = m_slowWater/LavaParameters[element]
    (the 10x less fluid set; when the game uses it is not established).
    raw=True skips the 1/8 temperature scaling (stored values as-is).
    border = (water surface, hard ground, soft ground) of the map's border band."""
    P = np.zeros(56, np.float32)
    recs = [stored_fluid_params(element)]
    if material == 'slow':
        recs += [stored_records('s_waterParameters')[element], stored_records('s_lavaParameters')[element]]
    for rec in recs:
        for k, v in rec.items():
            k = RENAME.get(k, k)
            if k in PI:
                P[PI[k]] = v
    if not raw:
        for i in TEMPERATURE_SCALED:
            P[i] *= np.float32(0.125)
    P[PI['BorderWaterSurfaceHeight']], P[PI['BorderHardgroundSurfaceHeight']], P[PI['BorderSoftgroundSurfaceHeight']] = border
    return P


def param_image(P, skip_border=0, param_i=None, param_u8=None, ganja=None):
    """0x360-byte Galileo::SimulatorParam (layout from the worker PDB)."""
    b = bytearray(0x360)
    b[0x10:0x10 + 224] = np.asarray(P, np.float32).tobytes()
    if param_i is not None:
        b[0xF0:0xF0 + 32] = np.asarray(param_i, np.int32).tobytes()
    if param_u8 is not None:
        b[0x110:0x110 + 48] = bytes(param_u8)
    if ganja is not None:
        b[0x140:0x140 + 512] = bytes(ganja)
    struct.pack_into('<III', b, 0x340, 9, 9, skip_border)
    return bytes(b)


def empty_grid():
    return np.zeros((8, N, N), np.uint32)


def setf(grid, comp, a):
    grid[comp] = np.asarray(a, np.float32).view(np.uint32)


def getf(grid, comp):
    return grid[comp].view(np.float32)


def pack(grid):
    """(8, 512, 512) -> game memory layout bytes."""
    g = grid.reshape(8, N, N // 16, 16).transpose(1, 2, 0, 3)  # y, group, comp, lane
    return np.ascontiguousarray(g).tobytes()


def unpack(raw):
    a = np.frombuffer(raw, np.uint32).reshape(N, N // 16, 8, 16).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(a).reshape(8, N, N)


def flux_pair(grid, comp):
    """Split a packed u16 pair: returns (low16, high16) as uint16 arrays."""
    v = grid[comp]
    return (v & 0xFFFF).astype(np.uint16), (v >> 16).astype(np.uint16)


def run(grid, P, steps=1, skip_border=0, dump_every_step=False, **kw):
    """Run the game's step `steps` times. Returns the final grid (or a list per step)."""
    with tempfile.TemporaryDirectory() as td:
        pin, gin, gout = (os.path.join(td, n) for n in ('param.bin', 'in.bin', 'out.bin'))
        open(pin, 'wb').write(param_image(P, skip_border, **kw))
        open(gin, 'wb').write(pack(grid))
        cmd = [SIMSTEP, WORKER, pin, gin, gout, str(steps)]
        if dump_every_step:
            cmd.append(os.path.join(td, 'step'))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr or r.stdout)
        if dump_every_step:
            return [unpack(open(os.path.join(td, 'step%d.bin' % (i + 1)), 'rb').read()) for i in range(steps)]
        return unpack(open(gout, 'rb').read())


def modifier_block(center, param, P, skip=2):
    """0x50-byte SimulatorModifierTemplateParam as SimulatorWater::Modify (0x10012060) builds it:
    rows / 16-cell column groups of the bounding box centre +- (radius + 1), clamped to the processed
    area; Center (col, row, radius as given, first row as float: Modify overwrites w); Param; (ParamF[17], WaterEnergy = ParamF[46],
    1 / WaterEnergy, 0). The grid pointer (+0x08) is filled in by modstep."""
    cx, cy, r = (int(np.rint(np.float32(v))) for v in center[:3])
    lo, hi = skip * 16, N - skip * 16 - 1
    row0 = max(cy - r - 1 if cy >= r + 1 else 0, lo)
    row1 = min(cy + r + 1, hi)
    col0 = max(cx - r - 1 if cx >= r + 1 else 0, lo)
    col1 = min(cx + r + 1, hi)
    b = bytearray(0x50)
    struct.pack_into('<8I', b, 0, row0, row1, 0, 9, row0, row1, col0 >> 4, (col1 + 15) >> 4)
    struct.pack_into('<4f', b, 0x20, center[0], center[1], center[2], float(row0))   # w = first row (the row loop's counter)
    struct.pack_into('<4f', b, 0x30, *param)
    e = np.float32(P[46])
    struct.pack_into('<4f', b, 0x40, P[17], e, np.float32(1) / e if e else 0, 0)
    return bytes(b)


def modify(grid, kind, center, param, P, skip=2):
    """Run the game's own modifier `kind` (MODIFIER_RVA key, e.g. 'RepulseFluid') once on grid
    (8, 512, 512) uint32 with Center (col, row, radius, w) and Param (4 floats). Returns the new grid."""
    with tempfile.TemporaryDirectory() as td:
        pb, gin, gout = (os.path.join(td, n) for n in ('block.bin', 'in.bin', 'out.bin'))
        open(pb, 'wb').write(modifier_block(center, param, P, skip))
        open(gin, 'wb').write(pack(grid))
        r = subprocess.run([MODSTEP, MODIFIER_DLL, '%X' % MODIFIER_RVA[kind], pb, gin, gout], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr or r.stdout)
        return unpack(open(gout, 'rb').read())


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    P = fluid_params()
    for i, n in enumerate(PARAM_NAMES):
        if P[i]:
            print('%2d %-42s %g' % (i, n, P[i]))
