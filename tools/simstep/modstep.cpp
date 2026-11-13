// Run one of From Dust's simulator modifiers (SSE41/GalileoSimulatorWater.dll) on a grid file.
//
// usage: modstep <GalileoSimulatorWater.dll> <rva hex> <block.bin> <grid_in.bin> <grid_out.bin>
//   rva        address of the modifier function relative to the DLL base (image base 0x10000000):
//              SoftGround A8F0, HardGround B420, RepulseFluid BEE0, AddLava DC60, RemoveLava EED0,
//              AddWater F800, RemoveWater 10B40 (SimulatorModifierTemplate::*, from the PDB)
//   block.bin  0x50-byte SimulatorModifierTemplateParam as SimulatorWater::Modify builds it; the grid
//              pointer at +0x08 is filled in here (see galileo.modifier_block)
//   grid_*.bin 8 MB grid (same layout as simstep); the modifier edits it in place
//
// The DLL has no imports and no entry point, so LoadLibrary + base + RVA is enough. The modifier
// functions are __stdcall with the block pointer as the only argument (RepulseFluid ends in ret 4;
// note SimulatorWater::Modify calls AddWater without a push, so AddWater may need checking).
// Build: build.bat (MSVC x86).
#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (__stdcall *ModFn)(const void *block);
static const size_t GRID_BYTES = 512 * 512 * 32;

static bool readFile(const char *path, void *dst, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    size_t got = fread(dst, 1, n, f);
    fclose(f);
    return got == n;
}

static bool writeFile(const char *path, const void *src, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    size_t put = fwrite(src, 1, n, f);
    fclose(f);
    return put == n;
}

static int run(ModFn fn, const void *block) {
    __try {
        fn(block);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return (int)GetExceptionCode();
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: modstep <dll> <rva hex> <block.bin> <grid_in.bin> <grid_out.bin>\n");
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "LoadLibrary failed (%lu)\n", GetLastError()); return 1; }
    ModFn fn = (ModFn)((unsigned char *)dll + strtoul(argv[2], NULL, 16));
    unsigned char *block = (unsigned char *)_aligned_malloc(0x50, 64);
    unsigned char *grid = (unsigned char *)_aligned_malloc(GRID_BYTES, 64);
    if (!readFile(argv[3], block, 0x50)) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    if (!readFile(argv[4], grid, GRID_BYTES)) { fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    *(unsigned char **)(block + 0x08) = grid;
    int ex = run(fn, block);
    if (ex) { fprintf(stderr, "exception 0x%08X\n", ex); return 1; }
    if (!writeFile(argv[5], grid, GRID_BYTES)) { fprintf(stderr, "cannot write %s\n", argv[5]); return 1; }
    printf("ok: modifier at %p\n", (void *)fn);
    return 0;
}
