// Run From Dust's own terrain brush (job 14 CallbackUpdateTexture, GalileoSimulatorWater_9_9.dll) on a grid.
//
// usage: brushstep <worker.dll> <grid_in.bin> <grid_out.bin> <texture.bin> <tex_log2>
//                  <x> <y> <size> <effect> <dir_x> <dir_y> <strength> <min> <max> [times]
//   grid_*.bin   8 MB grid as for simstep (galileo.pack)
//   texture.bin  the brush bitmap as the job reads it: 4 bytes per texel, mip chain from 2^tex_log2 down
//   x y size effect   ints (UpdateTextureParam XYST: cell column, row, diameter in cells,
//                     kBrushParamEffect_*: 1/2 hard relative add/sub, 3/4 hard absolute, 5-8 soft)
//   dir_x dir_y strength  floats (CosSinCoef: the brush's rotation as sin / cos, the curve value)
//   min max      ints (MinMax: the job's cell range, (min, max, min, max) as the game builds it)
//   times        run the job this many times (default 1)
// Parameter layout (SimulatorWater frame update, SSE41 DLL 0x100131F9; UpdateTextureParam in the PDB):
//   +0x00 XYST (u32 x4), +0x10 CosSinCoef (f32 x4), +0x20 grid*, +0x24 texture*, +0x28 tex_log2, +0x2C 0,
//   +0x30 MinMax (u32 x4).
// Build: build.bat. Python wrapper: ../terraform.py
#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long (__cdecl *JobQuery)(unsigned int);
typedef void (__cdecl *JobFn)(const void *param, void *scratch, unsigned long);

static const size_t GRID_BYTES = 512 * 512 * 32;

static long fileSize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

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

static int runJob(JobFn fn, const void *param, void *scratch) {
    __try {
        fn(param, scratch, 0);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return (int)GetExceptionCode();
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 15) {
        fprintf(stderr, "usage: brushstep <worker.dll> <grid_in> <grid_out> <texture> <tex_log2> <x> <y> <size> "
                        "<effect> <dir_x> <dir_y> <strength> <min> <max> [times]\n");
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "LoadLibrary failed (%lu)\n", GetLastError()); return 1; }
    JobQuery codeAddr = (JobQuery)GetProcAddress(dll, "sCodeAddrJob");
    JobQuery stackSize = (JobQuery)GetProcAddress(dll, "sStackSizeJob");
    if (!codeAddr || !stackSize) { fprintf(stderr, "exports not found\n"); return 1; }
    JobFn job = (JobFn)codeAddr(14);
    unsigned long scratchSize = stackSize(14);
    if (!job) { fprintf(stderr, "no job 14\n"); return 1; }

    long texBytes = fileSize(argv[4]);
    if (texBytes <= 0) { fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    unsigned char *grid = (unsigned char *)_aligned_malloc(GRID_BYTES, 64);
    unsigned char *tex = (unsigned char *)_aligned_malloc(texBytes + 64, 64);
    unsigned char *param = (unsigned char *)_aligned_malloc(0x40, 64);
    void *scratch = _aligned_malloc(scratchSize + 64, 64);
    if (!readFile(argv[2], grid, GRID_BYTES)) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    if (!readFile(argv[4], tex, texBytes)) { fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }

    unsigned int *pu = (unsigned int *)param;
    float *pf = (float *)param;
    memset(param, 0, 0x40);
    pu[0] = (unsigned int)atoi(argv[6]);     // x
    pu[1] = (unsigned int)atoi(argv[7]);     // y
    pu[2] = (unsigned int)atoi(argv[8]);     // size
    pu[3] = (unsigned int)atoi(argv[9]);     // effect
    pf[4] = (float)atof(argv[10]);           // dir x
    pf[5] = (float)atof(argv[11]);           // dir y
    pf[6] = (float)atof(argv[12]);           // strength
    pf[7] = 0.0f;
    *(unsigned char **)(param + 0x20) = grid;
    *(unsigned char **)(param + 0x24) = tex;
    pu[10] = (unsigned int)atoi(argv[5]);    // tex_log2
    pu[11] = 0;
    pu[12] = pu[14] = (unsigned int)atoi(argv[13]);
    pu[13] = pu[15] = (unsigned int)atoi(argv[14]);
    int times = argc > 15 ? atoi(argv[15]) : 1;
    for (int i = 0; i < times; i++) {
        memset(scratch, 0, scratchSize);
        int ex = runJob(job, param, scratch);
        if (ex) { fprintf(stderr, "exception 0x%08X\n", ex); return 1; }
    }
    if (!writeFile(argv[3], grid, GRID_BYTES)) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    printf("ok: job 14 x%d, scratch 0x%lX\n", times, scratchSize);
    return 0;
}
