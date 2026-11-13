// Run From Dust's own water/lava simulation step (GalileoSimulatorWater_9_9.dll) on a grid file.
//
// usage: simstep <worker.dll> <param.bin> <grid_in.bin> <grid_out.bin> <steps> [dump_prefix]
//   param.bin   0x360-byte Galileo::SimulatorParam image (pointers at +0x350/+0x354 are filled here)
//   grid_*.bin  8 MB grid: 512 rows x 32 groups of 16 cells x 8 components x 16 floats
//   steps       number of simulation steps; with dump_prefix, every step is written to
//               <dump_prefix><n>.bin as well
//
// The worker DLL has no imports. Job 0 from sCodeAddrJob is CallbackSimulator
// (SimulatorParam const*, __m128i* scratch, ulong unused); its scratch size comes from
// sStackSizeJob(0) (0x12600 = SimulatorWaterIteratorContainer<9>).
// Build: build.bat (MSVC x86). Python wrapper: ../galileo.py
#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

typedef unsigned long (__cdecl *JobQuery)(unsigned int);
typedef void (__cdecl *JobFn)(const void *param, void *scratch, unsigned long);

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

static int runJob(JobFn fn, const void *param, void *scratch) {
    __try {
        fn(param, scratch, 0);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return (int)GetExceptionCode();
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: simstep <worker.dll> <param.bin> <grid_in.bin> <grid_out.bin> <steps> [dump_prefix]\n");
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "LoadLibrary failed (%lu)\n", GetLastError()); return 1; }
    JobQuery codeAddr = (JobQuery)GetProcAddress(dll, "sCodeAddrJob");      // exported undecorated
    JobQuery stackSize = (JobQuery)GetProcAddress(dll, "sStackSizeJob");
    if (!codeAddr || !stackSize) { fprintf(stderr, "exports not found\n"); return 1; }
    JobFn sim = (JobFn)codeAddr(0);
    unsigned long scratchSize = stackSize(0);

    unsigned char *param = (unsigned char *)_aligned_malloc(0x360, 64);
    unsigned char *grid[2] = {(unsigned char *)_aligned_malloc(GRID_BYTES, 64),
                              (unsigned char *)_aligned_malloc(GRID_BYTES, 64)};
    void *scratch = _aligned_malloc(scratchSize + 64, 64);
    if (!readFile(argv[2], param, 0x360)) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    if (!readFile(argv[3], grid[0], GRID_BYTES)) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    memcpy(grid[1], grid[0], GRID_BYTES);  // rows the step skips keep their input values

    int steps = atoi(argv[5]);
    const char *dump = argc > 6 ? argv[6] : NULL;
    int cur = 0;
    for (int s = 0; s < steps; s++) {
        *(unsigned char **)(param + 0x350) = grid[cur];
        *(unsigned char **)(param + 0x354) = grid[cur ^ 1];
        memset(scratch, 0, scratchSize);
        int ex = runJob(sim, param, scratch);
        if (ex) { fprintf(stderr, "exception 0x%08X in step %d\n", ex, s); return 1; }
        cur ^= 1;
        if (dump) {
            char path[MAX_PATH];
            _snprintf(path, sizeof path, "%s%d.bin", dump, s + 1);
            writeFile(path, grid[cur], GRID_BYTES);
        }
    }
    if (!writeFile(argv[4], grid[cur], GRID_BYTES)) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    printf("ok: %d steps, scratch 0x%lX, job at %p\n", steps, scratchSize, (void *)sim);
    return 0;
}
