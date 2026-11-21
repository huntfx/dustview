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
// SIMSTEP_JOB=4 runs CallbackSimulatorGanja (vegetation / fire step) instead, same arguments.
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

// Resident mode: simstep --serve <worker.dll> <mapping name>
// The caller creates a named shared-memory block: SimulatorParam image at 0 (0x360 bytes), grid at
// SERVE_GRID_OFS (GRID_BYTES). Each stdin line "run <job> <steps>" runs that many steps of the job on the
// grid in place (rows the step skips keep their values) and answers "ok" (or "err ...") on stdout;
// "quit" or end of input exits. Used by galileo.Resident (DustView's vegetation step).
static const size_t SERVE_GRID_OFS = 0x1000;

static int serve(const char *dllPath, const char *mapName) {
    HMODULE dll = LoadLibraryA(dllPath);
    if (!dll) { printf("err LoadLibrary %lu\n", GetLastError()); return 1; }
    JobQuery codeAddr = (JobQuery)GetProcAddress(dll, "sCodeAddrJob");
    JobQuery stackSize = (JobQuery)GetProcAddress(dll, "sStackSizeJob");
    HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, mapName);
    if (!codeAddr || !stackSize || !h) { printf("err setup %lu\n", GetLastError()); return 1; }
    unsigned char *view = (unsigned char *)MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, SERVE_GRID_OFS + GRID_BYTES);
    if (!view) { printf("err map %lu\n", GetLastError()); return 1; }
    unsigned char *param = (unsigned char *)_aligned_malloc(0x360, 64);
    unsigned char *other = (unsigned char *)_aligned_malloc(GRID_BYTES, 64);
    void *scratch = NULL; size_t scratchCap = 0;
    printf("ready\n"); fflush(stdout);
    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        unsigned int job; int steps;
        if (strncmp(line, "quit", 4) == 0) break;
        if (sscanf(line, "run %u %d", &job, &steps) != 2) { printf("err bad command\n"); fflush(stdout); continue; }
        JobFn fn = (JobFn)codeAddr(job);
        unsigned long need = stackSize(job);
        if (!fn) { printf("err no job\n"); fflush(stdout); continue; }
        if (need + 64 > scratchCap) { _aligned_free(scratch); scratchCap = need + 64; scratch = _aligned_malloc(scratchCap, 64); }
        memcpy(param, view, 0x360);
        unsigned char *grid[2] = {view + SERVE_GRID_OFS, other};
        memcpy(other, grid[0], GRID_BYTES);
        int cur = 0, ex = 0;
        for (int s = 0; s < steps && !ex; s++) {
            *(unsigned char **)(param + 0x350) = grid[cur];
            *(unsigned char **)(param + 0x354) = grid[cur ^ 1];
            memset(scratch, 0, need);
            ex = runJob(fn, param, scratch);
            cur ^= 1;
        }
        if (cur) memcpy(grid[0], other, GRID_BYTES);
        if (ex) printf("err exception 0x%08X\n", ex); else printf("ok\n");
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && strcmp(argv[1], "--serve") == 0)
        return serve(argv[2], argv[3]);
    if (argc < 6) {
        fprintf(stderr, "usage: simstep <worker.dll> <param.bin> <grid_in.bin> <grid_out.bin> <steps> [dump_prefix]\n");
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "LoadLibrary failed (%lu)\n", GetLastError()); return 1; }
    JobQuery codeAddr = (JobQuery)GetProcAddress(dll, "sCodeAddrJob");      // exported undecorated
    JobQuery stackSize = (JobQuery)GetProcAddress(dll, "sStackSizeJob");
    if (!codeAddr || !stackSize) { fprintf(stderr, "exports not found\n"); return 1; }
    // Job index: 0 = CallbackSimulator (water/lava, default), 4 = CallbackSimulatorGanja
    // (vegetation / fire). Chosen with the environment variable SIMSTEP_JOB.
    const char *jobEnv = getenv("SIMSTEP_JOB");
    unsigned int job = jobEnv ? (unsigned int)atoi(jobEnv) : 0;
    JobFn sim = (JobFn)codeAddr(job);
    unsigned long scratchSize = stackSize(job);
    if (!sim) { fprintf(stderr, "no job %u\n", job); return 1; }

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
