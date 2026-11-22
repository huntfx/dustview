// Writes this CPU's rcpps result for every integer 0..65535 (as float) to the file given, 65536 float32.
// The vegetation step's display field TimeAnim a divides by a u16 phase length with rcpps (approximate,
// CPU-dependent), so ganja_step.py looks the reciprocal up in this table to match the DLL bit for bit.
#include <stdio.h>
#include <emmintrin.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: rcptable out.bin\n"); return 1; }
    FILE *f = fopen(argv[1], "wb");
    if (!f) return 1;
    for (int i = 0; i < 65536; i += 4) {
        __m128 v = _mm_cvtepi32_ps(_mm_setr_epi32(i, i + 1, i + 2, i + 3));
        float out[4];
        _mm_storeu_ps(out, _mm_rcp_ps(v));
        fwrite(out, 4, 4, f);
    }
    fclose(f);
    return 0;
}
