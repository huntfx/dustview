// d3dlog: read-only Direct3D 9 logging for dusthook (observes, never changes what the game draws).
//
// How: at load, a thread creates its own IDirect3D9 and patches CreateDevice in the (shared) IDirect3D9
// vtable. When the game creates its device, the device's vtable gets patched for SetTexture (65),
// DrawPrimitive / DrawIndexedPrimitive / ...UP (81-84), Present (17), SetPixelShader (107) and
// SetPixelShaderConstantF (109). Every hook calls the original first or last, unchanged.
//
// What it logs (to dusthook.log, about once every 10 s per pass):
//   - the WaterParams converter shader (fxcache 105 / 343): its g_BorderConsts c128-c130 whenever they change
//   - each new pixel shader that belongs to the water (its CTAB names: UVScrollParams, WaterParams, ...)
//   - for a draw with such a shader: the constant registers c36-c42 last set (WaterTime, SedimentColor,
//     UVScrollParams, ?, WaterNormalmapParams, ?, WaterDetailParams) and the textures bound to stages
//     0-15 (format, size, pool, usage)
//   - the terrain shader (fxcache 96, CTAB name af4_TerrainRampUV): constants c55-c70 and c128-c135 plus its
//     textures, same rate (c59 = terrain ramp rows, c131 = ground tiling / ramp distance term)
// F3 (in the game window): dump the textures bound to the next draw of each water pass to
//   dustdump_<pass>_s<stage>.bin next to the DLL (header: u32 width, height, format, pitch; then rows),
//   if they can be read (managed / system-memory / dynamic textures by LockRect, render targets by
//   GetRenderTargetData).

#include <windows.h>
#include <d3d9.h>
#include <stdio.h>
#include <string.h>

void Log(const char *fmt, ...);          // from dusthook.cpp
extern char g_logDir[MAX_PATH];
volatile LONG g_dumpRequest = 0;

typedef HRESULT (STDMETHODCALLTYPE *CreateDeviceFn)(IDirect3D9 *, UINT, D3DDEVTYPE, HWND, DWORD, D3DPRESENT_PARAMETERS *, IDirect3DDevice9 **);
typedef HRESULT (STDMETHODCALLTYPE *SetTextureFn)(IDirect3DDevice9 *, DWORD, IDirect3DBaseTexture9 *);
typedef HRESULT (STDMETHODCALLTYPE *DrawPrimitiveFn)(IDirect3DDevice9 *, D3DPRIMITIVETYPE, UINT, UINT);
typedef HRESULT (STDMETHODCALLTYPE *DrawIndexedPrimitiveFn)(IDirect3DDevice9 *, D3DPRIMITIVETYPE, INT, UINT, UINT, UINT, UINT);
typedef HRESULT (STDMETHODCALLTYPE *DrawPrimitiveUPFn)(IDirect3DDevice9 *, D3DPRIMITIVETYPE, UINT, CONST void *, UINT);
typedef HRESULT (STDMETHODCALLTYPE *DrawIndexedPrimitiveUPFn)(IDirect3DDevice9 *, D3DPRIMITIVETYPE, UINT, UINT, UINT, CONST void *, D3DFORMAT, CONST void *, UINT);
typedef HRESULT (STDMETHODCALLTYPE *PresentFn)(IDirect3DDevice9 *, CONST RECT *, CONST RECT *, HWND, CONST RGNDATA *);
typedef HRESULT (STDMETHODCALLTYPE *SetPixelShaderFn)(IDirect3DDevice9 *, IDirect3DPixelShader9 *);
typedef HRESULT (STDMETHODCALLTYPE *SetPSConstFn)(IDirect3DDevice9 *, UINT, CONST float *, UINT);

static CreateDeviceFn g_CreateDevice;
static SetTextureFn g_SetTexture;
static DrawPrimitiveFn g_DrawPrimitive;
static DrawIndexedPrimitiveFn g_DrawIndexedPrimitive;
static DrawPrimitiveUPFn g_DrawPrimitiveUP;
static DrawIndexedPrimitiveUPFn g_DrawIndexedPrimitiveUP;
static PresentFn g_Present;
static SetPixelShaderFn g_SetPixelShader;
static SetPSConstFn g_SetPSConst;

// ---- state shadow -----------------------------------------------------------------
static float g_psConst[256][4];
static IDirect3DBaseTexture9 *g_tex[16];
static IDirect3DPixelShader9 *g_curPS;
static DWORD g_frame;

// water pass kinds, recognised by names in the shader's constant table
enum { K_NONE = 0, K_UVSCROLL, K_WATER, K_FOAMPACK, K_PARAMS, K_OTHERWATER, K_CONVERT, K_TERRAIN, K_COUNT };
static const char *kKindName[K_COUNT] = { "none", "uvscroll", "water", "foampack", "params", "otherwater", "convert", "terrain" };
struct ShaderKind { IDirect3DPixelShader9 *ps; int kind; };
static ShaderKind g_kinds[512];
static int g_nkinds;
static DWORD g_lastLog[K_COUNT];
static LONG g_dumpPending[K_COUNT];

static bool Has(const BYTE *b, UINT n, const char *s) {
    size_t m = strlen(s);
    for (UINT i = 0; i + m <= n; i++)
        if (memcmp(b + i, s, m) == 0) return true;
    return false;
}

static int Classify(IDirect3DPixelShader9 *ps) {
    if (!ps) return K_NONE;
    for (int i = 0; i < g_nkinds; i++)
        if (g_kinds[i].ps == ps) return g_kinds[i].kind;
    int kind = K_NONE;
    UINT n = 0;
    if (SUCCEEDED(ps->GetFunction(NULL, &n)) && n > 0 && n < (1 << 20)) {
        BYTE *buf = (BYTE *)malloc(n);
        if (buf && SUCCEEDED(ps->GetFunction(buf, &n))) {
            UINT m = n < 4096 ? n : 4096;
            if (Has(buf, m, "af4_TerrainRampUV")) kind = K_TERRAIN;            // fxcache 96: sand / rock terrain
            else if (Has(buf, m, "samp_MatWeigthTex")) kind = K_CONVERT;          // fxcache 105 / 343: builds WaterParams
            else if (Has(buf, m, "g_SamplerPreviousUV")) kind = K_UVSCROLL;
            else if (Has(buf, m, "g_SamplerWaterFoam")) kind = K_FOAMPACK;
            else if (Has(buf, m, "PS_f4_UVScrollParams") && Has(buf, m, "g_SamplerWaterParams")) kind = K_WATER;
            else if (Has(buf, m, "WaterParams")) kind = K_PARAMS;
            else if (Has(buf, m, "Water")) kind = K_OTHERWATER;
            if (kind != K_NONE) Log("d3d: pixel shader %p = %s (%u bytes)", ps, kKindName[kind], n);
        }
        free(buf);
    }
    if (g_nkinds < 512) { g_kinds[g_nkinds].ps = ps; g_kinds[g_nkinds].kind = kind; g_nkinds++; }
    return kind;
}

static const char *PoolName(D3DPOOL p) {
    switch (p) { case D3DPOOL_DEFAULT: return "default"; case D3DPOOL_MANAGED: return "managed";
                 case D3DPOOL_SYSTEMMEM: return "sysmem"; case D3DPOOL_SCRATCH: return "scratch"; }
    return "?";
}

static void FormatName(D3DFORMAT f, char *out, size_t n) {
    if (f > 0xFFFF) { sprintf_s(out, n, "%c%c%c%c", f & 0xFF, (f >> 8) & 0xFF, (f >> 16) & 0xFF, (f >> 24) & 0xFF); return; }
    switch (f) {
    case D3DFMT_A8R8G8B8: strcpy_s(out, n, "A8R8G8B8"); return;
    case D3DFMT_X8R8G8B8: strcpy_s(out, n, "X8R8G8B8"); return;
    case D3DFMT_A16B16G16R16F: strcpy_s(out, n, "A16B16G16R16F"); return;
    case D3DFMT_G16R16F: strcpy_s(out, n, "G16R16F"); return;
    case D3DFMT_A32B32G32R32F: strcpy_s(out, n, "A32B32G32R32F"); return;
    case D3DFMT_G32R32F: strcpy_s(out, n, "G32R32F"); return;
    case D3DFMT_R32F: strcpy_s(out, n, "R32F"); return;
    case D3DFMT_R16F: strcpy_s(out, n, "R16F"); return;
    case D3DFMT_G16R16: strcpy_s(out, n, "G16R16"); return;
    case D3DFMT_A16B16G16R16: strcpy_s(out, n, "A16B16G16R16"); return;
    case D3DFMT_L8: strcpy_s(out, n, "L8"); return;
    case D3DFMT_A8: strcpy_s(out, n, "A8"); return;
    case D3DFMT_A8L8: strcpy_s(out, n, "A8L8"); return;
    case D3DFMT_A8B8G8R8: strcpy_s(out, n, "A8B8G8R8"); return;
    }
    sprintf_s(out, n, "fmt%u", (unsigned)f);
}

static bool Describe(IDirect3DBaseTexture9 *t, char *out, size_t n, IDirect3DTexture9 **tex2d) {
    *tex2d = NULL;
    if (!t) return false;
    if (t->GetType() != D3DRTYPE_TEXTURE) { sprintf_s(out, n, "%p type %d", t, t->GetType()); return true; }
    IDirect3DTexture9 *tx = (IDirect3DTexture9 *)t;
    D3DSURFACE_DESC d;
    if (FAILED(tx->GetLevelDesc(0, &d))) { sprintf_s(out, n, "%p ?", t); return true; }
    char f[32]; FormatName(d.Format, f, sizeof(f));
    sprintf_s(out, n, "%p %ux%u %s %s%s%s levels %u", t, d.Width, d.Height, f, PoolName(d.Pool),
              (d.Usage & D3DUSAGE_RENDERTARGET) ? " rendertarget" : "", (d.Usage & D3DUSAGE_DYNAMIC) ? " dynamic" : "",
              tx->GetLevelCount());
    *tex2d = tx;
    return true;
}

static void DumpTexture(IDirect3DDevice9 *dev, IDirect3DTexture9 *tx, const char *pass, int stage) {
    D3DSURFACE_DESC d;
    if (FAILED(tx->GetLevelDesc(0, &d))) return;
    IDirect3DSurface9 *src = NULL, *tmp = NULL;
    D3DLOCKED_RECT lr;
    bool locked = false;
    if (FAILED(tx->GetSurfaceLevel(0, &src))) return;
    if (d.Pool != D3DPOOL_DEFAULT || (d.Usage & D3DUSAGE_DYNAMIC)) {
        locked = SUCCEEDED(src->LockRect(&lr, NULL, D3DLOCK_READONLY));
    } else if (d.Usage & D3DUSAGE_RENDERTARGET) {
        if (SUCCEEDED(dev->CreateOffscreenPlainSurface(d.Width, d.Height, d.Format, D3DPOOL_SYSTEMMEM, &tmp, NULL)) &&
            SUCCEEDED(dev->GetRenderTargetData(src, tmp)))
            locked = SUCCEEDED(tmp->LockRect(&lr, NULL, D3DLOCK_READONLY));
    }
    if (!locked) {
        Log("d3d: dump %s s%d: not readable (pool %s, usage %08X)", pass, stage, PoolName(d.Pool), d.Usage);
    } else {
        char path[MAX_PATH];
        sprintf_s(path, "%sdustdump_%s_s%d.bin", g_logDir, pass, stage);
        FILE *f = NULL;
        if (fopen_s(&f, path, "wb") == 0 && f) {
            UINT hdr[4] = { d.Width, d.Height, (UINT)d.Format, (UINT)lr.Pitch };
            fwrite(hdr, 4, 4, f);
            UINT rows = d.Height;
            if (d.Format == D3DFMT_DXT1 || d.Format == D3DFMT_DXT5 || d.Format == D3DFMT_DXT3) rows = (d.Height + 3) / 4;
            for (UINT y = 0; y < rows; y++) fwrite((BYTE *)lr.pBits + y * lr.Pitch, 1, lr.Pitch, f);
            fclose(f);
            Log("d3d: dumped %s s%d to %s", pass, stage, path);
        }
        (tmp ? tmp : src)->UnlockRect();
    }
    if (tmp) tmp->Release();
    src->Release();
}

static void OnDraw(IDirect3DDevice9 *dev) {
    __try {
        int kind = Classify(g_curPS);
        if (kind == K_NONE || kind == K_OTHERWATER) return;
        if (kind == K_CONVERT) {                      // log the border constants whenever they change
            static float last[3][4];
            static bool first = true;
            if (first || memcmp(last, g_psConst[128], sizeof(last)) != 0) {
                memcpy(last, g_psConst[128], sizeof(last));
                first = false;
                Log("d3d: frame %u convert shader %p g_BorderConsts:", g_frame, g_curPS);
                for (int r = 128; r <= 130; r++)
                    Log("d3d:    c%d %12.6f %12.6f %12.6f %12.6f", r, g_psConst[r][0], g_psConst[r][1], g_psConst[r][2], g_psConst[r][3]);
            }
            if (!g_dumpPending[kind]) return;
        }
        DWORD now = GetTickCount();
        bool dump = InterlockedCompareExchange(&g_dumpPending[kind], 0, 1) == 1;
        if (!dump && now - g_lastLog[kind] < 10000) return;
        g_lastLog[kind] = now;
        Log("d3d: frame %u draw with %s shader %p", g_frame, kKindName[kind], g_curPS);
        if (kind == K_TERRAIN) {                      // terrain shader: c55-c70 (c59 = af4_TerrainRampUV, c67 detail) and c128-c135 (c131 tiling)
            for (int r = 55; r <= 70; r++)
                Log("d3d:    c%d %12.6f %12.6f %12.6f %12.6f", r, g_psConst[r][0], g_psConst[r][1], g_psConst[r][2], g_psConst[r][3]);
            for (int r = 128; r <= 135; r++)
                Log("d3d:    c%d %12.6f %12.6f %12.6f %12.6f", r, g_psConst[r][0], g_psConst[r][1], g_psConst[r][2], g_psConst[r][3]);
        }
        static const char *cname[7] = { "WaterTime", "SedimentColor", "UVScrollParams", "c39", "WaterNormalmapParams", "c41", "WaterDetailParams" };
        for (int r = 36; r <= 42 && kind != K_TERRAIN; r++)
            Log("d3d:    c%d %-20s %10.5f %10.5f %10.5f %10.5f", r, cname[r - 36], g_psConst[r][0], g_psConst[r][1], g_psConst[r][2], g_psConst[r][3]);
        for (int s = 0; s < 16; s++) {
            char desc[160]; IDirect3DTexture9 *tx;
            if (Describe(g_tex[s], desc, sizeof(desc), &tx)) {
                Log("d3d:    s%-2d %s", s, desc);
                if (dump && tx) DumpTexture(dev, tx, kKindName[kind], s);
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        Log("d3d: exception 0x%08X in OnDraw (ignored)", GetExceptionCode());
    }
}

// ---- hooks -------------------------------------------------------------------------
static HRESULT STDMETHODCALLTYPE Hk_SetTexture(IDirect3DDevice9 *d, DWORD stage, IDirect3DBaseTexture9 *t) {
    if (stage < 16) g_tex[stage] = t;
    return g_SetTexture(d, stage, t);
}
static HRESULT STDMETHODCALLTYPE Hk_SetPixelShader(IDirect3DDevice9 *d, IDirect3DPixelShader9 *ps) {
    g_curPS = ps;
    return g_SetPixelShader(d, ps);
}
static HRESULT STDMETHODCALLTYPE Hk_SetPSConst(IDirect3DDevice9 *d, UINT start, CONST float *v, UINT count) {
    if (v && start < 256) {
        UINT n = count; if (start + n > 256) n = 256 - start;
        memcpy(g_psConst[start], v, n * 16);
    }
    return g_SetPSConst(d, start, v, count);
}
static HRESULT STDMETHODCALLTYPE Hk_DrawPrimitive(IDirect3DDevice9 *d, D3DPRIMITIVETYPE t, UINT a, UINT b) {
    OnDraw(d); return g_DrawPrimitive(d, t, a, b);
}
static HRESULT STDMETHODCALLTYPE Hk_DrawIndexedPrimitive(IDirect3DDevice9 *d, D3DPRIMITIVETYPE t, INT a, UINT b, UINT c, UINT e, UINT f) {
    OnDraw(d); return g_DrawIndexedPrimitive(d, t, a, b, c, e, f);
}
static HRESULT STDMETHODCALLTYPE Hk_DrawPrimitiveUP(IDirect3DDevice9 *d, D3DPRIMITIVETYPE t, UINT a, CONST void *p, UINT s) {
    OnDraw(d); return g_DrawPrimitiveUP(d, t, a, p, s);
}
static HRESULT STDMETHODCALLTYPE Hk_DrawIndexedPrimitiveUP(IDirect3DDevice9 *d, D3DPRIMITIVETYPE t, UINT a, UINT b, UINT c, CONST void *i, D3DFORMAT f, CONST void *v, UINT s) {
    OnDraw(d); return g_DrawIndexedPrimitiveUP(d, t, a, b, c, i, f, v, s);
}
static HRESULT STDMETHODCALLTYPE Hk_Present(IDirect3DDevice9 *d, CONST RECT *a, CONST RECT *b, HWND w, CONST RGNDATA *r) {
    g_frame++;
    if (InterlockedExchange(&g_dumpRequest, 0)) {
        for (int k = 0; k < K_COUNT; k++) g_dumpPending[k] = 1;
        Log("d3d: F3 - dumping the textures of the next draw of each water pass");
    }
    return g_Present(d, a, b, w, r);
}

static void Patch(void **vt, int slot, void *hook, void **orig) {
    DWORD old;
    if (!VirtualProtect(&vt[slot], sizeof(void *), PAGE_EXECUTE_READWRITE, &old)) return;
    if (vt[slot] != hook) { *orig = vt[slot]; vt[slot] = hook; }
    VirtualProtect(&vt[slot], sizeof(void *), old, &old);
}

static HRESULT STDMETHODCALLTYPE Hk_CreateDevice(IDirect3D9 *self, UINT ad, D3DDEVTYPE type, HWND w, DWORD flags,
                                                 D3DPRESENT_PARAMETERS *pp, IDirect3DDevice9 **out) {
    HRESULT hr = g_CreateDevice(self, ad, type, w, flags, pp, out);
    Log("d3d: CreateDevice(type %d, flags %08X) -> %08X, device %p", type, flags, hr, (out && SUCCEEDED(hr)) ? *out : NULL);
    if (SUCCEEDED(hr) && out && *out) {
        void **vt = *(void ***)*out;
        Patch(vt, 17, (void *)Hk_Present, (void **)&g_Present);
        Patch(vt, 65, (void *)Hk_SetTexture, (void **)&g_SetTexture);
        Patch(vt, 81, (void *)Hk_DrawPrimitive, (void **)&g_DrawPrimitive);
        Patch(vt, 82, (void *)Hk_DrawIndexedPrimitive, (void **)&g_DrawIndexedPrimitive);
        Patch(vt, 83, (void *)Hk_DrawPrimitiveUP, (void **)&g_DrawPrimitiveUP);
        Patch(vt, 84, (void *)Hk_DrawIndexedPrimitiveUP, (void **)&g_DrawIndexedPrimitiveUP);
        Patch(vt, 107, (void *)Hk_SetPixelShader, (void **)&g_SetPixelShader);
        Patch(vt, 109, (void *)Hk_SetPSConst, (void **)&g_SetPSConst);
        Log("d3d: device vtable %p patched (Present, SetTexture, Draw*, SetPixelShader, SetPixelShaderConstantF)", vt);
    }
    return hr;
}

DWORD WINAPI D3DHookThread(LPVOID) {
    HMODULE m = LoadLibraryA("d3d9.dll");
    typedef IDirect3D9 *(WINAPI *Create9Fn)(UINT);
    Create9Fn create = m ? (Create9Fn)GetProcAddress(m, "Direct3DCreate9") : NULL;
    IDirect3D9 *d3d = create ? create(D3D_SDK_VERSION) : NULL;
    if (!d3d) { Log("d3d: could not create IDirect3D9, no graphics logging"); return 0; }
    void **vt = *(void ***)d3d;
    Patch(vt, 16, (void *)Hk_CreateDevice, (void **)&g_CreateDevice);
    Log("d3d: IDirect3D9 vtable %p: CreateDevice hooked (waiting for the game to create its device)", vt);
    d3d->Release();
    return 0;
}
