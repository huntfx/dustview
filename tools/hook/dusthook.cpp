// dusthook: minimal From Dust hook that toggles the game's built-in level editor.
//
// Loaded by the game as XAPOFX1_5.dll (the exe imports only CreateFX from it; we
// forward that to the real system DLL). A background thread waits for the game
// window and subclasses it; F2 during normal play (mode 2) switches CLevelParameters'
// game mode to editor (1), on the game's own UI thread. Inside the editor, F2 is left
// to the game (its own "F2 to InGame mode").
//
// Addresses (From Dust Steam build, exe linked 2011-09-05; see ../../docs/technical-notes.md):
//   exe 0x0094B4A4            IComClassManager* (vtable slot 7 = GetSingletonImpl)
//   "DustManagerImplementation" singleton +24 -> CLevelParameters*
//   CLevelParameters +0x94    m_gameMode (0 menu, 1 editor, 2 ingame, 3/4 editor variants)
//   dustai ChangeGameMode     found by byte pattern (RVA 0x34BA0 in win32.f, 0x34A50 in win32o.f)
// Also: read-only Direct3D 9 logging of the water shaders' constants and textures (d3dlog.cpp; F3 dumps
// the textures bound to the water passes).
// To remove: delete XAPOFX1_5.dll (and dusthook.log, dustdump_*.bin) from the game folder.

#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static FARPROC g_realCreateFX = NULL;
static WNDPROC g_origWndProc = NULL;
static char    g_logPath[MAX_PATH];
char           g_logDir[MAX_PATH];
extern volatile LONG g_dumpRequest;
DWORD WINAPI D3DHookThread(LPVOID);

void Log(const char *fmt, ...) {
    FILE *f = NULL;
    if (fopen_s(&f, g_logPath, "a") != 0 || !f) return;
    SYSTEMTIME t; GetLocalTime(&t);
    fprintf(f, "%02d:%02d:%02d.%03d ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list va; va_start(va, fmt); vfprintf(f, fmt, va); va_end(va);
    fputc('\n', f);
    fclose(f);
}

// ---- CreateFX forwarder -------------------------------------------------------
extern "C" __declspec(naked) void CreateFX_proxy() {
    __asm jmp dword ptr [g_realCreateFX]
}

// ---- Game structures -------------------------------------------------------------
static const DWORD kClassMgrPtr = 0x0094B4A4;
static const DWORD kExeLo = 0x00401000, kExeHi = 0x0089B000;   // exe .text range
static const unsigned char kChangeGameModePattern[] = {
    0x83,0xEC,0x24,0x56,0x8B,0xF1,0x8B,0x86,0x94,0x00,0x00,0x00,0x57,0x8B,0x7C,0x24,
    0x30,0x3B,0xF8,0x0F,0x84 };   // sub esp,24h; push esi; mov esi,ecx; mov eax,[esi+94h]; push edi; mov edi,[esp+30h]; cmp edi,eax; je ...

// Level-start helper called on DustManager+0x64 after entering the editor
// (dustai win32.f RVA 0x9A210; win32o.f found by the same pattern). 0x100 = wildcard (relocated absolute address).
static const short kEditorStartCallPattern[] = {
    0xA1,0x100,0x100,0x100,0x100, 0x8B,0x88,0x50,0x1C,0x00,0x00, 0x6A,0x37, 0x6A,0x28, 0xFF,0xD1,
    0x83,0xC4,0x08, 0x85,0xC0, 0x74,0x16, 0x8B,0x54,0x24,0x04, 0x52, 0x50,
    0xA1,0x100,0x100,0x100,0x100, 0x8B,0x88,0x30,0x0A,0x00,0x00, 0xFF,0xD1, 0x83,0xC4,0x08, 0xC2,0x04,0x00 };

// thiscall functions called through __fastcall with a dummy EDX (same stack cleanup).
typedef void *(__fastcall *GetSingletonFn)(void *self, void *edx, const char *name);
typedef void  (__fastcall *ChangeGameModeFn)(void *self, void *edx, int mode);
typedef void  (__fastcall *EditorStartCallFn)(void *self, void *edx, int arg);

static void *FindInModule(HMODULE mod, const short *pat, size_t n) {
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)mod;
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)((BYTE *)mod + dos->e_lfanew);
    BYTE *base = (BYTE *)mod;
    DWORD size = nt->OptionalHeader.SizeOfImage;
    void *hit = NULL;
    for (DWORD i = 0; i + n <= size; i++) {
        size_t k = 0;
        while (k < n && (pat[k] == 0x100 || base[i + k] == (BYTE)pat[k])) k++;
        if (k == n) {
            if (hit) return (void *)-1;   // ambiguous
            hit = base + i;
        }
    }
    return hit;
}

static HMODULE g_dustai = NULL;
static ChangeGameModeFn g_change = NULL;
static EditorStartCallFn g_editorStart = NULL;

static bool ResolveDustAi() {
    if (g_change && g_editorStart) return true;
    const char *names[] = { "dustai.win32.f.dll", "dustai.win32o.f.dll" };
    for (int i = 0; i < 2; i++) {
        HMODULE m = GetModuleHandleA(names[i]);
        if (!m) continue;
        short cgm[sizeof(kChangeGameModePattern)];
        for (size_t k = 0; k < sizeof(kChangeGameModePattern); k++) cgm[k] = kChangeGameModePattern[k];
        void *p = FindInModule(m, cgm, sizeof(kChangeGameModePattern));
        void *q = FindInModule(m, kEditorStartCallPattern, sizeof(kEditorStartCallPattern) / sizeof(short));
        Log("%s at %p: ChangeGameMode %p, EditorStartCall %p", names[i], m, p, q);
        if (!p || p == (void *)-1 || !q || q == (void *)-1) { Log("  pattern missing or ambiguous, refusing"); continue; }
        g_dustai = m; g_change = (ChangeGameModeFn)p; g_editorStart = (EditorStartCallFn)q;
        return true;
    }
    Log("no usable dustai module found");
    return false;
}

static void *g_dustMgr = NULL;

static BYTE *GetLevelParams() {
    void **mgr = *(void ***)kClassMgrPtr;
    if (!mgr) { Log("class manager not initialised yet"); return NULL; }
    void **vtbl = *(void ***)mgr;
    DWORD fn = (DWORD)vtbl[7];
    if (fn < kExeLo || fn >= kExeHi) { Log("GetSingletonImpl %08X outside exe code, refusing", fn); return NULL; }
    void *dustMgr = ((GetSingletonFn)fn)(mgr, NULL, "DustManagerImplementation");
    if (!dustMgr) { Log("DustManagerImplementation not found"); return NULL; }
    g_dustMgr = dustMgr;
    BYTE *lp = *(BYTE **)((BYTE *)dustMgr + 24);
    if (!lp) { Log("DustManager %p has no LevelParameters yet", dustMgr); return NULL; }
    return lp;
}

static void ToggleEditor() {
    __try {
        BYTE *lp = GetLevelParams();
        if (!lp) return;
        int *mode = (int *)(lp + 0x94), *prev = (int *)(lp + 0x98);
        int *editionLayer = (int *)(lp + 0xA0), *fakeLayer = (int *)(lp + 0xA4);
        Log("F2: LevelParameters %p, gameMode=%d (previous %d), editionLayer=%d, fakeLayer=%d",
            lp, *mode, *prev, *editionLayer, *fakeLayer);
        // Only act during normal play. Inside the editor (modes 1/3/4) the game handles F2 itself
        // ("F2 to InGame mode" on the editor HUD); acting there too made both fire at once.
        if (*mode != 2) { Log("  mode %d: leaving F2 to the game", *mode); return; }
        if (!ResolveDustAi()) return;
        // v3: plain ChangeGameMode(1), as in v1. v2 also copied the level-start extras (layer reset to
        // 1/-2 and the DustManager+0x64 -> EditorStartCall(0) engine call); the layer reset was a no-op
        // (values already 1/-2) and the cursor ring went missing with the extra call, so both are dropped.
        g_change(lp, NULL, 1);
        Log("  entered editor: gameMode=%d, editionLayer=%d, fakeLayer=%d", *mode, *editionLayer, *fakeLayer);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        Log("F2: exception 0x%08X caught, nothing changed by hook after that point", GetExceptionCode());
    }
}

// ---- Window subclass ------------------------------------------------------------
static LRESULT CALLBACK HookWndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_KEYDOWN && wp == VK_F2 && !(lp & (1 << 30)))   // ignore auto-repeat
        ToggleEditor();
    if (msg == WM_KEYDOWN && wp == VK_F3 && !(lp & (1 << 30)))
        InterlockedExchange(&g_dumpRequest, 1);              // d3dlog: dump water-pass textures
    return CallWindowProcA(g_origWndProc, h, msg, wp, lp);
}

static BOOL CALLBACK FindGameWindow(HWND h, LPARAM out) {
    DWORD pid; GetWindowThreadProcessId(h, &pid);
    if (pid != GetCurrentProcessId() || !IsWindowVisible(h)) return TRUE;
    char title[64]; GetWindowTextA(h, title, sizeof(title));
    if (strcmp(title, "Dust") != 0) return TRUE;
    *(HWND *)out = h;
    return FALSE;
}

static DWORD WINAPI SetupThread(LPVOID) {
    for (int i = 0; i < 600; i++) {   // up to 5 minutes
        HWND h = NULL;
        EnumWindows(FindGameWindow, (LPARAM)&h);
        if (h) {
            g_origWndProc = (WNDPROC)SetWindowLongPtrA(h, GWLP_WNDPROC, (LONG_PTR)HookWndProc);
            Log("window %p subclassed (orig WndProc %p). F2 toggles editor in a map, F3 dumps water textures.", h, g_origWndProc);
            return 0;
        }
        Sleep(500);
    }
    Log("game window not found, giving up");
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(inst);
        GetModuleFileNameA(inst, g_logPath, MAX_PATH);
        char *slash = strrchr(g_logPath, '\\');
        strcpy_s(g_logDir, g_logPath);
        char *dslash = strrchr(g_logDir, '\\');
        if (dslash) dslash[1] = 0; else g_logDir[0] = 0;
        strcpy_s(slash ? slash + 1 : g_logPath, MAX_PATH - (slash ? (slash + 1 - g_logPath) : 0), "dusthook.log");

        char sys[MAX_PATH];
        GetSystemDirectoryA(sys, MAX_PATH);   // WOW64 redirects this to SysWOW64
        strcat_s(sys, "\\XAPOFX1_5.dll");
        HMODULE real = LoadLibraryA(sys);
        g_realCreateFX = real ? GetProcAddress(real, "CreateFX") : NULL;
        Log("---- dusthook loaded; real XAPOFX1_5 %p, CreateFX %p", real, g_realCreateFX);
        if (!g_realCreateFX) return FALSE;   // refuse to load rather than break audio
        CloseHandle(CreateThread(NULL, 0, D3DHookThread, NULL, 0, NULL));   // before the game creates its device
        CloseHandle(CreateThread(NULL, 0, SetupThread, NULL, 0, NULL));
    }
    return TRUE;
}
