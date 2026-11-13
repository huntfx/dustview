"""Read-only memory scanner for a running 32-bit game process (Windows, ctypes, no dependencies).

Usage:
  python memscan.py params              # find From Dust's live simulator parameter array(s) and print them
  python memscan.py find <hex bytes>    # list addresses where a byte pattern occurs

Opens the process with PROCESS_VM_READ | PROCESS_QUERY_INFORMATION only; never writes.
Works from 64-bit Python on a 32-bit target (addresses stay below 4 GB).
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys

PROCESS_VM_READ, PROCESS_QUERY_INFORMATION = 0x10, 0x400
MEM_COMMIT = 0x1000
READABLE = {0x02, 0x04, 0x08, 0x20, 0x40, 0x80}   # PAGE_READONLY, READWRITE, WRITECOPY, EXECUTE_READ(+WRITE/+WC)
PAGE_GUARD = 0x100

k32 = ctypes.WinDLL('kernel32', use_last_error=True)


class MBI(ctypes.Structure):
    _fields_ = [('BaseAddress', ctypes.c_void_p), ('AllocationBase', ctypes.c_void_p),
                ('AllocationProtect', wt.DWORD), ('PartitionId', wt.WORD), ('RegionSize', ctypes.c_size_t),
                ('State', wt.DWORD), ('Protect', wt.DWORD), ('Type', wt.DWORD)]


k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_size_t)]
k32.OpenProcess.restype = wt.HANDLE


def find_pid(exe):
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq %s' % exe, '/FO', 'CSV', '/NH'],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if parts and parts[0].lower() == exe.lower():
            return int(parts[1])
    return None


class Proc:
    def __init__(self, exe='From_Dust.exe'):
        self.pid = find_pid(exe)
        if not self.pid:
            raise SystemExit('%s is not running' % exe)
        self.h = k32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, self.pid)
        if not self.h:
            raise SystemExit('OpenProcess failed (%d)' % ctypes.get_last_error())

    def regions(self):
        addr, mbi = 0, MBI()
        while addr < 0xFFFF0000 and k32.VirtualQueryEx(self.h, addr, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            size = mbi.RegionSize
            if mbi.State == MEM_COMMIT and (mbi.Protect & 0xFF) in READABLE and not mbi.Protect & PAGE_GUARD:
                yield addr, size
            addr += size

    def read(self, addr, n):
        buf = ctypes.create_string_buffer(n)
        got = ctypes.c_size_t()
        if not k32.ReadProcessMemory(self.h, addr, buf, n, ctypes.byref(got)):
            return None
        return buf.raw[:got.value]

    def find(self, pattern):
        hits = []
        for base, size in self.regions():
            data = self.read(base, size)
            if not data:
                continue
            i = data.find(pattern)
            while i >= 0:
                hits.append(base + i)
                i = data.find(pattern, i + 1)
        return hits


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = Proc()
    if sys.argv[1] == 'find':
        for a in p.find(bytes.fromhex(sys.argv[2])):
            print(hex(a))
    elif sys.argv[1] == 'params':
        sys.path.insert(0, __import__('os').path.dirname(__file__))
        from galileo import PARAM_NAMES
        sig = struct.pack('<4f', 0.999, 0.032, 0.4, 0.002)    # ParamF[0..3] for normal water
        slow = struct.pack('<4f', 0.99, 0.0032, 0.4, 0.002)   # same with the slow-water set
        for label, pat in [('normal', sig), ('slow', slow)]:
            for a in p.find(pat):
                vals = struct.unpack('<56f', p.read(a, 224))
                print('== %s pattern at %s' % (label, hex(a)))
                for i, (n, v) in enumerate(zip(PARAM_NAMES, vals)):
                    print('  %2d %-42s %g' % (i, n, v))


if __name__ == '__main__':
    main()
