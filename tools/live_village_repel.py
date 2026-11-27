"""Log village repel behaviour tick by tick from the running game (read-only), to find when a village
starts / stops repelling (the threat check) and the pulse timing.

Every game tick, for every built village (totem with a village):
  - fluid around the village from the grid: for radii 10..60 cells the max water depth, cells with water
    > 0.01 / > 2 (m_threatWaterHeightThreshold), max lava depth, cells with lava > 1.5, and the highest
    fluid surface above the village's own ground;
  - the repel knowledge entities near it (vtable 0x1015B248; +0x200 flags (bit 0 = active), +0x1EC size,
    +0x1FC age, +0x1F0 AI object) and the repel AI objects (water 0x1016C1CC / lava 0x1016C214; +0x84
    trigger, +0x88 ring count, +0x7C/+0x80 cell);
  - the RepulseFluid (type-1) entries of the per-tick modifier list within 3 cells of the village
    (only if the list address is known: data/live/modifier_list_address.txt or a running spring / vent).
Objects are rescanned in memory once a second (new ones appear when a threat starts).

Output: data/live/village_repel_<time>.tsv (one row per village per tick) and a .log of events
(object appeared / gone, active flag changes, repel radius / amount changes, with the fluid stats then).

Usage (game running, level loaded): .venv/Scripts/python tools/live_village_repel.py [minutes=10]
"""
import os, struct, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, live_generators as lg, live_check as lc, galileo as g  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOW_VT = 0x1015B248
RADII = (10, 15, 20, 26, 30, 40, 60)


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    stamp = time.strftime('%Y%m%d_%H%M%S')
    tsv = open(os.path.join(HERE, 'data', 'live', 'village_repel_%s.tsv' % stamp), 'w', encoding='utf-8')
    logf = open(os.path.join(HERE, 'data', 'live', 'village_repel_%s.log' % stamp), 'w', encoding='utf-8')
    t0 = time.time()

    def log(s):
        s = '%7.2fs  %s' % (time.time() - t0, s)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    while True:                              # wait for the game, a loaded level and a built village
        try:
            p = memscan.Proc(); d = lg.dustai_delta(p); base = lc.find_sim(p)
            vils = [t for t in lg.totems(p, d) if t['built']]
            if vils:
                break
            msg = 'level loaded, no built village yet'
        except BaseException as e:           # find_sim raises SystemExit while no level is loaded
            msg = 'waiting for a level: %s' % e
        if time.time() - t0 > 30 * 60:
            log('gave up waiting (%s)' % msg); return
        log(msg); time.sleep(5)
    t0 = time.time()
    log('built villages: %s' % [(t['id'], t['col'], t['row'], 'power %d' % t['power']) for t in vils])
    vec = lg.known_modifier_list(p)
    if vec is None:
        vec = lg.find_modifier_list(p, lg.generators(p, d))
    log('modifier list: %s' % ('%08X' % vec if vec else 'not found (repel entries come from the objects only)'))

    objs = {'know': {}, 'ai': {}}            # address -> static info

    def scan():
        while time.time() - t0 < minutes * 60:
            know, ai = {}, {}
            try:
                for a in p.find(struct.pack('<I', KNOW_VT + d)):
                    raw = p.read(a, 0x204)
                    if raw is None:
                        continue
                    size = struct.unpack_from('<f', raw, 0x1EC)[0]
                    if 1 < size < 200:
                        know[a] = struct.unpack_from('<I', raw, 0x54)[0]
                for kind, vt in lg.SONG_VTABLES.items():
                    for a in p.find(struct.pack('<I', vt + d)):
                        raw = p.read(a, 0x8C)
                        if raw is None:
                            continue
                        col, row = struct.unpack_from('<2I', raw, 0x7C)
                        if 0 < col < 512 and 0 < row < 512:
                            ai[a] = (kind, col, row, struct.unpack_from('<I', raw, 0x74)[0])
            except Exception as e:
                log('scan failed: %s' % e)
            for k, new in (('know', know), ('ai', ai)):
                for a in set(new) - set(objs[k]):
                    log('%s object APPEARED at %08X: %s' % (k, a, new[a] if k == 'ai' else 'handle %08X' % new[a]))
                for a in set(objs[k]) - set(new):
                    log('%s object GONE at %08X' % (k, a))
                objs[k] = new
            time.sleep(1)
    threading.Thread(target=scan, daemon=True).start()

    # disc masks around each village
    yy, xx = np.mgrid[0:512, 0:512]
    masks = {}
    for v in vils:
        dist2 = (xx - v['col']) ** 2 + (yy - v['row']) ** 2
        masks[v['id']] = {R: dist2 <= R * R for R in RADII}
    cols = ['t', 'tick', 'village', 'col', 'row']
    for R in RADII:
        cols += ['w%d_max' % R, 'w%d_n' % R, 'w%d_n2' % R, 'l%d_max' % R, 'l%d_n15' % R, 'surf%d' % R]
    cols += ['know', 'active', 'size', 'age', 'ai', 'trig', 'rings', 'repels']
    tsv.write('\t'.join(cols) + '\n')

    last_tick, state = None, {}
    while time.time() - t0 < minutes * 60:
        n = struct.unpack('<I', p.read(base + 0x92C, 4))[0]
        if n == last_tick:
            time.sleep(0.002); continue
        last_tick = n
        reps = [e for e in (lg.read_modifier_list(p, vec, settle=0.004) if vec else []) if e['type'] == 1]
        bufs = struct.unpack('<2I', p.read(base + 0x950, 8))
        raw = p.read(bufs[(n - 1) & 1], 512 * 512 * 32)
        if raw is None:
            continue
        G = g.unpack(raw)
        T = G[0].view(np.float32); W = G[2].view(np.float32)
        water = np.where(W > 0, W, 0); lava = np.where(W < 0, -W, 0)
        # object fields this tick
        kinfo = []
        for a in list(objs['know']):
            r = p.read(a, 0x204)
            if r is None:
                continue
            ai = struct.unpack_from('<I', r, 0x1F0)[0]
            kinfo.append(dict(a=a, flags=struct.unpack_from('<I', r, 0x200)[0], size=struct.unpack_from('<f', r, 0x1EC)[0],
                              age=struct.unpack_from('<f', r, 0x1FC)[0], ai=ai))
        ainfo = {}
        for a, (kind, col, row, holder) in list(objs['ai'].items()):
            r = p.read(a, 0x8C)
            if r is None:
                continue
            ainfo[holder] = dict(kind=kind, col=col, row=row, trig=struct.unpack_from('<I', r, 0x84)[0],
                                 rings=struct.unpack_from('<I', r, 0x88)[0])
        now = time.time() - t0
        for v in vils:
            vid, vc, vr = v['id'], v['col'], v['row']
            row = [('%.3f' % now), str(n), vid, str(vc), str(vr)]
            gz = float(T[vr, vc])
            stats = {}
            for R in RADII:
                m = masks[vid][R]
                w = water[m]; l = lava[m]
                wet = (W[m] != 0)
                surf = float(((T[m] + np.abs(W[m]))[wet]).max() - gz) if wet.any() else float('nan')
                st = (float(w.max()), int((w > 0.01).sum()), int((w > 2).sum()), float(l.max()), int((l > 1.5).sum()), surf)
                stats[R] = st
                row += ['%.3f' % st[0], str(st[1]), str(st[2]), '%.3f' % st[3], str(st[4]), '%.2f' % st[5]]
            # knowledge objects belonging to this village: AI object cell within 3 cells
            mine = [ainfo[k['a']] | k for k in kinfo if k['a'] in ainfo
                    and abs(ainfo[k['a']]['col'] - vc) <= 3 and abs(ainfo[k['a']]['row'] - vr) <= 3]
            vrep = sorted((round(e['radius'], 2), round(e['amount'], 2), int(e['f5'])) for e in reps
                          if abs(e['col'] - vc) <= 3 and abs(e['row'] - vr) <= 3)
            kdesc = ';'.join('%s:%x' % (k['kind'], k['flags']) for k in mine)
            row += [str(len(mine)), kdesc, ';'.join('%.2f' % k['size'] for k in mine), ';'.join('%.2f' % k['age'] for k in mine),
                    ';'.join(k['kind'] for k in mine), ';'.join(str(k['trig']) for k in mine), ';'.join(str(k['rings']) for k in mine),
                    ';'.join('%g/%g/%+d' % e for e in vrep)]
            tsv.write('\t'.join(row) + '\n')
            # events
            key = (kdesc, tuple(vrep))
            if state.get(vid) != key:
                s26 = stats[26]; s40 = stats[40]
                log('village %s (%d,%d) tick %d: objects [%s] repels %s | r26 water max %.2f n %d n>2 %d lava max %.2f surf %.2f | r40 water max %.2f n %d n>2 %d surf %.2f' % (
                    vid, vc, vr, n, kdesc or '-', vrep or '-', s26[0], s26[1], s26[2], s26[3], s26[5], s40[0], s40[1], s40[2], s40[5]))
                state[vid] = key
        tsv.flush()
    log('done')


if __name__ == '__main__':
    main()
