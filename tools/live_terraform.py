"""Log the ground at a level's TerraForming spots in the running game (read-only), to check
tools/terraform.py against the game.

Usage: .venv/Scripts/python tools/live_terraform.py [level=ALPHA_Map1_Erosion] [seconds=180] [per_second=6]
Start it before or after loading the level; it waits for the simulator. Every sample reads the game's
complete grid buffer (the one not being written, as DustView's LiveGame) and records, per spot, the
simulation tick, the ground at the centre, the highest ground within the brush's reach and the water
surface at the centre.
Output: data/live/terraform_<level>_<time>.log (readable, progress printed) and .npz (arrays for
compare: python live_terraform.py compare <npz>).
"""
import os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memscan, galileo as g, live_check as lc, sav_sources as ss, terraform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INGAME = os.path.join(ss.INSTALL, 'DUST', 'DustSave', 'Ingame')


def spots(level):
    """(col, row, reach) of every TerraForming effect in the level's script (reach = half the largest size)."""
    import sav
    _, u = sav.read_sav(os.path.join(INGAME, level + '.sav'))
    out = {}

    def walk(lst):
        for r in lst['rules']:
            for e in r['effects']:
                if e['name'] == 'Sequence':
                    walk(e)
                elif e['name'] == 'TerraForming':
                    f = terraform.effect_fields(e)
                    k = (f['col'], f['row'])
                    out[k] = max(out.get(k, 0), max(f['start_size'], f['end_size']) // 2)
    walk(ss.find_root(u))
    return [(c, r, h) for (c, r), h in sorted(out.items())]


def read_grid(p, base):
    """(tick, ground, fluid) from the complete buffer, or None if a step started meanwhile."""
    rd = lambda off, n: p.read(base + off, n)
    n = struct.unpack('<I', rd(0x92C, 4))[0]
    bufs = struct.unpack('<2I', rd(0x950, 8))
    raw = p.read(bufs[(n - 1) & 1], lc.GRID)
    if raw is None or struct.unpack('<I', rd(0x92C, 4))[0] != n:
        return None
    G = g.unpack(raw)
    return n, G[0].view(np.float32), G[2].view(np.float32)


def log_main(level, secs, rate):
    pts = spots(level)
    os.makedirs(os.path.join(HERE, 'data', 'live'), exist_ok=True)
    stem = os.path.join(HERE, 'data', 'live', time.strftime('terraform_%s_%%Y%%m%%d_%%H%%M%%S' % level))
    log = open(stem + '.log', 'w', encoding='utf-8')

    def out(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    out('%s: %d spots %s' % (level, len(pts), ' '.join('%d/%d' % (c, r) for c, r, _ in pts)))
    p = memscan.Proc()
    base = None
    while base is None:
        try:
            base = lc.find_sim(p)
        except SystemExit:
            out('waiting for a level (simulator not found yet) ...')
            time.sleep(3)
    out('simulator at 0x%X; logging %gs at %g per second to %s.log' % (base, secs, rate, stem))
    yy, xx = np.mgrid[0:512, 0:512]
    masks = [((xx - c) ** 2 + (yy - r) ** 2 <= h * h) for c, r, h in pts]
    T, ticks, centre, top, surf = [], [], [], [], []
    t0 = time.time()
    last_tick, last_print = None, 0
    while time.time() - t0 < secs:
        got = read_grid(p, base)
        if got is None:
            continue
        n, ground, fluid = got
        if n == last_tick:                       # paused: no new state
            time.sleep(0.2)
            continue
        last_tick = n
        T.append(time.time() - t0)
        ticks.append(n)
        centre.append([ground[r, c] for c, r, _ in pts])
        top.append([ground[m].max() for m in masks])
        surf.append([ground[r, c] + max(fluid[r, c], 0) for c, r, _ in pts])
        if time.time() - last_print >= 2:
            last_print = time.time()
            out('%6.1fs tick %7d | centre ground %s' % (T[-1], n, ' '.join('%5.1f' % v for v in centre[-1])))
        time.sleep(max(0.0, 1 / rate - 0.02))
    np.savez(stem + '.npz', level=level, spots=np.array(pts), t=np.array(T), tick=np.array(ticks),
             centre=np.array(centre), top=np.array(top), surface=np.array(surf))
    out('saved %s.npz (%d samples)' % (stem, len(T)))


def compare_main(path):
    """The model's prediction for each spot (level script + terraform brushes on the level file's ground,
    no water) against the log, aligned on each spot's first logged rise."""
    import dustsim, level_schedule, sav
    d = np.load(path)
    level = str(d['level'])
    pts = [tuple(x) for x in d['spots']]
    _, u = sav.read_sav(os.path.join(INGAME, level + '.sav'))
    lay = lambda k: np.frombuffer(u[0x80 + k * 0x100000:0x80 + (k + 1) * 0x100000], np.float32).reshape(512, 512)
    G = {'T': lay(0).copy(), 'soft': lay(1).copy()}
    sch = level_schedule.LevelSchedule(level)
    span = (d['tick'][-1] - d['tick'][0]) + 31 * 60
    pred = []
    for k in range(int(span)):
        sch.tick(1 / 31)
        for c in sch.brush_calls:
            terraform.apply_brush(G, c)
        pred.append([G['T'][r, x] for x, r, _ in pts])
    pred = np.array(pred)
    ticks = d['tick'] - d['tick'][0]
    for i, (x, r, h) in enumerate(pts):
        live = d['centre'][:, i]
        print('spot col %d row %d: live centre ground min %.2f max %.2f; model (from level start) min %.2f max %.2f'
              % (x, r, live.min(), live.max(), pred[:, i].min(), pred[:, i].max()))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    if a and a[0] == 'compare':
        compare_main(a[1])
    else:
        log_main(a[0] if a else 'ALPHA_Map1_Erosion', float(a[1]) if len(a) > 1 else 180, float(a[2]) if len(a) > 2 else 6)
