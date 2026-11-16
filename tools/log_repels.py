"""Log every RepulseFluid (type-1) entry in the live game's per-tick modifier list (read-only).

The game lists a village's repel only while fluid is near it, so this watches the list for a while and
writes each distinct entry once (cell, radius, mode, amount, sediment), with the nearest totem and
whether a song object (live_generators.songs) exists there. A background thread also rescans memory
for the repel objects (ENTITY_KNOWLEDGE_I = repel water, _J = repel lava) every ~2 s and logs each one
appearing / disappearing, to test whether they exist only while fluid threatens the village.

Usage (game running, level loaded):  .venv/Scripts/python tools/log_repels.py [minutes]
Log: data/live/repels_<time>.log
"""
import os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dustview as dv, live_generators  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 15
    path = os.path.join(HERE, 'data', 'live', time.strftime('repels_%Y%m%d_%H%M%S.log'))
    out = open(path, 'w', encoding='utf-8')
    log = lambda s: (print(s, flush=True), out.write(s + '\n'), out.flush())
    g = dv.LiveGame()
    t0 = time.time()
    while time.time() - t0 < 40 and (getattr(g, 'vec', None) is None or getattr(g, 'totems', None) is None
                                     or getattr(g, 'songs', None) is None):
        g.read(); time.sleep(0.2)
    log('level %s | totems %s' % (dv.level_number(g), [(t.get('power'), round(t['col']), round(t['row'])) for t in (getattr(g, 'totems', None) or [])]))
    log('song objects: %s' % [(s['kind'], round(s['col']), round(s['row']), s['radius']) for s in (getattr(g, 'songs', None) or [])])
    log('source list %s' % ('found' if getattr(g, 'vec', None) is not None else 'NOT found (needs a running, unburied source)'))
    seen, frames, last_note = {}, 0, time.time()

    def watch_objects():                     # repel objects appearing / disappearing
        delta = live_generators.dustai_delta(g.p)
        have = None
        while time.time() - t0 < minutes * 60:
            try:
                now = {(o['kind'], o['col'], o['row'], round(o['size'], 2)) for o in live_generators.songs(g.p, delta)}
            except Exception as e:
                log('object scan failed: %s' % e); time.sleep(2); continue
            if have is not None:
                for o in sorted(now - have):
                    log('%6.1fs  OBJECT APPEARED: repel %s at (%d, %d), size %s' % ((time.time() - t0,) + o))
                for o in sorted(have - now):
                    log('%6.1fs  OBJECT GONE: repel %s at (%d, %d), size %s' % ((time.time() - t0,) + o))
            else:
                log('%6.1fs  repel objects now: %s' % (time.time() - t0, sorted(now)))
            have = now
            time.sleep(1)
    threading.Thread(target=watch_objects, daemon=True).start()
    while time.time() - t0 < minutes * 60:
        if g.read() is not None:
            frames += 1
        for e in getattr(g, 'other_modifiers', None) or []:
            if e.get('type') != 1:
                continue
            key = (round(e['col']), round(e['row']), round(e['radius'], 2), e['f5'], round(e['amount'], 3), round(e['sediment'], 3))
            if key in seen:
                seen[key][1] += 1
                continue
            seen[key] = [time.time(), 1]
            tots = getattr(g, 'totems', None) or []
            near = min(tots, key=lambda t: (t['col'] - e['col']) ** 2 + (t['row'] - e['row']) ** 2) if tots else None
            dist = ((near['col'] - e['col']) ** 2 + (near['row'] - e['row']) ** 2) ** 0.5 if near else -1
            song = [s['kind'] for s in (getattr(g, 'songs', None) or []) if abs(s['col'] - e['col']) < 1.5 and abs(s['row'] - e['row']) < 1.5]
            log('%6.1fs  NEW repel at (%d, %d): radius %.2f, mode %+g (%s), amount %.3f, sediment %.3f | nearest totem p%s at %.1f cells | song object here: %s | raw %s' % (
                time.time() - t0, key[0], key[1], e['radius'], e['f5'], 'water' if e['f5'] < 0 else 'lava', e['amount'], e['sediment'],
                near.get('power') if near else '?', dist, song or 'none', {k: v for k, v in e.items() if k != '_cells'}))
        if time.time() - last_note > 60:
            last_note = time.time()
            log('%6.1fs  %d frames, %d distinct repel entries so far' % (time.time() - t0, frames, len(seen)))
        time.sleep(0.03)
    log('done: %d distinct entries; frames listed in: %s' % (len(seen), {k: v[1] for k, v in seen.items()}))
    g.stop()


if __name__ == '__main__':
    main()
