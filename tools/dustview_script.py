"""DustView Script tab: what the level script (tools/level_schedule.py) is doing in model mode.

Sections: Now (level time, sea level, tide), Running (TerraForming brushes with progress, springs / vents
switched on for a time, waves, tsunamis), Coming up (timed rules with countdowns, also the next run of a
looping sequence's rules; rules waiting on conditions with each condition met / not met), Recent (the
script log without tutorial / tooltip / encyclopedia noise) and a Preview graph for the selected item:
TerraForming = strength per frame and the height added / removed at the brush centre over the effect (with
a "now" marker while it runs), Tide = the sea level over one cycle. The Curves list previews any of the
TerraForming strength curves. Double-click an item with a position to centre the map on it (ring marker).
Read-only: it only reads the schedule the model thread runs (snapshots, refreshed ~3 times a second while
the tab is shown).
"""
import time
import tkinter as tk
from tkinter import ttk

NOISE = {'Tuto', 'Tooltip', 'BigTooltip', 'DustPediaEntry', 'CloseAllTutos', 'DisplayHUDElement', 'PlaySound',
         'AddWarning', 'Cinematic', 'CinematicPriority', 'HooverActivateLayer', 'RenderParam'}
W, H = 252, 150                                   # preview canvas


def _fmt_time(s):
    s = max(0.0, s)
    return '%d:%04.1f' % (s // 60, s % 60) if s >= 60 else '%.1fs' % s


def _effect_text(e):
    import sav_sources as ss, terraform
    n = e['name']
    if n == 'TerraForming':
        f = terraform.effect_fields(e)
        return 'TerraForming %s col %d row %d, size %d%s, %gs' % (
            'raise' if f['op'] else 'lower', f['col'], f['row'], f['start_size'],
            '' if f['end_size'] == f['start_size'] else '->%d' % f['end_size'], f['duration'])
    if n == 'Sequence':
        return 'Sequence (%d rules, %s)' % (len(e['rules']), 'loops every %gs' % e['duration'] if e['loops'] else 'once')
    return ss.describe_effect(e, [])[:90]


def _effect_pos(e, sch):
    """(row, col, radius) on the map for an effect, if it has one."""
    if e['name'] == 'TerraForming':
        import terraform
        f = terraform.effect_fields(e)
        return f['row'], f['col'], max(f['start_size'], f['end_size']) / 2
    if e['name'] == 'FluidGenerator':
        if e.get('col') is not None and e.get('row') is not None:
            return e['row'], e['col'], e.get('radius', 3)
        try:
            g = sch._target(e) if e.get('type') != 0 else None
        except Exception:
            g = None
        if g is not None:
            return g.row, g.col, g.size / 2
    return None


def _cond_state(sch, st):
    """(text, met) for an armed rule's condition state, without changing it."""
    c = st['c']
    n = c['name']
    t = sch.time
    try:
        if n == 'CreatedVillagesCount':
            import level_schedule as ls
            v = sch.villages(t)
            ok = ls.TYPETEST.get(c.get('test'), lambda a, b: False)(v, c.get('villages', 0))
            return 'villages %d %s %d' % (v, c.get('test'), c.get('villages', 0)), ok
        if n == 'RandomTime':
            return 'random wait %.1f / %.1fs' % (st['t'], st.get('wait', 0)), st['t'] > st.get('wait', 0)
        if n == 'CheckVegetationNumberOnce' and 'cells' in c:
            v = sch.vegetation(t)
            return 'vegetation %d >= %d' % (v, c['cells']), v >= c['cells']
        if n in ('Population', 'PopulationUnder') and 'population' in c:
            v = sch.population(t)
            return 'population %d %s %d' % (v, '>=' if n == 'Population' else '<', c['population']), \
                (v >= c['population']) if n == 'Population' else (v < c['population'])
        v = sch.conditions.get(n, False)
        return n + ('' if n in sch.conditions else ' (not modelled: false)'), bool(v(t) if callable(v) else v)
    except Exception:
        return n, False


class ScriptTab:
    def __init__(self, parent, app):
        self.app = app
        self.items = {}                            # tree item id -> dict(kind, e, pos, tf, ...)
        self.selected = None
        self.curve_choice = None
        self._last = 0.0
        self._sch = None
        f = self.frame = parent
        self.now_text = tk.StringVar()
        ttk.Label(f, textvariable=self.now_text, wraplength=250, justify='left').pack(fill='x')
        self.show_noise = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text='Show tutorial / UI events', variable=self.show_noise, takefocus=False,
                        command=self.refresh).pack(anchor='w')
        self.tree = ttk.Treeview(f, columns=('when',), height=14, selectmode='browse', takefocus=False)
        self.tree.heading('#0', text='Event')
        self.tree.heading('when', text='When')
        self.tree.column('#0', width=180, stretch=True)
        self.tree.column('when', width=62, anchor='e', stretch=False)
        self.tree.pack(fill='x', pady=(4, 0))
        for tag, colour in (('met', '#2a7a2a'), ('unmet', '#a03030'), ('dim', '#777777')):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._select())
        self.tree.bind('<Double-1>', lambda e: self._locate())
        self.sec = {k: self.tree.insert('', 'end', text=t, open=True) for k, t in
                    (('running', 'Running'), ('upcoming', 'Coming up'), ('waiting', 'Waiting on conditions'),
                     ('recent', 'Recent'))}
        row = ttk.Frame(f); row.pack(fill='x', pady=(6, 0))
        ttk.Label(row, text='Preview curve').pack(side='left')
        import terraform
        self.curve_var = tk.StringVar(value='(selected event)')
        names = ['(selected event)'] + ['%d %s' % (i, n[:-4]) for i, n in enumerate(terraform.CURVES)]
        cb = ttk.Combobox(row, textvariable=self.curve_var, values=names, state='readonly', width=26, takefocus=False)
        cb.pack(side='left', padx=(4, 0))
        cb.bind('<<ComboboxSelected>>', lambda e: self._curve_chosen())
        self.canvas = tk.Canvas(f, width=W, height=H, bg='white', highlightthickness=1, highlightbackground='#bbb')
        self.canvas.pack(pady=(4, 0))
        self.preview_text = tk.StringVar()
        ttk.Label(f, textvariable=self.preview_text, wraplength=250, justify='left', foreground='#555').pack(fill='x')
        ttk.Label(f, text='Double-click an event with a place to centre the map on it.', wraplength=250,
                  foreground='#666').pack(fill='x', pady=(4, 0))

    # ---------------------------------------------------------------- data
    def _schedule(self):
        return getattr(self.app.src, 'schedule', None)

    def sync(self, force=False):
        now = time.time()
        if not force and now - self._last < 0.33:
            return
        self._last = now
        self.refresh()

    def refresh(self):
        self._refreshing = True
        try:
            self._refresh()
        finally:
            self._refreshing = False

    def _refresh(self):
        sch = self._schedule()
        tree = self.tree
        for s in self.sec.values():
            tree.delete(*tree.get_children(s))
        self.items = {}
        if sch is None:
            self.now_text.set('The script runs in model mode: load a level or a save on the Map tab. (Taking over '
                              'from the live game has no script.)')
            self._draw_preview()
            return
        try:
            self._fill(sch)
        except (RuntimeError, KeyError, IndexError, AttributeError):
            pass                                   # the model thread changed something mid-read: next refresh
        self._draw_preview()

    def _add(self, sec, text, when='', tags=(), **info):
        iid = self.tree.insert(self.sec[sec], 'end', text=text, values=(when,), tags=tags)
        self.items[iid] = info
        key = info.get('key')
        if key is not None and key == getattr(self, '_sel_key', None):
            self.tree.selection_set(iid)
        return iid

    def _fill(self, sch):
        import terraform
        noise = not self.show_noise.get()
        td = sch.tide
        tide = ''
        if td is not None:
            tide = ', tide %s' % {1: 'falling', 2: 'low', 3: 'rising', 4: 'high'}.get(td['state'], '?')
        self.now_text.set('Level time %s | sea level %.2f%s | %d sequences running'
                          % (_fmt_time(sch.time), sch.border_level, tide, len(sch.lists) - 1))
        # running
        for tf in list(sch.terraforms):
            f = tf.f
            el = max(tf.elapsed, 0.0)
            self._add('running', 'TerraForming %s col %d row %d (%d%%)' % ('raise' if f['op'] else 'lower', f['col'],
                      f['row'], 100 * el / tf.T), '%s left' % _fmt_time(tf.T - el),
                      kind='tf', f=f, elapsed=el, pos=(f['row'], f['col'], max(f['start_size'], f['end_size']) / 2),
                      key=('tf', f['col'], f['row'], f['op']))
        for a in list(sch.active):
            g, e = a['gen'], a['e']
            self._add('running', 'generator #%d %s col %d row %d' % (g.index, 'lava' if g.temperature else 'water',
                      g.col, g.row), '%s left' % _fmt_time(e['duration'] - a['timer']), kind='gen',
                      pos=(g.row, g.col, g.size / 2), key=('gen', g.index))
        for w in list(sch.waves):
            self._add('running', '%s on edge %d, amplitude %g' % (w['kind'], w['side'], w['amp']),
                      '%s/%gs' % (_fmt_time(w['t']), w['dur']), kind='wave', key=('wave', w['side'], w['kind']))
        if td is not None:
            self._add('running', 'Tide (sea level %.2f)' % sch.border_level, {1: 'falling', 2: 'low', 3: 'rising',
                      4: 'high'}.get(td['state'], ''), kind='tide', tide=dict(td), level=sch.border_level, key=('tide',))
        # coming up / waiting
        up, wait = [], []
        for lst in list(sch.lists):
            if not lst.playing:
                continue
            for x in lst.rules:
                r = x['r']
                effs = [e for e in r['effects'] if not (noise and e['name'] in NOISE)]
                if not effs:
                    continue
                st = x['state']
                eta = None
                if st == 0:
                    eta = r['start'] - lst.t
                elif st == 3 and lst.loops and lst.duration >= 0:
                    eta = (lst.duration - lst.t) + r['start']        # the next run of the looping sequence
                if eta is not None:
                    for e in effs:
                        up.append((eta, e, r, bool(r['conditions'])))
                elif st == 1:
                    wait.append((r, effs, x['conds'] or []))
        up.sort(key=lambda u: u[0])
        for eta, e, r, conds in up[:40]:
            self._add('upcoming', _effect_text(e) + (' (if conditions)' if conds else ''), 'in ' + _fmt_time(eta),
                      kind='effect', e=e, eta=eta, pos=_effect_pos(e, sch),
                      key=('e', r['id'], e['name'], eta // 1))
        for r, effs, conds in wait[:20]:
            states = [_cond_state(sch, c) for c in conds]
            ok = all(m for _, m in states)
            iid = self._add('waiting', ' + '.join(_effect_text(e) for e in effs)[:120], 'met' if ok else 'not met',
                            tags=('met' if ok else 'unmet',), kind='effect', e=effs[0], pos=_effect_pos(effs[0], sch),
                            key=('w', r['id']))
            for text, met in states:
                self.tree.insert(iid, 'end', text=('yes: ' if met else 'no: ') + text, tags=('met' if met else 'unmet',))
        # recent
        n = 0
        for t, text in reversed(list(sch.events)):
            if noise and any(text.startswith(k) for k in NOISE):
                continue
            self._add('recent', text[:120], _fmt_time(t), tags=('dim',))
            n += 1
            if n >= 25:
                break

    # ---------------------------------------------------------------- selection / map
    _refreshing = False

    def _select(self):
        sel = self.tree.selection()
        info = self.items.get(sel[0]) if sel else None
        if info is not None and self._refreshing:   # the refresh re-selecting the same event: new data only
            self.selected = info
            return
        if info is not None:
            self.selected = info
            self._sel_key = info.get('key')
            self.curve_var.set('(selected event)')
            self.curve_choice = None
        self._draw_preview()

    def _locate(self):
        sel = self.tree.selection()
        info = self.items.get(sel[0]) if sel else None
        if info and info.get('pos'):
            r, c, rad = info['pos']
            self.app.centre_on(int(r), int(c), rad)

    def _curve_chosen(self):
        v = self.curve_var.get()
        self.curve_choice = None if v.startswith('(') else int(v.split()[0])
        self._draw_preview()

    # ---------------------------------------------------------------- preview graph
    def _axes(self, title, xmax, ymin, ymax, xlabel='s'):
        cv = self.canvas
        cv.delete('all')
        L, R, T, B = 34, W - 6, 14, H - 18
        cv.create_text(L, 2, text=title, anchor='nw', font=('Segoe UI', 8))
        cv.create_line(L, B, R, B, fill='#999')
        cv.create_line(L, T, L, B, fill='#999')
        span = (ymax - ymin) or 1.0
        for v in (ymin, ymax):
            y = B - (v - ymin) / span * (B - T)
            cv.create_text(L - 3, y, text='%.3g' % v, anchor='e', font=('Segoe UI', 7))
        cv.create_text(L, B + 2, text='0', anchor='n', font=('Segoe UI', 7))
        cv.create_text(R, B + 2, text='%g%s' % (xmax, xlabel), anchor='ne', font=('Segoe UI', 7))
        return lambda x, y: (L + x / (xmax or 1) * (R - L), B - (y - ymin) / span * (B - T))

    def _plot(self, tr, xs, ys, colour):
        pts = []
        for x, y in zip(xs, ys):
            pts += tr(x, y)
        if len(pts) >= 4:
            self.canvas.create_line(*pts, fill=colour, width=2)

    def _marker(self, tr, x, ymin, ymax):
        x0, y0 = tr(x, ymin)
        _, y1 = tr(x, ymax)
        self.canvas.create_line(x0, y0, x0, y1, fill='#e07000', dash=(3, 2))
        self.canvas.create_text(x0 + 2, y1, text='now', anchor='nw', fill='#e07000', font=('Segoe UI', 7))

    def _draw_preview(self):
        import terraform
        self.canvas.delete('all')
        if self.curve_choice is not None:
            c = terraform.curve(self.curve_choice)
            xs = [i / 100 for i in range(101)]
            ys = [c(x) for x in xs]
            tr = self._axes('Curve %d %s (y range %g)' % (self.curve_choice, terraform.CURVES[self.curve_choice][:-4],
                            getattr(c, 'y_range', 1)), 1, min(0.0, min(ys)), max(ys) or 1, xlabel='')
            self._plot(tr, xs, ys, '#1f6fd0')
            self.preview_text.set('Strength (per frame, times the brush texel / 256) over the effect, x = fraction of '
                                  'its duration.')
            return
        info = self.selected
        if not info:
            self.preview_text.set('Select a TerraForming event or the tide for a graph, or pick a curve.')
            return
        f = None
        if info.get('kind') == 'tf':
            f = info['f']
        elif info.get('kind') == 'effect' and info['e']['name'] == 'TerraForming':
            f = terraform.effect_fields(info['e'])
        if f is not None:
            self._preview_tf(f, info.get('elapsed'))
        elif info.get('kind') == 'tide':
            self._preview_tide(info['tide'], info['level'])
        else:
            self.preview_text.set('No graph for this event.')

    def _preview_tf(self, f, elapsed):
        import terraform, numpy as np
        T = float(f['duration'])
        c = terraform.curve(f['curve'])
        n = max(1, int(T / terraform.FRAME))
        xs = [k * terraform.FRAME for k in range(n)]
        strength = [c(x / T) for x in xs]
        levels = terraform.brush_levels(f['brush']) if f['brush'] < len(terraform.BRUSHES) else None
        cum, tot = [], 0.0
        for x, s in zip(xs, strength):
            size = int(f['start_size'] + (f['end_size'] - f['start_size']) * x / T)
            L, k = 64, 0
            while size < L and L > 4:
                L >>= 1
                k += 1
            texel = float(levels[k][L // 2, L // 2]) if levels is not None else 255.0
            tot += s * texel / 256 * (1 if f['op'] else -1)
            cum.append(tot)
        lo, hi = min(0.0, min(cum)), max(0.0, max(cum))
        tr = self._axes('%s at col %d row %d: height at the centre' % ('Raise' if f['op'] else 'Lower', f['col'], f['row']),
                        T, lo, hi or 1)
        self._plot(tr, xs, cum, '#1f6fd0')
        smax = max(strength) or 1
        self._plot(tr, xs, [lo + s / smax * (hi - lo) * 0.5 for s in strength], '#9a9a9a')
        if elapsed is not None:
            self._marker(tr, min(elapsed, T), lo, hi or 1)
        self.preview_text.set('Blue: ground change at the brush centre (%+.1f in total over %gs). Grey: curve %d '
                              '(peak %.2f per frame x texel / 256). Size %d%s cells, brush %d. Relative to the ground; '
                              'lowering stops at 0.' % (cum[-1], T, f['curve'], smax, f['start_size'],
                              '' if f['end_size'] == f['start_size'] else ' -> %d' % f['end_size'], f['brush']))

    def _preview_tide(self, td, level):
        h0, amp, tr_t, low, high = td['h0'], td['amp'], td['trans'], td['low'], td['high']
        lo = max(0.0, h0 - amp)
        pts = [(0, h0), (tr_t, lo), (tr_t + low, lo), (2 * tr_t + low, h0), (2 * tr_t + low + high, h0)]
        period = pts[-1][0]
        tr = self._axes('Tide: sea level over one cycle (%gs)' % period, period, lo - 1, h0 + 1)
        self._plot(tr, [p[0] for p in pts], [p[1] for p in pts], '#1f6fd0')
        st, timer = td['state'], td['timer']
        if st == 1:
            x = (h0 - level) / max(amp / tr_t, 1e-9)
        elif st == 2:
            x = tr_t + timer
        elif st == 3:
            x = tr_t + low + (level - lo) / max(amp / tr_t, 1e-9)
        else:
            x = 2 * tr_t + low + timer
        self._marker(tr, min(max(x, 0), period), lo - 1, h0 + 1)
        self.preview_text.set('Falls %g over %gs, stays low %gs, rises over %gs, stays high %gs. Now %.2f.'
                              % (amp, tr_t, low, tr_t, high, level))
