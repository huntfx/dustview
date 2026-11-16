"""DustView control panel: the box of controls on the right of the map.

Everything here calls the same App actions as the keyboard shortcuts (shown in brackets), and `sync()`
(called every frame) copies the App's state back into the widgets, so keys and panel never disagree.
Layout: what is shown (live game or the Python model) and the live / takeover buttons always at the top,
then tabs: Map (start a level file, load one of the player's saves), Model (pause, step, speed,
villages), View (views, overlays, display).
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog

VIEW_LABELS = (('1', 'natural', 'Natural'), ('2', 'height', 'Height'), ('3', 'water depth', 'Water depth'),
               ('4', 'lava temperature', 'Lava temperature'), ('5', 'sediment', 'Sediment'),
               ('6', 'flow speed', 'Flow speed'), ('7', 'unexplained changes', 'Unexplained changes (live)'),
               ('8', 'flow texture inputs', 'Flow texture inputs'))
FLOW_INPUTS = ('foam', 'velocity', 'drag')
SPEED_LABELS = ((1, '1x'), (2, '2x'), (4, '4x'), (0, 'max'))


class ControlPanel:
    def __init__(self, root, app, dv):
        self.dv, self.app, self.root = dv, app, root
        f = self.frame = ttk.Frame(root, padding=(6, 4))
        self._totem_key = None
        self._shown_src = None

        # ---- source
        box = self._section('Showing')
        self.source_text = tk.StringVar()
        ttk.Label(box, textvariable=self.source_text, wraplength=240, justify='left').pack(fill='x')
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        self._button(row, 'Live game (L)', app.connect_live).pack(side='left')
        self.takeover_btn = self._button(row, 'Continue in model (M)', app.take_over)
        self.takeover_btn.pack(side='left', padx=(4, 0))

        # ---- tabs: Map / Model / View
        self.tabs = ttk.Notebook(f, takefocus=False)
        self.tabs.pack(fill='both', expand=True)
        tab_map, tab_model, tab_view = (self._tab(t) for t in ('Map', 'Model', 'View'))
        self.tabs.bind('<<NotebookTabChanged>>', lambda e: self.root.focus_set())

        # ---- level files
        box = self._section('Start a level (fresh map)', tab_map)
        self.levels = dv.level_names()
        self.level = tk.StringVar(value=self._default_level())
        self._combo(box, self.level, self.levels).pack(fill='x')
        self._button(box, 'Start level', lambda: app.load_level(self.level.get())).pack(anchor='w', pady=(4, 0))

        # ---- player saves
        box = self._section('Load my save (map as I left it)', tab_map)
        self.saves = dv.player_saves()
        self.save = tk.StringVar(value=self.saves[-1][1] if self.saves else '')
        self._combo(box, self.save, [s[1] for s in self.saves]).pack(fill='x')
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        self._button(row, 'Load save', self._load_save).pack(side='left')
        self._button(row, 'Open file...', self._open_file).pack(side='left', padx=(4, 0))
        ttk.Label(box, text="Sources and the level script come from the level file; the script starts again "
                            "from 0 s.", wraplength=240, foreground='#666').pack(fill='x', pady=(2, 0))

        # ---- model
        box = self.model_box = self._section('Run', tab_model)
        row = ttk.Frame(box); row.pack(fill='x')
        self.pause_btn = self._button(row, 'Pause (Space)', app.toggle_pause)
        self.pause_btn.pack(side='left')
        self.step_btn = self._button(row, 'Step (N)', app.step_model)
        self.step_btn.pack(side='left', padx=(4, 0))
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='Speed (T):').pack(side='left')
        self.speed = tk.IntVar(value=1)
        self.speed_btns = [self._radio(row, text, self.speed, val, lambda: app.set_speed(self.speed.get()))
                           for val, text in SPEED_LABELS]
        for b in self.speed_btns:
            b.pack(side='left', padx=(4, 0))
        box = self._section('Villages (or click a totem on the map)', tab_model)
        self.totem_frame = ttk.Frame(box); self.totem_frame.pack(fill='x')
        self.totem_vars = []

        # ---- view
        box = self._section('Map view', tab_view)
        self.view = tk.StringVar(value=app.view)
        for key, view, text in VIEW_LABELS:
            self._radio(box, '%s  %s' % (key, text), self.view, view, lambda: app.set_view(self.view.get())).pack(anchor='w')
        row = ttk.Frame(box); row.pack(fill='x', padx=(18, 0))
        ttk.Label(row, text='input:').pack(side='left')
        self.flow_input = tk.StringVar(value=FLOW_INPUTS[0])
        self.flow_combo = self._combo(row, self.flow_input, FLOW_INPUTS, self._set_flow_input, width=10)
        self.flow_combo.pack(side='left', padx=(4, 0))

        # ---- overlays
        box = self._section('Show', tab_view)
        self.checks = {}
        for attr, text in (('overlay', 'Sources, totems, repulsors (O)'), ('flow_on', 'Flow texture, natural view (F)'),
                           ('show_events', 'Terrain-change markers, live (D)')):
            v = self.checks[attr] = tk.BooleanVar(value=getattr(app, attr))
            self._check(box, text, v, lambda a=attr: self._set_flag(a)).pack(anchor='w')

        # ---- display
        box = self._section('Display', tab_view)
        self.flip = tk.BooleanVar(value=app.flip)
        row = ttk.Frame(box); row.pack(fill='x')
        self._check(row, 'Flip vertically (V)', self.flip, lambda: app.set_flip(self.flip.get())).pack(side='left')
        self._button(row, 'Rotate (R)', app.rotate_view).pack(side='left', padx=(8, 0))
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='Zoom:').pack(side='left')
        self.scale = tk.IntVar(value=app.scale)
        for s in (1, 2, 3):
            self._radio(row, '%dx' % s, self.scale, s, lambda: app.set_scale(self.scale.get())).pack(side='left', padx=(4, 0))
        self._button(box, 'Screenshot (S)', app.screenshot).pack(anchor='w', pady=(4, 0))
        ttk.Label(f, text='Hover the map for cell values.  Esc quits.', foreground='#666').pack(anchor='w', pady=(6, 0))

    # ---- widget helpers (no keyboard focus, so Space / letters keep working as shortcuts)
    def _section(self, title, parent=None):
        box = ttk.LabelFrame(parent or self.frame, text=title, padding=(6, 3))
        box.pack(fill='x', pady=(0, 5))
        return box

    def _tab(self, title):
        page = ttk.Frame(self.tabs, padding=(4, 6))
        self.tabs.add(page, text=title)
        return page

    def _button(self, parent, text, cmd):
        return ttk.Button(parent, text=text, command=cmd, takefocus=False)

    def _radio(self, parent, text, var, value, cmd=None):
        return ttk.Radiobutton(parent, text=text, variable=var, value=value, command=cmd, takefocus=False)

    def _check(self, parent, text, var, cmd):
        return ttk.Checkbutton(parent, text=text, variable=var, command=cmd, takefocus=False)

    def _combo(self, parent, var, values, on_select=None, width=36):
        c = ttk.Combobox(parent, textvariable=var, values=list(values), state='readonly', width=width,
                         height=min(30, max(1, len(values))))

        def selected(_e):
            self.root.focus_set()                    # give the keys back to the map
            if on_select:
                on_select()
        c.bind('<<ComboboxSelected>>', selected)
        return c

    def _default_level(self):
        n = self.dv.level_number(self.app.src)
        hit = [x for x in self.levels if n and x.startswith('ALPHA_Map%d_' % n) and 'Demo' not in x]
        return hit[0] if hit else (self.levels[0] if self.levels else '')

    # ---- actions
    def _set_flag(self, attr):
        setattr(self.app, attr, self.checks[attr].get())
        self.app.dirty = True

    def _set_flow_input(self):
        self.app.flow_debug = FLOW_INPUTS.index(self.flow_input.get())
        self.app.set_view('flow texture inputs') if self.app.view != 'flow texture inputs' else None
        self.app.dirty = True

    def _load_save(self):
        hit = [p for p, label in self.saves if label == self.save.get()]
        if hit:
            self.app.load_save(hit[0])

    def _open_file(self):
        import sav_sources, compare_sim
        path = filedialog.askopenfilename(parent=self.root, title='Open a From Dust map or save',
                                          initialdir=sav_sources.PROGRESS_DIR,
                                          filetypes=[('From Dust maps / saves', '*.sav *.SAV'), ('All files', '*.*')])
        self.root.focus_set()
        if not path:
            return
        if os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(compare_sim.SAVES)):
            self.app.load_level(os.path.splitext(os.path.basename(path))[0])
        else:
            self.app.load_save(path)

    # ---- state -> widgets (every frame; only writes what changed)
    @staticmethod
    def _put(var, value):
        if var.get() != value:
            var.set(value)

    def sync(self):
        app, dv = self.app, self.dv
        src = app.src
        model = app.is_model()
        if model:
            what = 'Python model: %s' % (src.label or 'model')
        elif isinstance(src, dv.LiveGame):
            what = 'Live game (read-only)'
        else:
            what = 'Nothing loaded'
        if app.loading:
            what += '\nloading %s ...' % app.loading
        self._put(self.source_text, what)
        self._enable(self.takeover_btn, isinstance(src, dv.LiveGame) and app.frame is not None)
        for w in [self.pause_btn, self.step_btn] + self.speed_btns:
            self._enable(w, model)
        if model:
            label = 'Resume (Space)' if src.paused else 'Pause (Space)'
            if self.pause_btn.cget('text') != label:
                self.pause_btn.configure(text=label)
            self._put(self.speed, src.speed)
        self._put(self.view, app.view)
        self._enable(self.flow_combo, app.view == 'flow texture inputs', readonly=True)
        self._put(self.flow_input, FLOW_INPUTS[app.flow_debug % 3])
        for attr, v in self.checks.items():
            self._put(v, bool(getattr(app, attr)))
        self._put(self.flip, bool(app.flip))
        self._put(self.scale, app.scale)
        self._sync_totems(src if model else None)
        if src is not self._shown_src:              # a new map: point the level list at it
            self._shown_src = src
            name = getattr(src, 'level_name', None)
            if name in self.levels:
                self._put(self.level, name)

    @staticmethod
    def _enable(w, on, readonly=False):
        want = ('readonly',) if (on and readonly) else (('!disabled',) if on else ('disabled',))
        if on != (not w.instate(['disabled'])):
            w.state(want)

    def _sync_totems(self, src):
        """One checkbox per totem (model mode): ticked = village built."""
        totems = getattr(src, 'totems', None) or []
        key = (id(src), id(totems), len(totems))
        if key != self._totem_key:
            self._totem_key = key
            for w in self.totem_frame.winfo_children():
                w.destroy()
            self.totem_vars = []
            if not totems:
                ttk.Label(self.totem_frame, text='(model mode, levels with totems)', foreground='#666').pack(anchor='w')
            for i, t in enumerate(totems):
                v = tk.BooleanVar(value=bool(t.get('built')))
                self._check(self.totem_frame, '', v, lambda t=t, v=v: self._toggle_totem(t, v)).grid(row=i, column=0, sticky='w')
                self.totem_vars.append((t, v))
        for i, (t, v) in enumerate(self.totem_vars):
            self._put(v, bool(t.get('built')))
            w = self.totem_frame.grid_slaves(row=i, column=0)
            text = self._totem_text(i, t)
            if w and w[0].cget('text') != text:
                w[0].configure(text=text)

    @staticmethod
    def _totem_text(i, t):
        songs = '+'.join(k for k in ('water', 'lava') if t.get(k))
        return 'totem p%d at %d, %d%s' % (t.get('power', i), t['col'], t['row'], (', repels ' + songs) if songs else '')

    def _toggle_totem(self, t, v):
        if bool(t.get('built')) != v.get():
            self.app.toggle_totem(t)
