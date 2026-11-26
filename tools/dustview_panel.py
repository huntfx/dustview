"""DustView control panel: the box of controls on the right of the map.

Everything here calls the same App actions as the keyboard shortcuts (shown in brackets), and `sync()`
(called every frame) copies the App's state back into the widgets, so keys and panel never disagree.
Layout: what is shown (live game or the Python model) and the live / takeover buttons always at the top,
then tabs: Map (pick a level, then one of its saves or New, all profiles), Model (pause, step, speed,
villages), Edit (paint water / lava / sand / rock with the game's own modifiers, model mode), View (views
incl. the engine-texture dropdown, overlays, display), Slice (pick two points on the map for a side-on
cross-section in its own window: rock, sand, water / lava, air; tools/dustslice.py).
"""
import os, sys
import tkinter as tk
from tkinter import ttk, filedialog
import level_names
import engine_textures

VIEW_LABELS = (('natural', 'Natural'), ('height', 'Height'), ('water depth', 'Water depth'),
               ('lava temperature', 'Lava temperature'), ('sediment', 'Sediment'), ('flow speed', 'Flow speed'),
               ('unexplained changes', 'Unexplained changes (live)'), ('engine texture', 'Engine texture:'))
TOOL_LABELS = (('inspect', 'View (drag pans, wheel zooms; click a totem: village on / off)'), ('water', 'Water'),
               ('sand', 'Sand (soft ground)'), ('rock', 'Rock (hard ground)'), ('lava', 'Lava'))
EMPTY_MAP = 'New (the level as shipped, no villages)'
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
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        self.compare_btn = self._button(row, 'Compare with model (C)', app.compare)
        self.compare_btn.pack(side='left')
        self.show_btn = self._button(row, 'Next: model (G)', app.toggle_compare_show)
        self.show_btn.pack(side='left', padx=(4, 0))

        # ---- tabs: Map / Model / View
        self.tabs = ttk.Notebook(f, takefocus=False)
        self.tabs.pack(fill='both', expand=True)
        tab_map, tab_model, tab_edit, tab_view, tab_slice = (self._tab(t) for t in ('Map', 'Model', 'Edit', 'View', 'Slice'))
        self.tab_slice = tab_slice
        self.tabs.bind('<<NotebookTabChanged>>', lambda e: self._tab_changed())

        # ---- map: pick a level, then its empty map or one of its saves
        box = self._section('Map', tab_map)
        self.levels = dv.level_names()
        self.saves = dv.player_saves()
        self.level_labels = {}                       # shown text -> file name
        for n in self.levels:
            self.level_labels[level_names.label(n)] = n
        row = ttk.Frame(box); row.pack(fill='x')
        ttk.Label(row, text='Level:').pack(side='left')
        self.level = tk.StringVar(value=self._level_label(self._default_level()))
        self._combo(row, self.level, list(self.level_labels), self._show_saves, width=32).pack(side='left', padx=(4, 0), fill='x', expand=True)
        ttk.Label(box, text='Saves').pack(anchor='w', pady=(8, 0))
        self.save_list = tk.Listbox(box, height=7, activestyle='none', exportselection=False, takefocus=0)
        self.save_list.pack(fill='x')
        self.save_list.bind('<Double-Button-1>', lambda e: self._load_selected())
        self.save_list.bind('<<ListboxSelect>>', lambda e: self.root.focus_set())
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        self._button(row, 'Load', self._load_selected).pack(side='left')
        self._button(row, 'Open file...', self._open_file).pack(side='left', padx=(4, 0))
        ttk.Label(box, text="Saves of the chosen level from every profile, newest at the top; New = the level as shipped (double-click loads). A save "
                            "is the map as it was left (terrain, water, lava, villages); sources and the level script "
                            "come from the level file, and the script starts again from 0 s.",
                  wraplength=250, foreground='#666').pack(fill='x', pady=(4, 0))
        self.shown_saves = []
        self._show_saves()

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
        dustview = sys.modules['__main__'] if hasattr(sys.modules['__main__'], 'VEGETATION_STEP') else __import__('dustview')
        box = self._section('Simulation', tab_model)
        self.step_vars = {}
        for attr, label, choices in (('FLUID_STEP', 'Water / lava / sand', ('off', 'game', 'model')),
                                     ('SOURCE_STEP', 'Sources / repels', ('off', 'game', 'model')),
                                     ('WAVE_STEP', 'Waves / tide', ('off', 'model')),
                                     ('VEGETATION_STEP', 'Vegetation / fire', ('off', 'game', 'model'))):
            row = ttk.Frame(box); row.pack(fill='x', pady=(2, 0))
            ttk.Label(row, text=label, width=17).pack(side='left')
            var = self.step_vars[attr] = tk.StringVar(value=getattr(dustview, attr))
            for val in choices:
                self._radio(row, {'off': 'Disabled', 'game': 'Game', 'model': 'Model'}[val], var, val,
                            lambda a=attr, v=var: setattr(dustview, a, v.get())).pack(side='left', padx=(4, 0))
        ttk.Label(box, text="Game = the game's own simulation code, run in a helper program (matches the game "
                            "exactly). Model = DustView's re-implementation. Water, lava and sand are one step in "
                            "the game, so they share one choice. Sources / repels = springs, lava vents, drains "
                            "and village repels. Waves / tide = the level script's edge waves and sea level "
                            "(Disabled keeps the current level). Painting works whatever is chosen.",
                  wraplength=250, foreground='#666').pack(fill='x', pady=(4, 0))
        box = self._section('Villages (or click a totem on the map)', tab_model)
        self.totem_frame = ttk.Frame(box); self.totem_frame.pack(fill='x')
        self.totem_vars = []

        # ---- edit (model mode): tool, brush size, strength
        box = self._section('Tool', tab_edit)
        self.tool = tk.StringVar(value=app.tool)
        for tool, text in TOOL_LABELS:
            self._radio(box, text, self.tool, tool, lambda: app.set_tool(self.tool.get())).pack(anchor='w')
        ttk.Label(box, text='Left button adds, middle smooths, right removes, wheel = brush size. '
                            'Ctrl + left-drag pans, Ctrl + wheel zooms.', wraplength=250,
                  foreground='#666').pack(fill='x', pady=(4, 0))
        box = self._section('Brush', tab_edit)
        self.brush_radius = tk.IntVar(value=app.brush_radius)
        self.brush_strength = tk.DoubleVar(value=app.brush_strength)
        self.lava_temperature = tk.DoubleVar(value=app.lava_temperature * 8)   # shown x 8
        self.radius_text, self.strength_text, self.temp_text = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for label, var, text, lo, hi, cmd in (
                ('Size ([ ])', self.brush_radius, self.radius_text, dv.BRUSH_MIN, dv.BRUSH_MAX,
                 lambda v: app.set_brush(radius=round(float(v)))),
                ('Strength', self.brush_strength, self.strength_text, 1, 60,
                 lambda v: app.set_brush(strength=round(float(v), 1))),
                ('Lava heat', self.lava_temperature, self.temp_text, 100, 2040,
                 lambda v: app.set_brush(temperature=round(float(v)) / 8))):
            row = ttk.Frame(box); row.pack(fill='x', pady=(2, 0))
            ttk.Label(row, text=label, width=10).pack(side='left')
            ttk.Scale(row, from_=lo, to=hi, variable=var, command=cmd, takefocus=False).pack(side='left', fill='x', expand=True)
            ttk.Label(row, textvariable=text, width=7).pack(side='left')
        ttk.Label(box, text='Size = radius in cells. Strength = height added or removed per cell per second '
                            '(a spring adds ~25); for smoothing, how fast (10: halfway to the local average in 0.2 s). Lava heat: 2040 = a vent\'s fresh lava, it hardens at 100 (the '
                            'shipped maps hold 100-2040). The game\'s own rules: water on lava cools it to rock, lava '
                            'on water boils it off and hardens; sand and rock raise the ground. The fixed '
                            'border band is not edited.',
                  wraplength=250, foreground='#666').pack(fill='x', pady=(4, 0))
        ttk.Label(tab_edit, text='Painting works on the Python model (Map tab, or "Continue in model" from the '
                                 'live game); the game itself is never written to.',
                  wraplength=260, foreground='#666').pack(fill='x', pady=(4, 0))

        # ---- view
        box = self._section('Map view', tab_view)
        self.view = tk.StringVar(value=app.view)
        for view, text in VIEW_LABELS:
            self._radio(box, text, self.view, view, lambda: app.set_view(self.view.get())).pack(anchor='w')
        self.tex_labels = [label for _, label in engine_textures.ENTRIES]
        self.texture = tk.StringVar(value=engine_textures.LABELS[app.texture])
        self.tex_combo = self._combo(box, self.texture, self.tex_labels, self._set_texture, width=40)
        self.tex_combo.pack(fill='x', padx=(18, 0))

        # ---- overlays
        box = self._section('Show', tab_view)
        self.checks = {}
        for attr, text in (('overlay', 'Sources, totems, repulsors (O)'), ('flow_on', 'Flow texture, natural view (F)'),
                           ('ground_tex', 'Sand / rock textures, natural view'),
                           ('show_plants', 'Plants (textured view: the game grass layer)'),
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
        self._button(row, '-', lambda: app.zoom_step(-1)).pack(side='left', padx=(4, 0))
        self._button(row, '+', lambda: app.zoom_step(1)).pack(side='left', padx=(2, 0))
        self._button(row, 'Whole map', lambda: app.set_zoom(1)).pack(side='left', padx=(4, 0))
        self.zoom_text = tk.StringVar()
        ttk.Label(row, textvariable=self.zoom_text).pack(side='left', padx=(6, 0))
        self.smooth = tk.BooleanVar(value=app.smooth)
        self._check(box, 'Smooth when zoomed (off: crisp cells)', self.smooth,
                    lambda: (setattr(app, 'smooth', self.smooth.get()), setattr(app, 'dirty', True))).pack(anchor='w')
        ttk.Label(box, text='Mouse wheel zooms at the cursor (Ctrl + wheel with an edit tool); left-drag pans with the View tool, '
                            'Ctrl + left-drag with an edit tool.',
                  wraplength=250, foreground='#666').pack(fill='x')
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='Window size:').pack(side='left')
        self.scale = tk.IntVar(value=app.scale)
        for s in (1, 2, 3):
            self._radio(row, '%dx' % s, self.scale, s, lambda: app.set_scale(self.scale.get())).pack(side='left', padx=(4, 0))
        self._button(box, 'Screenshot (S)', app.screenshot).pack(anchor='w', pady=(4, 0))
        # ---- slice: two points on the map -> side-on cross-section window
        box = self._section('Slice', tab_slice)
        ttk.Label(box, text='While this tab is open: left-click the map for point A, right-click for point B '
                            '(hold and drag to move them). The slice opens in its own window and follows the '
                            'simulation. Middle-drag still pans.',
                  wraplength=250).pack(fill='x')
        self.slice_text = tk.StringVar()
        ttk.Label(box, textvariable=self.slice_text, justify='left').pack(anchor='w', pady=(4, 0))
        row = ttk.Frame(box); row.pack(fill='x', pady=(4, 0))
        self.slice_open_btn = self._button(row, 'Open window', app.open_slice)
        self.slice_open_btn.pack(side='left')
        self._button(row, 'Swap A / B', app.swap_slice).pack(side='left', padx=(4, 0))
        self._button(row, 'Clear', app.clear_slice).pack(side='left', padx=(4, 0))
        box = self._section('Height scale', tab_slice)
        self.slice_whole = tk.BooleanVar(value=app.slice_whole)
        self._radio(box, 'Fit to the slice', self.slice_whole, False, lambda: app.set_slice_whole(False)).pack(anchor='w')
        self._radio(box, "Whole map's heights (compare slices)", self.slice_whole, True,
                    lambda: app.set_slice_whole(True)).pack(anchor='w')
        ttk.Label(tab_slice, text='Bottom to top: rock (grey, darker deeper down), sand, water (darker deeper, '
                                  'browner with sediment) or lava (by temperature), plants (a band on dry ground), '
                                  'air. One column per cell the line crosses; heights are stretched to fit, so '
                                  'slopes look steeper than in the game. Hover the slice for the values of a column '
                                  '(marked on the map in yellow).',
                  wraplength=260, foreground='#666').pack(fill='x', pady=(4, 0))

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
    def _tab_changed(self):
        """The Slice tab turns map clicks into placing its points; leaving it goes back to View."""
        self.root.focus_set()
        on_slice = self.tabs.select() == str(self.tab_slice)
        if on_slice and self.app.tool != 'slice':
            self.app.set_tool('slice')
        elif not on_slice and self.app.tool == 'slice':
            self.app.set_tool('inspect')

    def _set_flag(self, attr):
        setattr(self.app, attr, self.checks[attr].get())
        if attr == 'show_plants':
            import flowtex
            flowtex.SHOW_VEGETATION = self.checks[attr].get()
        self.app.dirty = True

    def _set_texture(self):
        i = self.tex_labels.index(self.texture.get())
        while engine_textures.ENTRIES[i][0] == '--':          # a group header: take the first entry below it
            i += 1
        self.app.set_texture(engine_textures.ENTRIES[i][0])

    def _level_label(self, name):
        return next((k for k, v in self.level_labels.items() if v == name), name)

    def _current_level(self):
        return self.level_labels.get(self.level.get(), self.level.get())

    def _show_saves(self):
        """List the chosen level's saves (all profiles, newest at the top), then New (the empty map).
        The newest save is selected (New when there are none)."""
        name = self._current_level()
        self.shown_saves = [d for d in self.saves if d['level'] == name]
        self.save_list.delete(0, 'end')
        for d in self.shown_saves:
            self.save_list.insert('end', d['label'])
        self.save_list.insert('end', EMPTY_MAP)
        self.save_list.selection_set(0)

    def _load_selected(self):
        sel = self.save_list.curselection()
        i = sel[0] if sel else len(self.shown_saves)
        if i < len(self.shown_saves):
            self.app.load_save(self.shown_saves[i]['path'])
        else:
            self.app.load_level(self._current_level())

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
            t = level_names.title(getattr(src, 'level_name', None))
            what = 'Python model: %s%s' % ((t + ' | ') if t else '', src.label or 'model')
        elif isinstance(src, dv.CompareSource):
            what = 'Live game + model in lockstep (showing: %s)' % app.compare_mode()
        elif isinstance(src, dv.LiveGame):
            what = 'Live game (read-only)'
        else:
            what = 'Nothing loaded'
        if app.loading:
            what += '\nloading %s ...' % app.loading
        self._put(self.source_text, what)
        self._enable(self.takeover_btn, isinstance(src, dv.LiveGame) and app.frame is not None)
        comparing = isinstance(src, dv.CompareSource)
        self._enable(self.compare_btn, comparing or (isinstance(src, dv.LiveGame) and app.frame is not None))
        label = 'Re-sync model (C)' if comparing else 'Compare with model (C)'
        if self.compare_btn.cget('text') != label:
            self.compare_btn.configure(text=label)
        self._enable(self.show_btn, comparing)
        nxt = {'game': 'model', 'model': 'differences', 'differences': 'game'}.get(app.compare_mode(), 'model')
        label = 'Next: %s (G)' % nxt
        if self.show_btn.cget('text') != label:
            self.show_btn.configure(text=label)
        for w in [self.pause_btn, self.step_btn] + self.speed_btns:
            self._enable(w, model)
        if model:
            label = 'Resume (Space)' if src.paused else 'Pause (Space)'
            if self.pause_btn.cget('text') != label:
                self.pause_btn.configure(text=label)
            self._put(self.speed, src.speed)
        self._put(self.view, app.view)
        self._put(self.texture, engine_textures.LABELS.get(app.texture, app.texture))
        self._put(self.tool, app.tool)
        self._put(self.brush_radius, app.brush_radius)
        self._put(self.radius_text, '%d' % app.brush_radius)
        self._put(self.strength_text, '%.1f/s' % app.brush_strength)
        self._put(self.temp_text, '%.0f' % (app.lava_temperature * 8))
        self._put(self.zoom_text, '%gx' % app.zoom)
        self._put(self.smooth, bool(app.smooth))
        for attr, v in self.checks.items():
            self._put(v, bool(getattr(app, attr)))
        self._put(self.flip, bool(app.flip))
        self._put(self.scale, app.scale)
        self._sync_totems(src if model else None)
        a, b = app.slice_a, app.slice_b
        fmt = lambda c: 'col %d, row %d' % (c[1], c[0]) if c else '(not set)'
        text = 'A: %s\nB: %s' % (fmt(a), fmt(b))
        if a and b:
            text += '\nlength %.0f cells' % ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        self._put(self.slice_text, text)
        self._enable(self.slice_open_btn, bool(a and b) and app.slice_win is None)
        self._put(self.slice_whole, bool(app.slice_whole))
        name = getattr(src, 'level_name', None)
        if name is None and not model:              # live / compare: the story level from the totems (once found)
            n = self.dv.level_number(src)
            hit = [x for x in self.levels if n and x.startswith('ALPHA_Map%d_' % n) and 'Demo' not in x]
            name = hit[0] if hit else None
        if (src, name) != self._shown_src:          # a new map or a new live level: point the level list at it
            self._shown_src = (src, name)
            if name in self.levels and self._current_level() != name:
                self._put(self.level, self._level_label(name))
                self._show_saves()

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
