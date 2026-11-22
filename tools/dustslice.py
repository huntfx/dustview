"""DustView Slice: a side-on cross-section of the map between two points (Slice tab).

The grid stores, per cell: ground surface (component 0, includes the sand), sand thickness (1) and
fluid (2: water > 0, lava < 0) with sediment / temperature (3); plants are component 6. A column of the
slice is therefore, bottom to top: rock up to ground - sand, sand up to ground, then water or lava up to
ground + fluid, then air. One column per cell crossed (nearest cell along the line), so the picture is
blocky sideways like the grid itself; heights are drawn exactly.

`sample(fr, a, b)` reads the cells along A -> B; `render(...)` draws them; `SliceWindow` is the resizable
window DustView opens for it (hover shows the cell's values and marks it on the map).
"""
import math
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

N = 512
LEFT, BOTTOM, TOP, RIGHT = 46, 20, 8, 10            # plot margins in pixels (axis labels)

SKY_TOP, SKY_LOW = np.array([120, 165, 215], np.float32), np.array([210, 225, 238], np.float32)
ROCK, ROCK_DEEP = np.array([125, 114, 104], np.float32), np.array([62, 55, 50], np.float32)
SAND, SAND_DEEP = np.array([222, 194, 140], np.float32), np.array([176, 140, 92], np.float32)
WATER_TOP, WATER_DEEP = np.array([80, 160, 200], np.float32), np.array([16, 44, 100], np.float32)
MUD = np.array([150, 118, 80], np.float32)
WATER_SURFACE = np.array([205, 238, 252], np.float32)
ROCK_DEPTH, SAND_DEPTH, WATER_DEPTH = 40.0, 6.0, 12.0   # height over which each layer darkens to its deep colour


def sample(fr, a, b):
    """Cells along the line from a = (row, col) to b: one per step of the longer axis (nearest cell)."""
    (r0, c0), (r1, c1) = a, b
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    t = np.linspace(0.0, 1.0, n)
    rows = np.clip(np.rint(r0 + (r1 - r0) * t).astype(np.intp), 0, N - 1)
    cols = np.clip(np.rint(c0 + (c1 - c0) * t).astype(np.intp), 0, N - 1)
    p = {'rows': rows, 'cols': cols, 'n': n, 'length': math.hypot(r1 - r0, c1 - c0),
         'ground': fr.ground[rows, cols].astype(np.float32), 'soft': fr.soft[rows, cols].astype(np.float32),
         'water': fr.water[rows, cols].astype(np.float32), 'lava': fr.lava[rows, cols].astype(np.float32),
         'sediment': fr.sediment[rows, cols].astype(np.float32),
         'temperature': fr.temperature[rows, cols].astype(np.float32), 'plant_cov': None}
    grid = getattr(fr, 'grid', None)
    if grid is not None and grid.shape[0] > 6 and grid[6].any():
        import ganja
        p['plant_col'], p['plant_cov'] = ganja.colours(grid[6][rows, cols])
    return p


def height_range(p, whole=None):
    """Vertical range to show: the slice's own (rock surface to fluid top, with margins) or `whole`."""
    if whole is not None:
        lo, hi = whole
    else:
        lo = float((p['ground'] - p['soft']).min())
        hi = float((p['ground'] + p['water'] + p['lava']).max())
    span = max(hi - lo, 4.0)
    return max(0.0, lo - 0.25 * span), hi + 0.12 * span


def _noise(rows, cols, q):
    """Fixed per-(cell, height step) noise in -0.5..0.5, so the texture stays put as layers move."""
    x = np.sin((rows * 12.9898 + cols * 78.233).astype(np.float32) + q * np.float32(37.719)) * np.float32(4375.85453)
    return x - np.floor(x) - np.float32(0.5)


def _nice_step(span, pixels, min_px=28):
    raw = span * min_px / max(pixels, 1)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _layers(p, i, ph, lo, hi, lava_xs, lava_cs):
    """Colour of each (pixel row, slice column i): (ph, len(i), 3) uint8."""
    f32 = np.float32
    g, s = p['ground'][i], p['soft'][i]
    hard, w, lv = g - s, p['water'][i], p['lava'][i]
    top = g + w + lv
    hv = (f32(hi) - (np.arange(ph, dtype=f32) + f32(0.5)) * f32((hi - lo) / ph))[:, None]   # height of each row
    upp = f32((hi - lo) / ph)                                  # height units per pixel
    cw = len(i)
    out = np.empty((ph, cw, 3), f32)
    fy = (np.arange(ph, dtype=f32) / f32(ph))[:, None, None]
    out[:] = SKY_TOP + (SKY_LOW - SKY_TOP) * fy                 # air

    rock = hv < hard
    sand = (hv >= hard) & (hv < g)
    fluid = (hv >= g) & (hv < top)
    ground = rock | sand
    yy, xx = np.nonzero(ground)
    h = hv[yy, 0]
    q = np.floor(h / max(upp * 3, f32(0.25)))                   # texture grain: ~3 px of height
    grain = _noise(p['rows'][i][xx], p['cols'][i][xx], q)[:, None]
    is_rock = rock[yy, xx][:, None]
    d_rock = np.clip((hard[xx] - h) / f32(ROCK_DEPTH), 0, 1)[:, None]
    d_sand = np.clip((g[xx] - h) / f32(SAND_DEPTH), 0, 1)[:, None]
    out[yy, xx] = np.where(is_rock, (ROCK + (ROCK_DEEP - ROCK) * d_rock) * (1 + f32(0.16) * grain),
                           (SAND + (SAND_DEEP - SAND) * d_sand) * (1 + f32(0.10) * grain))

    if p['plant_cov'] is not None:                             # plants: a band on the dry surface
        cov = p['plant_cov'][i] * ((w <= 0.01) & (lv <= 0))
        band = ground & (hv >= g - max(f32(1.2), 3 * upp)) & (cov > 0)
        yy, xx = np.nonzero(band)
        pc = p['plant_col'][i][xx]
        out[yy, xx] += (pc - out[yy, xx]) * cov[xx][:, None]

    yy, xx = np.nonzero(fluid)
    if len(yy):
        h = hv[yy, 0]
        below = (top[xx] - h)[:, None]                         # depth under the fluid's surface
        wd = np.clip(below / f32(WATER_DEPTH), 0, 1)
        wc = WATER_TOP + (WATER_DEEP - WATER_TOP) * wd
        mud = (np.clip(p['sediment'][i] / (f32(0.02) * w + f32(1e-6)), 0, 1) * f32(0.6))[xx][:, None]
        wc += (MUD - wc) * mud
        wc = np.where(below < 1.5 * upp, WATER_SURFACE, wc)    # a bright line on the surface
        temp = p['temperature'][i]
        lc = np.stack([np.interp(temp, lava_xs, lava_cs[:, k]) for k in range(3)], -1).astype(f32)[xx]
        lc *= 1 - f32(0.35) * np.clip(below / f32(6.0), 0, 1)  # glows at the top, darker below
        out[yy, xx] = np.where((lv[xx] > 0)[:, None], lc, wc)
    return np.clip(out, 0, 255).astype(np.uint8)


def render(p, width, height, lo, hi, lava_xs, lava_cs, hover=None):
    """The slice as an RGB image of width x height (axes included). hover = column index to mark."""
    pw, ph = max(width - LEFT - RIGHT, 10), max(height - TOP - BOTTOM, 10)
    n = p['n']
    cw = min(n, pw)                                            # computed columns (one per cell when they fit)
    i = np.minimum((np.arange(cw) * n) // cw, n - 1)
    plot = Image.fromarray(_layers(p, i, ph, lo, hi, lava_xs, lava_cs), 'RGB')
    if cw != pw:
        plot = plot.resize((pw, ph), Image.NEAREST)            # blocky columns, like the grid
    img = Image.new('RGB', (width, height), (32, 32, 32))
    img.paste(plot, (LEFT, TOP))
    dr = ImageDraw.Draw(img)

    # axes: heights on the left, distance along the line at the bottom
    step = _nice_step(hi - lo, ph)
    v = math.ceil(lo / step) * step
    while v <= hi:
        y = TOP + (hi - v) / (hi - lo) * ph
        dr.line([LEFT - 4, y, LEFT, y], fill=(200, 200, 200))
        dr.text((2, y - 6), ('%g' % round(v, 6)).rjust(6), fill=(200, 200, 200))
        v += step
    length = max(p['length'], 1e-6)
    step = _nice_step(length, pw, 50)
    v = 0.0
    while v <= length + 1e-6:
        x = LEFT + v / length * pw
        dr.line([x, TOP + ph, x, TOP + ph + 4], fill=(200, 200, 200))
        dr.text((x + 2, TOP + ph + 4), '%g' % round(v, 6), fill=(200, 200, 200))
        v += step
    dr.text((LEFT + 3, TOP + 2), 'A', fill=(255, 255, 255))
    dr.text((LEFT + pw - 10, TOP + 2), 'B', fill=(255, 255, 255))
    if hover is not None and 0 <= hover < n:
        x0, x1 = LEFT + hover * pw / n, LEFT + (hover + 1) * pw / n
        dr.rectangle([x0, TOP, max(x0, x1 - 1), TOP + ph - 1], outline=(255, 255, 0))
    return img


def describe(p, k):
    """Text readout of column k."""
    r, c = int(p['rows'][k]), int(p['cols'][k])
    g, s, w, lv = (float(p[x][k]) for x in ('ground', 'soft', 'water', 'lava'))
    t = 'col %d row %d | %.1f cells from A | rock to %.2f, sand %.2f thick (to %.2f)' % (
        c, r, k / max(p['n'] - 1, 1) * p['length'], g - s, s, g)
    if lv > 0:
        t += ' | lava %.2f deep (to %.2f), %.0f deg' % (lv, g + lv, p['temperature'][k])
    elif w > 0:
        t += ' | water %.2f deep (to %.2f), sediment %.3f' % (w, g + w, p['sediment'][k])
    else:
        t += ' | dry'
    return t


class SliceWindow:
    """Resizable window showing the slice; DustView calls update(p, lo, hi) when the frame or points change."""
    def __init__(self, root, on_hover, on_close, lava_xs, lava_cs):
        self.on_hover, self.on_close = on_hover, on_close
        self.lava_xs, self.lava_cs = lava_xs, lava_cs
        self.win = tk.Toplevel(root)
        self.win.title('DustView - Slice A to B')
        self.win.configure(bg='#202020')
        self.win.geometry('1000x380')
        self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.label = tk.Label(self.win, bg='#202020', borderwidth=0)
        self.label.pack(fill='both', expand=True)
        self.info = tk.Label(self.win, anchor='w', font=('Consolas', 9), bg='#202020', fg='#e0e0e0')
        self.info.pack(fill='x')
        self.label.bind('<Motion>', self._motion)
        self.label.bind('<Leave>', lambda e: self._set_hover(None))
        self.label.bind('<Configure>', lambda e: self._redraw())
        self.photo, self.p, self.hover, self.range = None, None, None, (0.0, 1.0)
        self.open = True

    def close(self):
        self.open = False
        self.win.destroy()
        self.on_close()

    def _motion(self, e):
        if self.p is None:
            return
        pw = max(self.label.winfo_width() - LEFT - RIGHT, 10)
        k = int((e.x - LEFT) * self.p['n'] // pw)
        self._set_hover(k if 0 <= k < self.p['n'] and e.x >= LEFT else None)

    def _set_hover(self, k):
        if k != self.hover:
            self.hover = k
            self.on_hover(None if k is None else (int(self.p['rows'][k]), int(self.p['cols'][k])))
            self._redraw()

    def update(self, p, lo, hi):
        self.p, self.range = p, (lo, hi)
        if self.hover is not None and self.hover >= p['n']:
            self.hover = None
        self._redraw()

    def _redraw(self):
        if not self.open or self.p is None:
            return
        w, h = self.label.winfo_width(), self.label.winfo_height()
        if w < 60 or h < 60:
            return
        img = render(self.p, w, h, *self.range, self.lava_xs, self.lava_cs, self.hover)
        if self.photo is not None and (self.photo.width(), self.photo.height()) == img.size:
            self.photo.paste(img)
        else:
            self.photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self.photo)
        text = describe(self.p, self.hover) if self.hover is not None else \
            'Hover for values. Rock (grey), sand, water (browner = more sediment), lava, plants (green band on dry ground).'
        if text != self.info.cget('text'):
            self.info.configure(text=text)
