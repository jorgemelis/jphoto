"""
jphoto/window.py — the jphoto photo widget, rebuilt from scratch.

Self-contained: it does NOT inherit from diapos. The layout is grown by hand so
nothing is inherited that we don't want — a left column with the Original|Result
panes + a zoom bar, a right column of control tabs, and a draggable splitter
between them. v1 ships a solid viewer (zoom that goes BELOW fit, so the photo
shrinks inside its pane without fighting the window) plus the two recovery tools
that matter first: eyedropper white balance and per-channel curves.

All image maths lives in jphoto.engine (pure numpy); this file is only Qt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QAction, QShortcut, QKeySequence,
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPlainTextEdit,
    QTabWidget, QScrollArea, QSplitter, QMenu, QMessageBox, QSizePolicy,
    QDialog, QFrame,
)

# Shared RAW decode (sips) + EXIF/IPTC metadata (mdls); repo root is on the
# path via the launcher, but add it defensively in case window.py is imported
# on its own.
import os as _os
sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import rawmeta

from . import engine as E
from . import state

PROXY_MAX = 1100        # longest edge of the live-preview proxy


def np_to_qpixmap(arr01: np.ndarray) -> QPixmap:
    u8 = np.ascontiguousarray(E.to_uint8(arr01))
    h, w, _ = u8.shape
    img = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())     # copy() detaches from the numpy buffer


# --- preview pane -----------------------------------------------------------

class PreviewPane(QLabel):
    """An image shown scaled-to-fit, with synced zoom/pan and an optional
    eyedropper. zoom == 1 fits the pane; zoom < 1 shrinks the image inside it;
    zoom > 1 enlarges for pixel-peeping. Click (no drag) emits a normalized
    image coord; drag pans; wheel zooms."""

    picked = Signal(float, float)        # normalized (fx, fy) of an eyedropper click
    panned = Signal(float, float)        # normalized pan delta, synced to both panes
    zoomed = Signal(float)               # wheel zoom multiplier
    paint_begin = Signal(float, float, bool)  # (fx, fy, alt): stroke start / set source
    paint_to = Signal(float, float)      # (fx, fy): continue the stroke
    paint_finish = Signal()              # stroke released → commit one undo step

    def __init__(self, title: str, clickable: bool = False):
        super().__init__()
        self.setMinimumSize(120, 120)    # small floor so the window can shrink freely
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; color:#777; border:1px solid #333;")
        self.setText(title)
        self._clickable = clickable
        self._full = None                # QPixmap of the current image
        self._zoom = 1.0
        self._cx, self._cy = 0.5, 0.5    # view centre, normalized image coords
        self._disp = None                # (x0,y0,dw,dh) drawn rect, widget px
        self._src = None                 # (sx,sy,sw,sh) source rect, image px
        self._press = None
        self._moved = False
        self._marker = None              # (fx,fy) of the last eyedropper pick
        self._marker_visible = True      # toggled from the GUI
        self._paint_mode = False         # retouch active: drag paints, no pan
        self._painting = False
        self._brush_r_img = None         # brush radius in IMAGE px (None = hide cursor)
        self._cursor = None              # (px,py) last mouse pos, for the brush ring
        self._src_marker = None          # (fx,fy) clone-source anchor

    def set_image(self, arr01):
        if arr01 is None:
            self._full = None
            self._disp = self._src = None
            self.clear()
            return
        self._full = np_to_qpixmap(arr01)
        self._render()

    def set_view(self, zoom, cx, cy):
        self._zoom, self._cx, self._cy = zoom, cx, cy
        self._render()

    def set_marker(self, fx, fy):
        self._marker = None if fx is None else (fx, fy)
        self.update()

    def set_marker_visible(self, on):
        self._marker_visible = on
        self.update()

    def set_paint_mode(self, on):
        self._paint_mode = on
        self._painting = False
        self.setMouseTracking(on)        # get the brush ring under the cursor
        if not on:
            self._cursor = None
        self.update()

    def set_brush_radius_img(self, r):
        self._brush_r_img = r
        self.update()

    def set_src_marker(self, fx, fy):
        self._src_marker = None if fx is None else (fx, fy)
        self.update()

    def _render(self):
        if self._full is None:
            return
        W, H = self._full.width(), self._full.height()
        pw, ph = max(1, self.width()), max(1, self.height())
        scale = min(pw / W, ph / H) * self._zoom          # fit * zoom (zoom may be <1)
        sw = min(W, pw / scale)                            # visible source size (image px)
        sh = min(H, ph / scale)
        sx = min(max(self._cx * W - sw / 2, 0.0), W - sw)
        sy = min(max(self._cy * H - sh / 2, 0.0), H - sh)
        crop = self._full.copy(int(sx), int(sy), max(1, int(round(sw))), max(1, int(round(sh))))
        scaled = crop.scaled(max(1, int(round(sw * scale))), max(1, int(round(sh * scale))),
                             Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        x0 = (pw - scaled.width()) // 2
        y0 = (ph - scaled.height()) // 2
        self._src = (sx, sy, sw, sh)
        self._disp = (x0, y0, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def _to_img_norm(self, px, py):
        if self._disp is None or self._src is None:
            return None
        x0, y0, dw, dh = self._disp
        sx, sy, sw, sh = self._src
        if not (x0 <= px < x0 + dw and y0 <= py < y0 + dh):
            return None
        ix = sx + (px - x0) / dw * sw
        iy = sy + (py - y0) / dh * sh
        return ix / self._full.width(), iy / self._full.height()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()

    def mousePressEvent(self, ev):
        if self._paint_mode and ev.button() == Qt.MouseButton.LeftButton:
            n = self._to_img_norm(ev.position().x(), ev.position().y())
            if n is not None:
                alt = bool(ev.modifiers() & Qt.KeyboardModifier.AltModifier)
                self._painting = not alt                   # alt-click only sets source
                self.paint_begin.emit(n[0], n[1], alt)
            return
        self._press = (ev.position().x(), ev.position().y())
        self._moved = False

    def mouseMoveEvent(self, ev):
        px, py = ev.position().x(), ev.position().y()
        if self._paint_mode:
            self._cursor = (px, py)                        # keep the brush ring following
            if self._painting:
                n = self._to_img_norm(px, py)
                if n is not None:
                    self.paint_to.emit(n[0], n[1])
            self.update()
            return
        if self._press is None or self._disp is None:
            return
        lx, ly = self._press
        if abs(px - lx) + abs(py - ly) > 2:
            self._moved = True
        self._press = (px, py)
        x0, y0, dw, dh = self._disp
        sx, sy, sw, sh = self._src
        dcx = -(px - lx) / dw * sw / self._full.width()    # pan opposite to drag
        dcy = -(py - ly) / dh * sh / self._full.height()
        self.panned.emit(dcx, dcy)

    def mouseReleaseEvent(self, ev):
        if self._paint_mode:
            if self._painting:
                self._painting = False
                self.paint_finish.emit()
            return
        if self._press is None:
            return
        px, py = ev.position().x(), ev.position().y()
        if self._clickable and not self._moved:
            n = self._to_img_norm(px, py)                  # click (no drag) → pick
            if n is not None:
                self.picked.emit(n[0], n[1])
        self._press = None

    def leaveEvent(self, ev):
        if self._paint_mode:
            self._cursor = None
            self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        self.zoomed.emit(1.2 if ev.angleDelta().y() > 0 else 1 / 1.2)

    def _img_to_widget(self, fx, fy):
        """Map a normalized image coord to widget px, or None if off-view."""
        if self._disp is None or self._src is None or self._full is None:
            return None
        x0, y0, dw, dh = self._disp
        sx, sy, sw, sh = self._src
        ix, iy = fx * self._full.width(), fy * self._full.height()
        if not (sx <= ix < sx + sw and sy <= iy < sy + sh):
            return None
        return int(x0 + (ix - sx) / sw * dw), int(y0 + (iy - sy) / sh * dh)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._disp is None or self._src is None or self._full is None:
            return
        p = QPainter(self)
        # eyedropper / probe pick marker (red cross)
        if self._marker_visible and self._marker is not None:
            w = self._img_to_widget(*self._marker)
            if w is not None:
                px, py = w
                p.setPen(QPen(QColor(255, 60, 60), 2))
                p.drawEllipse(QPoint(px, py), 8, 8)
                p.drawLine(px - 13, py, px + 13, py)
                p.drawLine(px, py - 13, px, py + 13)
        # clone-source anchor (green cross)
        if self._src_marker is not None:
            w = self._img_to_widget(*self._src_marker)
            if w is not None:
                px, py = w
                p.setPen(QPen(QColor(60, 220, 90), 2))
                p.drawEllipse(QPoint(px, py), 7, 7)
                p.drawLine(px - 11, py, px + 11, py)
                p.drawLine(px, py - 11, px, py + 11)
        # brush ring under the cursor, sized to the actual brush footprint
        if self._paint_mode and self._cursor is not None and self._brush_r_img:
            x0, y0, dw, dh = self._disp
            sx, sy, sw, sh = self._src
            r_screen = max(2, int(self._brush_r_img * dw / sw))
            p.setPen(QPen(QColor(255, 255, 255, 180), 1))
            p.drawEllipse(QPoint(int(self._cursor[0]), int(self._cursor[1])),
                          r_screen, r_screen)


# --- per-channel curve editor ----------------------------------------------

class CurveEditor(QWidget):
    """Compact draggable curve editor. Click empty space to add a point, drag to
    move, double-click a point to remove it. Endpoints (x=0, x=1) move only
    vertically. `changed` fires live as points move."""

    changed = Signal()
    _COLORS = {"L": "#dddddd", "R": "#ff5555", "G": "#55ff55", "B": "#5599ff"}
    _M = 8   # margin in px

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.curves = {k: list(E.IDENTITY) for k in "LRGB"}
        self.channel = "L"
        self._drag = None        # index of the point being dragged

    def set_channel(self, ch):
        self.channel = ch
        self.update()

    def reset_current(self):
        self.curves[self.channel] = list(E.IDENTITY)
        self.update()
        self.changed.emit()

    # widget px <-> data [0,1]
    def _to_px(self, x, y):
        w, h, m = self.width(), self.height(), self._M
        return m + x * (w - 2 * m), (h - m) - y * (h - 2 * m)

    def _to_data(self, px, py):
        w, h, m = self.width(), self.height(), self._M
        x = (px - m) / max(1, w - 2 * m)
        y = ((h - m) - py) / max(1, h - 2 * m)
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _hit(self, px, py):
        pts = self.curves[self.channel]
        for i, (x, y) in enumerate(pts):
            gx, gy = self._to_px(x, y)
            if (gx - px) ** 2 + (gy - py) ** 2 <= 100:     # 10 px radius
                return i
        return None

    def mousePressEvent(self, ev):
        px, py = ev.position().x(), ev.position().y()
        i = self._hit(px, py)
        pts = self.curves[self.channel]
        if i is None:
            x, y = self._to_data(px, py)
            pts.append((x, y))
            pts.sort()
            i = pts.index((x, y))
        self._drag = i
        self.update()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        pts = self.curves[self.channel]
        x, y = self._to_data(ev.position().x(), ev.position().y())
        if self._drag == 0:
            pts[0] = (0.0, y)                              # endpoints: vertical only
        elif self._drag == len(pts) - 1:
            pts[-1] = (1.0, y)
        else:
            lo = pts[self._drag - 1][0] + 1e-3
            hi = pts[self._drag + 1][0] - 1e-3
            pts[self._drag] = (min(max(x, lo), hi), y)
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        i = self._hit(ev.position().x(), ev.position().y())
        pts = self.curves[self.channel]
        if i is not None and 0 < i < len(pts) - 1:         # never the endpoints
            pts.pop(i)
            self.update()
            self.changed.emit()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#181818"))
        w, h, m = self.width(), self.height(), self._M
        p.setPen(QPen(QColor("#333"), 1))
        for f in (0.25, 0.5, 0.75):
            x = m + f * (w - 2 * m)
            y = (h - m) - f * (h - 2 * m)
            p.drawLine(int(x), m, int(x), h - m)
            p.drawLine(m, int(y), w - m, int(y))
        p.setPen(QPen(QColor("#444"), 1, Qt.PenStyle.DashLine))
        p.drawLine(*[int(v) for v in (*self._to_px(0, 0), *self._to_px(1, 1))])

        pts = self.curves[self.channel]
        col = QColor(self._COLORS[self.channel])
        xs = np.linspace(0, 1, 128)
        ys = np.interp(xs, [pp[0] for pp in pts], [pp[1] for pp in pts])
        p.setPen(QPen(col, 2))
        prev = None
        for x, y in zip(xs, ys):
            gx, gy = self._to_px(x, y)
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(gx), int(gy))
            prev = (gx, gy)
        p.setBrush(col)
        for x, y in pts:
            gx, gy = self._to_px(x, y)
            p.drawEllipse(QPoint(int(gx), int(gy)), 4, 4)


# --- the jphoto widget -----------------------------------------------------

class JPhotoWidget(QWidget):
    """Standalone photo-restoration widget: viewer + white balance + curves."""

    ZOOM_MIN, ZOOM_MAX = 0.2, 8.0        # multipliers of "fit"; <1 shrinks the image

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base = None                 # full-res working image, sRGB float [0,1]
        self.proxy = None                # downscaled copy for live preview
        self.orig_base = None            # pristine full-res, to revert retouch on Reset
        self.path = None
        self.neutral = None              # sampled neutral RGB (white balance source)
        self.zoom, self.pan_cx, self.pan_cy = 1.0, 0.5, 0.5
        self._last_dir = state.load().get("last_dir", str(Path.home()))

        # retouch (local clone / highlight / recolour) state
        self._retouch = "off"            # "off" | "clone" | "highlight" | "recolor"
        self._brush_r = 20               # brush radius, IMAGE px
        self._clone_src = None           # (y,x) source anchor, image px
        self._clone_off = None           # (dy,dx) source-minus-dab offset, image px
        self._recolor_target = None      # chosen destination colour, RGB [0,1]
        self._stroke_base = None         # full-res snapshot at stroke start (undo src)
        self._stroke_bbox = None         # union bbox touched this stroke
        self._undo = []                  # [(y0,y1,x0,x1, before_patch)] per stroke

        self._build_ui()

        # coalesce rapid curve/slider edits into one re-render
        self._kick_timer = QTimer(self)
        self._kick_timer.setSingleShot(True)
        self._kick_timer.setInterval(30)
        self._kick_timer.timeout.connect(self._render_result)

        sizes = state.load().get("splitter")
        if sizes and len(sizes) == 2:
            self.splitter.setSizes([int(s) for s in sizes])
        self.splitter.splitterMoved.connect(
            lambda *_: state.update(splitter=list(self.splitter.sizes())))

    # -- public API (used by the launcher) --------------------------------
    def open_path(self, p):
        p = Path(p)
        if p.is_file():
            self._load(p)
        elif p.is_dir():
            self._last_dir = str(p)

    def close(self):
        try:
            state.update(splitter=list(self.splitter.sizes()))
        except Exception:
            pass

    # -- UI ----------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # left: open row, images, zoom bar, log
        left = QVBoxLayout()
        top = QHBoxLayout()
        self.open_btn = QPushButton("Open image…")
        self.open_btn.clicked.connect(self._open)
        self.recent_btn = QPushButton("Recent ▾")
        self._recent_menu = QMenu(self.recent_btn)
        self._recent_menu.aboutToShow.connect(self._build_recent_menu)
        self.recent_btn.setMenu(self._recent_menu)
        self.info_btn = QPushButton("Info ⓘ")
        self.info_btn.setToolTip("Properties — EXIF / IPTC / file info (I)")
        self.info_btn.clicked.connect(self._show_properties)
        self.eyedrop_cb = QCheckBox("Eyedropper")
        self.eyedrop_cb.setToolTip("Click a point that should be neutral grey/white "
                                   "on the ORIGINAL to set the white balance.")
        self.probe_cb = QCheckBox("Probe cast")
        self.probe_cb.setToolTip(
            "Click points on the Original to READ the colour tint there (changes "
            "nothing). On a surface you KNOW is neutral that tint IS the cast — "
            "use it for WB. Compare points: a shared tint is the cast, an outlier "
            "is a real colour.")
        # eyedropper and probe are mutually exclusive (both act on Original clicks)
        self.eyedrop_cb.toggled.connect(lambda on: on and self.probe_cb.setChecked(False))
        self.probe_cb.toggled.connect(lambda on: on and self.eyedrop_cb.setChecked(False))
        self.mark_cb = QCheckBox("Mark")
        self.mark_cb.setChecked(True)
        self.mark_cb.setToolTip("Show the red cross where you last picked / probed.")
        self.mark_cb.toggled.connect(lambda on: self.orig_pane.set_marker_visible(on))
        top.addWidget(self.open_btn)
        top.addWidget(self.recent_btn)
        top.addWidget(self.info_btn)
        top.addWidget(self.eyedrop_cb)
        top.addWidget(self.probe_cb)
        top.addWidget(self.mark_cb)
        top.addStretch()
        self.path_lbl = QLabel("—")
        self.path_lbl.setStyleSheet("color:#888;")
        top.addWidget(self.path_lbl)
        left.addLayout(top)

        self.orig_pane = PreviewPane("Original", clickable=True)
        self.orig_pane.picked.connect(self._on_pick)
        self.res_pane = PreviewPane("Result")
        self.res_pane.paint_begin.connect(self._on_paint_begin)
        self.res_pane.paint_to.connect(self._on_paint_to)
        self.res_pane.paint_finish.connect(self._on_paint_finish)
        for p in (self.orig_pane, self.res_pane):
            p.panned.connect(self._on_pan)
            p.zoomed.connect(self._on_wheel_zoom)
        img_row = QHBoxLayout()
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.addWidget(self.orig_pane)
        img_row.addWidget(self.res_pane)
        left.addLayout(img_row, stretch=1)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(int(self.ZOOM_MIN * 100), int(self.ZOOM_MAX * 100))
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(lambda v: self._set_zoom(v / 100.0))
        zoom_row.addWidget(self.zoom_slider, stretch=1)
        self.zoom_lbl = QLabel("1.0×")
        self.zoom_lbl.setStyleSheet("color:#888;")
        zoom_row.addWidget(self.zoom_lbl)
        fit = QPushButton("Fit")
        fit.setMaximumWidth(56)
        fit.clicked.connect(self._zoom_fit)
        zoom_row.addWidget(fit)
        left.addLayout(zoom_row)

        self.probe_lbl = QLabel("")
        self.probe_lbl.setVisible(False)
        self.probe_lbl.setStyleSheet(
            "font-family:Menlo,monospace; font-size:12px; font-weight:bold; "
            "color:#eaffff; background:#243042; padding:4px; border-radius:4px;")
        left.addWidget(self.probe_lbl)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(96)
        self.log.setStyleSheet("font-family:Menlo,monospace; font-size:10px;")
        left.addWidget(self.log)

        left_box = QWidget()
        left_box.setLayout(left)

        # right: control tabs + actions
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.addTab(self._page([self._wb_group()]), "White balance")
        self.ctrl_tabs.addTab(self._page([self._curves_group()]), "Curves")
        self.ctrl_tabs.addTab(self._page([self._retouch_group()]), "Retouch")

        right = QVBoxLayout()
        right.addWidget(self.ctrl_tabs, stretch=1)
        actions = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn = QPushButton("Save result…")
        self.save_btn.clicked.connect(self._save)
        actions.addWidget(self.reset_btn)
        actions.addStretch()
        actions.addWidget(self.save_btn)
        right.addLayout(actions)
        right_box = QWidget()
        right_box.setLayout(right)
        right_box.setMinimumWidth(260)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_box)
        self.splitter.addWidget(right_box)
        self.splitter.setStretchFactor(0, 1)     # images take the slack on resize
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1000, 300])
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter)

        undo_sc = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_sc.activated.connect(self._undo_retouch)

        info_sc = QShortcut(QKeySequence("I"), self)
        info_sc.activated.connect(self._show_properties)

    def _page(self, groups) -> QScrollArea:
        page = QWidget()
        lay = QVBoxLayout(page)
        for g in groups:
            lay.addWidget(g)
        lay.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(page)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return sc

    def _wb_group(self) -> QGroupBox:
        g = QGroupBox("White balance (eyedropper)")
        lay = QGridLayout(g)
        hint = QLabel("Turn on Eyedropper, click a neutral point on the Original, "
                      "then dial how strongly to apply it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        lay.addWidget(hint, 0, 0, 1, 3)
        lay.addWidget(QLabel("Strength"), 1, 0)
        self.wb_slider = QSlider(Qt.Orientation.Horizontal)
        self.wb_slider.setRange(0, 100)
        self.wb_slider.setValue(100)
        self.wb_slider.valueChanged.connect(self._on_wb_strength)
        lay.addWidget(self.wb_slider, 1, 1)
        self.wb_lbl = QLabel("100%")
        self.wb_lbl.setStyleSheet("color:#9bd; font-family:Menlo,monospace;")
        lay.addWidget(self.wb_lbl, 1, 2)
        clr = QPushButton("Clear pick")
        clr.clicked.connect(self._clear_neutral)
        lay.addWidget(clr, 2, 0, 1, 3)
        return g

    def _curves_group(self) -> QGroupBox:
        g = QGroupBox("Per-channel curves")
        lay = QVBoxLayout(g)
        row = QHBoxLayout()
        self.curve_ch = QComboBox()
        self.curve_ch.addItems(["L (master)", "R", "G", "B"])
        self.curve_ch.currentIndexChanged.connect(
            lambda i: self.curve.set_channel(["L", "R", "G", "B"][i]))
        row.addWidget(QLabel("Channel"))
        row.addWidget(self.curve_ch)
        reset = QPushButton("Reset curve")
        reset.clicked.connect(lambda: self.curve.reset_current())
        row.addWidget(reset)
        row.addStretch()
        lay.addLayout(row)
        self.curve = CurveEditor()
        self.curve.changed.connect(self._kick)
        lay.addWidget(self.curve)
        return g

    def _retouch_group(self) -> QGroupBox:
        g = QGroupBox("Retouch spots (moles)")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("Tool"), 0, 0)
        self.retouch_tool = QComboBox()
        self.retouch_tool.addItems(["Off", "Hide (clone)", "Emphasise", "Recolour"])
        self.retouch_tool.currentIndexChanged.connect(self._on_retouch_tool)
        lay.addWidget(self.retouch_tool, 0, 1, 1, 2)

        self.retouch_hint = QLabel("")
        self.retouch_hint.setWordWrap(True)
        self.retouch_hint.setStyleSheet("color:#888;")
        lay.addWidget(self.retouch_hint, 1, 0, 1, 3)

        self.swatch_row = QWidget()
        srow = QHBoxLayout(self.swatch_row)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(QLabel("Target colour"))
        self.recolor_swatch = QLabel()
        self.recolor_swatch.setFixedSize(46, 18)
        self.recolor_swatch.setStyleSheet(
            "background:#333; border:1px solid #666; border-radius:3px;")
        srow.addWidget(self.recolor_swatch)
        srow.addStretch()
        self.swatch_row.setVisible(False)
        lay.addWidget(self.swatch_row, 2, 0, 1, 3)

        lay.addWidget(QLabel("Tolerance"), 3, 0)
        self.recolor_tol = QSlider(Qt.Orientation.Horizontal)
        self.recolor_tol.setRange(2, 100)
        self.recolor_tol.setValue(30)
        self.recolor_tol.valueChanged.connect(
            lambda v: self.recolor_tol_lbl.setText(f"{v}"))
        lay.addWidget(self.recolor_tol, 3, 1)
        self.recolor_tol_lbl = QLabel("30")
        self.recolor_tol_lbl.setStyleSheet("color:#9bd; font-family:Menlo,monospace;")
        lay.addWidget(self.recolor_tol_lbl, 3, 2)
        self.tol_row_widgets = (self.recolor_tol, self.recolor_tol_lbl)

        self.brush_row_lbl = QLabel("Brush")
        lay.addWidget(self.brush_row_lbl, 4, 0)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 200)
        self.brush_slider.setValue(self._brush_r)
        self.brush_slider.valueChanged.connect(self._on_brush_size)
        lay.addWidget(self.brush_slider, 4, 1)
        self.brush_lbl = QLabel(f"{self._brush_r}px")
        self.brush_lbl.setStyleSheet("color:#9bd; font-family:Menlo,monospace;")
        lay.addWidget(self.brush_lbl, 4, 2)

        lay.addWidget(QLabel("Strength"), 5, 0)
        self.retouch_strength = QSlider(Qt.Orientation.Horizontal)
        self.retouch_strength.setRange(0, 100)
        self.retouch_strength.setValue(85)
        self.retouch_strength.valueChanged.connect(
            lambda v: self.retouch_str_lbl.setText(f"{v}%"))
        lay.addWidget(self.retouch_strength, 5, 1)
        self.retouch_str_lbl = QLabel("85%")
        self.retouch_str_lbl.setStyleSheet("color:#9bd; font-family:Menlo,monospace;")
        lay.addWidget(self.retouch_str_lbl, 5, 2)

        undo = QPushButton("Undo stroke (⌘Z)")
        undo.clicked.connect(self._undo_retouch)
        lay.addWidget(undo, 6, 0, 1, 3)
        self._on_retouch_tool(0)                 # set the initial hint
        return g

    # -- load --------------------------------------------------------------
    def _open(self):
        start = self._last_dir if Path(self._last_dir).is_dir() else str(Path.home())
        raw_glob = " ".join(f"*{e}" for e in sorted(rawmeta.RAW_EXTENSIONS))
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open image", start,
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp "
            f"{raw_glob});;RAW ({raw_glob});;All files (*)")
        if fn:
            self._load(Path(fn))

    def _show_properties(self):
        if not self.path:
            self._append("No image open — nothing to inspect.")
            return
        try:
            sections = rawmeta.read_metadata(str(self.path))
        except Exception as exc:
            sections = [("Error", [("read_metadata", str(exc))])]
        PropertiesDialog(str(self.path), sections, self).exec()

    def _load(self, path: Path):
        try:
            self.base = E.load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.path = path
        self.orig_base = self.base.copy()        # pristine, for reverting retouch
        self.proxy = E.make_proxy(self.base, PROXY_MAX)
        self.neutral = None
        self._undo = []
        self._clone_src = self._clone_off = None
        self._recolor_target = None
        if hasattr(self, "recolor_swatch"):
            self._update_swatch()
        self.res_pane.set_src_marker(None, None)
        self.zoom, self.pan_cx, self.pan_cy = 1.0, 0.5, 0.5
        self._sync_zoom_ui()
        self.curve.curves = {k: list(E.IDENTITY) for k in "LRGB"}
        self.curve.update()
        self.orig_pane.set_marker(None, None)
        self.orig_pane.set_image(self.proxy)
        self._render_result()
        self._last_dir = str(path.parent)
        h, w = self.base.shape[:2]
        self.path_lbl.setText(f"{path.name}   {w}×{h}")
        self._append(f"Opened {path.name}  ({w}×{h})")
        state.record_open(path)
        # fit the window's height to the photo so the panes don't letterbox.
        # deferred so the layout has settled and the panes report real widths.
        QTimer.singleShot(0, self._fit_window_to_image)

    def _fit_window_to_image(self):
        """Resize the top-level window's HEIGHT so the Original pane matches the
        photo's aspect — kills the black bars without touching the width or the
        user's left/right split. Width is left alone, and the result is clamped
        to the screen so a portrait photo can't push it off-screen."""
        if self.base is None:
            return
        win = self.window()
        pw = self.orig_pane.width()
        if pw < 50:                              # not laid out yet — skip quietly
            return
        H, W = self.base.shape[:2]
        target = pw * H / W                      # the pane height that exactly fits
        delta = int(round(target - self.orig_pane.height()))
        if abs(delta) < 8:
            return
        scr = win.screen().availableGeometry()
        new_h = min(max(win.height() + delta, 360), scr.height() - 40)
        win.resize(win.width(), new_h)

    def _build_recent_menu(self):
        self._recent_menu.clear()
        imgs = state.load().get("recent_images", [])
        if not imgs:
            act = self._recent_menu.addAction("(no recent images)")
            act.setEnabled(False)
            return
        for p in imgs:
            act = QAction(f"{Path(p).name}     {Path(p).parent.name}/", self._recent_menu)
            act.triggered.connect(lambda _=False, pp=p: self._load(Path(pp)))
            act.setEnabled(Path(p).exists())
            self._recent_menu.addAction(act)

    # -- probe / white balance --------------------------------------------
    def _on_pick(self, fx, fy):
        if self.base is None:
            return
        if self.probe_cb.isChecked():
            self._probe(fx, fy)
            return
        if not self.eyedrop_cb.isChecked():
            return
        H, W = self.base.shape[:2]
        y, x = int(fy * H), int(fx * W)
        r = 6
        patch = self.base[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
        self.neutral = patch.mean(0)
        self.orig_pane.set_marker(fx, fy)
        rgb = self.neutral * 255
        self._append(f"Neutral pick → R{rgb[0]:.0f} G{rgb[1]:.0f} B{rgb[2]:.0f}")
        self._render_result()

    def _probe(self, fx, fy):
        """Read the colour tint of a patch and say how far it is from neutral —
        so a real grey can be told from one the light has tinted. Reads only."""
        H, W = self.base.shape[:2]
        y, x = int(fy * H), int(fx * W)
        r = 10
        rgb = self.base[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3).mean(0) * 255.0
        spread = float(rgb.max() - rgb.min())
        g = float(rgb.mean())
        gain = [g / c if c > 1e-6 else 1.0 for c in rgb]
        if spread < 6:
            cast = "neutral here"
        else:
            dev = rgb - g                       # which channel is the odd one out
            i = int(np.argmax(np.abs(dev)))
            if dev[i] >= 0:
                hue = ("warm/red", "green", "cool/blue")[i]
            else:                               # a channel sunk → its complement
                hue = ("cyan", "magenta", "yellow")[i]
            cast = f"{hue} Δ{spread:.0f}"
        line = (f"R{rgb[0]:3.0f} G{rgb[1]:3.0f} B{rgb[2]:3.0f}  {cast:15s} "
                f"→WB {gain[0]:.2f}/{gain[1]:.2f}/{gain[2]:.2f}")
        self._probes = ([line] + getattr(self, "_probes", []))[:5]
        self.orig_pane.set_marker(fx, fy)
        self.probe_lbl.setVisible(True)
        self.probe_lbl.setText(
            "PROBE — tint here. On a KNOWN neutral the tint IS the cast → use for WB.\n"
            "        Compare points: shared tint = cast, an outlier = real colour.\n"
            + "\n".join(self._probes))

    def _on_wb_strength(self, v):
        self.wb_lbl.setText(f"{v}%")
        self._kick()

    def _clear_neutral(self):
        self.neutral = None
        self.orig_pane.set_marker(None, None)
        self._append("White balance cleared.")
        self._render_result()

    # -- retouch: local clone (hide) / highlight (emphasise) ---------------
    _HINTS = {
        "off": "Pick a tool. Retouch is painted on the RESULT pane; pan with the "
               "Original pane.",
        "clone": "Alt/Option-click clean skin on the Result to set the source "
                 "(green cross), then drag over the mole to cover it with that "
                 "skin.",
        "highlight": "Drag over the mole on the Result to deepen it and make it "
                     "stand out.",
        "recolor": "Alt/Option-click the Result to pick the target colour, then "
                   "click any colour: everything like it turns into the target. "
                   "Tolerance = how wide a colour range counts.",
    }

    def _on_retouch_tool(self, idx):
        self._retouch = ("off", "clone", "highlight", "recolor")[idx]
        on = self._retouch != "off"
        uses_brush = self._retouch in ("clone", "highlight")
        self.res_pane.set_paint_mode(on)
        self.res_pane.set_brush_radius_img(self._brush_r if uses_brush else None)
        if self._retouch != "clone":
            self._clone_src = self._clone_off = None
            self.res_pane.set_src_marker(None, None)
        if hasattr(self, "retouch_hint"):
            self.retouch_hint.setText(self._HINTS[self._retouch])
            recolor = self._retouch == "recolor"
            self.swatch_row.setVisible(recolor)
            for w in self.tol_row_widgets:
                w.setVisible(recolor)
            self.brush_row_lbl.setVisible(uses_brush)
            self.brush_slider.setVisible(uses_brush)
            self.brush_lbl.setVisible(uses_brush)

    def _update_swatch(self):
        if self._recolor_target is None:
            self.recolor_swatch.setStyleSheet(
                "background:#333; border:1px solid #666; border-radius:3px;")
            return
        r, g, b = (int(c * 255) for c in self._recolor_target)
        self.recolor_swatch.setStyleSheet(
            f"background:rgb({r},{g},{b}); border:1px solid #666; border-radius:3px;")

    def _on_brush_size(self, v):
        self._brush_r = v
        self.brush_lbl.setText(f"{v}px")
        if self._retouch != "off":
            self.res_pane.set_brush_radius_img(v)

    def _strength(self):
        return self.retouch_strength.value() / 100.0

    def _sample_patch(self, fx, fy, r=4):
        H, W = self.base.shape[:2]
        y, x = int(fy * H), int(fx * W)
        return self.base[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3).mean(0)

    def _on_paint_begin(self, fx, fy, alt):
        if self.base is None or self._retouch == "off":
            return
        if self._retouch == "recolor":
            self._recolor(fx, fy, alt)
            return
        H, W = self.base.shape[:2]
        y, x = fy * H, fx * W
        if self._retouch == "clone" and alt:
            self._clone_src = (y, x)             # anchor clean skin as the source
            self._clone_off = None
            self.res_pane.set_src_marker(fx, fy)
            self._append("Clone source set.")
            return
        if self._retouch == "clone" and self._clone_src is None:
            self._append("Set a clone source first: Alt/Option-click clean skin.")
            return
        self._stroke_base = self.base.copy()     # snapshot for a single-stroke undo
        self._stroke_bbox = None
        if self._retouch == "clone":
            sy, sx = self._clone_src
            self._clone_off = (sy - y, sx - x)   # fix the aligned-clone offset
        self._dab(y, x)

    def _on_paint_to(self, fx, fy):
        if self._stroke_base is None:
            return
        H, W = self.base.shape[:2]
        self._dab(fy * H, fx * W)

    def _on_paint_finish(self):
        if self._stroke_base is None or self._stroke_bbox is None:
            self._stroke_base = None
            return
        y0, y1, x0, x1 = self._stroke_bbox
        self._undo.append((y0, y1, x0, x1, self._stroke_base[y0:y1, x0:x1].copy()))
        self._undo = self._undo[-40:]
        self._stroke_base = None

    def _dab(self, y, x):
        cy, cx = int(round(y)), int(round(x))
        if self._retouch == "clone":
            dy, dx = self._clone_off
            bbox = E.clone_dab(self.base, cy, cx, int(round(y + dy)),
                               int(round(x + dx)), self._brush_r, 0.5, self._strength())
        else:
            bbox = E.highlight_dab(self.base, cy, cx, self._brush_r, 0.5, self._strength())
        if bbox is None:
            return
        self._grow_bbox(bbox)
        self._sync_proxy_region(*bbox)
        self._refresh_panes()

    def _grow_bbox(self, bbox):
        if self._stroke_bbox is None:
            self._stroke_bbox = bbox
        else:
            y0, y1, x0, x1 = self._stroke_bbox
            self._stroke_bbox = (min(y0, bbox[0]), max(y1, bbox[1]),
                                 min(x0, bbox[2]), max(x1, bbox[3]))

    def _sync_proxy_region(self, y0, y1, x0, x1):
        """Re-derive the edited base region into the preview proxy so both stay
        in step without rebuilding the whole proxy every dab."""
        if self.proxy is None or self.base is None:
            return
        Hb, Wb = self.base.shape[:2]
        Hp, Wp = self.proxy.shape[:2]
        if (Hp, Wp) == (Hb, Wb):
            self.proxy[y0:y1, x0:x1] = self.base[y0:y1, x0:x1]
            return
        sx, sy = Wp / Wb, Hp / Hb
        pad = 2
        py0 = max(0, int(np.floor(y0 * sy)) - pad)
        py1 = min(Hp, int(np.ceil(y1 * sy)) + pad)
        px0 = max(0, int(np.floor(x0 * sx)) - pad)
        px1 = min(Wp, int(np.ceil(x1 * sx)) + pad)
        by0 = max(0, int(np.floor(py0 / sy)))
        by1 = min(Hb, int(np.ceil(py1 / sy)))
        bx0 = max(0, int(np.floor(px0 / sx)))
        bx1 = min(Wb, int(np.ceil(px1 / sx)))
        tw, th = px1 - px0, py1 - py0
        if tw < 1 or th < 1 or by1 <= by0 or bx1 <= bx0:
            return
        tile = Image.fromarray(E.to_uint8(self.base[by0:by1, bx0:bx1]))
        tile = tile.resize((tw, th), Image.LANCZOS)
        self.proxy[py0:py1, px0:px1] = np.asarray(tile, np.float32) / 255.0

    def _refresh_panes(self):
        """Rebuild both panes from the current proxy (retouch is baked into the
        source, so Original updates too), keeping the shared view."""
        if self.proxy is None:
            return
        self.orig_pane.set_image(self.proxy)
        self.res_pane.set_image(self._process(self.proxy))
        self._apply_views()

    def _undo_retouch(self):
        if not self._undo:
            return
        y0, y1, x0, x1, before = self._undo.pop()
        self.base[y0:y1, x0:x1] = before
        self._sync_proxy_region(y0, y1, x0, x1)
        self._refresh_panes()
        self._append("Undid retouch stroke.")

    def _recolor(self, fx, fy, alt):
        colour = self._sample_patch(fx, fy)
        if alt:                                  # pick the destination colour
            self._recolor_target = colour.copy()
            self._update_swatch()
            rgb = colour * 255
            self._append(f"Target colour → R{rgb[0]:.0f} G{rgb[1]:.0f} B{rgb[2]:.0f}")
            return
        if self._recolor_target is None:
            self._append("Pick a target colour first: Alt/Option-click the colour "
                         "you want.")
            return
        pre = self.base.copy()                   # transient, for one undo step
        tol = 0.02 + self.recolor_tol.value() / 100.0 * 0.58
        bbox = E.recolor(self.base, colour, self._recolor_target, tol, self._strength())
        if bbox is None:
            self._append("No pixels matched that colour.")
            return
        y0, y1, x0, x1 = bbox
        self._undo.append((y0, y1, x0, x1, pre[y0:y1, x0:x1].copy()))
        self._undo = self._undo[-40:]
        self._sync_proxy_region(*bbox)
        self._refresh_panes()
        self._append(f"Recoloured a {y1 - y0}×{x1 - x0} region.")

    # -- render ------------------------------------------------------------
    def _kick(self):
        self._kick_timer.start()

    def _process(self, arr):
        out = arr
        if self.neutral is not None:
            out = E.white_balance(out, self.neutral, self.wb_slider.value() / 100.0)
        out = E.apply_curves(out, self.curve.curves)
        return out

    def _render_result(self):
        if self.proxy is None:
            return
        self.res_pane.set_image(self._process(self.proxy))
        self._apply_views()

    def _apply_views(self):
        self.orig_pane.set_view(self.zoom, self.pan_cx, self.pan_cy)
        self.res_pane.set_view(self.zoom, self.pan_cx, self.pan_cy)

    # -- zoom / pan --------------------------------------------------------
    def _set_zoom(self, z):
        z = min(max(z, self.ZOOM_MIN), self.ZOOM_MAX)
        self.zoom = z
        self._sync_zoom_ui()
        self._apply_views()

    def _sync_zoom_ui(self):
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(self.zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_lbl.setText(f"{self.zoom:.1f}×")

    def _on_wheel_zoom(self, factor):
        self._set_zoom(self.zoom * factor)

    def _zoom_fit(self):
        self.pan_cx, self.pan_cy = 0.5, 0.5
        self._set_zoom(1.0)

    def _on_pan(self, dcx, dcy):
        if self.base is None:
            return
        self.pan_cx = min(max(self.pan_cx + dcx, 0.0), 1.0)
        self.pan_cy = min(max(self.pan_cy + dcy, 0.0), 1.0)
        self._apply_views()

    # -- reset / save ------------------------------------------------------
    def _reset(self):
        self.neutral = None
        self.wb_slider.setValue(100)
        self.curve.curves = {k: list(E.IDENTITY) for k in "LRGB"}
        self.curve.update()
        self.orig_pane.set_marker(None, None)
        if self.orig_base is not None:           # discard all retouch strokes
            self.base = self.orig_base.copy()
            self.proxy = E.make_proxy(self.base, PROXY_MAX)
        self._undo = []
        self._clone_src = self._clone_off = None
        self.res_pane.set_src_marker(None, None)
        self._append("Reset all adjustments.")
        self._refresh_panes()

    def _save(self):
        if self.base is None:
            return
        out = self._process(self.base)          # full-res, not the proxy
        default = str(Path(self._last_dir) / f"{self.path.stem}_jphoto.jpg")
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save result", default, "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif)")
        if not fn:
            return
        try:
            Image.fromarray(E.to_uint8(out)).save(fn, quality=95)
            self._append(f"Saved → {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # -- log ---------------------------------------------------------------
    def _append(self, msg):
        self.log.appendPlainText(msg)


class PropertiesDialog(QDialog):
    """Read-only EXIF / IPTC / file-info inspector (fed by rawmeta)."""

    def __init__(self, path, sections, parent=None):
        super().__init__(parent)
        self._path = path
        self._sections = sections
        self.setWindowTitle(f"Properties — {Path(path).name}")
        self.resize(500, 620)
        self.setStyleSheet("""
            QDialog, QScrollArea, QWidget { background:#0e0e0e; }
            QLabel { color:#f2f2f2; font-size:13px; }
            QLabel[role="section"] { color:#8fbcff; font-weight:bold;
                                     font-size:14px; padding-top:10px; }
            QLabel[role="key"] { color:#aab4bd; }
            QLabel[role="val"] { color:#ffffff; }
            QPushButton { background:#2a2a2a; color:#f2f2f2; border:1px solid #555;
                          border-radius:4px; padding:4px 12px; }
            QPushButton:hover { background:#46648a; }
        """)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(3)

        row = 0
        for title, rows in sections:
            head = QLabel(title)
            head.setProperty("role", "section")
            grid.addWidget(head, row, 0, 1, 2)
            row += 1
            for label, value in rows:
                k = QLabel(label)
                k.setProperty("role", "key")
                k.setAlignment(Qt.AlignTop | Qt.AlignRight)
                v = QLabel(str(value))
                v.setProperty("role", "val")
                v.setWordWrap(True)
                v.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(k, row, 0)
                grid.addWidget(v, row, 1)
                row += 1
        grid.setRowStretch(row, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        btns = QHBoxLayout()
        btns.addStretch(1)
        copy_btn = QPushButton("Copy all")
        copy_btn.clicked.connect(self._copy_all)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(copy_btn)
        btns.addWidget(close_btn)
        outer.addLayout(btns)

    def _copy_all(self):
        from PySide6.QtWidgets import QApplication
        try:
            text = rawmeta.metadata_text(self._path)
        except Exception:
            text = "\n".join(
                f"{lbl}: {val}" for _, rows in self._sections for lbl, val in rows)
        QApplication.clipboard().setText(text)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_I):
            self.accept()
        else:
            super().keyPressEvent(event)
