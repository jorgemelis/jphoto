#!/usr/bin/env python3
"""
jphoto — standalone photo-restoration app.

Built on the diapos phase-1 engine (non-destructive stack in linear light) with
the CLAHE / crop / upscale / preset grafts from the jjtools photo_restoration
tab. Runs entirely from this repo — the phase-2 ML scripts and models are local,
no dependency on /Volumes/n01.

Usage:
    python jphoto.py [image_or_directory]
"""

import sys
from pathlib import Path

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QApplication, QMainWindow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jphoto.window import JPhotoWidget  # noqa: E402
from jphoto import state  # noqa: E402


class MainWindow(QMainWindow):
    def __init__(self, start=None):
        super().__init__()
        self.setWindowTitle("jphoto — photo restoration")
        self.widget = JPhotoWidget(self)
        self.setCentralWidget(self.widget)
        self._restore_state()
        if start is not None:
            self.widget.open_path(start)
        else:
            last = state.load().get("last_image")
            if last and Path(last).exists():
                self.widget.open_path(Path(last))   # reopen the last photo

    def _restore_state(self):
        scr = QApplication.primaryScreen().availableGeometry()
        geom = state.load().get("geometry")
        restored = False
        if geom:
            try:
                self.restoreGeometry(QByteArray.fromHex(geom.encode()))
                restored = True
            except Exception:
                pass
        if not restored:
            self.resize(min(1500, scr.width() - 40), min(950, scr.height() - 60))
        self._fit_to_screen(scr)

    def _fit_to_screen(self, scr=None):
        """Never let the window exceed the screen, and keep it fully on-screen —
        a stale saved geometry was making it wider than the display."""
        if scr is None:
            scr = QApplication.primaryScreen().availableGeometry()
        w = min(self.width(), scr.width() - 20)
        h = min(self.height(), scr.height() - 40)
        self.resize(w, h)
        x = min(max(self.x(), scr.x() + 10), scr.x() + scr.width() - w - 10)
        y = min(max(self.y(), scr.y() + 10), scr.y() + scr.height() - h - 10)
        self.move(x, y)

    def closeEvent(self, ev):
        try:
            state.update(geometry=bytes(self.saveGeometry().toHex()).decode())
        except Exception:
            pass
        self.widget.close()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    start = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            start = p
    win = MainWindow(start)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
