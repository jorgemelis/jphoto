"""
jphoto/state.py — tiny persisted state (last image, recent images/dirs, window
geometry). Both the launcher and the widget read/write it, so writes MERGE into
the existing file instead of clobbering it.
"""

import json
from pathlib import Path

STATE = Path(__file__).resolve().parent / "data" / "jphoto_state.json"

MAX_IMAGES = 15
MAX_DIRS = 10


def load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def update(**kwargs) -> None:
    """Merge kwargs into the state file (other keys preserved)."""
    d = load()
    d.update(kwargs)
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


def record_open(path) -> None:
    """Remember `path` as the last image and push it onto the recent lists."""
    path = str(path)
    parent = str(Path(path).parent)
    d = load()
    imgs = [p for p in d.get("recent_images", []) if p != path]
    imgs.insert(0, path)
    dirs = [p for p in d.get("recent_dirs", []) if p != parent]
    dirs.insert(0, parent)
    update(last_image=path, last_dir=parent,
           recent_images=imgs[:MAX_IMAGES], recent_dirs=dirs[:MAX_DIRS])
