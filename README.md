# jphoto

A small, self-contained macOS app for restoring faded or colour-cast photographs.
It develops one image by hand — white balance, per-channel curves, local retouch —
rather than running a batch pipeline. Built with PySide6; the image maths is plain
NumPy on sRGB float images, so the engine stays readable and easy to extend.

## Features

- **Original / Result split view** with synced zoom and pan.
- **White balance** by eyedropper: click a point that should be neutral grey/white.
- **Probe cast** (read-only): sample points to read the colour tint; a tint shared
  across surfaces you know are neutral is the cast, an outlier is a real colour.
- **Per-channel curves** (L / R / G / B).
- **Local retouch**: clone / heal strokes with undo.
- **RAW support**: opens Sony ARW, Canon CR2/CR3, Nikon NEF, Adobe DNG, Fujifilm
  RAF, Panasonic RW2, Olympus ORF and more, decoded through macOS' native
  ImageIO (`sips`) — no libraw/rawpy needed.
- **Properties inspector** (`I`): file characteristics + EXIF + IPTC, read from
  Spotlight metadata (`mdls`) and `sips`. Works for JPEG/TIFF and RAW alike.
- **Save** the full-resolution result as JPEG / PNG / TIFF.

## Requirements

- **macOS** — RAW decoding and metadata use the built-in `sips` and `mdls` tools.
- **Python 3.11+**
- `PySide6`, `numpy`, `Pillow` (see `requirements.txt`).

## Install & run

```bash
git clone https://github.com/jorgemelis/jphoto.git
cd jphoto
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jphoto.py [image_or_directory]
```

Double-clicking `jphoto.app` runs the same thing: its launcher resolves the repo
root from its own location and prefers a `.venv/` next to it, falling back to
`python3` on your `PATH`.

## Keyboard

| Key | Action |
|-----|--------|
| `I` | Properties (EXIF / IPTC / file info) |
| `⌘Z` | Undo retouch stroke |

## How it works

`jphoto/engine.py` is a headless, pure-NumPy module: every operation takes and
returns a float32 `(H, W, 3)` array in `[0, 1]`, with Qt conversion confined to
the GUI. `rawmeta.py` is the only OS-specific piece — it shells out to `sips`
(RAW → RGB) and `mdls` + `sips -g all` (metadata), so the rest of the app never
has to know a file was RAW.

## License

MIT — see [LICENSE](LICENSE).
