# jphoto — standalone photo restoration

Standalone PySide6 app for restoring faded / colour-cast photos. Launched with
`python jphoto.py [image_or_directory]` (the `.app` bundle wraps the same).

## Status (2026-06-27): rebuilt from scratch, self-contained

jphoto **no longer inherits from `diapos`** (it used to subclass `DiaposTab`).
That coupling dragged in diapos's whole inherited layout and a zoom clamped at
"fit" (1.0×), which made the image impossible to shrink. It was scrapped and
rebuilt as an independent widget that owns its own minimal engine. Nothing under
`diapos/` is imported anymore.

Built incrementally — features are added only as they prove necessary, not ported
wholesale.

### Files

- `jphoto.py` (repo root) — `QMainWindow` launcher. Restores/saves window
  geometry; reopens the last image. **Auto-fits the window height to the photo's
  aspect on open** (`JPhotoWidget._fit_window_to_image`) so landscape photos
  don't letterbox in a tall window.
- `jphoto/window.py` — the GUI (`QWidget`, no diapos). Contains `PreviewPane`,
  `CurveEditor`, `JPhotoWidget`. Qt only; all maths is in the engine.
- `jphoto/engine.py` — minimal pure-numpy engine on sRGB float [0,1]. No diapos,
  no `/Volumes` dependency. `load_image`, `to_uint8`, `make_proxy`,
  `white_balance`, `apply_curves`.
- `jphoto/state.py` — tiny persisted state (last image, recents, splitter sizes,
  window geometry) in `jphoto/data/jphoto_state.json`. Writes MERGE.

### Layout

`QSplitter` (saved/restored): left = open/recent + tool toggles · Original|Result
panes · zoom bar · probe readout · log. Right = control tabs (White balance,
Curves) + Reset / Save.

### Implemented (v1)

- **Viewer**: Original|Result panes, synced zoom + pan. Zoom **0.2×–8.0×** —
  below 1.0× ("fit") the image shrinks *inside* the pane, so it can be made small
  without resizing the window. `Fit` button recentres at 1.0×.
- **Eyedropper white balance**: click a neutral on Original → gray-world per-
  channel gains, blended by a Strength slider (0–100%). Live preview on the proxy;
  full-res only applied on Save.
- **Probe cast** (read-only): click points on Original to read RGB + cast hue +
  Δspread + suggested WB gains; keeps the last 5 readings. Mutually exclusive with
  Eyedropper. Use to tell a real neutral (consistent WB across points) from a real
  colour (outlier).
- **Per-channel curves**: master L + R/G/B. Click to add a point, drag to move,
  double-click to remove; endpoints move vertically only.
- **Mark** toggle: show/hide the red cross at the last pick/probe point.
- **Retouch spots (moles)**: two local brushes painted on the **Result** pane
  (pan meanwhile on the Original pane). Brush-size + strength sliders; a live
  ring shows the footprint. Per-stroke undo (`⌘Z` or the button); Reset reverts
  every stroke to the pristine image. Retouch edits the source `base`/`proxy` in
  place *before* WB/curves, so it shows on both panes.
  - **Hide (clone)** — Alt/Option-click clean skin to anchor a source (green
    cross), then drag over the mole: an aligned soft-edged clone of real skin
    covers it. Copies actual pixels, never an average — that's what keeps it
    skin-coloured instead of the flat grey a neighbourhood blur produces.
  - **Emphasise** — drag over the mole: within the soft disk it lifts local
    contrast about the patch mean, deepens (×0.70) and boosts saturation, all
    amplifying the pixels' own colour so the spot stands out without greying.
    Tuning lives in `engine.highlight_dab` (the 1.6 / 0.70 / 1.5 factors).
  - **Recolour** — Alt/Option-click to pick a target colour (swatch), then click
    any colour: every pixel within `Tolerance` of it takes the target's hue+sat,
    and its brightness is *shifted* by the target-minus-source difference (not
    kept verbatim) so per-pixel shading survives while black/dark areas actually
    recolour — keeping value would leave black black (HSV value 0 is black at any
    hue). Global, not brushed; one undo step. `engine.recolor` (selection is an
    RGB distance smoothstep). Vectorised `_rgb_to_hsv`/`_hsv_to_rgb`.
- Recent-images menu; Reset; Save (full-res, JPEG/PNG/TIFF).

### Not yet ported (from the old diapos-based version)

CLAHE · interactive crop · tone presets (fade/cast) · phase-2 ML (colorize /
inpaint / denoise / face restore / upscale). Add on demand.

### Conventions

- All image ops take/return float32 (H,W,3) in [0,1]; Qt conversion stays in the GUI.
- Live preview runs on a proxy (longest edge `PROXY_MAX = 1100`); Save processes
  the full-res `base`.
- Project-local `jphoto/tmp/` and `jphoto/data/`; never `/tmp`.
