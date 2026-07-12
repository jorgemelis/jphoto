#!/usr/bin/env python3
"""
jphoto/engine.py — minimal, self-contained photo engine.

Pure-numpy operations on sRGB float images in [0,1]. No dependency on diapos
(or anything under /Volumes): jphoto owns its image maths and grows it as the
GUI needs it. v1 covers what a "viewer + white balance + curves" front-end
asks for: load, proxy downscale, eyedropper white balance, per-channel curves.

Convention: every op takes and returns a float32 array of shape (H, W, 3) with
values in [0, 1]; Qt conversion lives in the GUI, so this module stays headless.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


# --- load / convert ---------------------------------------------------------

def load_image(path) -> np.ndarray:
    """Read an image as sRGB float32 [0,1], honouring EXIF orientation.

    RAW files (Sony ARW, Canon CR2/CR3, Nikon NEF, DNG, …) are decoded through
    ``rawmeta`` (macOS sips); everything else goes straight through Pillow.
    """
    import rawmeta
    img = rawmeta.open_pil(path)              # Image.open for non-RAW, sips for RAW
    img = ImageOps.exif_transpose(img)        # rotate per camera orientation tag
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    """Clamp and quantize a float [0,1] image to uint8 for display / saving."""
    return (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_proxy(arr01: np.ndarray, max_edge: int = 1100) -> np.ndarray:
    """Downscale so the longest edge is `max_edge`, for fast live preview.
    Images already small enough are returned unchanged."""
    h, w = arr01.shape[:2]
    s = max_edge / max(h, w)
    if s >= 1.0:
        return arr01
    img = Image.fromarray(to_uint8(arr01)).resize(
        (max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# --- white balance ----------------------------------------------------------

def white_balance(arr01: np.ndarray, neutral_rgb, strength: float = 1.0) -> np.ndarray:
    """Scale each channel so the sampled `neutral_rgb` becomes neutral grey.

    `neutral_rgb` is the average colour of a patch the user clicked on something
    that should be grey/white. The per-channel gain that maps it to its own mean
    removes the cast; `strength` (0..1) dials how much of that correction to keep.
    """
    neutral = np.clip(np.asarray(neutral_rgb, dtype=np.float32), 1e-4, None)
    gains = float(neutral.mean()) / neutral          # gray-world gain per channel
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(arr01 * gains, 0.0, 1.0)


# --- curves -----------------------------------------------------------------

IDENTITY = [(0.0, 0.0), (1.0, 1.0)]


def is_identity(pts) -> bool:
    return len(pts) == 2 and tuple(pts[0]) == (0.0, 0.0) and tuple(pts[1]) == (1.0, 1.0)


def _apply_curve(v: np.ndarray, pts) -> np.ndarray:
    """Map values through a piecewise-linear curve defined by control points."""
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    order = np.argsort(xs)
    return np.interp(v, xs[order], ys[order]).astype(np.float32)


def apply_curves(arr01: np.ndarray, curves: dict) -> np.ndarray:
    """Apply a master (L) curve to every channel, then each R/G/B channel's own.

    `curves` maps "L"/"R"/"G"/"B" to lists of (x, y) control points in [0,1].
    Identity curves are skipped, so an untouched editor costs nothing.
    """
    master = curves.get("L")
    do_master = master and not is_identity(master)
    out = np.empty_like(arr01)
    for i, ch in enumerate("RGB"):
        v = arr01[:, :, i]
        if do_master:
            v = _apply_curve(v, master)
        pts = curves.get(ch)
        if pts and not is_identity(pts):
            v = _apply_curve(v, pts)
        out[:, :, i] = v
    return np.clip(out, 0.0, 1.0)


# --- local retouch: clone (hide) / highlight (emphasise) --------------------
#
# Two brushes for spots like moles. Both edit the image IN PLACE over a soft
# circular disk and return the touched bbox (y0, y1, x0, x1), or None if the
# dab fell entirely outside the image. Working on real pixels — cloned skin or
# the spot's own colour pushed further — is what keeps the result skin-coloured
# instead of the flat grey you get from averaging a neighbourhood.

def _disk_mask(radius: int, softness: float) -> np.ndarray:
    """A soft circular alpha mask (2R+1 square), 1 in the core, feathered to 0
    at the edge. `softness` in (0,1] is the fraction of the radius spent
    fading — 0.5 feathers the outer half, a small value gives a crisp edge."""
    R = int(max(1, radius))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(xx * xx + yy * yy) / R                 # 0 centre .. 1 at radius
    a = np.clip((1.0 - d) / max(1e-3, softness), 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)                         # smoothstep for a soft ramp
    a[d > 1.0] = 0.0
    return a.astype(np.float32)


def clone_dab(img, cy, cx, syc, sxc, radius, softness, opacity):
    """Copy a soft disk of skin from source centre (syc,sxc) onto (cy,cx).

    A plain aligned clone: real pixels from clean skin the user pointed at,
    blended in through the feathered mask. No averaging, so it can't go grey.
    """
    R = int(max(1, round(radius)))
    H, W = img.shape[:2]
    mask = _disk_mask(R, softness) * float(opacity)
    dy = np.arange(cy - R, cy + R + 1)
    dx = np.arange(cx - R, cx + R + 1)
    sy = np.arange(syc - R, syc + R + 1)
    sx = np.arange(sxc - R, sxc + R + 1)
    vy = (dy >= 0) & (dy < H) & (sy >= 0) & (sy < H)   # keep rows valid in BOTH
    vx = (dx >= 0) & (dx < W) & (sx >= 0) & (sx < W)   # dst and src windows
    if not vy.any() or not vx.any():
        return None
    dy, dx, sy, sx = dy[vy], dx[vx], sy[vy], sx[vx]
    m = mask[np.ix_(vy, vx)][..., None]
    dst = img[np.ix_(dy, dx)]
    src = img[np.ix_(sy, sx)]
    img[np.ix_(dy, dx)] = dst * (1.0 - m) + src * m
    return (int(dy[0]), int(dy[-1] + 1), int(dx[0]), int(dx[-1] + 1))


def highlight_dab(img, cy, cx, radius, softness, strength):
    """Make the spot under the brush stand out: deepen it and lift local
    contrast + saturation, softly masked. Works on the pixels' own colour, so a
    dark mole gets darker and richer rather than washing toward grey."""
    R = int(max(1, round(radius)))
    H, W = img.shape[:2]
    ys = np.arange(max(0, cy - R), min(H, cy + R + 1))
    xs = np.arange(max(0, cx - R), min(W, cx + R + 1))
    if ys.size == 0 or xs.size == 0:
        return None
    full = _disk_mask(R, softness)
    m = (full[np.ix_(ys - (cy - R), xs - (cx - R))] * float(strength))[..., None]
    reg = img[np.ix_(ys, xs)]
    lum = reg.mean(2, keepdims=True)
    mean_l = float((lum * m).sum() / (m.sum() + 1e-6))  # patch's average brightness
    out = mean_l + (reg - mean_l) * 1.6                 # local contrast about it
    out = out * 0.70                                    # deepen the spot
    plum = out.mean(2, keepdims=True)
    out = plum + (out - plum) * 1.5                     # richer colour, not grey
    out = np.clip(out, 0.0, 1.0)
    img[np.ix_(ys, xs)] = reg * (1.0 - m) + out * m
    return (int(ys[0]), int(ys[-1] + 1), int(xs[0]), int(xs[-1] + 1))


# --- recolour: replace one colour with another everywhere -------------------
#
# Select every pixel near a picked colour and push it to a chosen colour's hue
# and saturation while keeping each pixel's own brightness — so shading and
# texture survive and only the colour changes (a flat fill would kill both).

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV on an (..., 3) float array in [0,1]; H,S,V in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-12
    im = nz & (mx == r)
    h[im] = ((g[im] - b[im]) / d[im]) % 6.0
    im = nz & (mx == g)
    h[im] = (b[im] - r[im]) / d[im] + 2.0
    im = nz & (mx == b)
    h[im] = (r[im] - g[im]) / d[im] + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1).astype(np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of `_rgb_to_hsv`."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = (i.astype(int)) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def recolor(img, source_rgb, target_rgb, tolerance, strength):
    """Recolour every pixel within `tolerance` of `source_rgb` toward
    `target_rgb`'s hue + saturation, keeping brightness. Edits img in place;
    returns the affected bbox, or None if nothing matched.

    `tolerance` is the RGB selection radius in (0,1]; the selection is feathered
    so edges blend. `strength` in [0,1] is how fully matched pixels take the new
    colour.

    Brightness is *shifted* by the target-minus-source difference rather than
    kept as-is: each pixel's own shading (its deviation from the picked colour)
    survives, but the overall level moves to the target's. This is what lets a
    dark or black area actually recolour — keeping value verbatim would leave
    black black, since HSV value 0 is black at any hue.
    """
    src = np.asarray(source_rgb, dtype=np.float32)
    dist = np.sqrt(((img - src) ** 2).sum(-1) / 3.0)      # 0..1 colour distance
    tol = max(1e-3, float(tolerance))
    m = np.clip(1.0 - dist / tol, 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m) * float(strength)          # smoothstep + strength
    ys, xs = np.where(m > 0.004)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)
    reg = img[y0:y1, x0:x1]
    mm = m[y0:y1, x0:x1][..., None]
    t_hsv = _rgb_to_hsv(np.asarray(target_rgb, np.float32).reshape(1, 1, 3))[0, 0]
    s_v = float(np.max(src))                               # brightness of picked colour
    dv = float(t_hsv[2]) - s_v                             # shift toward target's level
    hsv = _rgb_to_hsv(reg)
    hsv[..., 0] = t_hsv[0]                                  # adopt target hue
    hsv[..., 1] = t_hsv[1]                                  # and saturation
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0.0, 1.0)      # lift/lower brightness, keep shading
    rec = _hsv_to_rgb(hsv)
    img[y0:y1, x0:x1] = np.clip(reg * (1.0 - mm) + rec * mm, 0.0, 1.0)
    return (y0, y1, x0, x1)
