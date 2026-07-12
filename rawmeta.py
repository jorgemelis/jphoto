#!/usr/bin/env python3
"""
rawmeta.py — shared RAW decoding + image metadata for jsee and jphoto.

Zero external dependencies: it shells out to macOS' native tools.

- RAW decode uses ``sips`` (Apple ImageIO), which renders Sony ARW, Canon
  CR2/CR3, Nikon NEF, Adobe DNG, etc. without libraw/rawpy.
- Metadata comes from ``mdls`` (Spotlight — surfaces EXIF/IPTC that ImageIO
  indexed) merged with ``sips -g all`` and ``os.stat``.

Qt (PySide6) and PIL are imported lazily, so the metadata half of this module
works in a headless context and neither app pays for imports it doesn't use.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

# RAW containers macOS' ImageIO can decode via sips. Lower-case, with dot.
RAW_EXTENSIONS = {
    ".arw", ".srf", ".sr2",              # Sony
    ".cr2", ".cr3", ".crw",              # Canon
    ".nef", ".nrw",                      # Nikon
    ".dng",                              # Adobe / generic
    ".raf",                              # Fujifilm
    ".rw2",                              # Panasonic
    ".orf",                              # Olympus
    ".pef",                              # Pentax
    ".srw",                              # Samsung
    ".arq",                              # Sony pixel-shift
    ".3fr", ".fff",                      # Hasselblad
    ".erf",                              # Epson
    ".raw", ".rwl", ".dcr", ".kdc",      # Leica / Kodak / misc
    ".mos", ".mrw", ".x3f",              # Leaf / Minolta / Sigma
}

_TMP_DIR = Path(__file__).resolve().parent / "tmp"


def is_raw(path) -> bool:
    """True if *path* has a RAW extension we route through sips."""
    return Path(path).suffix.lower() in RAW_EXTENSIONS


# --- RAW decode -------------------------------------------------------------

def render_raw(path, max_px: int | None = None, timeout: int = 60) -> bytes | None:
    """Decode a RAW file to JPEG bytes via ``sips``.

    *max_px* caps the longer edge (for thumbnails/proxies); ``None`` renders at
    full resolution. Returns ``None`` if sips is missing or fails.
    """
    src = str(path)
    try:
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        fd, out = tempfile.mkstemp(suffix=".jpg", dir=str(_TMP_DIR))
        os.close(fd)
    except OSError:
        return None
    try:
        cmd = ["sips"]
        if max_px:
            cmd += ["-Z", str(int(max_px))]
        cmd += ["-s", "format", "jpeg", src, "--out", out]
        subprocess.run(
            cmd, check=True, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        data = Path(out).read_bytes()
        return data or None
    except (subprocess.SubprocessError, OSError):
        return None
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def load_pixmap(path, max_px: int | None = None):
    """Return a QPixmap for *path*, decoding RAW via sips when needed.

    For non-RAW files this is just ``QPixmap(path)`` — same behaviour as before,
    so callers can route every image through here. Returns a null/na QPixmap on
    failure (mirrors Qt's own behaviour), or ``None`` if Qt is unavailable.
    """
    try:
        from PySide6.QtGui import QPixmap
    except ImportError:
        return None
    if not is_raw(path):
        return QPixmap(str(path))
    data = render_raw(path, max_px=max_px)
    pm = QPixmap()
    if data:
        pm.loadFromData(data)
    return pm


def open_pil(path):
    """Return a PIL.Image for *path*, decoding RAW via sips when needed.

    Drop-in for ``Image.open(path)``; raises if PIL is missing or decode fails.
    """
    from PIL import Image
    if not is_raw(path):
        return Image.open(path)
    data = render_raw(path)
    if not data:
        raise OSError(f"sips could not decode RAW file: {path}")
    return Image.open(io.BytesIO(data))


# --- metadata ---------------------------------------------------------------

def _run(cmd, timeout: int = 20) -> str:
    try:
        return subprocess.run(
            cmd, timeout=timeout, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _parse_mdls(out: str) -> dict[str, str]:
    """Parse ``mdls FILE`` output into a flat {key: value} dict.

    Handles scalar values and parenthesised arrays that span several lines.
    Drops ``(null)``. Strips surrounding quotes from scalars.
    """
    result: dict[str, str] = {}
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "=" not in line:
            i += 1
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        rest = rest.strip()
        if rest == "(":  # array — accumulate until the closing ')'
            items = []
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                items.append(lines[i].strip().rstrip(",").strip().strip('"'))
                i += 1
            items = [x for x in items if x and x != "(null)"]
            if items:
                result[key] = ", ".join(items)
        else:
            val = rest.strip('"')
            if val and val != "(null)":
                result[key] = val
        i += 1
    return result


def _parse_sips(out: str) -> dict[str, str]:
    """Parse ``sips -g all FILE`` output (``  key: value`` lines)."""
    result: dict[str, str] = {}
    for line in out.splitlines():
        s = line.strip()
        if ":" in s and not s.endswith(":"):
            key, _, val = s.partition(":")
            result[key.strip()] = val.strip()
    return result


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{n} B"


def _fmt_shutter(v: str) -> str:
    try:
        t = float(v)
    except (TypeError, ValueError):
        return v
    if t <= 0:
        return v
    if t >= 1:
        return f"{t:g} s"
    return f"1/{round(1 / t)} s"


def _fmt_float(v: str, suffix: str = "") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return f"{v}{suffix}"
    txt = f"{f:g}"
    return f"{txt}{suffix}"


_EXPOSURE_PROGRAM = {
    "0": "Not defined", "1": "Manual", "2": "Program AE",
    "3": "Aperture priority", "4": "Shutter priority", "5": "Creative",
    "6": "Action", "7": "Portrait", "8": "Landscape",
}
_WHITE_BALANCE = {"0": "Auto", "1": "Manual"}
_ORIENTATION = {
    "1": "Normal", "2": "Mirror horizontal", "3": "Rotate 180",
    "4": "Mirror vertical", "5": "Mirror horizontal + rotate 270 CW",
    "6": "Rotate 90 CW", "7": "Mirror horizontal + rotate 90 CW",
    "8": "Rotate 270 CW",
}


def read_metadata(path):
    """Return image metadata as ordered sections.

    Result: ``list[tuple[str, list[tuple[str, str]]]]`` — ``(section_title,
    [(label, value), ...])``. Empty sections are omitted. Works for RAW and
    ordinary images alike (mdls/sips read both).
    """
    p = Path(path)
    md = _parse_mdls(_run(["mdls", str(p)]))
    sp = _parse_sips(_run(["sips", "-g", "all", str(p)]))
    try:
        st = p.stat()
    except OSError:
        st = None

    def get(*keys):
        for k in keys:
            if k in md and md[k]:
                return md[k]
            if k in sp and sp[k]:
                return sp[k]
        return None

    sections: list[tuple[str, list[tuple[str, str]]]] = []

    def section(title, rows):
        rows = [(lbl, val) for lbl, val in rows if val]
        if rows:
            sections.append((title, rows))

    # --- File ---
    file_rows = [("Name", md.get("kMDItemFSName") or p.name), ("Path", str(p))]
    if st:
        file_rows.append(("Size", f"{_human_size(st.st_size)}  ({st.st_size:,} bytes)"))
    file_rows += [
        ("Kind", get("kMDItemKind")),
        ("Type", get("typeIdentifier", "format")),
    ]
    if st:
        import datetime as _dt
        file_rows.append(
            ("Modified", _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")))
    file_rows.append(("Added", get("kMDItemFSCreationDate")))
    section("File", file_rows)

    # --- Image ---
    w = get("pixelWidth", "kMDItemPixelWidth")
    h = get("pixelHeight", "kMDItemPixelHeight")
    dims = f"{w} × {h}" if w and h else None
    mp = None
    try:
        mp = f"{int(w) * int(h) / 1e6:.1f} MP"
    except (TypeError, ValueError):
        pass
    bits = get("bitsPerSample")
    spp = get("samplesPerPixel")
    depth = f"{bits}-bit × {spp} ch" if bits and spp else bits
    dpi = get("dpiWidth", "kMDItemResolutionWidthDPI")
    orient = md.get("kMDItemOrientation")
    section("Image", [
        ("Dimensions", dims),
        ("Megapixels", mp),
        ("Color space", get("space", "kMDItemColorSpace")),
        ("Color profile", get("profile", "kMDItemProfileName")),
        ("Bit depth", depth),
        ("Resolution", f"{_fmt_float(dpi)} dpi" if dpi else None),
        ("Orientation", _ORIENTATION.get(orient, orient) if orient else None),
    ])

    # --- Camera / EXIF ---
    prog = md.get("kMDItemExposureProgram")
    wb = md.get("kMDItemWhiteBalance")
    flash = md.get("kMDItemFlashOnOff")
    st_v = get("kMDItemExposureTimeSeconds")
    fn = get("kMDItemFNumber")
    iso = get("kMDItemISOSpeed")
    fl = get("kMDItemFocalLength")
    fl35 = get("kMDItemFocalLength35mm")
    section("Camera / EXIF", [
        ("Make", get("make", "kMDItemAcquisitionMake")),
        ("Model", get("model", "kMDItemAcquisitionModel")),
        ("Lens", get("kMDItemLensModel")),
        ("Software", get("software", "kMDItemCreator")),
        ("Capture date", get("creation", "kMDItemContentCreationDate")),
        ("Shutter", _fmt_shutter(st_v) if st_v else None),
        ("Aperture", f"f/{_fmt_float(fn)}" if fn else None),
        ("ISO", f"ISO {_fmt_float(iso)}" if iso else None),
        ("Focal length", f"{_fmt_float(fl)} mm" if fl else None),
        ("Focal length (35mm)", f"{_fmt_float(fl35)} mm" if fl35 else None),
        ("Exposure program", _EXPOSURE_PROGRAM.get(prog, prog) if prog else None),
        ("Exposure bias", get("kMDItemExposureBiasValue")),
        ("Metering", get("kMDItemMeteringMode")),
        ("White balance", _WHITE_BALANCE.get(wb, wb) if wb else None),
        ("Flash", ("Fired" if flash == "1" else "Did not fire") if flash is not None else None),
    ])

    # --- Location ---
    section("Location", [
        ("Latitude", get("kMDItemLatitude")),
        ("Longitude", get("kMDItemLongitude")),
        ("Altitude", get("kMDItemAltitude")),
        ("GPS date", get("kMDItemGPSDateStamp")),
        ("Place", get("kMDItemNamedLocation")),
        ("City", get("kMDItemCity")),
        ("State", get("kMDItemStateOrProvince")),
        ("Country", get("kMDItemCountry")),
    ])

    # --- Description / IPTC ---
    section("Description / IPTC", [
        ("Title", get("kMDItemTitle", "kMDItemDisplayName")),
        ("Headline", get("kMDItemHeadline")),
        ("Description", get("kMDItemDescription", "kMDItemFinderComment")),
        ("Keywords", get("kMDItemKeywords")),
        ("Authors", get("kMDItemAuthors")),
        ("Copyright", get("kMDItemCopyright", "kMDItemRights")),
        ("Credit", get("kMDItemCredit")),
        ("Contact", get("kMDItemContactKeywords")),
    ])

    return sections


def metadata_text(path) -> str:
    """Plain-text rendering of :func:`read_metadata`, for logs/tooltips."""
    out = []
    for title, rows in read_metadata(path):
        out.append(f"— {title} —")
        width = max((len(lbl) for lbl, _ in rows), default=0)
        for lbl, val in rows:
            out.append(f"  {lbl.ljust(width)}   {val}")
        out.append("")
    return "\n".join(out).rstrip()


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(f"### {arg}")
        print(metadata_text(arg))
        print()
