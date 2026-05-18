import hashlib
import io
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageFilter, ImageOps, ExifTags
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

RAW_EXTENSIONS = {
    ".cr2", ".cr3",           # Canon
    ".nef", ".nrw",           # Nikon
    ".arw", ".srf", ".sr2",   # Sony
    ".raf",                   # Fujifilm
    ".orf",                   # Olympus / OM System
    ".rw2",                   # Panasonic
    ".pef", ".ptx",           # Pentax
    ".dng",                   # Adobe DNG (Leica, Ricoh, phones…)
    ".rwl",                   # Leica legacy
    ".3fr",                   # Hasselblad
    ".iiq",                   # Phase One
    ".x3f",                   # Sigma
}

ALL_PHOTO_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS

BLUR_THRESHOLD = 500.0
SIMILARITY_THRESHOLD = 10

# Central database — one file for all projects
CONFIG_DIR = Path.home() / ".config" / "photo_sorter"
DB_PATH = CONFIG_DIR / "photos.db"
THUMB_DIR = Path.home() / ".cache" / "photo_sorter" / "thumbs"
THUMB_SIZE = 300  # px on longest edge

# In-memory caches keyed by absolute path
analysis_cache: dict = {}
raw_preview_cache: dict[str, bytes] = {}  # RAW path -> JPEG bytes


# ── Database ──────────────────────────────────────────────────────────────────

def thumb_path(photo_path: str) -> Path:
    key = hashlib.md5(photo_path.encode()).hexdigest()
    return THUMB_DIR / f"{key}.jpg"


def build_thumbnail(photo_path: str) -> Path:
    dst = thumb_path(photo_path)
    if dst.exists():
        # Invalidate if source is newer
        if Path(photo_path).stat().st_mtime <= dst.stat().st_mtime:
            return dst
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    img = open_as_pil(photo_path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    img.convert("RGB").save(dst, format="JPEG", quality=82, optimize=True)
    return dst


def init_db() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            path            TEXT PRIMARY KEY,
            folder          TEXT NOT NULL,
            name            TEXT,
            size_mb         REAL,
            width           INTEGER,
            height          INTEGER,
            blur_score      REAL,
            suggest_delete  INTEGER,
            file_hash       TEXT,
            dhash           TEXT,
            exif_json       TEXT,
            classification  TEXT,
            classified_at   TEXT,
            raw_companion   TEXT
        );
        CREATE TABLE IF NOT EXISTS projects (
            folder       TEXT PRIMARY KEY,
            name         TEXT,
            last_opened  TEXT,
            total        INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_folder    ON photos(folder);
        CREATE INDEX IF NOT EXISTS idx_dhash     ON photos(dhash);
        CREATE INDEX IF NOT EXISTS idx_file_hash ON photos(file_hash);
    """)
    conn.commit()
    conn.close()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_stats(folder: str) -> dict:
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT classification, COUNT(*) AS n FROM photos "
                "WHERE folder=? AND classification IS NOT NULL GROUP BY classification",
                (folder,),
            ).fetchall()
        return {r["classification"]: r["n"] for r in rows}
    except Exception:
        return {}


# ── Projects registry ─────────────────────────────────────────────────────────

def registry_upsert(folder: str, total: int) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO projects (folder, name, last_opened, total)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(folder) DO UPDATE SET
                name=excluded.name,
                last_opened=excluded.last_opened,
                total=excluded.total
            """,
            (folder, Path(folder).name, datetime.now(timezone.utc).isoformat(), total),
        )


# ── Security ──────────────────────────────────────────────────────────────────

def validate_path(path: str, folder: str) -> None:
    real_path = str(Path(path).resolve())
    real_folder = str(Path(folder).resolve())
    if not real_path.startswith(real_folder + os.sep):
        raise HTTPException(403, "Access denied")


def require_valid_folder(folder: str) -> str:
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        raise HTTPException(400, "Directory not found")
    return folder


# ── Image analysis ────────────────────────────────────────────────────────────

def is_raw(path: str) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def open_as_pil(path: str) -> Image.Image:
    """Open any photo (including RAW) as a PIL Image using the embedded JPEG thumbnail."""
    if is_raw(path):
        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(thumb.data)).copy()
            # Fallback: render full postprocessed image (slower)
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            return Image.fromarray(rgb)
    return Image.open(path)


def raw_preview_jpeg(path: str) -> bytes:
    """Return cached JPEG bytes for a RAW file (for browser display)."""
    if path not in raw_preview_cache:
        img = open_as_pil(path)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        raw_preview_cache[path] = buf.getvalue()
    return raw_preview_cache[path]


def find_raw_companion(path: str) -> str:
    p = Path(path)
    for ext in RAW_EXTENSIONS:
        for candidate in (p.with_suffix(ext), p.with_suffix(ext.upper())):
            if candidate.exists() and candidate.resolve() != p.resolve():
                return str(candidate)
    return ""


def compute_dhash(path: str) -> str:
    try:
        img = open_as_pil(path).convert("L").resize((9, 8), Image.LANCZOS)
        arr = np.array(img)
        diff = arr[:, 1:] > arr[:, :-1]
        val = 0
        for bit in diff.flatten():
            val = (val << 1) | int(bit)
        return f"{val:016x}"
    except Exception:
        return ""


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def hamming(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 999
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def compute_blur_score(path: str) -> float:
    try:
        img = open_as_pil(path)
        img.thumbnail((512, 512), Image.LANCZOS)
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        return float(np.array(edges, dtype=np.float64).var())
    except Exception:
        return -1.0


def get_exif(path: str) -> dict:
    try:
        img = open_as_pil(path) if is_raw(path) else Image.open(path)
        raw_exif = img.getexif()
        if not raw_exif:
            return {}
        wanted = {"Make", "Model", "DateTime", "DateTimeOriginal",
                  "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings",
                  "FocalLength", "FocalLengthIn35mmFilm"}
        result = {}
        # Merge ExifIFD sub-IFD — this is where FNumber, ExposureTime, ISO, FocalLength live
        all_tags: dict = dict(raw_exif)
        try:
            all_tags.update(raw_exif.get_ifd(0x8769))
        except Exception:
            pass
        for tag_id, val in all_tags.items():
            tag = ExifTags.TAGS.get(tag_id, "")
            if tag not in wanted:
                continue
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
                result[tag] = val
                continue
            try:
                if tag == "FNumber":
                    result[tag] = f"f/{float(val):.1f}"
                elif tag == "ExposureTime":
                    fval = float(val)
                    result[tag] = f"1/{round(1/fval)}s" if fval > 0 and fval < 1 else f"{fval:.1f}s"
                elif tag in ("FocalLength", "FocalLengthIn35mmFilm"):
                    result[tag] = f"{float(val):.0f}mm"
                else:
                    result[tag] = str(val)
            except Exception:
                result[tag] = str(val)
        return result
    except Exception:
        return {}


def analyze(path: str, folder: str) -> dict:
    if path in analysis_cache:
        return analysis_cache[path]

    f = Path(path)
    stat = f.stat()
    score = compute_blur_score(path)
    exif = get_exif(path)
    dhash = compute_dhash(path)
    file_hash = compute_file_hash(path)
    raw_companion = find_raw_companion(path)

    try:
        img = open_as_pil(path)
        w, h = img.size
    except Exception:
        w, h = 0, 0

    # Cross-folder exact duplicates (same file_hash, different folder)
    cross_dupes = []
    if file_hash:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT path, folder FROM photos WHERE file_hash=? AND path!=?",
                (file_hash, path),
            ).fetchall()
        cross_dupes = [
            {"path": r["path"], "folder": r["folder"], "name": Path(r["path"]).name}
            for r in rows
            if r["folder"] != folder
        ]

    result = {
        "path": path,
        "name": f.name,
        "size_mb": round(stat.st_size / 1_048_576, 2),
        "width": w,
        "height": h,
        "blur_score": round(score, 1),
        "suggest_delete": 0 <= score < BLUR_THRESHOLD,
        "file_hash": file_hash,
        "dhash": dhash,
        "exif": exif,
        "raw_companion": raw_companion,
        "raw_name": Path(raw_companion).name if raw_companion else "",
        "cross_folder_duplicates": cross_dupes,
    }
    analysis_cache[path] = result

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO photos
                (path, folder, name, size_mb, width, height, blur_score,
                 suggest_delete, file_hash, dhash, exif_json, raw_companion)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                folder=excluded.folder,
                size_mb=excluded.size_mb, width=excluded.width,
                height=excluded.height, blur_score=excluded.blur_score,
                suggest_delete=excluded.suggest_delete,
                file_hash=excluded.file_hash, dhash=excluded.dhash,
                exif_json=excluded.exif_json, raw_companion=excluded.raw_companion
            """,
            (path, folder, f.name, result["size_mb"], w, h, round(score, 1),
             int(result["suggest_delete"]), file_hash, dhash, json.dumps(exif), raw_companion),
        )
    return result


# ── Similarity ────────────────────────────────────────────────────────────────

def cluster_by_dhash(rows: list) -> list:
    assigned = set()
    groups = []
    for i, a in enumerate(rows):
        if a["path"] in assigned or not a["dhash"]:
            continue
        group = [a["path"]]
        for j, b in enumerate(rows):
            if i == j or b["path"] in assigned or not b["dhash"]:
                continue
            if hamming(a["dhash"], b["dhash"]) <= SIMILARITY_THRESHOLD:
                group.append(b["path"])
        if len(group) > 1:
            for p in group:
                assigned.add(p)
            groups.append(group)
    return sorted(groups, key=len, reverse=True)


def scan_photos(folder: str) -> list:
    all_files = [f for f in Path(folder).iterdir()
                 if f.is_file() and f.suffix.lower() in ALL_PHOTO_EXTENSIONS]
    # Stems covered by a JPEG/standard image — RAW files with these stems are companions, not primary
    jpeg_stems = {f.stem.upper() for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS}
    return sorted(
        str(f) for f in all_files
        if f.suffix.lower() in IMAGE_EXTENSIONS
        or f.stem.upper() not in jpeg_stems  # RAW only included when no JPEG companion exists
    )


# ── Move helper ───────────────────────────────────────────────────────────────

def _move_to_trash(src: Path, trash: Path) -> str | None:
    dst = trash / src.name
    counter = 1
    while dst.exists():
        dst = trash / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    try:
        shutil.move(str(src), str(dst))
        return src.name
    except Exception:
        return None


# ── Models ────────────────────────────────────────────────────────────────────

class ClassifyReq(BaseModel):
    folder: str
    path: str
    action: str


class ExecuteReq(BaseModel):
    folder: str


class RotateReq(BaseModel):
    folder: str
    path: str
    degrees: int


class PrefetchReq(BaseModel):
    folder: str
    paths: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT p.folder, p.name, p.last_opened, p.total, "
            "   COUNT(CASE WHEN ph.classification='favorite' THEN 1 END) AS fav, "
            "   COUNT(CASE WHEN ph.classification='keep'     THEN 1 END) AS keep, "
            "   COUNT(CASE WHEN ph.classification='delete'   THEN 1 END) AS del, "
            "   COUNT(CASE WHEN ph.classification='unsure'   THEN 1 END) AS unsure "
            "FROM projects p "
            "LEFT JOIN photos ph ON ph.folder=p.folder "
            "GROUP BY p.folder "
            "ORDER BY p.last_opened DESC LIMIT 20"
        ).fetchall()
    return {
        "projects": [
            {
                "folder": r["folder"],
                "name": r["name"],
                "last_opened": r["last_opened"],
                "total": r["total"],
                "stats": {
                    "favorite": r["fav"],
                    "keep": r["keep"],
                    "delete": r["del"],
                    "unsure": r["unsure"],
                },
            }
            for r in rows
        ]
    }


@app.get("/api/folder")
def open_folder(path: str):
    folder = require_valid_folder(path)
    photos = scan_photos(folder)

    # Persist companions eagerly (pure path checks, no image decode)
    companions: dict = {}
    for photo_path in photos:
        raw = find_raw_companion(photo_path)
        if raw:
            companions[photo_path] = raw

    with get_db() as conn:
        for photo_path, raw_path in companions.items():
            conn.execute(
                "INSERT INTO photos (path, folder, name, raw_companion) VALUES (?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET raw_companion=excluded.raw_companion",
                (photo_path, folder, Path(photo_path).name, raw_path),
            )

        placeholders = ",".join("?" * len(photos)) if photos else "''"
        rows = conn.execute(
            f"SELECT path, classification FROM photos WHERE path IN ({placeholders})",
            photos,
        ).fetchall() if photos else []

    classifications = {r["path"]: r["classification"] for r in rows if r["classification"]}
    registry_upsert(folder, len(photos))

    return {
        "photos": photos,
        "count": len(photos),
        "classifications": classifications,
        "companions": companions,
        "stats": db_stats(folder),
    }


@app.get("/api/analyze")
def get_analyze(path: str, folder: str):
    folder = require_valid_folder(folder)
    validate_path(path, folder)
    return analyze(path, folder)


@app.get("/api/image")
def get_image(path: str, folder: str):
    folder = require_valid_folder(folder)
    validate_path(path, folder)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    if is_raw(path):
        jpeg_bytes = raw_preview_jpeg(path)
        return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")
    return FileResponse(path)


@app.get("/api/thumbnail")
def get_thumbnail(path: str, folder: str):
    folder = require_valid_folder(folder)
    validate_path(path, folder)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    try:
        dst = build_thumbnail(path)
    except Exception:
        raise HTTPException(500, "Could not generate thumbnail")
    return FileResponse(dst, media_type="image/jpeg")


@app.post("/api/rotate")
def rotate_photo(req: RotateReq):
    if req.degrees not in (90, 180, 270):
        raise HTTPException(400, "degrees must be 90, 180, or 270")
    folder = require_valid_folder(req.folder)
    validate_path(req.path, folder)
    if not os.path.isfile(req.path):
        raise HTTPException(404, "Not found")
    if is_raw(req.path):
        raise HTTPException(400, "RAW files cannot be rotated in place")
    try:
        img = Image.open(req.path)
        img = ImageOps.exif_transpose(img)        # bake in any existing EXIF orientation first
        rotated = img.rotate(-req.degrees, expand=True)
        ext = Path(req.path).suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            rotated.save(req.path, format="JPEG", quality=95, optimize=True)
        else:
            rotated.save(req.path)
    except Exception as e:
        raise HTTPException(500, f"Could not rotate: {e}")
    analysis_cache.pop(req.path, None)
    raw_preview_cache.pop(req.path, None)
    thumb_path(req.path).unlink(missing_ok=True)
    return {"ok": True}


def _warm_image_cache(path: str) -> None:
    try:
        if is_raw(path):
            raw_preview_jpeg(path)
    except Exception:
        pass


@app.post("/api/prefetch")
def prefetch_images(req: PrefetchReq, background_tasks: BackgroundTasks):
    folder = require_valid_folder(req.folder)
    for path in req.paths[:5]:
        try:
            validate_path(path, folder)
            if os.path.isfile(path):
                background_tasks.add_task(_warm_image_cache, path)
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/classify")
def classify(req: ClassifyReq):
    if req.action not in ("keep", "delete", "unsure", "favorite"):
        raise HTTPException(400, "Invalid action")
    folder = require_valid_folder(req.folder)
    validate_path(req.path, folder)
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO photos (path, folder, name, classification, classified_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                classification=excluded.classification,
                classified_at=excluded.classified_at
            """,
            (req.path, folder, Path(req.path).name, req.action, now),
        )
    # Invalidate cache so cross_folder_duplicates refresh on next analyze
    analysis_cache.pop(req.path, None)
    return {"ok": True}


@app.get("/api/groups")
def get_groups(folder: str):
    folder = require_valid_folder(folder)
    photos = scan_photos(folder)
    if not photos:
        return {"groups": []}

    placeholders = ",".join("?" * len(photos))
    with get_db() as conn:
        db_rows = conn.execute(
            f"SELECT path, dhash, classification FROM photos WHERE path IN ({placeholders})",
            photos,
        ).fetchall()

    existing = {r["path"]: dict(r) for r in db_rows}

    for path in photos:
        if not existing.get(path, {}).get("dhash"):
            dhash = compute_dhash(path)
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO photos (path, folder, name, dhash) VALUES (?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET dhash=excluded.dhash",
                    (path, folder, Path(path).name, dhash),
                )
            existing.setdefault(path, {})["dhash"] = dhash

    rows = [{"path": p, "dhash": existing.get(p, {}).get("dhash", "")} for p in photos]
    clusters = cluster_by_dhash(rows)

    with get_db() as conn:
        class_map = {
            r["path"]: r["classification"]
            for r in conn.execute(
                f"SELECT path, classification FROM photos WHERE path IN ({placeholders})",
                photos,
            ).fetchall()
        }

    return {
        "groups": [
            [{"path": p, "classification": class_map.get(p)} for p in cluster]
            for cluster in clusters
        ]
    }


@app.post("/api/execute")
def execute(req: ExecuteReq):
    folder = require_valid_folder(req.folder)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT path, raw_companion FROM photos WHERE folder=? AND classification='delete'",
            (folder,),
        ).fetchall()

    trash = Path(folder) / "_trash"
    trash.mkdir(exist_ok=True)
    moved, raw_moved, errors = [], [], []

    for row in rows:
        src = Path(row["path"])
        if not src.exists():
            continue

        name = _move_to_trash(src, trash)
        if name:
            moved.append(name)
            analysis_cache.pop(str(src), None)
        else:
            errors.append({"name": src.name, "error": "could not move"})
            continue

        raw_path = row["raw_companion"] or find_raw_companion(str(src))
        if raw_path:
            raw_src = Path(raw_path)
            if raw_src.exists():
                raw_name = _move_to_trash(raw_src, trash)
                if raw_name:
                    raw_moved.append(raw_name)
                else:
                    errors.append({"name": raw_src.name, "error": "could not move RAW"})

    registry_upsert(folder, len(scan_photos(folder)))
    return {"moved": moved, "raw_moved": raw_moved, "errors": errors, "trash": str(trash)}


@app.post("/api/execute_unsure")
def execute_unsure(req: ExecuteReq):
    folder = require_valid_folder(req.folder)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT path, raw_companion FROM photos WHERE folder=? AND classification='unsure'",
            (folder,),
        ).fetchall()

    unsure_dir = Path(folder) / "_unsure"
    unsure_dir.mkdir(exist_ok=True)
    moved, raw_moved, errors = [], [], []

    for row in rows:
        src = Path(row["path"])
        if not src.exists():
            continue

        name = _move_to_trash(src, unsure_dir)
        if name:
            moved.append(name)
            analysis_cache.pop(str(src), None)
        else:
            errors.append({"name": src.name, "error": "could not move"})
            continue

        raw_path = row["raw_companion"] or find_raw_companion(str(src))
        if raw_path:
            raw_src = Path(raw_path)
            if raw_src.exists():
                raw_name = _move_to_trash(raw_src, unsure_dir)
                if raw_name:
                    raw_moved.append(raw_name)
                else:
                    errors.append({"name": raw_src.name, "error": "could not move RAW"})

    registry_upsert(folder, len(scan_photos(folder)))
    return {"moved": moved, "raw_moved": raw_moved, "errors": errors, "unsure_dir": str(unsure_dir)}


@app.get("/api/stats")
def get_stats(folder: str, classification: str = ""):
    folder = require_valid_folder(folder)
    photos = scan_photos(folder)
    total = len(photos)
    empty = {"total": total, "matched": 0, "with_exif": 0,
             "focal_lengths": [], "apertures": [], "isos": [], "shutter_speeds": []}
    if not photos:
        return empty

    placeholders = ",".join("?" * len(photos))
    with get_db() as conn:
        if classification == "unclassified":
            classified = {r["path"] for r in conn.execute(
                f"SELECT path FROM photos WHERE path IN ({placeholders}) AND classification IS NOT NULL",
                photos,
            ).fetchall()}
            subset = [p for p in photos if p not in classified]
        elif classification in ("favorite", "keep", "delete", "unsure"):
            subset = [r["path"] for r in conn.execute(
                f"SELECT path FROM photos WHERE path IN ({placeholders}) AND classification=?",
                photos + [classification],
            ).fetchall()]
        else:
            subset = photos

        matched = len(subset)
        if not subset:
            return {**empty, "matched": 0}

        sp = ",".join("?" * len(subset))
        rows = conn.execute(
            f"SELECT path, exif_json FROM photos WHERE path IN ({sp}) AND exif_json IS NOT NULL",
            subset,
        ).fetchall()

    EXPOSURE_KEYS = ("FocalLength", "FNumber", "ISOSpeedRatings", "ExposureTime")

    # Re-read EXIF from disk for photos missing exposure fields — happens when photos were
    # analyzed before the ExifIFD fix. Update the DB so this only runs once per photo.
    stale: list[tuple[str, str]] = []
    refreshed: dict[str, dict] = {}
    for row in rows:
        try:
            exif = json.loads(row["exif_json"])
        except Exception:
            continue
        if not any(exif.get(k) for k in EXPOSURE_KEYS) and os.path.isfile(row["path"]):
            fresh = get_exif(row["path"])
            if any(fresh.get(k) for k in EXPOSURE_KEYS):
                refreshed[row["path"]] = fresh
                stale.append((json.dumps(fresh), row["path"]))

    if stale:
        with get_db() as conn:
            for exif_json, path in stale:
                conn.execute("UPDATE photos SET exif_json=? WHERE path=?", (exif_json, path))

    focal_lengths: dict[str, int] = {}
    apertures: dict[str, int] = {}
    isos: dict[str, int] = {}
    shutter_speeds: dict[str, int] = {}
    with_exif = 0

    for row in rows:
        try:
            exif = refreshed.get(row["path"]) or json.loads(row["exif_json"])
        except Exception:
            continue
        has = False
        if v := exif.get("FocalLength"):
            focal_lengths[v] = focal_lengths.get(v, 0) + 1
            has = True
        if v := exif.get("FNumber"):
            apertures[v] = apertures.get(v, 0) + 1
            has = True
        if v := exif.get("ISOSpeedRatings"):
            isos[v] = isos.get(v, 0) + 1
            has = True
        if v := exif.get("ExposureTime"):
            shutter_speeds[v] = shutter_speeds.get(v, 0) + 1
            has = True
        if has:
            with_exif += 1

    def sorted_items(d: dict, key_fn) -> list:
        items = [{"label": k, "count": v} for k, v in d.items()]
        try:
            items.sort(key=lambda x: key_fn(x["label"]))
        except Exception:
            items.sort(key=lambda x: -x["count"])
        return items

    def parse_shutter(s: str) -> float:
        t = s.replace("s", "")
        if "/" in t:
            n, d = t.split("/")
            return float(n) / float(d)
        return float(t)

    return {
        "total": total,
        "matched": matched,
        "with_exif": with_exif,
        "focal_lengths": sorted_items(focal_lengths, lambda s: float(s.replace("mm", ""))),
        "apertures":     sorted_items(apertures,     lambda s: float(s.replace("f/", ""))),
        "isos":          sorted_items(isos,           lambda s: int(s)),
        "shutter_speeds":sorted_items(shutter_speeds, parse_shutter),
    }


@app.get("/api/browse")
def browse(path: str = ""):
    target = Path(path).expanduser() if path else Path.home()
    target = target.resolve()
    if not target.is_dir():
        raise HTTPException(400, "Not a directory")
    try:
        dirs = sorted(
            [{"name": e.name, "path": str(e)} for e in target.iterdir()
             if e.is_dir() and not e.name.startswith(".")],
            key=lambda d: d["name"].lower(),
        )
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    parent = str(target.parent) if target != target.parent else None
    return {"path": str(target), "parent": parent, "dirs": dirs}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

init_db()

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=True)
