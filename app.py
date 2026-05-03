import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ExifTags
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

# RAW formats paired with viewable images (same stem, different extension)
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
BLUR_THRESHOLD = 500.0
SIMILARITY_THRESHOLD = 10

# Per-project DBs live inside each folder; the projects registry lives here.
REGISTRY_PATH = Path.home() / ".photo_sorter" / "projects.json"

# In-memory analysis cache — keyed by absolute path, shared across folders.
analysis_cache: dict = {}


# ── Projects registry ─────────────────────────────────────────────────────────

def load_registry() -> list:
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return []


def save_registry(projects: list) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(projects, indent=2))


def registry_upsert(folder: str, total: int, stats: dict) -> None:
    projects = load_registry()
    entry = next((p for p in projects if p["folder"] == folder), None)
    record = {
        "folder": folder,
        "name": Path(folder).name,
        "last_opened": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "stats": stats,
    }
    if entry:
        entry.update(record)
    else:
        projects.insert(0, record)
    save_registry(projects[:20])


# ── Per-folder SQLite ─────────────────────────────────────────────────────────

def init_db(folder: str) -> None:
    db_path = Path(folder) / "_photo_sorter.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            path            TEXT PRIMARY KEY,
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
    """)
    # Migration: add column for databases created before RAW support
    try:
        conn.execute("ALTER TABLE photos ADD COLUMN raw_companion TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.close()


def get_db(folder: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(folder) / "_photo_sorter.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_stats(folder: str) -> dict:
    try:
        with get_db(folder) as conn:
            rows = conn.execute(
                "SELECT classification, COUNT(*) AS n FROM photos "
                "WHERE classification IS NOT NULL GROUP BY classification"
            ).fetchall()
        return {r["classification"]: r["n"] for r in rows}
    except Exception:
        return {}


# ── Security ──────────────────────────────────────────────────────────────────

def validate_path(path: str, folder: str) -> None:
    """Raise 403 unless path is strictly inside folder."""
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

def compute_dhash(path: str) -> str:
    try:
        img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
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
        img = Image.open(path)
        img.thumbnail((512, 512), Image.LANCZOS)
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        return float(np.array(edges, dtype=np.float64).var())
    except Exception:
        return -1.0


def get_exif(path: str) -> dict:
    try:
        raw = Image.open(path).getexif()
        if not raw:
            return {}
        wanted = {"Make", "Model", "DateTime", "DateTimeOriginal"}
        result = {}
        for tag_id, val in raw.items():
            tag = ExifTags.TAGS.get(tag_id, "")
            if tag in wanted:
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                result[tag] = str(val)
        return result
    except Exception:
        return {}


def find_raw_companion(path: str) -> str:
    """Return the path of a RAW sidecar with the same stem, or empty string."""
    p = Path(path)
    for ext in RAW_EXTENSIONS:
        for candidate in (p.with_suffix(ext), p.with_suffix(ext.upper())):
            if candidate.exists() and candidate.resolve() != p.resolve():
                return str(candidate)
    return ""


def analyze(path: str, folder: str) -> dict:
    if path in analysis_cache:
        return analysis_cache[path]

    f = Path(path)
    stat = f.stat()
    score = compute_blur_score(path)
    exif = get_exif(path)
    dhash = compute_dhash(path)
    file_hash = compute_file_hash(path)

    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        w, h = 0, 0

    raw_companion = find_raw_companion(path)

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
    }
    analysis_cache[path] = result

    with get_db(folder) as conn:
        conn.execute(
            """
            INSERT INTO photos
                (path, name, size_mb, width, height, blur_score,
                 suggest_delete, file_hash, dhash, exif_json, raw_companion)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                size_mb=excluded.size_mb, width=excluded.width,
                height=excluded.height, blur_score=excluded.blur_score,
                suggest_delete=excluded.suggest_delete,
                file_hash=excluded.file_hash, dhash=excluded.dhash,
                exif_json=excluded.exif_json, raw_companion=excluded.raw_companion
            """,
            (path, f.name, result["size_mb"], w, h, round(score, 1),
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
    return sorted(
        str(f) for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ── Models ────────────────────────────────────────────────────────────────────

class ClassifyReq(BaseModel):
    folder: str
    path: str
    action: str


class ExecuteReq(BaseModel):
    folder: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    return {"projects": load_registry()}


@app.get("/api/folder")
def open_folder(path: str):
    folder = require_valid_folder(path)
    init_db(folder)

    photos = scan_photos(folder)

    # Eagerly find RAW companions for every photo (fast — just path checks, no image decode)
    companions: dict = {}
    for photo_path in photos:
        raw = find_raw_companion(photo_path)
        if raw:
            companions[photo_path] = raw

    # Persist companions to DB (upsert only those found, don't overwrite existing full analysis)
    if companions:
        with get_db(folder) as conn:
            for photo_path, raw_path in companions.items():
                conn.execute(
                    "INSERT INTO photos (path, name, raw_companion) VALUES (?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET raw_companion=excluded.raw_companion",
                    (photo_path, Path(photo_path).name, raw_path),
                )

    classifications = {}
    if photos:
        placeholders = ",".join("?" * len(photos))
        with get_db(folder) as conn:
            rows = conn.execute(
                f"SELECT path, classification FROM photos WHERE path IN ({placeholders})",
                photos,
            ).fetchall()
        classifications = {r["path"]: r["classification"] for r in rows if r["classification"]}

    stats = db_stats(folder)
    registry_upsert(folder, len(photos), stats)

    return {
        "photos": photos,
        "count": len(photos),
        "classifications": classifications,
        "companions": companions,
        "stats": stats,
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
    return FileResponse(path)


@app.post("/api/classify")
def classify(req: ClassifyReq):
    if req.action not in ("keep", "delete", "unsure", "favorite"):
        raise HTTPException(400, "Invalid action")
    folder = require_valid_folder(req.folder)
    validate_path(req.path, folder)
    now = datetime.now(timezone.utc).isoformat()
    with get_db(folder) as conn:
        conn.execute(
            """
            INSERT INTO photos (path, name, classification, classified_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                classification=excluded.classification,
                classified_at=excluded.classified_at
            """,
            (req.path, Path(req.path).name, req.action, now),
        )
    return {"ok": True}


@app.get("/api/groups")
def get_groups(folder: str):
    folder = require_valid_folder(folder)
    photos = scan_photos(folder)
    if not photos:
        return {"groups": []}

    placeholders = ",".join("?" * len(photos))
    with get_db(folder) as conn:
        db_rows = conn.execute(
            f"SELECT path, dhash, classification FROM photos WHERE path IN ({placeholders})",
            photos,
        ).fetchall()

    existing = {r["path"]: dict(r) for r in db_rows}

    for path in photos:
        if not existing.get(path, {}).get("dhash"):
            dhash = compute_dhash(path)
            with get_db(folder) as conn:
                conn.execute(
                    "INSERT INTO photos (path, name, dhash) VALUES (?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET dhash=excluded.dhash",
                    (path, Path(path).name, dhash),
                )
            existing.setdefault(path, {})["dhash"] = dhash

    rows = [{"path": p, "dhash": existing.get(p, {}).get("dhash", "")} for p in photos]
    clusters = cluster_by_dhash(rows)

    with get_db(folder) as conn:
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


def _move_to_trash(src: Path, trash: Path) -> str | None:
    """Move src into trash, avoiding collisions. Returns destination name or None on error."""
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


@app.post("/api/execute")
def execute(req: ExecuteReq):
    folder = require_valid_folder(req.folder)
    with get_db(folder) as conn:
        rows = conn.execute(
            "SELECT path, raw_companion FROM photos WHERE classification='delete'"
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
        else:
            errors.append({"name": src.name, "error": "could not move"})
            continue  # skip companion if main file failed

        # Move RAW companion — prefer DB value, fall back to live filesystem check
        raw_path = row["raw_companion"] or find_raw_companion(str(src))
        if raw_path:
            raw_src = Path(raw_path)
            if raw_src.exists():
                raw_name = _move_to_trash(raw_src, trash)
                if raw_name:
                    raw_moved.append(raw_name)
                else:
                    errors.append({"name": raw_src.name, "error": "could not move RAW"})

    registry_upsert(folder, len(scan_photos(folder)), db_stats(folder))
    return {"moved": moved, "raw_moved": raw_moved, "errors": errors, "trash": str(trash)}


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False)
