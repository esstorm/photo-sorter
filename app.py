import hashlib
import json
import os
import shutil
import sqlite3
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
BLUR_THRESHOLD = 500.0
SIMILARITY_THRESHOLD = 10  # max Hamming bits apart to be considered "similar"

session: dict = {"folder": None, "photos": [], "classifications": {}, "db_path": None}
analysis_cache: dict = {}


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(folder: str) -> str:
    db_path = str(Path(folder) / "_photo_sorter.db")
    conn = sqlite3.connect(db_path)
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
            classified_at   TEXT
        );
    """)
    conn.commit()
    conn.close()
    return db_path


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(session["db_path"], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Hashing & analysis ────────────────────────────────────────────────────────

def compute_dhash(path: str) -> str:
    """8x8 difference hash — resize to 9×8, compare adjacent columns."""
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


def analyze(path: str) -> dict:
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
    }
    analysis_cache[path] = result

    if session["db_path"]:
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO photos
                    (path, name, size_mb, width, height, blur_score,
                     suggest_delete, file_hash, dhash, exif_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    size_mb=excluded.size_mb, width=excluded.width,
                    height=excluded.height, blur_score=excluded.blur_score,
                    suggest_delete=excluded.suggest_delete,
                    file_hash=excluded.file_hash, dhash=excluded.dhash,
                    exif_json=excluded.exif_json
                """,
                (path, f.name, result["size_mb"], w, h, round(score, 1),
                 int(result["suggest_delete"]), file_hash, dhash, json.dumps(exif)),
            )

    return result


def within_folder(path: str) -> bool:
    if not session["folder"]:
        return False
    return os.path.realpath(path).startswith(os.path.realpath(session["folder"]))


# ── Similarity groups ─────────────────────────────────────────────────────────

def cluster_by_dhash(rows: list) -> list:
    """Greedy O(n²) grouping. Fine for typical photo collections."""
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


# ── Pydantic models ───────────────────────────────────────────────────────────

class FolderReq(BaseModel):
    folder: str


class ClassifyReq(BaseModel):
    path: str
    action: str


# ── Routes ────────────────────────────────────────────────────────────────────

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


@app.post("/api/folder")
def set_folder(req: FolderReq):
    folder = req.folder.strip()
    if not os.path.isdir(folder):
        raise HTTPException(400, "Directory not found")

    session["folder"] = folder
    session["db_path"] = init_db(folder)
    analysis_cache.clear()

    photos = sorted(
        str(f) for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    session["photos"] = photos

    classifications = {}
    if photos:
        with db_conn() as conn:
            placeholders = ",".join("?" * len(photos))
            rows = conn.execute(
                f"SELECT path, classification FROM photos WHERE path IN ({placeholders})",
                photos,
            ).fetchall()
        classifications = {r["path"]: r["classification"] for r in rows if r["classification"]}

    session["classifications"] = classifications
    return {"photos": photos, "count": len(photos), "classifications": classifications}


@app.get("/api/analyze")
def get_analyze(path: str):
    if not within_folder(path):
        raise HTTPException(403, "Access denied")
    return analyze(path)


@app.get("/api/image")
def get_image(path: str):
    if not within_folder(path):
        raise HTTPException(403, "Access denied")
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path)


@app.post("/api/classify")
def classify(req: ClassifyReq):
    if req.action not in ("keep", "delete", "unsure", "favorite"):
        raise HTTPException(400, "Invalid action")
    if not within_folder(req.path):
        raise HTTPException(403, "Access denied")
    session["classifications"][req.path] = req.action
    now = datetime.now(timezone.utc).isoformat()
    with db_conn() as conn:
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
def get_groups():
    photos = session["photos"]
    if not photos or not session["db_path"]:
        return {"groups": []}

    placeholders = ",".join("?" * len(photos))
    with db_conn() as conn:
        db_rows = conn.execute(
            f"SELECT path, dhash, classification FROM photos WHERE path IN ({placeholders})",
            photos,
        ).fetchall()

    existing = {r["path"]: {"dhash": r["dhash"], "classification": r["classification"]} for r in db_rows}

    # Compute dhash for any photos not yet stored
    needs_hash = [p for p in photos if not existing.get(p, {}).get("dhash")]
    for path in needs_hash:
        dhash = compute_dhash(path)
        f = Path(path)
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO photos (path, name, dhash) VALUES (?,?,?)
                ON CONFLICT(path) DO UPDATE SET dhash=excluded.dhash
                """,
                (path, f.name, dhash),
            )
        existing.setdefault(path, {})["dhash"] = dhash

    rows = [{"path": p, "dhash": existing.get(p, {}).get("dhash", "")} for p in photos]
    clusters = cluster_by_dhash(rows)

    # Enrich with live classifications
    result = []
    for cluster in clusters:
        result.append([
            {"path": p, "classification": session["classifications"].get(p)}
            for p in cluster
        ])

    return {"groups": result}


@app.post("/api/execute")
def execute():
    folder = session["folder"]
    if not folder:
        raise HTTPException(400, "No folder set")
    trash = Path(folder) / "_trash"
    trash.mkdir(exist_ok=True)
    moved, errors = [], []
    for path, action in session["classifications"].items():
        if action != "delete":
            continue
        src = Path(path)
        dst = trash / src.name
        counter = 1
        while dst.exists():
            dst = trash / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            shutil.move(str(src), str(dst))
            moved.append(src.name)
        except Exception as e:
            errors.append({"name": src.name, "error": str(e)})
    return {"moved": moved, "errors": errors, "trash": str(trash)}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False)
