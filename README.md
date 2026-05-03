# Photo Sorter

A keyboard-first local web app for quickly triaging photo libraries. Classify photos at speed, group burst shots, and detect blurry images — nothing is permanently deleted until you explicitly confirm.

## Features

- **Keyboard-driven** — `K` keep · `F` favorite · `U` unsure · `D` delete · `←/→` navigate · `Z` undo · `G` groups
- **Blur detection** — Laplacian edge variance flags likely-blurry photos with a suggested action
- **Similarity groups** — perceptual hashing (dHash) clusters burst shots and near-duplicates; classify a whole group at once
- **Persistent state** — classifications and analysis are stored in a SQLite database (`_photo_sorter.db`) inside each folder, so reopening the same folder restores exactly where you left off
- **File hashing** — SHA-256 stored per photo for exact duplicate identification
- **Safe deletion** — marked photos are moved to a `_trash/` subfolder, never permanently removed until you empty it manually
- **Filesystem browser** — navigate to any folder from the UI without typing paths

## Requirements

- Python 3.9+
- pip

## Setup

```bash
cd photo_sorter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
source .venv/bin/activate
python app.py
```

Then open [http://localhost:8765](http://localhost:8765) in your browser.

## Workflow

1. **Open a folder** — type the path or use the Browse button
2. **Sort** — use keyboard shortcuts to classify each photo; the app advances automatically
3. **Check similar groups** — press `G` to see clusters of visually similar photos; classify a whole burst at once
4. **Review** — click "Review & Finish" to see a summary and a grid of everything marked for deletion
5. **Confirm** — photos marked delete are moved to `_trash/` inside the source folder

## File structure

```
photo_sorter/
├── app.py              # FastAPI backend
├── requirements.txt
└── static/
    └── index.html      # Alpine.js + Tailwind frontend (single file)
```

At runtime, each sorted folder gets:

```
your-photos/
├── _photo_sorter.db    # SQLite: analysis cache, hashes, classifications
└── _trash/             # photos moved here on confirm (recoverable)
```

## How blur detection works

Each photo is resized to 512×512, converted to grayscale, and passed through PIL's `FIND_EDGES` filter. The variance of the result is the sharpness score — low variance means soft/uniform edges (blurry). The threshold is 500; photos below 100 are flagged "Very blurry".

## How similarity grouping works

A difference hash (dHash) is computed for each photo: resize to 9×8 grayscale, compare adjacent pixel columns to produce a 64-bit fingerprint. Photos within a Hamming distance of 10 bits are considered similar. Grouping uses a greedy O(n²) pass — fast enough for typical library sizes (hundreds to low thousands of photos).

Hashes are computed once and stored in `_photo_sorter.db`, so the Groups screen is instant on subsequent opens.
