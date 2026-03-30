# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the development server:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Or directly:
```bash
python app.py
```

**Activate the local virtual environment (Windows):**
```bash
source myenv/Scripts/activate
```

## Architecture

This is a single-file FastAPI application (`app.py`) that implements a multi-step wall mockup tool. There are no separate modules, templates, or static files — all HTML, CSS, and JavaScript is rendered inline as Python f-strings.

### User flow (3 steps)

1. **`GET /`** — Upload form. User provides a room photo, an artwork photo, a wall measurement in inches, and print orientation (H/V).
2. **`POST /mockup/picker`** — Validates and stores both images server-side (in-memory `_store` dict with UUID keys), then serves an interactive canvas page where the user clicks two reference points on the room photo to establish a pixels-per-inch scale.
3. **`POST /mockup/editor`** — Receives the two clicked pixel coordinates, computes `ppi_orig = span_px / wall_measurement`, then serves a full-viewport editor where the user drags their artwork overlay, sets print size in inches, adjusts shadow direction, and downloads the composited mockup. The download is done client-side via Canvas `toBlob()`.

### Image storage

Images are stored in a module-level `_store: dict[str, tuple[bytes, str, float]]` (UUID → bytes, MIME, timestamp). Entries expire after 1 hour (`_STORE_TTL = 3600`). Pruning happens lazily on each `_save_image()` call. Images are served via `GET /mockup/img/{uid}`.

**Important:** The server never trusts image data from form fields — dimensions and pixel data are always recomputed from `_store` to prevent tampering.

### Key constraints
- Max upload: 10 MB per file
- Supported formats: JPEG, PNG, TIFF, BMP, WebP, and optionally HEIC/HEIF (if `pillow-heif` is installed — app works without it)
- The picker GET route (`GET /mockup/picker`) exists solely to support the "Re-mark Wall" back-link without triggering a browser resubmit warning

### No tests, no linter config
There is no test suite or linter configuration in this repo.
