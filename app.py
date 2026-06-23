from __future__ import annotations

import base64
import html as html_mod
import json
import os
import logging
import re
import time
import urllib.parse
import urllib.request
import uuid
from io import BytesIO

# ── Load SMTP.env for local dev ───────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SMTP.env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            # Support both KEY=value and KEY:value formats
            for _sep in ("=", ":"):
                if _sep in _line:
                    _k, _, _v = _line.partition(_sep)
                    os.environ.setdefault(_k.strip(), _v.strip())
                    break

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from PIL import Image, ImageOps

# ── Optional HEIF/HEIC support ────────────────────────────────────────────────
try:
    from pillow_heif import register_heif_opener          # type: ignore
    register_heif_opener()
    _HEIF_OK = True
except ImportError:
    _HEIF_OK = False

# ── App & constants ───────────────────────────────────────────────────────────
app = FastAPI(title="Wall Mockup Tool")

class FrameAllowMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        slug = request.query_params.get("gallery", "")
        if slug:
            tenant = _get_tenant(slug)
            origins = tenant.get("allowed_origins") or []
        else:
            origins = []
        csp = ("frame-ancestors 'self' " + " ".join(origins)) if origins else "frame-ancestors 'none'"
        response.headers["Content-Security-Policy"] = csp
        return response

app.add_middleware(FrameAllowMiddleware)

MAX_FILE_SIZE = 10 * 1024 * 1024          # 10 MB
ALLOWED_EXT: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
if _HEIF_OK:
    ALLOWED_EXT |= {".heic", ".heif"}

_MIME_MAP: dict[str, str] = {
    "JPEG":  "image/jpeg",
    "PNG":   "image/png",
    "GIF":   "image/gif",
    "WEBP":  "image/webp",
    "BMP":   "image/bmp",
    "TIFF":  "image/tiff",
    "HEIF":  "image/heic",
    "HEIC":  "image/heic",
}

# ── Server-side image store ───────────────────────────────────────────────────
# uid → (raw_bytes, mime_string, monotonic_timestamp)
_store: dict[str, tuple[bytes, str, float]] = {}
_STORE_TTL = 3_600  # seconds — prune uploads older than 1 hour

# ── SMTP config (loaded from env) ─────────────────────────────────────────────
_FROM_EMAIL  = os.environ.get("SMTP_USER", "info@kenhoehn.ca")
_RESEND_KEY  = os.environ.get("RESEND_API_KEY", "")

logger = logging.getLogger("wallymock")

# ── Supabase client ───────────────────────────────────────────────────────────
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_sb = None
if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        from supabase import create_client
        _sb = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception as _e:
        logger.warning("Supabase init failed: %s", _e)

# Simple in-memory tenant cache (5-minute TTL) to avoid a DB hit on every request
_tenant_cache: dict[str, tuple[dict, float]] = {}
_TENANT_TTL = 300

def _get_tenant(slug: str) -> dict:
    """Fetch tenant config from Supabase with short-lived cache. Falls back to defaults."""
    now = time.monotonic()
    if slug in _tenant_cache:
        data, ts = _tenant_cache[slug]
        if now - ts < _TENANT_TTL:
            return data
    if _sb and slug:
        try:
            res = _sb.table("tenants").select("*").eq("slug", slug).eq("subscription_status", "active").single().execute()
            if res.data:
                _tenant_cache[slug] = (res.data, now)
                return res.data
        except Exception as e:
            logger.warning("Tenant lookup failed for %r: %s", slug, e)
    # Default fallback — keeps Ken's existing behaviour if Supabase is unreachable
    return {
        "slug": slug or "kenhoehn",
        "gallery_name": "Ken Hoehn Gallery",
        "phone": "403-675-6677",
        "email": "info@kenhoehn.ca",
        "bcc_email": "info@kenhoehn.ca",
        "email_intro": (
            "This is a visual of the beautiful Ken Hoehn picture you chose for your home or office. "
            "We\u2019re really excited to have the piece on your wall, right where you have it placed in "
            "the mock up, in that exact size!"
        ),
        "email_body": (
            "We encourage you to call us at 403-675-6677 so we can answer additional questions you "
            "might have and make recommendations for the finish you\u2019d like, clarify shipping options, "
            "offer production time estimates and present a great way for you to save money with our "
            "incredible collector program. You can also email us at info@kenhoehn.ca if you like."
        ),
        "email_signature": "Thanks so much,\nKen",
        "subscription_status": "active",
    }


def _save_image(data: bytes, mime: str) -> str:
    """Persist image bytes server-side; return a fresh UUID key."""
    uid = str(uuid.uuid4())
    now = time.monotonic()
    _store[uid] = (data, mime, now)
    # Prune stale entries while we're here
    cutoff = now - _STORE_TTL
    stale = [k for k, v in list(_store.items()) if v[2] < cutoff]
    for k in stale:
        _store.pop(k, None)
    return uid


def _load_image(uid: str) -> tuple[bytes, str]:
    """Retrieve stored image; 410 if expired or unknown."""
    entry = _store.get(uid)
    if not entry:
        raise HTTPException(
            410,
            "Your upload session has expired. Please start over from the beginning.",
        )
    return entry[0], entry[1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_mime(pil_img: Image.Image) -> str:
    """Return MIME type from a PIL image's format field."""
    return _MIME_MAP.get((pil_img.format or "").upper(), "image/jpeg")


_MAX_DIM = 1800  # cap longest edge to keep base64 payloads small

# Image keys for non-rectangular artwork that must be stored as PNG to preserve transparency.
# Add new keys here whenever Ken adds a circle/oval/custom-shape product.
_PNG_IMAGE_KEYS: set[str] = {
    "7c8ac3e2-f61d-4c24-aa49-e842af9f55d6",  # oval
    "e4e65e05-3a9b-47f2-835c-89dda531c9cb",  # non-rectangular
}


def _to_jpeg(pil_img: Image.Image, quality: int = 80) -> bytes:
    """Convert a PIL image to JPEG bytes, flattening any alpha channel onto white.
    Resizes so the longest edge is at most _MAX_DIM pixels."""
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[-1])
        pil_img = bg
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    if max(w, h) > _MAX_DIM:
        if w >= h:
            pil_img = pil_img.resize((_MAX_DIM, round(h * _MAX_DIM / w)), Image.LANCZOS)
        else:
            pil_img = pil_img.resize((round(w * _MAX_DIM / h), _MAX_DIM), Image.LANCZOS)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _to_png(pil_img: Image.Image) -> bytes:
    """Resize and save as PNG, preserving transparency for non-rectangular artwork."""
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode == "P":
        pil_img = pil_img.convert("RGBA")
    elif pil_img.mode not in ("RGBA", "RGB", "LA"):
        pil_img = pil_img.convert("RGBA")
    w, h = pil_img.size
    if max(w, h) > _MAX_DIM:
        if w >= h:
            pil_img = pil_img.resize((_MAX_DIM, round(h * _MAX_DIM / w)), Image.LANCZOS)
        else:
            pil_img = pil_img.resize((round(w * _MAX_DIM / h), _MAX_DIM), Image.LANCZOS)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


async def read_and_validate(upload: UploadFile) -> tuple[bytes, Image.Image]:
    """
    Read an UploadFile, enforce size / extension limits, and open with PIL.
    Returns (raw_bytes, PIL_image).
    Raises HTTPException(400) on any failure.
    """
    data = await upload.read()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            400,
            f"'{upload.filename}' is {len(data) // (1024*1024)} MB — "
            "the maximum allowed size is 10 MB.",
        )

    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXT:
        supported = ", ".join(sorted(ALLOWED_EXT))
        raise HTTPException(
            400,
            f"File type '{ext}' is not supported. "
            f"Please upload one of: {supported}.",
        )

    # Open and lightly validate the image (PIL is lazy; verify() forces header parse)
    try:
        pil = Image.open(BytesIO(data))
        pil.verify()                    # raises on truncated / corrupt files
    except Exception:
        raise HTTPException(
            400,
            f"'{upload.filename}' could not be opened as an image. "
            "The file may be corrupt or in an unsupported format.",
        )
    # verify() exhausts the internal stream — must re-open for further use
    pil = Image.open(BytesIO(data))
    return data, pil


# ── Error page ────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_error_handler(request, exc: HTTPException) -> HTMLResponse:  # type: ignore[override]
    return HTMLResponse(
        content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#f5f2ec">
<title>Error — Wall Mockup</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'DM Sans',sans-serif;font-weight:300;background:#f5f2ec;
       min-height:100vh;display:flex;flex-direction:column;align-items:center;
       justify-content:center;padding:40px 20px}}
  .card{{background:#fff;border:1px solid #d8d4cc;border-radius:18px;
         padding:44px 40px;max-width:480px;width:100%;text-align:center}}
  h1{{font-family:'DM Serif Display',serif;font-size:1.8rem;color:#c8440a;margin-bottom:12px}}
  p{{color:#555;line-height:1.65;font-size:0.93rem;margin-bottom:28px}}
  .code{{display:inline-block;background:#f5f2ec;border-radius:6px;
          padding:2px 10px;font-size:0.8rem;color:#8a8880;margin-bottom:18px}}
  a{{display:inline-block;padding:12px 28px;background:#c8440a;color:#fff;
     border-radius:9px;text-decoration:none;font-weight:500;font-size:0.92rem}}
  a:hover{{opacity:0.88}}
</style>
</head>
<body>
<div class="card">
  <h1>Something went wrong</h1>
  <div class="code">HTTP {exc.status_code}</div>
  <p>{html_mod.escape(str(exc.detail))}</p>
  <a href="/">← Start Over</a>
</div>
</body>
</html>""",
        status_code=exc.status_code,
    )


# ── Thumbnail helper endpoint ─────────────────────────────────────────────────

@app.post("/mockup/thumbnail")
async def make_thumbnail(file: UploadFile = File(...)) -> Response:
    """Accept any supported image, return a small JPEG thumbnail."""
    data, pil = await read_and_validate(file)
    pil = Image.open(BytesIO(_to_jpeg(pil)))
    pil.thumbnail((600, 300), Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ── Step 1 — Upload Form ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(gallery: str = Query("kenhoehn")) -> HTMLResponse:
    _html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#f5f2ec">
<title>WallyMock</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ink: #000; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #000; --border: #d8d4cc;
  }
  body {
    background: var(--paper); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-weight: 300;
    height: 100vh; overflow: hidden;
    display: grid; grid-template-columns: 1fr 400px;
  }
  .col-info {
    padding: 40px 48px 40px 60px;
    display: flex; flex-direction: column; justify-content: center;
    overflow: hidden;
  }
  .col-form {
    border-left: 1px solid var(--border); background: #fff;
    padding: 32px 36px;
    display: flex; flex-direction: column; justify-content: center;
    overflow-y: auto;
  }
  .logo {
    font-family: Georgia, serif;
    font-size: clamp(2.2rem, 4.5vw, 3.4rem);
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .tagline { font-size: 1.15rem; font-style: italic; margin-bottom: 36px; }

  .how-section { width: 100%; margin-bottom: 0; }
  .how-section h2 {
    font-family: Georgia, serif; font-size: 1.55rem;
    margin-bottom: 16px;
  }
  .steps { display: grid; grid-template-columns: 1fr; gap: 12px; }
  .step-card {
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 16px;
  }
  .step-num {
    width: 26px; height: 26px; border-radius: 50%; background: var(--accent); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 500; margin-bottom: 8px;
  }
  .step-card h3 { font-size: 1.08rem; font-weight: 500; margin-bottom: 4px; }
  .step-card p  { font-size: 0.95rem; line-height: 1.55; }

  .upload-card { width: 100%; }
  .upload-card h2 { font-family: Georgia, serif; font-size: 1.7rem; margin-bottom: 4px; }
  .upload-card .card-sub {
    font-size: 1rem; font-style: italic; margin-bottom: 20px;
  }
  label.field-label {
    display: block; font-size: 0.9rem; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 6px; margin-top: 16px;
  }
  label.field-label:first-of-type { margin-top: 0; }
  @media (max-width: 800px) {
    body { grid-template-columns: 1fr; height: auto; overflow-y: auto; }
    .col-info { padding: 36px 24px 24px; }
    .col-form { border-left: none; border-top: 1px solid var(--border); padding: 28px 24px 40px; justify-content: flex-start; }
    .steps { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    .logo { font-size: 2.8rem; }
    .tagline { font-size: 1.2rem; }
    .step-card h3 { font-size: 1.15rem; }
    .step-card p { font-size: 1.05rem; line-height: 1.6; }
    .upload-card h2 { font-size: 1.9rem; }
    .upload-card .card-sub { font-size: 1.05rem; }
    label.field-label { font-size: 1rem; }
    input[type="file"] { font-size: 1.1rem; padding: 13px 14px; }
    input[type="number"] { font-size: 1.1rem; padding: 13px 14px; }
    .orient-card .lbl { font-size: 1.08rem; }
    .orient-card .sublbl { font-size: 0.98rem; }
    button[type="submit"] { font-size: 1.2rem; padding: 16px; }
  }

  input[type="file"] {
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 1rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s; cursor: pointer;
  }
  input[type="number"] {
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 1rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s;
  }
  input:focus { border-color: var(--accent); }
  input.field-error { border-color: #c8440a; background: #fff5f2; animation: errorShake .3s; }
  @keyframes errorShake {
    0%,100%{transform:translateX(0)} 25%{transform:translateX(-6px)} 75%{transform:translateX(6px)}
  }

  .preview-thumb {
    max-width: 100%; max-height: 110px; border-radius: 0; margin-top: 10px;
    object-fit: cover; border: 1px solid var(--border); display: block;
  }

  .orient-row { display: flex; gap: 12px; margin-top: 6px; }
  .orient-row input[type="radio"] {
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
  }
  .orient-card {
    flex: 1; border: 2px solid var(--border); border-radius: 10px;
    padding: 13px 10px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s; background: var(--paper);
    user-select: none; display: block;
  }
  .orient-row input[type="radio"]:checked + .orient-card {
    border-color: var(--accent); background: #fff3ee;
  }
  .orient-card .icon  { font-size: 1.6rem; display: block; margin-bottom: 5px; line-height: 1; }
  .orient-card .lbl   { font-size: 1rem; font-weight: 500; }
  .orient-card .sublbl { font-size: 0.9rem; margin-top: 2px; }

  button[type="submit"] {
    margin-top: 28px; width: 100%; padding: 14px;
    background: var(--accent); color: #fff; border: none;
    border-radius: 9px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 1.05rem; font-weight: 500; cursor: pointer;
    transition: opacity .2s, transform .15s;
  }
  button[type="submit"]:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
  button[type="submit"]:disabled { opacity: 0.55; cursor: not-allowed; }
  .btn-spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
    border-radius: 50%; animation: btnSpin .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes btnSpin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="col-info">
  <div class="logo">WallyMock<sup style="font-size:0.45em;vertical-align:super;letter-spacing:0;">TM</sup></div>
  <p class="tagline">See any print to scale on any wall — in seconds</p>

  <div class="how-section">
    <h2>How it works</h2>
    <div class="steps">
      <div class="step-card">
        <div class="step-num" id="stepOneBadge">1</div>
        <h3>Upload your photos</h3>
        <p>Upload a photo of your room or wall, then upload your artwork or print file. Choose the print orientation — horizontal for wide prints, vertical for tall ones. Enter the real-world width or height of the wall you measured in inches.</p>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <h3>Mark the measurement</h3>
        <p>On the next screen, click two points on the room photo that correspond to your wall measurement — for example, the left and right edges of the wall, or the floor directly below to the ceiling directly above a single spot.</p>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <h3>Set print size</h3>
        <p>Enter the longest edge of your print in inches. The tool uses your marked measurement to calculate the exact scale and places the artwork on the wall at true size.</p>
      </div>
      <div class="step-card">
        <div class="step-num">4</div>
        <h3>Position &amp; download</h3>
        <p>Drag the print to your desired position on the wall. Adjust the shadow direction to match your room's lighting. When you're happy with the placement, click Download to save your mockup.</p>
      </div>
    </div>
  </div>
</div>

<div class="col-form">
<div class="upload-card">
  <h2>Create a Mockup</h2>
  <p class="card-sub">Step 1 of 4 — Upload &amp; configure</p>

  <form action="/mockup/picker?gallery=__GALLERY__" enctype="multipart/form-data" method="post" id="mainForm">
    <input type="hidden" name="gallery" value="__GALLERY__">

    <label class="field-label" for="roomFile">Room / Wall Photo</label>
    <input type="file" name="room_file" id="roomFile" accept="image/*"
           autocomplete="off" required
           onchange="previewFile(this,'roomPrev')">
    <div id="roomPrev"></div>

    <label class="field-label" for="artFile">Artwork / Print File</label>
    <input type="file" name="art_file" id="artFile" accept="image/*"
           autocomplete="off" required
           onchange="previewFile(this,'artPrev')">
    <div id="artPrev"></div>

    <label class="field-label">Print Orientation</label>
    <div class="orient-row">
      <input type="radio" name="orientation" id="orH" value="H" onchange="updateMeasLabel()">
      <label class="orient-card" for="orH">
        <span class="icon">⬛️</span>
        <div class="lbl">Horizontal</div>
        <div class="sublbl">Landscape / wide print</div>
      </label>

      <input type="radio" name="orientation" id="orV" value="V" onchange="updateMeasLabel()">
      <label class="orient-card" for="orV">
        <span class="icon">▮</span>
        <div class="lbl">Vertical</div>
        <div class="sublbl">Portrait / tall print</div>
      </label>
    </div>

    <div id="measWrap" style="display:none;margin-top:0;">
      <label class="field-label" id="measLabel" for="wallMeasInput"> </label>
      <input type="number" name="wall_measurement" id="wallMeasInput"
             step="0.1" min="1" max="9999" inputmode="decimal"
             placeholder="e.g. 96">
    </div>

    <button type="submit" id="submitBtn">Next: Mark the Measurement →</button>
  </form>
</div>
</div>

<script>
function previewFile(input, containerId) {
  const file = input.files[0];
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const tempImg = new Image();
    tempImg.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = tempImg.naturalWidth;
      canvas.height = tempImg.naturalHeight;
      canvas.getContext('2d').drawImage(tempImg, 0, 0);
      const img = document.createElement('img');
      img.src = canvas.toDataURL('image/jpeg', 0.85);
      img.className = 'preview-thumb';
      img.alt = 'Preview';
      container.appendChild(img);
    };
    tempImg.onerror = () => {
      // Browser can't decode this format (e.g. TIFF, HEIC) — ask the server
      const fd = new FormData();
      fd.append('file', file);
      fetch('/mockup/thumbnail', { method: 'POST', body: fd })
        .then(r => r.ok ? r.blob() : Promise.reject())
        .then(blob => {
          const img = document.createElement('img');
          img.src = URL.createObjectURL(blob);
          img.className = 'preview-thumb';
          img.alt = 'Preview';
          container.appendChild(img);
        })
        .catch(() => {});
    };
    tempImg.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function updateMeasLabel() {
  const isV  = document.getElementById('orV').checked;
  const wrap  = document.getElementById('measWrap');
  const label = document.getElementById('measLabel');
  label.textContent = isV ? 'Vertical wall height (inches)' : 'Horizontal wall width (inches)';
  wrap.style.display = 'block';
  document.getElementById('wallMeasInput').focus();
  checkStep1Complete();
}

function replaceFile(input, file) {
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
}

async function compressImage(file, maxDim = 2000, quality = 0.88) {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        let w = img.naturalWidth, h = img.naturalHeight;
        if (Math.max(w, h) > maxDim) {
          if (w >= h) { h = Math.round(h * maxDim / w); w = maxDim; }
          else        { w = Math.round(w * maxDim / h); h = maxDim; }
        }
        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        canvas.getContext('2d').drawImage(img, 0, 0, w, h);
        canvas.toBlob(blob => {
          resolve(new File([blob], file.name.replace(/\.[^.]+$/, '.jpg'), { type: 'image/jpeg' }));
        }, 'image/jpeg', quality);
      };
      img.onerror = () => resolve(file);
      img.src = e.target.result;
    };
    reader.onerror = () => resolve(file);
    reader.readAsDataURL(file);
  });
}

document.getElementById('mainForm').addEventListener('submit', async e => {
  e.preventDefault();
  const roomEl = document.getElementById('roomFile');
  const artEl  = document.getElementById('artFile');
  const measEl = document.getElementById('wallMeasInput');
  const orientOk = document.getElementById('orH').checked || document.getElementById('orV').checked;
  const missing = [
    [roomEl, roomEl.files.length === 0],
    [artEl,  artEl.files.length === 0],
    [measEl, measEl.value.trim() === ''],
  ].filter(([, bad]) => bad).map(([el]) => el);
  if (!orientOk || missing.length) {
    if (!orientOk) {
      document.querySelector('.orient-row').style.outline = '2px solid #c8440a';
      document.querySelector('.orient-row').style.borderRadius = '10px';
    }
    missing.forEach(el => {
      el.classList.add('field-error');
      el.addEventListener('change', () => el.classList.remove('field-error'), { once: true });
      el.addEventListener('input',  () => el.classList.remove('field-error'), { once: true });
    });
    if (missing.length) missing[0].focus();
    return;
  }
  const btn = document.getElementById('submitBtn');
  const THRESHOLD = 4 * 1024 * 1024;
  const roomFile = roomEl.files[0];
  const artFile  = artEl.files[0];
  const needsCompression = roomFile.size > THRESHOLD || artFile.size > THRESHOLD;
  btn.innerHTML = needsCompression ? '<span class="btn-spinner"></span>Compressing…' : '<span class="btn-spinner"></span>Uploading…';
  btn.disabled = true;
  if (needsCompression) {
    const [cRoom, cArt] = await Promise.all([
      roomFile.size > THRESHOLD ? compressImage(roomFile) : Promise.resolve(roomFile),
      artFile.size  > THRESHOLD ? compressImage(artFile)  : Promise.resolve(artFile),
    ]);
    replaceFile(roomEl, cRoom);
    replaceFile(artEl,  cArt);
    btn.innerHTML = '<span class="btn-spinner"></span>Uploading…';
  }
  document.getElementById('mainForm').submit();
});

['orH','orV'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    document.querySelector('.orient-row').style.outline = '';
  });
});

function checkStep1Complete() {
  const roomOk   = document.getElementById('roomFile').files.length > 0;
  const artOk    = document.getElementById('artFile').files.length > 0;
  const measOk   = document.getElementById('wallMeasInput').value.trim() !== '';
  const orientOk = document.getElementById('orH').checked || document.getElementById('orV').checked;
  const badge = document.getElementById('stepOneBadge');
  if (roomOk && artOk && measOk && orientOk) {
    badge.textContent = '✓';
    badge.style.background = '#16a34a';
  } else {
    badge.textContent = '1';
    badge.style.background = '';
  }
}
document.getElementById('roomFile').addEventListener('change', checkStep1Complete);
document.getElementById('artFile').addEventListener('change', checkStep1Complete);
document.getElementById('wallMeasInput').addEventListener('input', checkStep1Complete);
</script>
</body>
</html>"""
    return HTMLResponse(content=_html.replace("__GALLERY__", html_mod.escape(gallery)))


def _back_btn(ref: str, base_url: str = "") -> str:
    """Fixed top-left 'Back to Artwork' button, only rendered when ref is set."""
    if not ref or not base_url:
        return ""
    url = html_mod.escape(base_url + ref)
    return (
        f'<a href="{url}" '
        'style="position:fixed;top:12px;left:12px;z-index:9999;background:#fff;'
        'border:1px solid #d8d4cc;border-radius:6px;padding:8px 14px;font-size:0.88rem;'
        'text-decoration:none;color:#222;font-weight:500;'
        'box-shadow:0 1px 4px rgba(0,0,0,.12);" '
        'onclick="event.preventDefault();window.parent.location.href=this.href;">'
        '\u2190 Back to Artwork</a>'
    )


def _img_key(url: str) -> str:
    """Extract Squarespace CDN image key from a URL (second-to-last path segment)."""
    if not url:
        return ""
    try:
        parts = [p for p in urllib.parse.urlparse(url).path.split("/") if p]
        return parts[-2] if len(parts) >= 2 else ""
    except Exception:
        return ""


def _fetch_variants(ref: str, base_url: str = "") -> str:
    """Fetch Squarespace product JSON and return all variants as a JSON string."""
    if not ref or not ref.startswith("/") or not base_url:
        return "[]"
    try:
        url = base_url + ref + "?format=json-pretty"
        req = urllib.request.Request(url, headers={"User-Agent": "WallyMock/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read(2 * 1024 * 1024))
    except Exception as e:
        logging.warning("_fetch_variants failed for %s: %s", ref, e)
        return "[]"
    # Try multiple JSON paths Squarespace uses for variants
    raw = (
        data.get("variants")
        or data.get("item", {}).get("variants")
        or data.get("item", {}).get("structuredContent", {}).get("variants")
        or []
    )
    logging.info("_fetch_variants %s: top-level keys=%s raw_count=%d", ref, list(data.keys()), len(raw))
    item_id = data.get("item", {}).get("id", "")
    variants = []
    for v in raw:
        attrs = v.get("attributes", {})
        size_str = attrs.get("Size", "")
        frame = attrs.get("Frame", "")
        image_key = _img_key(v.get("mainImage", {}).get("assetUrl", ""))
        price_val = v.get("priceMoney", {}).get("value", "")
        try:
            price_str = f"CA${float(price_val):,.0f}" if price_val else ""
        except Exception:
            price_str = ""
        # Format A: "W x H in." or "W x H" (open editions, or limited with explicit dims)
        m = re.match(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)", size_str, re.IGNORECASE)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            long_edge = max(w, h)
            short_edge = min(w, h)
            orientation = attrs.get("Orientation")
            aspect = round(long_edge / short_edge, 4) if short_edge > 0 else 1.0
        else:
            # Format B: "N in." + "Aspect Ratio": "W:H Aspect Ratio"
            m2 = re.match(r"(\d+(?:\.\d+)?)\s*in\.", size_str, re.IGNORECASE)
            ar_m = re.match(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", attrs.get("Aspect Ratio", ""))
            if m2 and ar_m:
                long_edge = float(m2.group(1))
                ar_a, ar_b = float(ar_m.group(1)), float(ar_m.group(2))
                ar_norm = max(ar_a, ar_b) / min(ar_a, ar_b) if min(ar_a, ar_b) > 0 else 1.0
                aspect = round(ar_norm, 4)
                short_edge = round(long_edge / ar_norm, 2)
                w, h = long_edge, short_edge
                orientation = None
            else:
                # Non-numeric aspect ratio (e.g. "Oval Aspect") — store with null aspect
                m2 = re.match(r"(\d+(?:\.\d+)?)\s*in\.", size_str, re.IGNORECASE)
                if m2:
                    long_edge = float(m2.group(1))
                    w, h = long_edge, long_edge
                    short_edge = long_edge
                    aspect = None
                    orientation = None
                else:
                    logging.warning("_fetch_variants: unrecognised size format %r attrs=%r", size_str, attrs)
                    continue
        variants.append({
            "id": v.get("id", ""),
            "sku": v.get("sku", ""),
            "itemId": item_id,
            "size": size_str,
            "w": w,
            "h": h,
            "orientation": orientation,
            "price": price_str,
            "aspect": aspect,
            "long": long_edge,
            "frame": frame,
            "imageKey": image_key,
        })
    return json.dumps(variants)


# ── Step 1b — Prefill flow (artwork supplied via URL) ─────────────────────────

def _build_prefill_html(art_id: str, art_thumb_b64: str, ref: str = "", variants_json: str = "[]", frame: str = "", coa_field: str = "", image_key: str = "", gallery: str = "kenhoehn", base_url: str = "") -> str:
    """Return Step-1 HTML with artwork pre-loaded; client only supplies room + measurement."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#f5f2ec">
<title>WallyMock</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #000; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #000; --border: #d8d4cc;
  }}
  body {{
    background: var(--paper); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-weight: 300;
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    padding: 40px 20px;
  }}
  .card {{
    background: #fff; border: 1px solid var(--border); border-radius: 18px;
    padding: 40px 40px; max-width: 480px; width: 100%;
  }}
  .logo {{ font-family: Georgia, serif; font-size: 1.6rem; letter-spacing: -0.02em; margin-bottom: 4px; }}
  .card-sub {{ font-size: 1rem; font-style: italic; margin-bottom: 28px; }}
  .art-preview-box {{
    border: 1px solid var(--border); border-radius: 10px;
    background: var(--paper); padding: 12px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 14px;
  }}
  .art-preview-box img {{
    max-width: 90px; max-height: 90px; border-radius: 6px;
    object-fit: contain; flex-shrink: 0;
  }}
  .art-preview-text {{ font-size: 0.9rem; line-height: 1.5; }}
  .art-preview-text strong {{ display: block; margin-bottom: 2px; }}
  .art-check {{ color: #16a34a; font-size: 1.1rem; }}
  label.field-label {{
    display: block; font-size: 0.9rem; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 6px; margin-top: 20px;
  }}
  label.field-label:first-of-type {{ margin-top: 0; }}
  input[type="file"] {{
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 1rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s; cursor: pointer;
  }}
  input[type="number"] {{
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 1rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s;
  }}
  input:focus {{ border-color: var(--accent); }}
  input.field-error {{ border-color: #c8440a; background: #fff5f2; animation: errorShake .3s; }}
  @keyframes errorShake {{
    0%,100%{{transform:translateX(0)}} 25%{{transform:translateX(-6px)}} 75%{{transform:translateX(6px)}}
  }}
  .preview-thumb {{
    max-width: 100%; max-height: 110px; border-radius: 0; margin-top: 10px;
    object-fit: cover; border: 1px solid var(--border); display: block;
  }}
  .orient-row {{ display: flex; gap: 12px; margin-top: 6px; }}
  .orient-row input[type="radio"] {{
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
  }}
  .orient-card {{
    flex: 1; border: 2px solid var(--border); border-radius: 10px;
    padding: 13px 10px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s; background: var(--paper);
    user-select: none; display: block;
  }}
  .orient-row input[type="radio"]:checked + .orient-card {{
    border-color: var(--accent); background: #fff3ee;
  }}
  .orient-card .icon  {{ font-size: 1.6rem; display: block; margin-bottom: 5px; line-height: 1; }}
  .orient-card .lbl   {{ font-size: 1rem; font-weight: 500; }}
  .orient-card .sublbl {{ font-size: 0.9rem; margin-top: 2px; }}
  button[type="submit"] {{
    margin-top: 28px; width: 100%; padding: 14px;
    background: var(--accent); color: #fff; border: none;
    border-radius: 9px; font-family: inherit;
    font-size: 1.05rem; font-weight: 500; cursor: pointer;
    transition: opacity .2s, transform .15s;
  }}
  button[type="submit"]:hover:not(:disabled) {{ opacity: 0.88; transform: translateY(-1px); }}
  button[type="submit"]:disabled {{ opacity: 0.55; cursor: not-allowed; }}
  .btn-spinner {{
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
    border-radius: 50%; animation: btnSpin .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }}
  @keyframes btnSpin {{ to {{ transform: rotate(360deg); }} }}
  @media (max-width: 600px) {{
    .card {{ padding: 28px 20px; }}
    input[type="file"], input[type="number"] {{ font-size: 1.1rem; padding: 13px 14px; }}
    button[type="submit"] {{ font-size: 1.2rem; padding: 16px; }}
    label.field-label {{ font-size: 1rem; }}
  }}
</style>
</head>
<body>
{_back_btn(ref, base_url)}
<div class="card">
  <div class="logo">WallyMock<sup style="font-size:0.45em;vertical-align:super;letter-spacing:0;">TM</sup></div>
  <p class="card-sub">Step 1 of 4 — Upload &amp; configure</p>

  <div class="art-preview-box">
    <img src="data:image/jpeg;base64,{art_thumb_b64}" alt="Selected artwork">
    <div class="art-preview-text">
      <strong><span class="art-check">✓</span> Artwork pre-loaded</strong>
      Your selected print is ready. Just upload a photo of your wall and enter the measurement below.
    </div>
  </div>

  <form action="/mockup/picker_prefill?gallery={html_mod.escape(gallery)}" enctype="multipart/form-data" method="post" id="mainForm">
    <input type="hidden" name="art_id" value="{art_id}">
    <input type="hidden" name="ref" value="{html_mod.escape(ref)}">
    <input type="hidden" name="variants_json" value="{html_mod.escape(variants_json)}">
    <input type="hidden" name="frame" value="{html_mod.escape(frame)}">
    <input type="hidden" name="coa_field" value="{html_mod.escape(coa_field)}">
    <input type="hidden" name="image_key" value="{html_mod.escape(image_key)}">
    <input type="hidden" name="gallery" value="{html_mod.escape(gallery)}">

    <label class="field-label" for="roomFile">Room / Wall Photo</label>
    <input type="file" name="room_file" id="roomFile" accept="image/*"
           autocomplete="off"
           onchange="previewFile(this,'roomPrev')">
    <div id="roomPrev"></div>

    <label class="field-label">Print Orientation</label>
    <div class="orient-row">
      <input type="radio" name="orientation" id="orH" value="H" onchange="updateMeasLabel()">
      <label class="orient-card" for="orH">
        <span class="icon">⬛️</span>
        <div class="lbl">Horizontal</div>
        <div class="sublbl">Landscape / wide print</div>
      </label>
      <input type="radio" name="orientation" id="orV" value="V" onchange="updateMeasLabel()">
      <label class="orient-card" for="orV">
        <span class="icon">▮</span>
        <div class="lbl">Vertical</div>
        <div class="sublbl">Portrait / tall print</div>
      </label>
    </div>

    <div id="measWrap" style="display:none;margin-top:0;">
      <label class="field-label" id="measLabel" for="wallMeasInput"> </label>
      <input type="number" name="wall_measurement" id="wallMeasInput"
             step="0.1" min="1" max="9999" inputmode="decimal"
             placeholder="e.g. 96">
    </div>

    <button type="submit" id="submitBtn">Next: Mark the Measurement →</button>
  </form>
</div>

<script>
function updateMeasLabel() {{
  const isV  = document.getElementById('orV').checked;
  const wrap  = document.getElementById('measWrap');
  const label = document.getElementById('measLabel');
  label.textContent = isV ? 'Vertical wall height (inches)' : 'Horizontal wall width (inches)';
  wrap.style.display = 'block';
  document.getElementById('wallMeasInput').focus();
}}

function previewFile(input, containerId) {{
  const file = input.files[0];
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {{
    const tempImg = new Image();
    tempImg.onload = () => {{
      const canvas = document.createElement('canvas');
      canvas.width = tempImg.naturalWidth;
      canvas.height = tempImg.naturalHeight;
      canvas.getContext('2d').drawImage(tempImg, 0, 0);
      const img = document.createElement('img');
      img.src = canvas.toDataURL('image/jpeg', 0.85);
      img.className = 'preview-thumb';
      img.alt = 'Preview';
      container.appendChild(img);
    }};
    tempImg.onerror = () => {{
      const fd = new FormData();
      fd.append('file', file);
      fetch('/mockup/thumbnail', {{ method: 'POST', body: fd }})
        .then(r => r.ok ? r.blob() : Promise.reject())
        .then(blob => {{
          const img = document.createElement('img');
          img.src = URL.createObjectURL(blob);
          img.className = 'preview-thumb';
          img.alt = 'Preview';
          container.appendChild(img);
        }})
        .catch(() => {{}});
    }};
    tempImg.src = e.target.result;
  }};
  reader.readAsDataURL(file);
}}

function replaceFile(input, file) {{
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
}}

async function compressImage(file, maxDim = 2000, quality = 0.88) {{
  return new Promise(resolve => {{
    const reader = new FileReader();
    reader.onload = e => {{
      const img = new Image();
      img.onload = () => {{
        let w = img.naturalWidth, h = img.naturalHeight;
        if (Math.max(w, h) > maxDim) {{
          if (w >= h) {{ h = Math.round(h * maxDim / w); w = maxDim; }}
          else        {{ w = Math.round(w * maxDim / h); h = maxDim; }}
        }}
        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        canvas.getContext('2d').drawImage(img, 0, 0, w, h);
        canvas.toBlob(blob => {{
          resolve(new File([blob], file.name.replace(/\.[^.]+$/, '.jpg'), {{ type: 'image/jpeg' }}));
        }}, 'image/jpeg', quality);
      }};
      img.onerror = () => resolve(file);
      img.src = e.target.result;
    }};
    reader.onerror = () => resolve(file);
    reader.readAsDataURL(file);
  }});
}}

document.getElementById('mainForm').addEventListener('submit', async e => {{
  e.preventDefault();
  const roomEl   = document.getElementById('roomFile');
  const measEl   = document.getElementById('wallMeasInput');
  const orientOk = document.getElementById('orH').checked || document.getElementById('orV').checked;
  const missing = [
    [roomEl, roomEl.files.length === 0],
    [measEl, measEl.value.trim() === ''],
  ].filter(([, bad]) => bad).map(([el]) => el);
  if (!orientOk || missing.length) {{
    if (!orientOk) {{
      document.querySelector('.orient-row').style.outline = '2px solid #c8440a';
      document.querySelector('.orient-row').style.borderRadius = '10px';
    }}
    missing.forEach(el => {{
      el.classList.add('field-error');
      el.addEventListener('change', () => el.classList.remove('field-error'), {{ once: true }});
      el.addEventListener('input',  () => el.classList.remove('field-error'), {{ once: true }});
    }});
    if (missing.length) missing[0].focus();
    return;
  }}
  const btn = document.getElementById('submitBtn');
  const THRESHOLD = 4 * 1024 * 1024;
  const roomFile = roomEl.files[0];
  const needsCompression = roomFile.size > THRESHOLD;
  btn.innerHTML = needsCompression ? '<span class="btn-spinner"></span>Compressing…' : '<span class="btn-spinner"></span>Uploading…';
  btn.disabled = true;
  if (needsCompression) {{
    const cRoom = await compressImage(roomFile);
    replaceFile(roomEl, cRoom);
    btn.innerHTML = '<span class="btn-spinner"></span>Uploading…';
  }}
  document.getElementById('mainForm').submit();
}});

['orH','orV'].forEach(id => {{
  document.getElementById(id).addEventListener('change', () => {{
    document.querySelector('.orient-row').style.outline = '';
  }});
}});
</script>
</body>
</html>"""


def _build_restart_html(
    room_id: str,
    art_id: str,
    wall_measurement: float,
    orientation: str,
    room_thumb_b64: str,
    art_thumb_b64: str,
    ref: str = "",
    frame: str = "",
    coa_field: str = "",
    image_key: str = "",
    base_url: str = "",
    gallery: str = "kenhoehn",
) -> str:
    """Return Step-1 HTML with both images pre-loaded from the store."""
    orient_h_checked = "checked" if orientation == "H" else ""
    orient_v_checked = "checked" if orientation == "V" else ""
    meas_display = "block" if orientation in ("H", "V") else "none"
    meas_label = "Vertical wall height (inches)" if orientation == "V" else "Horizontal wall width (inches)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#f5f2ec">
<title>WallyMock</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #000; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #000; --border: #d8d4cc;
  }}
  body {{
    background: var(--paper); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-weight: 300;
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    padding: 40px 20px;
  }}
  .card {{
    background: #fff; border: 1px solid var(--border); border-radius: 18px;
    padding: 40px 40px; max-width: 480px; width: 100%;
  }}
  .logo {{ font-family: Georgia, serif; font-size: 1.6rem; letter-spacing: -0.02em; margin-bottom: 4px; }}
  .card-sub {{ font-size: 1rem; font-style: italic; margin-bottom: 24px; }}
  .img-preview-box {{
    border: 1px solid var(--border); border-radius: 10px;
    background: var(--paper); padding: 12px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 14px;
  }}
  .img-preview-box img {{
    max-width: 80px; max-height: 80px; border-radius: 6px;
    object-fit: contain; flex-shrink: 0;
  }}
  .img-preview-text {{ font-size: 0.9rem; line-height: 1.5; }}
  .img-preview-text strong {{ display: block; margin-bottom: 2px; }}
  .img-check {{ color: #16a34a; font-size: 1.1rem; }}
  label.field-label {{
    display: block; font-size: 0.9rem; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 6px; margin-top: 20px;
  }}
  input[type="number"] {{
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 1rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s;
  }}
  input:focus {{ border-color: var(--accent); }}
  .orient-row {{ display: flex; gap: 12px; margin-top: 6px; }}
  .orient-row input[type="radio"] {{
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
  }}
  .orient-card {{
    flex: 1; border: 2px solid var(--border); border-radius: 10px;
    padding: 13px 10px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s; background: var(--paper);
    user-select: none; display: block;
  }}
  .orient-row input[type="radio"]:checked + .orient-card {{
    border-color: var(--accent); background: #fff3ee;
  }}
  .orient-card .icon  {{ font-size: 1.6rem; display: block; margin-bottom: 5px; line-height: 1; }}
  .orient-card .lbl   {{ font-size: 1rem; font-weight: 500; }}
  .orient-card .sublbl {{ font-size: 0.9rem; margin-top: 2px; }}
  button[type="submit"] {{
    margin-top: 28px; width: 100%; padding: 14px;
    background: var(--accent); color: #fff; border: none;
    border-radius: 9px; font-family: inherit;
    font-size: 1.05rem; font-weight: 500; cursor: pointer;
    transition: opacity .2s, transform .15s;
  }}
  button[type="submit"]:hover {{ opacity: 0.88; transform: translateY(-1px); }}
  .btn-spinner {{
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
    border-radius: 50%; animation: btnSpin .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }}
  @keyframes btnSpin {{ to {{ transform: rotate(360deg); }} }}
  @media (max-width: 600px) {{
    .card {{ padding: 28px 20px; }}
    button[type="submit"] {{ font-size: 1.2rem; padding: 16px; }}
  }}
</style>
</head>
<body>
{_back_btn(ref, base_url)}
<div class="card">
  <div class="logo">WallyMock<sup style="font-size:0.45em;vertical-align:super;letter-spacing:0;">TM</sup></div>
  <p class="card-sub">Step 1 of 4 — Review &amp; continue</p>

  <div class="img-preview-box" id="roomPreviewBox">
    <img id="roomThumb" src="data:image/jpeg;base64,{room_thumb_b64}" alt="Room photo">
    <div class="img-preview-text">
      <strong><span class="img-check">✓</span> Room photo loaded</strong>
      <span id="roomPhotoSub">Your wall photo is ready to use.</span><br>
      <a href="#" style="font-size:0.85rem;color:#c8440a;text-decoration:none;"
         onclick="document.getElementById('newRoomFile').click();return false;">
        Use a different photo
      </a>
    </div>
  </div>

  <div class="img-preview-box">
    <img src="data:image/jpeg;base64,{art_thumb_b64}" alt="Artwork">
    <div class="img-preview-text">
      <strong><span class="img-check">✓</span> Artwork loaded</strong>
      Your selected print is ready.
    </div>
  </div>

  <form action="/mockup/picker_restart?gallery={html_mod.escape(gallery)}" enctype="multipart/form-data" method="post" id="restartForm">
    <input type="hidden" name="room_id"    id="roomIdField" value="{room_id}">
    <input type="hidden" name="art_id"    value="{art_id}">
    <input type="hidden" name="ref"       value="{html_mod.escape(ref)}">
    <input type="hidden" name="frame"     value="{html_mod.escape(frame)}">
    <input type="hidden" name="coa_field" value="{html_mod.escape(coa_field)}">
    <input type="hidden" name="image_key" value="{html_mod.escape(image_key)}">
    <input type="file" id="newRoomFile" name="room_file" accept="image/*"
           style="display:none" onchange="onNewRoom(this)">

    <label class="field-label">Print Orientation</label>
    <div class="orient-row">
      <input type="radio" name="orientation" id="orH" value="H" {orient_h_checked} onchange="updateMeasLabel()">
      <label class="orient-card" for="orH">
        <span class="icon">⬛️</span>
        <div class="lbl">Horizontal</div>
        <div class="sublbl">Landscape / wide print</div>
      </label>
      <input type="radio" name="orientation" id="orV" value="V" {orient_v_checked} onchange="updateMeasLabel()">
      <label class="orient-card" for="orV">
        <span class="icon">▮</span>
        <div class="lbl">Vertical</div>
        <div class="sublbl">Portrait / tall print</div>
      </label>
    </div>

    <div id="measWrap" style="display:{meas_display};margin-top:0;">
      <label class="field-label" id="measLabel" for="wallMeasInput">{meas_label}</label>
      <input type="number" name="wall_measurement" id="wallMeasInput"
             step="0.1" min="1" max="9999" inputmode="decimal"
             value="{wall_measurement}" placeholder="e.g. 96">
    </div>

    <button type="submit" id="submitBtn">Next: Mark the Measurement →</button>
  </form>
</div>

<script>
function updateMeasLabel() {{
  const isV  = document.getElementById('orV').checked;
  const wrap  = document.getElementById('measWrap');
  const label = document.getElementById('measLabel');
  label.textContent = isV ? 'Vertical wall height (inches)' : 'Horizontal wall width (inches)';
  wrap.style.display = 'block';
  document.getElementById('wallMeasInput').focus();
}}

function onNewRoom(input) {{
  const file = input.files[0]; if (!file) return;
  document.getElementById('roomIdField').value = '';
  const reader = new FileReader();
  reader.onload = e => {{
    document.getElementById('roomThumb').src = e.target.result;
    document.getElementById('roomPhotoSub').textContent = file.name;
  }};
  reader.readAsDataURL(file);
}}
document.getElementById('restartForm').addEventListener('submit', e => {{
  const measEl   = document.getElementById('wallMeasInput');
  const orientOk = document.getElementById('orH').checked || document.getElementById('orV').checked;
  if (!orientOk || measEl.value.trim() === '') {{
    e.preventDefault();
    if (!orientOk) {{
      document.querySelector('.orient-row').style.outline = '2px solid #c8440a';
      document.querySelector('.orient-row').style.borderRadius = '10px';
    }}
    if (measEl.value.trim() === '') {{
      measEl.style.borderColor = '#c8440a';
      measEl.focus();
    }}
    return;
  }}
  const btn = document.getElementById('submitBtn');
  btn.innerHTML = '<span class="btn-spinner"></span>Loading…';
  btn.disabled = true;
}});
</script>
</body>
</html>"""


@app.get("/mockup/restart", response_class=HTMLResponse)
async def mockup_restart(
    room_id:          str   = Query(...),
    art_id:           str   = Query(...),
    wall_measurement: float = Query(...),
    orientation:      str   = Query("H"),
    ref:              str   = Query(""),
    frame:            str   = Query(""),
    coa_field:        str   = Query(""),
    image_key:        str   = Query(""),
    gallery:          str   = Query("kenhoehn"),
) -> HTMLResponse:
    """Re-render step 1 with both images pre-loaded (used by all back-to-start links)."""
    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"
    try:
        room_data, _ = _load_image(room_id)
        art_data,  _ = _load_image(art_id)
    except HTTPException:
        return HTMLResponse(
            content='<meta http-equiv="refresh" content="0;url=/">',
            status_code=200,
        )
    room_pil = Image.open(BytesIO(room_data))
    room_pil.thumbnail((400, 220), Image.LANCZOS)
    room_buf = BytesIO()
    room_pil.save(room_buf, format="JPEG", quality=82)
    room_thumb_b64 = base64.b64encode(room_buf.getvalue()).decode("utf-8")

    art_pil = Image.open(BytesIO(art_data))
    art_pil.thumbnail((400, 220), Image.LANCZOS)
    art_buf = BytesIO()
    art_pil.save(art_buf, format="JPEG", quality=82)
    art_thumb_b64 = base64.b64encode(art_buf.getvalue()).decode("utf-8")

    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    return HTMLResponse(content=_build_restart_html(
        room_id, art_id, wall_measurement, orientation, room_thumb_b64, art_thumb_b64, ref, frame, coa_field, image_key, base_url, gallery
    ))


@app.post("/mockup/picker_restart", response_class=HTMLResponse)
async def mockup_picker_restart(
    art_id:           str                    = Form(...),
    wall_measurement: float                  = Form(...),
    orientation:      str                    = Form("H"),
    room_id:          str                    = Form(""),
    room_file:        UploadFile | None      = File(None),
    ref:              str                    = Form(""),
    frame:            str                    = Form(""),
    coa_field:        str                    = Form(""),
    image_key:        str                    = Form(""),
    gallery:          str                    = Form("kenhoehn"),
) -> HTMLResponse:
    """Accept stored image IDs from the restart page; optionally replace the room photo."""
    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"
    if not (1 <= wall_measurement <= 9999):
        raise HTTPException(400, "Measurement must be between 1 and 9999 inches.")
    _load_image(art_id)
    if room_file and room_file.filename:
        room_bytes, room_pil = await read_and_validate(room_file)
        room_id = _save_image(_to_jpeg(room_pil), "image/jpeg")
    else:
        _load_image(room_id)
    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    variants_json = _fetch_variants(ref, base_url)
    return HTMLResponse(content=_build_picker_html(room_id, art_id, wall_measurement, orientation, ref, variants_json, frame, coa_field, image_key, gallery, base_url))


@app.get("/mockup/prefill", response_class=HTMLResponse)
async def mockup_prefill(art_url: str = Query(...), ref: str = Query(""), frame: str = Query(""), coa_field: str = Query(""), image_key: str = Query(""), gallery: str = Query("kenhoehn")) -> HTMLResponse:
    """Fetch artwork from a URL, store it, return a pre-populated Step-1 page."""
    parsed = urllib.parse.urlparse(art_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "art_url must be an HTTP or HTTPS URL.")

    try:
        req = urllib.request.Request(
            art_url,
            headers={"User-Agent": "WallyMock/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(MAX_FILE_SIZE + 1)
    except Exception as exc:
        raise HTTPException(400, f"Could not fetch artwork from the provided URL. ({exc})")

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, "Artwork image exceeds the 10 MB limit.")

    try:
        pil = Image.open(BytesIO(data))
        pil.verify()
    except Exception:
        raise HTTPException(400, "The URL did not return a valid image file.")

    pil = Image.open(BytesIO(data))
    if image_key in _PNG_IMAGE_KEYS:
        art_bytes = _to_png(pil)
        art_id = _save_image(art_bytes, "image/png")
    else:
        art_bytes = _to_jpeg(pil)
        art_id = _save_image(art_bytes, "image/jpeg")

    # Small thumbnail for the preview card (always JPEG for compact preview)
    thumb = Image.open(BytesIO(art_bytes))
    thumb.thumbnail((400, 220), Image.LANCZOS)
    thumb_buf = BytesIO()
    if thumb.mode in ("RGBA", "LA"):
        thumb.save(thumb_buf, format="PNG")
    else:
        thumb.save(thumb_buf, format="JPEG", quality=82)
    art_thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode("utf-8")

    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    variants_json = _fetch_variants(ref, base_url)
    return HTMLResponse(content=_build_prefill_html(art_id, art_thumb_b64, ref, variants_json, frame, coa_field, image_key, gallery, base_url))


# ── Shared picker-page builder ────────────────────────────────────────────────

def _build_picker_html(
    room_id: str,
    art_id:  str,
    wall_measurement: float,
    orientation: str,
    ref: str = "",
    variants_json: str = "[]",
    frame: str = "",
    coa_field: str = "",
    image_key: str = "",
    gallery: str = "kenhoehn",
    base_url: str = "",
) -> str:
    """Return the full Step-2 picker HTML string."""
    room_data, _ = _load_image(room_id)

    try:
        room_pil = Image.open(BytesIO(room_data))
        orig_w, orig_h = room_pil.size
    except Exception:
        raise HTTPException(400, "Room image could not be opened from storage.")

    room_b64 = base64.b64encode(room_data).decode("utf-8")

    is_vertical = (orientation == "V")

    restart_url = (
        f"/mockup/restart?room_id={urllib.parse.quote(room_id)}"
        f"&art_id={urllib.parse.quote(art_id)}"
        f"&wall_measurement={wall_measurement}"
        f"&orientation={orientation}"
        f"&gallery={urllib.parse.quote(gallery)}"
        + (f"&ref={urllib.parse.quote(ref)}" if ref else "")
        + (f"&frame={urllib.parse.quote(frame)}" if frame else "")
        + (f"&coa_field={urllib.parse.quote(coa_field)}" if coa_field else "")
        + (f"&image_key={urllib.parse.quote(image_key)}" if image_key else "")
    )

    if is_vertical:
        point1_color  = "#7c3aed";  point2_color  = "#c8440a"
        point1_badge  = "TOP";      point2_badge  = "BOT"
        click_desc    = "the <strong>ceiling/top point</strong>"
        click_desc2   = "the <strong>floor/bottom point</strong>"
        orient_label  = "Vertical (portrait)"
        tip_extra     = "For vertical: click a point on the ceiling directly above the spot you measured, then the floor directly below it."
    else:
        point1_color  = "#2563eb";  point2_color  = "#c8440a"
        point1_badge  = "L";        point2_badge  = "R"
        click_desc    = "the <strong>left edge</strong> of your measured space"
        click_desc2   = "the <strong>right edge</strong> of your measured space"
        orient_label  = "Horizontal (landscape)"
        tip_extra     = "For horizontal: click the leftmost point of your measured span, then the rightmost point."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#f5f2ec">
<title>Mark Measurement Points</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #000; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #000; --border: #d8d4cc;
    --p1: {point1_color}; --p2: {point2_color};
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-weight: 300;
    background: var(--paper); color: var(--ink);
    height: 100vh; overflow: hidden; padding: 16px 20px 0;
    display: flex; flex-direction: column; align-items: center;
  }}
  .page-title {{
    font-family: Georgia, serif; font-size: 1.75rem;
    margin-bottom: 2px; text-align: center;
  }}
  .page-sub {{
    font-size: 0.95rem; font-style: italic;
    margin-bottom: 12px; text-align: center;
  }}
  .instr-bar {{
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 11px 18px; max-width: 960px; width: 100%; margin-bottom: 10px;
    display: flex; align-items: flex-start; gap: 24px; flex-wrap: wrap;
  }}
  .instr-step {{ display: flex; align-items: flex-start; gap: 10px; flex: 1; min-width: 200px; }}
  .badge {{
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 500; font-size: 0.72rem; margin-top: 1px;
    transition: background .25s, color .25s;
  }}
  .badge-waiting  {{ background: var(--border); color: var(--muted); }}
  .badge-active   {{ background: var(--p1);     color: #fff; }}
  .badge-active-2 {{ background: var(--p2);     color: #fff; }}
  .badge-done     {{ background: #16a34a;        color: #fff; }}
  .instr-text {{ font-size: 0.94rem; line-height: 1.55; padding-top: 4px; }}
  .instr-text.muted {{ color: var(--ink); }}
  .instr-meta {{
    margin-left: auto; display: flex; flex-direction: column;
    align-items: flex-end; gap: 5px;
  }}
  .meta-chip {{
    background: var(--paper); border: 1px solid var(--border); border-radius: 6px;
    padding: 3px 10px; white-space: nowrap; font-size: 0.82rem;
  }}
  .ppi-chip {{
    background: #2563eb; color: #fff; border: none;
    border-radius: 6px; padding: 3px 12px; font-size: 0.73rem;
    font-weight: 500; display: none; white-space: nowrap;
  }}
  .tips {{
    background: #f0f6ff; border: 1px solid #bdd3f5; border-radius: 10px;
    padding: 9px 14px; max-width: 960px; width: 100%; margin-bottom: 10px;
    font-size: 0.88rem; color: #000; line-height: 1.55;
  }}
  .tips strong {{ font-weight: 500; color: #000; }}
  .tips ul {{ margin: 4px 0 0 14px; }}
  .canvas-wrap {{
    position: relative; max-width: 960px; width: 100%;
    border-radius: 12px; overflow: hidden;
    border: 2px solid var(--border);
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
    cursor: crosshair; user-select: none; background: #e8e5de;
    flex: 1; min-height: 0;
    display: flex; align-items: center; justify-content: center;
  }}
  canvas {{ display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; }}
  .canvas-loader {{
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center;
  }}
  .spinner {{
    width: 36px; height: 36px; border: 3px solid rgba(0,0,0,0.12);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .controls {{ display: flex; gap: 12px; margin-top: 10px; padding-bottom: 14px; max-width: 960px; width: 100%; }}
  @media (max-width: 700px) {{
    body {{ height: auto; overflow-y: auto; overflow-x: hidden; padding-bottom: 20px; }}
    .canvas-wrap {{ flex: none; min-height: 72vw; }}
    .instr-bar {{ gap: 12px; }}
    .page-title {{ font-size: 2rem; }}
    .page-sub {{ font-size: 1.1rem; }}
    .instr-text {{ font-size: 1.05rem; }}
    .tips {{ font-size: 1rem; line-height: 1.65; }}
    .meta-chip {{ font-size: 0.98rem; }}
    .btn {{ font-size: 1.15rem; padding: 14px 24px; }}
    .ready-msg {{ font-size: 1.05rem; }}
    .back-row a {{ font-size: 1.1rem; padding: 14px 24px; }}
  }}
  .btn {{
    padding: 12px 24px; border-radius: 9px; border: none;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-size: 1rem; font-weight: 500;
    cursor: pointer; transition: opacity .2s, transform .15s;
  }}
  .btn:hover {{ opacity: 0.86; transform: translateY(-1px); }}
  .btn:disabled {{ opacity: 0.32; cursor: not-allowed; transform: none; }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-ghost   {{ background: #fff; color: var(--ink); border: 1px solid var(--border); }}
  .ready-msg {{
    flex: 1; background: #f0fff4; border: 1px solid #86efac; border-radius: 9px;
    padding: 10px 16px; font-size: 0.92rem; color: #166534;
    text-align: center; display: none; align-items: center; justify-content: center;
  }}
  .back-row {{ max-width: 960px; width: 100%; padding-bottom: 16px; }}
  .back-row a {{
    display: inline-block; padding: 11px 22px; border-radius: 9px;
    border: 1px solid var(--border); background: #fff; color: var(--ink);
    text-decoration: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 0.95rem; font-weight: 500;
    transition: opacity .2s, transform .15s;
  }}
  .back-row a:hover {{ opacity: 0.82; transform: translateY(-1px); }}
  .zoom-bar {{
    display: flex; align-items: center; gap: 6px;
    max-width: 960px; width: 100%; margin-bottom: 4px;
    justify-content: flex-end;
  }}
  .zoom-btn {{
    width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
    border: 1.5px solid var(--border); background: #fff;
    font-size: 1.05rem; cursor: pointer; display: flex;
    align-items: center; justify-content: center;
    transition: border-color .15s, background .15s, color .15s;
    user-select: none; line-height: 1;
  }}
  .zoom-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .zoom-btn.active {{ border-color: var(--accent); background: var(--accent); color: #fff; }}
  .zoom-level {{ font-size: 0.8rem; color: #666; min-width: 38px; text-align: center; }}
</style>
</head>
<body>
{_back_btn(ref, base_url)}
<h1 class="page-title">Step 2 — Mark Your Measurement</h1>
<p class="page-sub">Click two points on the photo that span your {wall_measurement}" measurement</p>

<div class="instr-bar">
  <div class="instr-step">
    <div class="badge badge-active" id="b1">1</div>
    <div class="instr-text" id="t1">
      Click {click_desc} on the photo below.
    </div>
  </div>
  <div class="instr-step">
    <div class="badge badge-waiting" id="b2">2</div>
    <div class="instr-text muted" id="t2">
      Then click {click_desc2}.
    </div>
  </div>
  <div class="instr-meta">
    <span class="meta-chip">📐 {wall_measurement}" · {orient_label}</span>
    <span class="ppi-chip" id="ppiChip">–</span>
  </div>
</div>

<div class="tips">
  <strong>How to mark your measurement:</strong>
  <ul>
    <li>{tip_extra}</li>
    <li>Click precisely on the two points that define the distance you entered on the previous step.</li>
    <li>Pick points at the same depth in the photo — both should lie flat on the wall surface, not on corners that angle away.</li>
    <li>For best accuracy, zoom in on the photo before clicking. Use the Reset button if a click was off.</li>
    <li>The further apart your two points are in the photo, the more accurate the final scale will be.</li>
  </ul>
</div>

<div class="zoom-bar">
  <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&#x2212;</button>
  <span class="zoom-level" id="zoomLevel">100%</span>
  <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">&#x2b;</button>
  <button class="zoom-btn" id="panBtn" onclick="togglePan()" title="Pan / hand mode">&#x270b;</button>
  <button class="zoom-btn" onclick="resetView()" title="Reset view" style="font-size:0.75rem">&#x21ba;</button>
</div>

<div class="canvas-wrap" id="canvasWrap">
  <canvas id="c"></canvas>
  <div class="canvas-loader" id="canvasLoader"><div class="spinner"></div></div>
</div>

<div class="controls">
  <button class="btn btn-ghost" onclick="reset()">↺ Reset</button>
  <div class="ready-msg" id="readyMsg">Both points set — click&nbsp;<strong>Next</strong>&nbsp;to see your mockup</div>
  <button class="btn btn-primary" id="nextBtn" disabled onclick="proceed()">
    Next
  </button>
</div>
<div class="back-row">
  <a href="{restart_url}">← Back to Step 1</a>
</div>

<form id="rf" action="/mockup/editor?gallery={html_mod.escape(gallery)}" method="post" style="display:none">
  <input type="hidden" name="room_id"          value="{room_id}">
  <input type="hidden" name="art_id"           value="{art_id}">
  <input type="hidden" name="wall_measurement" value="{wall_measurement}">
  <input type="hidden" name="orientation"      value="{orientation}">
  <input type="hidden" name="ref"              value="{html_mod.escape(ref)}">
  <input type="hidden" name="variants_json"    value="{html_mod.escape(variants_json)}">
  <input type="hidden" name="frame"            value="{html_mod.escape(frame)}">
  <input type="hidden" name="coa_field"        value="{html_mod.escape(coa_field)}">
  <input type="hidden" name="image_key"        value="{html_mod.escape(image_key)}">
  <input type="hidden" name="gallery"          value="{html_mod.escape(gallery)}">
  <input type="hidden" name="pt1_x"            id="fPt1x">
  <input type="hidden" name="pt1_y"            id="fPt1y">
  <input type="hidden" name="pt2_x"            id="fPt2x">
  <input type="hidden" name="pt2_y"            id="fPt2y">
</form>

<script>
const ROOM_B64   = "{room_b64}";
const IS_VERT    = {'true' if is_vertical else 'false'};
const WALL_MEAS  = {wall_measurement};
const P1_COLOR   = "{point1_color}";
const P2_COLOR   = "{point2_color}";
const P1_BADGE   = "{point1_badge}";
const P2_BADGE   = "{point2_badge}";

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();
let pt1 = null, pt2 = null, phase = 1;

// ── View transform ────────────────────────────────────────────────────
let viewScale = 1, viewOffX = 0, viewOffY = 0;
let panMode = false, panning = false, panLastX = 0, panLastY = 0;
const MIN_SCALE = 1, MAX_SCALE = 10;

function clampView() {{
  const minX = canvas.width  * (1 - viewScale);
  const minY = canvas.height * (1 - viewScale);
  viewOffX = Math.max(minX, Math.min(0, viewOffX));
  viewOffY = Math.max(minY, Math.min(0, viewOffY));
}}

function updateZoomUI() {{
  document.getElementById('zoomLevel').textContent = Math.round(viewScale * 100) + '%';
  canvas.style.cursor = panMode ? 'grab' : 'crosshair';
}}

function zoomAt(factor, pivotX, pivotY) {{
  const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, viewScale * factor));
  viewOffX = pivotX - (pivotX - viewOffX) * (newScale / viewScale);
  viewOffY = pivotY - (pivotY - viewOffY) * (newScale / viewScale);
  viewScale = newScale;
  if (viewScale <= MIN_SCALE) {{ viewOffX = 0; viewOffY = 0; }}
  clampView(); draw(); updateZoomUI();
}}

function zoomIn()  {{ zoomAt(1.5, canvas.width/2, canvas.height/2); }}
function zoomOut() {{ zoomAt(1/1.5, canvas.width/2, canvas.height/2); }}

function resetView() {{
  viewScale = 1; viewOffX = 0; viewOffY = 0;
  updateZoomUI(); draw();
}}

function togglePan() {{
  panMode = !panMode;
  document.getElementById('panBtn').classList.toggle('active', panMode);
  updateZoomUI();
}}

// Canvas display coords → image pixel coords
function toImgCoords(clientX, clientY) {{
  const rect = canvas.getBoundingClientRect();
  const cx = (clientX - rect.left) * (canvas.width  / rect.width);
  const cy = (clientY - rect.top)  * (canvas.height / rect.height);
  return [(cx - viewOffX) / viewScale, (cy - viewOffY) / viewScale];
}}

// Canvas display coords (no image-space transform)
function toCanvasCoords(clientX, clientY) {{
  const rect = canvas.getBoundingClientRect();
  return [
    (clientX - rect.left) * (canvas.width  / rect.width),
    (clientY - rect.top)  * (canvas.height / rect.height),
  ];
}}

img.onload = () => {{
  document.getElementById('canvasLoader').style.display = 'none';
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  draw();
}};
img.onerror = () => {{
  document.getElementById('canvasLoader').innerHTML =
    '<p style="color:#c8440a;font-size:0.85rem;padding:20px">Could not load room image.</p>';
}};
img.src = 'data:image/jpeg;base64,' + ROOM_B64;

// ── Draw ─────────────────────────────────────────────────────────────
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.setTransform(viewScale, 0, 0, viewScale, viewOffX, viewOffY);
  ctx.drawImage(img, 0, 0);

  // Scale-compensated sizes so overlays appear visually consistent
  const invS = 1 / viewScale;

  if (pt1 && pt2) {{
    const midX = (pt1.x + pt2.x) / 2;
    const midY = (pt1.y + pt2.y) / 2;
    const lw   = Math.max(2, canvas.width * 0.002) * invS;

    if (IS_VERT) {{
      ctx.fillStyle = 'rgba(124,58,237,0.10)';
      ctx.fillRect(0, Math.min(pt1.y,pt2.y), canvas.width, Math.abs(pt2.y-pt1.y));
      ctx.strokeStyle = P1_COLOR; ctx.lineWidth = lw; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(midX, pt1.y); ctx.lineTo(midX, pt2.y); ctx.stroke();
      arrowV(midX, pt1.y, -1, lw * 5);
      arrowV(midX, pt2.y,  1, lw * 5);
      ctx.fillStyle = P1_COLOR;
      ctx.font = `600 ${{Math.max(12, canvas.width * 0.016) * invS}}px DM Sans, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(WALL_MEAS + '"', midX + canvas.width * 0.025, midY);
    }} else {{
      const x1 = Math.min(pt1.x,pt2.x), x2 = Math.max(pt1.x,pt2.x);
      ctx.fillStyle = 'rgba(37,99,235,0.10)';
      ctx.fillRect(x1, 0, x2 - x1, canvas.height);
      const lineY = canvas.height * 0.07;
      ctx.strokeStyle = P1_COLOR; ctx.lineWidth = lw; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(pt1.x, lineY); ctx.lineTo(pt2.x, lineY); ctx.stroke();
      arrowH(pt1.x, lineY, -1, lw * 5);
      arrowH(pt2.x, lineY,  1, lw * 5);
      ctx.fillStyle = P1_COLOR;
      ctx.font = `600 ${{Math.max(12, canvas.width * 0.016) * invS}}px DM Sans, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(WALL_MEAS + '"', midX, lineY - canvas.height * 0.012);
    }}
  }}
  if (pt1) pinPoint(pt1, P1_COLOR, P1_BADGE);
  if (pt2) pinPoint(pt2, P2_COLOR, P2_BADGE);

  ctx.restore();
}}

function arrowH(x, y, dir, size) {{
  ctx.fillStyle = P1_COLOR;
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - dir*size, y - size*0.5); ctx.lineTo(x - dir*size, y + size*0.5);
  ctx.closePath(); ctx.fill();
}}
function arrowV(x, y, dir, size) {{
  ctx.fillStyle = P1_COLOR;
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - size*0.5, y - dir*size); ctx.lineTo(x + size*0.5, y - dir*size);
  ctx.closePath(); ctx.fill();
}}
function pinPoint(pt, color, label) {{
  const invS = 1 / viewScale;
  const r  = Math.max(12, canvas.width * 0.015) * invS;
  const lw = Math.max(2, canvas.width * 0.003) * invS;
  ctx.strokeStyle = color + 'cc'; ctx.lineWidth = lw; ctx.setLineDash([8 * invS, 5 * invS]);
  ctx.beginPath();
  if (IS_VERT) {{ ctx.moveTo(0, pt.y); ctx.lineTo(canvas.width, pt.y); }}
  else         {{ ctx.moveTo(pt.x, 0); ctx.lineTo(pt.x, canvas.height); }}
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = `600 ${{Math.max(9, r * 1.0)}}px DM Sans, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, pt.x, pt.y);
  ctx.textBaseline = 'alphabetic';
}}

// ── Pan events ───────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {{
  if (!panMode) return;
  panning = true;
  [panLastX, panLastY] = toCanvasCoords(e.clientX, e.clientY);
  canvas.style.cursor = 'grabbing'; e.preventDefault();
}});
window.addEventListener('mousemove', e => {{
  if (!panning) return;
  const [cx, cy] = toCanvasCoords(e.clientX, e.clientY);
  viewOffX += cx - panLastX; viewOffY += cy - panLastY;
  panLastX = cx; panLastY = cy;
  clampView(); draw();
}});
window.addEventListener('mouseup', () => {{
  if (panning) {{ panning = false; canvas.style.cursor = panMode ? 'grab' : 'crosshair'; }}
}});

// Touch pan (single finger in pan mode)
let touchPanActive = false, pinchActive = false;
canvas.addEventListener('touchstart', e => {{
  if (!panMode || e.touches.length !== 1) return;
  touchPanActive = true;
  const t = e.touches[0];
  [panLastX, panLastY] = toCanvasCoords(t.clientX, t.clientY);
  e.preventDefault();
}}, {{passive: false}});
window.addEventListener('touchmove', e => {{
  if (!touchPanActive) return;
  const t = e.touches[0];
  const [cx, cy] = toCanvasCoords(t.clientX, t.clientY);
  viewOffX += cx - panLastX; viewOffY += cy - panLastY;
  panLastX = cx; panLastY = cy;
  clampView(); draw(); e.preventDefault();
}}, {{passive: false}});
window.addEventListener('touchend', (e) => {{ touchPanActive = false; if (e.touches.length === 0) pinchActive = false; }});

// Pinch-to-zoom (two fingers)
let pinchStartDist = 0, pinchStartScale = 1, pinchStartMidX = 0, pinchStartMidY = 0, pinchStartOffX = 0, pinchStartOffY = 0;
canvas.addEventListener('touchstart', e => {{
  if (e.touches.length !== 2) return;
  pinchActive = true;
  e.preventDefault();
  const t1 = e.touches[0], t2 = e.touches[1];
  pinchStartDist  = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
  pinchStartScale = viewScale;
  pinchStartOffX  = viewOffX;
  pinchStartOffY  = viewOffY;
  const rect = canvas.getBoundingClientRect();
  pinchStartMidX = ((t1.clientX + t2.clientX)/2 - rect.left) * (canvas.width  / rect.width);
  pinchStartMidY = ((t1.clientY + t2.clientY)/2 - rect.top)  * (canvas.height / rect.height);
}}, {{passive: false}});
window.addEventListener('touchmove', e => {{
  if (e.touches.length !== 2) return;
  e.preventDefault();
  const t1 = e.touches[0], t2 = e.touches[1];
  const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
  if (pinchStartDist < 1) return;
  const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, pinchStartScale * dist / pinchStartDist));
  viewOffX = pinchStartMidX - (pinchStartMidX - pinchStartOffX) * (newScale / pinchStartScale);
  viewOffY = pinchStartMidY - (pinchStartMidY - pinchStartOffY) * (newScale / pinchStartScale);
  viewScale = newScale;
  if (viewScale <= MIN_SCALE) {{ viewOffX = 0; viewOffY = 0; }}
  clampView(); draw(); updateZoomUI();
}}, {{passive: false}});

// Scroll-wheel zoom
document.getElementById('canvasWrap').addEventListener('wheel', e => {{
  e.preventDefault();
  const [cx, cy] = toCanvasCoords(e.clientX, e.clientY);
  zoomAt(e.deltaY < 0 ? 1.2 : 1/1.2, cx, cy);
}}, {{passive: false}});

// ── Click handling ──────────────────────────────────────────────────
canvas.addEventListener('click', e => {{
  if (panMode || canvas.width === 0) return;
  const [ix, iy] = toImgCoords(e.clientX, e.clientY);
  if (ix < 0 || ix > img.naturalWidth || iy < 0 || iy > img.naturalHeight) return;
  if (phase === 1) {{
    pt1 = {{x: ix, y: iy}}; phase = 2;
    setStep(1,'done'); setStep(2,'active2');
    document.getElementById('t2').classList.remove('muted');
  }} else if (phase === 2) {{
    pt2 = {{x: ix, y: iy}}; phase = 3;
    setStep(2,'done');
    const span = IS_VERT ? Math.abs(pt2.y - pt1.y) : Math.abs(pt2.x - pt1.x);
    const ppi  = (span / WALL_MEAS).toFixed(1);
    const chip = document.getElementById('ppiChip');
    chip.textContent = ppi + ' px/in · ready ✓';
    chip.style.display = 'block';
    document.getElementById('nextBtn').disabled = false;
    document.getElementById('readyMsg').style.display = 'flex';
  }}
  draw();
}});

// Touch tap for point placement (non-pan mode)
canvas.addEventListener('touchend', e => {{
  if (panMode || touchPanActive || pinchActive || e.changedTouches.length !== 1) return;
  const t = e.changedTouches[0];
  const [ix, iy] = toImgCoords(t.clientX, t.clientY);
  if (ix < 0 || ix > img.naturalWidth || iy < 0 || iy > img.naturalHeight) return;
  e.preventDefault(); // prevent synthetic click from double-placing a point
  if (phase === 1) {{
    pt1 = {{x: ix, y: iy}}; phase = 2;
    setStep(1,'done'); setStep(2,'active2');
    document.getElementById('t2').classList.remove('muted');
  }} else if (phase === 2) {{
    pt2 = {{x: ix, y: iy}}; phase = 3;
    setStep(2,'done');
    const span = IS_VERT ? Math.abs(pt2.y - pt1.y) : Math.abs(pt2.x - pt1.x);
    const ppi  = (span / WALL_MEAS).toFixed(1);
    const chip = document.getElementById('ppiChip');
    chip.textContent = ppi + ' px/in · ready ✓';
    chip.style.display = 'block';
    document.getElementById('nextBtn').disabled = false;
    document.getElementById('readyMsg').style.display = 'flex';
  }}
  draw();
}});

function setStep(n, state) {{
  const el = document.getElementById('b' + n);
  if (state === 'done')    {{ el.className = 'badge badge-done';     el.textContent = '✓'; }}
  if (state === 'active')  {{ el.className = 'badge badge-active';   el.textContent = String(n); }}
  if (state === 'active2') {{ el.className = 'badge badge-active-2'; el.textContent = String(n); }}
  if (state === 'waiting') {{ el.className = 'badge badge-waiting';  el.textContent = String(n); }}
}}

function reset() {{
  pt1 = null; pt2 = null; phase = 1;
  setStep(1,'active'); setStep(2,'waiting');
  document.getElementById('t2').classList.add('muted');
  document.getElementById('ppiChip').style.display = 'none';
  document.getElementById('readyMsg').style.display = 'none';
  document.getElementById('nextBtn').disabled = true;
  draw();
}}

function proceed() {{
  document.getElementById('fPt1x').value = pt1.x.toFixed(2);
  document.getElementById('fPt1y').value = pt1.y.toFixed(2);
  document.getElementById('fPt2x').value = pt2.x.toFixed(2);
  document.getElementById('fPt2y').value = pt2.y.toFixed(2);
  document.getElementById('nextBtn').textContent = 'Loading editor…';
  document.getElementById('nextBtn').disabled = true;
  document.getElementById('rf').submit();
}}
</script>
</body>
</html>"""


# ── Step 2 routes ─────────────────────────────────────────────────────────────

@app.post("/mockup/picker", response_class=HTMLResponse)
async def mockup_picker_post(
    room_file:        UploadFile = File(...),
    art_file:         UploadFile = File(...),
    wall_measurement: float      = Form(...),
    orientation:      str        = Form("H"),
    gallery:          str        = Form("kenhoehn"),
) -> HTMLResponse:
    """Validate uploads, store them, return the measurement-picker page."""
    if not (1 <= wall_measurement <= 9999):
        raise HTTPException(400, "Measurement must be between 1 and 9999 inches.")

    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"

    room_bytes, room_pil = await read_and_validate(room_file)
    art_bytes,  art_pil  = await read_and_validate(art_file)

    room_bytes = _to_jpeg(room_pil)
    art_bytes  = _to_jpeg(art_pil)

    room_id = _save_image(room_bytes, "image/jpeg")
    art_id  = _save_image(art_bytes,  "image/jpeg")

    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    return HTMLResponse(content=_build_picker_html(room_id, art_id, wall_measurement, orientation, gallery=gallery, base_url=base_url))


@app.get("/mockup/picker", response_class=HTMLResponse)
async def mockup_picker_get(
    room_id:          str   = Query(...),
    art_id:           str   = Query(...),
    wall_measurement: float = Query(...),
    orientation:      str   = Query("H"),
    ref:              str   = Query(""),
    frame:            str   = Query(""),
    coa_field:        str   = Query(""),
    image_key:        str   = Query(""),
    gallery:          str   = Query("kenhoehn"),
) -> HTMLResponse:
    """Re-render the picker from stored images (used by 'Re-mark Wall' back-link)."""
    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"
    if not (1 <= wall_measurement <= 9999):
        raise HTTPException(400, "Invalid measurement.")
    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    variants_json = _fetch_variants(ref, base_url)
    return HTMLResponse(content=_build_picker_html(room_id, art_id, wall_measurement, orientation, ref, variants_json, frame, coa_field, image_key, gallery, base_url))


@app.post("/mockup/picker_prefill", response_class=HTMLResponse)
async def mockup_picker_prefill(
    room_file:        UploadFile = File(...),
    art_id:           str        = Form(...),
    wall_measurement: float      = Form(...),
    orientation:      str        = Form("H"),
    ref:              str        = Form(""),
    variants_json:    str        = Form("[]"),
    frame:            str        = Form(""),
    coa_field:        str        = Form(""),
    image_key:        str        = Form(""),
    gallery:          str        = Form("kenhoehn"),
) -> HTMLResponse:
    """Prefill variant: artwork already stored; only room file is uploaded here."""
    if not (1 <= wall_measurement <= 9999):
        raise HTTPException(400, "Measurement must be between 1 and 9999 inches.")

    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"

    # Confirm the artwork is still in the store (raises 410 if expired)
    _load_image(art_id)

    room_bytes, room_pil = await read_and_validate(room_file)
    room_bytes = _to_jpeg(room_pil)
    room_id = _save_image(room_bytes, "image/jpeg")

    base_url = (_get_tenant(gallery).get("allowed_origins") or [""])[0]
    return HTMLResponse(content=_build_picker_html(room_id, art_id, wall_measurement, orientation, ref, variants_json, frame, coa_field, image_key, gallery, base_url))


# ── Step 3 — Interactive Editor ───────────────────────────────────────────────

@app.post("/mockup/editor", response_class=HTMLResponse)
async def mockup_editor(
    room_id:          str   = Form(...),
    art_id:           str   = Form(...),
    wall_measurement: float = Form(...),
    orientation:      str   = Form("H"),
    pt1_x:            float = Form(...),
    pt1_y:            float = Form(...),
    pt2_x:            float = Form(...),
    pt2_y:            float = Form(...),
    ref:              str   = Form(""),
    variants_json:    str   = Form("[]"),
    frame:            str   = Form(""),
    coa_field:        str   = Form(""),
    image_key:        str   = Form(""),
    gallery:          str   = Form("kenhoehn"),
) -> HTMLResponse:
    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"

    if not (1 <= wall_measurement <= 9999):
        raise HTTPException(400, "Invalid measurement value.")

    tenant = _get_tenant(gallery)
    origins = tenant.get("allowed_origins") or []
    parent_origin = origins[0] if origins else ""

    # Load both images from server-side store (never trusts form for image data)
    room_data, _ = _load_image(room_id)
    art_data,  _ = _load_image(art_id)

    # Compute orig dimensions server-side — not from form fields (prevents tampering)
    try:
        room_pil = Image.open(BytesIO(room_data))
        orig_w, orig_h = room_pil.size
    except Exception:
        raise HTTPException(400, "Room image could not be read from storage.")

    try:
        art_pil   = Image.open(BytesIO(art_data))
        art_w, art_h = art_pil.size
    except Exception:
        raise HTTPException(400, "Art image could not be read from storage.")

    if art_w == 0 or art_h == 0:
        raise HTTPException(400, "Art image has zero width or height.")
    art_aspect = art_w / art_h

    # Encode images as base64 so editor page is self-contained (no _store HTTP round-trips)
    orig_w, orig_h = room_pil.size
    room_mime = detect_mime(room_pil)
    art_mime  = detect_mime(art_pil)
    room_b64  = base64.b64encode(room_data).decode("utf-8")
    art_b64   = base64.b64encode(art_data).decode("utf-8")

    is_vertical = (orientation == "V")
    span_px = abs(pt2_y - pt1_y) if is_vertical else abs(pt2_x - pt1_x)
    if span_px < 1:
        raise HTTPException(
            400,
            "The two clicked points are too close together. "
            "Please go back and re-mark a wider span."
        )
    ppi_orig = span_px / wall_measurement   # original image pixels per inch

    # Parse and filter product variants for Add to Cart
    try:
        variants_list = json.loads(variants_json)
    except Exception:
        variants_list = []

    orientation_filter = "Portrait" if is_vertical else "Landscape"
    art_aspect_norm = max(art_aspect, 1.0 / art_aspect) if art_aspect > 0 else 1.0

    if image_key:
        # Exact image key match — most precise, used when user clicked a specific thumbnail.
        # Bypasses aspect ratio tolerance entirely; correct for framed images and non-standard shapes.
        matching_variants = [v for v in variants_list if v.get("imageKey") == image_key]
    else:
        # Fallback: frame + orientation + aspect ratio matching
        matching_variants = []
        for v in variants_list:
            v_frame = v.get("frame", "").lower()
            if frame:
                if v_frame != frame.lower():
                    continue
            else:
                if v_frame != "unframed":
                    continue
            if v.get("orientation") is not None and v["orientation"] != orientation_filter:
                continue
            v_aspect = v.get("aspect")
            if v_aspect is None:
                continue  # skip non-numeric aspect variants when no image key available
            if abs(v_aspect - art_aspect_norm) / max(art_aspect_norm, 0.001) > 0.07:
                continue
            matching_variants.append(v)

    has_product_sizes = bool(matching_variants)
    variants_js = json.dumps(matching_variants)

    if has_product_sizes:
        preset_parts = []
        for v in matching_variants:
            sid = v["id"].replace("'", "\\'")
            sprice = v["price"].replace("'", "\\'")
            label = v["size"].replace(" in.", "") + (f' — {v["price"]}' if v["price"] else "")
            preset_parts.append(
                f'<button class="preset-btn" onclick="selectProductSize(\'{sid}\',\'{sprice}\',{v["long"]},this)">'
                f'{label}</button>'
            )
        preset_html = "\n    ".join(preset_parts)
        disclaimer_html = (
            '<p style="font-size:0.78rem;color:#888;font-style:italic;margin:8px 0 2px;">'
            'Prices shown are for unframed prints. To order a framed print, visit the product page.</p>\n'
        ) if not frame or frame.lower() == "unframed" else ""
        coa_panel_html = (
            '<div id="coaPanel" style="display:none;margin-top:8px;border:1px solid #d8d4cc;'
            'border-radius:9px;padding:12px 14px;background:#fafaf8;">'
            '<label style="display:block;font-size:0.82rem;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:6px;color:#444;">Name for Certificate of Authenticity</label>'
            '<input type="text" id="coaName" placeholder="Your full name" '
            'style="width:100%;padding:9px 12px;border:1px solid #d8d4cc;border-radius:7px;'
            'font-size:1rem;font-family:inherit;background:#fff;color:#000;outline:none;" '
            'oninput="updateCoaBtn()">'
            '<button class="btn btn-cart" id="confirmCartBtn" '
            'style="margin-top:8px;width:100%;" disabled onclick="confirmAddToCart()">'
            'Confirm \u0026 Add to Cart</button>'
            '</div>\n'
        ) if coa_field else ""
        cart_ui_html = (
            disclaimer_html
            + '  <button class="btn btn-cart" id="addToCartBtn" style="display:none;margin-top:6px;" onclick="addToCart()">'
            'Add to Cart</button>\n'
            + coa_panel_html
            + f'  <a class="btn btn-ghost" id="contactSizeBtn" href="{html_mod.escape(parent_origin)}/contact" '
            'target="_blank" rel="noopener" style="display:none;margin-top:6px;text-align:center;'
            'text-decoration:none;">Contact Us for a Custom Size \u2192</a>'
        )
    else:
        preset_html = (
            '<button class="preset-btn" onclick="applyPreset(24, this)">24"</button>\n'
            '    <button class="preset-btn" onclick="applyPreset(36, this)">36"</button>\n'
            '    <button class="preset-btn" onclick="applyPreset(48, this)">48"</button>\n'
            '    <button class="preset-btn" onclick="applyPreset(60, this)">60"</button>\n'
            '    <button class="preset-btn" onclick="applyPreset(72, this)">72"</button>'
        )
        cart_ui_html = ""

    # Panel label copy
    if is_vertical:
        long_axis_label = "Height (longest edge)"
        size_placeholder = "e.g. 60"
        size_tip_body = (
            "Because <strong>Vertical</strong> is selected, the number you enter "
            "is the <strong>height</strong> of the print in inches. "
            "The width scales automatically."
        )
    else:
        long_axis_label = "Width (longest edge)"
        size_placeholder = "e.g. 48"
        size_tip_body = (
            "Because <strong>Horizontal</strong> is selected, the number you enter "
            "is the <strong>width</strong> of the print in inches. "
            "The height scales automatically."
        )

    # Back-to-picker URL (GET — avoids history.back() after POST)
    back_url = (
        f"/mockup/picker?room_id={urllib.parse.quote(room_id)}"
        f"&art_id={urllib.parse.quote(art_id)}"
        f"&wall_measurement={wall_measurement}"
        f"&orientation={orientation}"
        f"&gallery={urllib.parse.quote(gallery)}"
        + (f"&ref={urllib.parse.quote(ref)}" if ref else "")
        + (f"&frame={urllib.parse.quote(frame)}" if frame else "")
        + (f"&coa_field={urllib.parse.quote(coa_field)}" if coa_field else "")
        + (f"&image_key={urllib.parse.quote(image_key)}" if image_key else "")
    )

    restart_url = (
        f"/mockup/restart?room_id={urllib.parse.quote(room_id)}"
        f"&art_id={urllib.parse.quote(art_id)}"
        f"&wall_measurement={wall_measurement}"
        f"&orientation={orientation}"
        f"&gallery={urllib.parse.quote(gallery)}"
        + (f"&ref={urllib.parse.quote(ref)}" if ref else "")
        + (f"&frame={urllib.parse.quote(frame)}" if frame else "")
        + (f"&coa_field={urllib.parse.quote(coa_field)}" if coa_field else "")
        + (f"&image_key={urllib.parse.quote(image_key)}" if image_key else "")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#1a1a18">
<title>Position Your Print</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #000; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #000; --border: #d8d4cc; --panel: #fff;
  }}

  /* ── Desktop: side-by-side full-viewport ── */
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-weight: 300;
    background: var(--paper); color: var(--ink);
    height: 100vh; overflow: hidden;
    display: grid; grid-template-columns: 1fr 320px;
  }}
  .stage {{ position: relative; overflow: hidden; background: #111; }}
  #mainCanvas {{ display: block; cursor: default; touch-action: none; }}
  .stage.dragging {{ cursor: grabbing !important; }}
  .panel {{
    background: var(--panel); border-left: 1px solid var(--border);
    display: flex; flex-direction: column; padding: 16px 18px;
    overflow-y: auto; gap: 0;
  }}

  /* ── Mobile: stacked layout ── */
  @media (max-width: 700px) {{
    body {{
      grid-template-columns: 1fr;
      grid-template-rows: 55vh auto;
      height: auto;
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
    }}
    .stage {{ height: 55vh; min-height: 220px; overflow: hidden; }}
    .panel {{
      border-left: none;
      border-top: 1px solid var(--border);
      height: auto;
      overflow-y: visible;
      padding-bottom: 40px;
    }}
    .hint {{ margin-top: 20px; }}
    .piece-list {{ max-height: none; }}
    .panel-title {{ font-size: 1.8rem; }}
    .panel-sub {{ font-size: 1.08rem; }}
    .field-label {{ font-size: 1rem; }}
    .size-input {{ font-size: 1.25rem; width: 110px; }}
    .size-unit {{ font-size: 1.15rem; }}
    .size-other {{ font-size: 1rem; }}
    .tip {{ font-size: 1rem; line-height: 1.65; }}
    .toggle-label {{ font-size: 1.08rem; }}
    .btn {{ font-size: 1.1rem; padding: 14px; }}
    .piece-name {{ font-size: 1.02rem; }}
    .piece-size {{ font-size: 0.92rem; }}
  }}

  /* ── Panel typography ── */
  .panel-title {{
    font-family: Georgia, serif; font-size: 1.45rem;
    margin-bottom: 3px; line-height: 1.2;
  }}
  .panel-sub {{ font-size: 0.9rem; font-style: italic; margin-bottom: 12px; }}
  .field-label {{
    font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 6px; margin-top: 12px; display: block;
  }}

  /* ── Size input ── */
  .size-row {{ display: flex; align-items: center; gap: 10px; }}
  .size-input {{
    width: 90px; padding: 10px 12px; border: 2px solid var(--accent);
    border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 1.05rem; font-weight: 500; color: var(--ink);
    background: #fff3ee; outline: none; transition: border-color .2s, box-shadow .2s;
    box-shadow: 0 0 0 3px rgba(200,68,10,0.12);
  }}
  .size-input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 4px rgba(200,68,10,0.22); }}
  .preset-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .preset-btn {{
    padding: 6px 11px; border-radius: 7px; border: 1.5px solid var(--border);
    background: var(--paper); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 0.88rem; font-weight: 500; color: var(--ink);
    cursor: pointer; transition: border-color .15s, background .15s, color .15s;
  }}
  .preset-btn:hover {{ border-color: var(--accent); color: var(--accent); background: #fff3ee; }}
  .preset-btn.active {{ border-color: var(--accent); background: var(--accent); color: #fff; }}
  .size-unit {{ font-size: 0.95rem; }}
  .size-other {{ font-size: 0.82rem; color: var(--muted); font-style: italic; }}
  .tip {{
    font-size: 0.82rem; color: #000; line-height: 1.5;
    margin-top: 6px; padding: 0;
  }}

  /* ── Light dial ── */
  .dial-wrap {{
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 8px 0 4px;
  }}
  .dial-label-row {{
    display: flex; justify-content: space-between; align-items: center;
    width: 100%; font-size: 0.82rem;
  }}
  .dial-angle-val {{
    font-size: 0.82rem; font-weight: 500;
    background: var(--paper); border: 1px solid var(--border);
    border-radius: 5px; padding: 2px 9px;
  }}
  #dialCanvas {{ cursor: crosshair; border-radius: 50%; box-shadow: 0 2px 8px rgba(0,0,0,0.10); }}

  /* ── Toggles ── */
  .toggle-row {{ display: flex; align-items: center; justify-content: space-between; margin-top: 2px; }}
  .toggle-label {{ font-size: 0.94rem; }}
  .toggle {{
    position: relative; width: 40px; height: 22px; background: var(--border);
    border-radius: 11px; cursor: pointer; transition: background .2s; flex-shrink: 0;
  }}
  .toggle.on {{ background: var(--accent); }}
  .toggle::after {{
    content: ''; position: absolute; top: 3px; left: 3px;
    width: 16px; height: 16px; border-radius: 50%; background: #fff;
    transition: transform .2s; box-shadow: 0 1px 3px rgba(0,0,0,0.25);
  }}
  .toggle.on::after {{ transform: translateX(18px); }}

  hr {{ border: none; border-top: 1px solid var(--border); margin: 8px 0; }}

  /* ── Buttons ── */
  .btn {{
    width: 100%; padding: 11px; border-radius: 9px; border: none;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-size: 0.94rem; font-weight: 500;
    cursor: pointer; transition: opacity .2s, transform .15s; margin-top: 6px;
  }}
  .btn:hover {{ opacity: 0.88; transform: translateY(-1px); }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-ghost   {{ background: var(--paper); color: var(--ink); border: 1px solid var(--border); }}
  .btn-sm      {{ padding: 7px 12px !important; font-size: 0.84rem !important; margin-top: 5px !important; }}
  .btn-cart  {{ background: #0a3a14; color: #fff; border: none; width: 100%; }}
  .btn-cart:hover {{ background: #061f0b; opacity: 1 !important; }}
  .btn-cart:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  .btn-danger  {{
    background: #fee2e2 !important; color: #991b1b !important;
    border: 1px solid #fca5a5 !important;
  }}
  .btn-danger:hover {{ background: #fecaca !important; opacity: 1 !important; }}

  /* ── Piece list ── */
  .piece-list {{
    display: flex; flex-direction: column; gap: 4px;
    margin-top: 4px; max-height: 110px; overflow-y: auto; padding-right: 2px;
  }}
  .piece-card {{
    display: flex; align-items: center; gap: 9px;
    background: var(--paper); border: 1.5px solid var(--border);
    border-radius: 9px; padding: 6px 10px; cursor: pointer;
    transition: border-color .15s, background .15s; flex-shrink: 0;
  }}
  .piece-card:hover  {{ background: #f9f6f0; }}
  .piece-card.active {{ border-color: var(--accent); background: #fff3ee; }}
  .piece-thumb {{ border-radius: 4px; background: #ddd8cc; flex-shrink: 0; display: block; }}
  .piece-info  {{ flex: 1; min-width: 0; }}
  .piece-name  {{
    font-size: 0.88rem; font-weight: 500; color: var(--ink);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .piece-size  {{ font-size: 0.76rem; margin-top: 1px; }}
  /* ── Perspective info ── */
  .persp-info {{
    background: #fff8f0; border: 1px solid #f0c090; border-radius: 8px;
    padding: 8px 12px; font-size: 0.82rem; color: #000; line-height: 1.5;
    margin-top: 6px; display: none;
  }}
  .persp-info strong {{ font-weight: 500; }}

  /* ── Hint footer ── */
  .hint {{ display: none; }}

  /* ── Stage loader overlay ── */
  #loadOverlay {{
    position: absolute; inset: 0; background: rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center; z-index: 10;
  }}
  .spinner {{
    width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.2);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .7s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
{_back_btn(ref, parent_origin)}
<!-- ═══ Stage ═══ -->
<div class="stage" id="stage">
  <canvas id="mainCanvas"></canvas>
  <div id="loadOverlay"><div class="spinner"></div></div>
</div>

<!-- ═══ Side Panel ═══ -->
<div class="panel">
  <div class="panel-title">Gallery Wall</div>
  <div class="panel-sub">Step 3 — add &amp; position prints</div>

  <div class="piece-list" id="pieceList"></div>
  <hr>

  <span class="field-label" id="activePieceLabel">{long_axis_label} (inches)</span>
  <div class="size-row">
    <input class="size-input" type="number" id="sizeInInput"
           min="1" max="999" step="0.5" placeholder="{size_placeholder}"
           inputmode="decimal" oninput="onSizeInput(this.value)">
    <span class="size-unit">in</span>
    <span class="size-other" id="sizeOther"></span>
  </div>

  <div class="preset-row" id="presetRow">
    {preset_html}
  </div>
  {cart_ui_html}

  <div class="tip" id="sizeTip">
    {size_tip_body}
    &nbsp;·&nbsp; <strong>Space:</strong> {wall_measurement}" &nbsp;·&nbsp;
    <strong>PPI:</strong> {ppi_orig:.1f}
  </div>

  <hr>

  <span class="field-label">Drop Shadow</span>
  <div class="toggle-row">
    <span class="toggle-label">Enable shadow</span>
    <div class="toggle on" id="shadowToggle" onclick="toggleShadow()"></div>
  </div>

  <span class="field-label">Light Direction</span>
  <div class="dial-wrap">
    <div class="dial-label-row">
      <span>Drag to set light source</span>
      <span class="dial-angle-val" id="dialAngleVal">315°</span>
    </div>
    <canvas id="dialCanvas" width="120" height="120"></canvas>
  </div>

  <span class="field-label">Perspective Adjust</span>
  <div class="toggle-row">
    <span class="toggle-label">Skew picture corners to match wall angle</span>
    <div class="toggle" id="perspToggle" ontouchstart="event.preventDefault();togglePersp();" onclick="togglePersp()"></div>
  </div>
  <div class="persp-info" id="perspInfo">
    <strong>Perspective mode on.</strong> Drag a corner handle to warp the print.
    Drag inside to reposition. Toggle off — the warp is <strong>preserved</strong>
    until you click Reset.
  </div>
  <button class="btn btn-ghost btn-sm" onclick="resetAdjustments()">↺ Reset Adjustments</button>
  <button class="btn btn-danger btn-sm" id="removeBtn" onclick="removeActivePiece()">
    🗑 Remove This Print
  </button>

  <hr>

  <button class="btn btn-primary" onclick="openEmailModal()">✉ Email Mockup</button>
  <a href="{back_url}" class="btn btn-ghost" style="text-decoration:none;display:block;
     text-align:center;margin-top:8px;">← Re-mark Wall</a>
  <a href="{restart_url}" class="btn btn-ghost" style="text-decoration:none;display:block;
     text-align:center;margin-top:4px;font-size:0.85rem;">← Back to Step 1</a>

  <p class="hint">
    • Click a print in the list — or on the canvas — to select it.<br>
    • <strong>Perspective:</strong> toggle on, drag corners, toggle off. Warp stays until Reset.<br>
    • <strong>Reset Adjustments</strong> returns the print to flat &amp; centred.
  </p>
</div>

<script>
// ── Constants ─────────────────────────────────────────────────────────────
const VARIANTS         = {variants_js};
const HAS_PRODUCT_SIZES = {'true' if has_product_sizes else 'false'};
const COA_FIELD        = "{html_mod.escape(coa_field)}";
const GALLERY_SLUG     = "{html_mod.escape(gallery)}";
const PARENT_ORIGIN    = "{html_mod.escape(parent_origin)}";
const ROOM_B64         = "{room_b64}";
const ROOM_MIME        = "{room_mime}";
const FIRST_ART_B64    = "{art_b64}";
const FIRST_ART_MIME   = "{art_mime}";
const FIRST_ART_ASPECT = {art_aspect:.6f};
const WALL_MEAS        = {wall_measurement};
const PPI_ORIG         = {ppi_orig:.6f};   // px/inch in original image space
const ORIG_W           = {orig_w};
const ORIG_H           = {orig_h};

// ── Canvas / stage ────────────────────────────────────────────────────────
const canvas  = document.getElementById('mainCanvas');
const ctx     = canvas.getContext('2d');
const stage   = document.getElementById('stage');
const roomImg = new Image();
let displayScale = 1, ppiDisp = PPI_ORIG;
let lightAngleDeg = 315;

// ── Pieces ────────────────────────────────────────────────────────────────
let pieces = [], activePieceIdx = -1, pieceIdCounter = 0;

function makePiece(img, aspect, name) {{
  return {{
    id: ++pieceIdCounter, name: name || ('Print ' + pieceIdCounter),
    img, aspect,
    artX: 0, artY: 0, artW: 0, artH: 0,
    corners: [],         // [] = flat rect; [4] = warp quad
    perspMode: false,    // true = corner handles visible
    shadowOn: true,
    lastSizeIn: null,
    initialPlaced: false,
  }};
}}

// ── Boot ──────────────────────────────────────────────────────────────────
let roomLoaded = false, firstArtLoaded = false;

roomImg.onload = () => {{ roomLoaded = true; tryStart(); }};
roomImg.onerror = () => {{
  document.getElementById('loadOverlay').innerHTML =
    '<p style="color:#fff;padding:24px;text-align:center">Room image failed to load.<br>Please go back and try again.</p>';
}};

const firstArtImg = new Image();
firstArtImg.onload = () => {{
  firstArtLoaded = true;
  pieces.push(makePiece(firstArtImg, FIRST_ART_ASPECT, 'Print 1'));
  activePieceIdx = 0;
  renderPieceList();
  tryStart();
}};
firstArtImg.onerror = () => {{
  // Still start — user can size later even if art fails
  firstArtLoaded = true;
  tryStart();
}};

roomImg.src     = 'data:' + ROOM_MIME      + ';base64,' + ROOM_B64;
firstArtImg.src = 'data:' + FIRST_ART_MIME + ';base64,' + FIRST_ART_B64;

function tryStart() {{
  if (!roomLoaded || !firstArtLoaded) return;
  document.getElementById('loadOverlay').style.display = 'none';
  drawDial();
  syncPanelToActive();
  (function waitForLayout() {{
    if (stage.clientWidth === 0 || stage.clientHeight === 0) {{
      requestAnimationFrame(waitForLayout); return;
    }}
    initLayout();
    startLoop();
    const inp = document.getElementById('sizeInInput');
    inp.value = '24';
    onSizeInput('24');
    document.querySelectorAll('.preset-btn').forEach(b => {{
      b.classList.toggle('active', b.textContent === '24"');
    }});
  }})();
}}

// ── Drag state ────────────────────────────────────────────────────────────
let dragging = false, dragOffX = 0, dragOffY = 0;
let activeCorner = -1, hoverIdx = -1;

// ── Layout ────────────────────────────────────────────────────────────────
function initLayout() {{
  const stageW = stage.clientWidth, stageH = stage.clientHeight;
  if (stageW === 0 || stageH === 0) return;
  const natW = roomImg.naturalWidth || ORIG_W;
  const natH = roomImg.naturalHeight || ORIG_H;
  const roomAsp = natW / natH;
  let dispW, dispH;
  if (roomAsp > stageW / stageH) {{ dispW = stageW; dispH = stageW / roomAsp; }}
  else                            {{ dispH = stageH; dispW = stageH * roomAsp; }}
  canvas.width  = dispW;
  canvas.height = dispH;
  canvas.style.position = 'absolute';
  canvas.style.top  = Math.max(0, (stageH - dispH) / 2) + 'px';
  canvas.style.left = Math.max(0, (stageW - dispW) / 2) + 'px';
  canvas.style.width  = dispW + 'px';
  canvas.style.height = dispH + 'px';
  displayScale = dispW / natW;
  ppiDisp      = PPI_ORIG * displayScale;
}}

// ── Piece helpers ─────────────────────────────────────────────────────────
function active() {{ return pieces[activePieceIdx] || null; }}

function selectPiece(idx) {{
  activePieceIdx = idx;
  renderPieceList();
  syncPanelToActive();
}}

function renderPieceList() {{
  const ul = document.getElementById('pieceList');
  ul.innerHTML = '';
  pieces.forEach((p, i) => {{
    const card = document.createElement('div');
    card.className = 'piece-card' + (i === activePieceIdx ? ' active' : '');
    card.onclick = () => selectPiece(i);

    const thumb = document.createElement('canvas');
    thumb.className = 'piece-thumb';
    thumb.width = 36; thumb.height = 36;
    const tc = thumb.getContext('2d');
    const a = p.aspect;
    let tw, th;
    if (a >= 1) {{ tw = 36; th = Math.round(36 / a); }}
    else        {{ th = 36; tw = Math.round(36 * a); }}
    tc.drawImage(p.img, (36 - tw) / 2, (36 - th) / 2, tw, th);
    card.appendChild(thumb);

    const info = document.createElement('div'); info.className = 'piece-info';
    const nameEl = document.createElement('div'); nameEl.className = 'piece-name';
    nameEl.textContent = p.name;
    const sizeEl = document.createElement('div'); sizeEl.className = 'piece-size';
    sizeEl.textContent = p.lastSizeIn != null
      ? (p.aspect < 1 ? p.lastSizeIn.toFixed(1) + '" H' : p.lastSizeIn.toFixed(1) + '" W')
      : 'Not sized yet';
    info.appendChild(nameEl); info.appendChild(sizeEl);
    card.appendChild(info);
    ul.appendChild(card);
  }});
  // Show remove button only when there are multiple pieces
  document.getElementById('removeBtn').style.display = pieces.length > 1 ? '' : 'none';
}}

function syncPanelToActive() {{
  const p = active();
  if (!p) {{
    document.getElementById('activePieceLabel').textContent = 'No print selected';
    document.getElementById('sizeInInput').value = '';
    document.getElementById('sizeOther').textContent = '';
    document.getElementById('shadowToggle').classList.remove('on');
    document.getElementById('perspToggle').classList.remove('on');
    document.getElementById('perspInfo').style.display = 'none';
    return;
  }}
  const isPort = p.aspect < 1;
  document.getElementById('activePieceLabel').textContent =
    (isPort ? 'Height' : 'Width') + ' — longest edge (inches)';
  document.getElementById('sizeTip').innerHTML =
    'Enter the <strong>' + (isPort ? 'height' : 'width') + '</strong> of the print '
    + "in inches. The other dimension scales from the artwork's aspect ratio."
    + '<br><br><strong>Measured space:</strong> ' + WALL_MEAS + '"'
    + ' &nbsp;·&nbsp; <strong>Calibrated PPI:</strong> ' + (PPI_ORIG).toFixed(1);

  const inp = document.getElementById('sizeInInput');
  inp.value = p.lastSizeIn != null ? p.lastSizeIn : '';
  const otherEl = document.getElementById('sizeOther');
  if (p.lastSizeIn != null) {{
    const isPort = p.aspect < 1;
    const other = isPort ? (p.lastSizeIn * p.aspect) : (p.lastSizeIn / p.aspect);
    otherEl.textContent = '× ' + other.toFixed(1) + '" ' + (isPort ? 'W' : 'H');
  }} else {{
    otherEl.textContent = '';
  }}
  document.getElementById('shadowToggle').classList.toggle('on', p.shadowOn);
  document.getElementById('perspToggle').classList.toggle('on', p.perspMode);
  document.getElementById('perspInfo').style.display = p.perspMode ? 'block' : 'none';
}}

// ── Size input ─────────────────────────────────────────────────────────────
function onSizeInput(val) {{
  const v = parseFloat(val), p = active();
  if (!p || !v || v <= 0) return;
  p.lastSizeIn = v;
  _applySizeIn(p, v);
  renderPieceList();
  // Sync preset button active state
  document.querySelectorAll('.preset-btn').forEach(b => {{
    b.classList.toggle('active', parseFloat(b.textContent) === v);
  }});
}}

function applyPreset(size, btn) {{
  const inp = document.getElementById('sizeInInput');
  inp.value = size;
  onSizeInput(String(size));
}}

function _applySizeIn(p, sizeIn) {{
  const isPort = p.aspect < 1;
  let newW, newH;
  if (isPort) {{ newH = sizeIn * ppiDisp; newW = newH * p.aspect; }}
  else        {{ newW = sizeIn * ppiDisp; newH = newW / p.aspect; }}

  if (!p.initialPlaced) {{
    p.artX = (canvas.width  - newW) / 2;
    p.artY = (canvas.height - newH) / 2;
    p.artW = newW; p.artH = newH;
    p.initialPlaced = true;
    _clampPiece(p);
    if (p.corners.length === 4) _resetCorners(p);
  }} else if (p.corners.length === 4) {{
    _scaleCornersTo(p, newW, newH);
    p.artW = newW; p.artH = newH;
    _syncBoundsFromCorners(p);
  }} else {{
    const cx = p.artX + p.artW / 2, cy = p.artY + p.artH / 2;
    p.artX = cx - newW / 2; p.artY = cy - newH / 2;
    p.artW = newW; p.artH = newH;
    _clampPiece(p);
  }}

  if (p === active()) {{
    const isPort2 = p.aspect < 1;
    const other = isPort2 ? (sizeIn * p.aspect) : (sizeIn / p.aspect);
    document.getElementById('sizeOther').textContent =
      '× ' + other.toFixed(1) + '" ' + (isPort2 ? 'W' : 'H');
  }}
}}

function _clampPiece(p) {{
  p.artX = p.artW >= canvas.width  ? (canvas.width  - p.artW)/2 : Math.max(0, Math.min(p.artX, canvas.width  - p.artW));
  p.artY = p.artH >= canvas.height ? (canvas.height - p.artH)/2 : Math.max(0, Math.min(p.artY, canvas.height - p.artH));
}}

// ── Perspective helpers ───────────────────────────────────────────────────
function _resetCorners(p) {{
  p.corners = [
    {{x: p.artX,          y: p.artY         }},
    {{x: p.artX + p.artW, y: p.artY         }},
    {{x: p.artX + p.artW, y: p.artY + p.artH}},
    {{x: p.artX,          y: p.artY + p.artH}},
  ];
}}
function _cornersCenter(p) {{
  const c = p.corners;
  return {{ x: (c[0].x+c[1].x+c[2].x+c[3].x)/4, y: (c[0].y+c[1].y+c[2].y+c[3].y)/4 }};
}}
function _scaleCornersTo(p, newW, newH) {{
  if (!p.corners.length) return;
  const ctr = _cornersCenter(p);
  const sx = p.artW > 0 ? newW/p.artW : 1;
  const sy = p.artH > 0 ? newH/p.artH : 1;
  p.corners = p.corners.map(c => ({{x: ctr.x + (c.x-ctr.x)*sx, y: ctr.y + (c.y-ctr.y)*sy}}));
}}
function _syncBoundsFromCorners(p) {{
  const xs = p.corners.map(c=>c.x), ys = p.corners.map(c=>c.y);
  p.artX = Math.min(...xs); p.artY = Math.min(...ys);
}}
function _cornerHitTest(p, cx, cy) {{
  const isTouch = navigator.maxTouchPoints > 0;
  const r = Math.max(isTouch ? 40 : 18, canvas.width * 0.018);
  for (let i = 0; i < p.corners.length; i++) {{
    const dx = cx - p.corners[i].x, dy = cy - p.corners[i].y;
    if (dx*dx + dy*dy < r*r) return i;
  }}
  return -1;
}}

// ── Point-in-shape ────────────────────────────────────────────────────────
function _triSign(ax,ay,bx,by,cx,cy) {{ return (ax-cx)*(by-cy)-(bx-cx)*(ay-cy); }}
function _ptInTri(px,py,ax,ay,bx,by,cx,cy) {{
  const d1=_triSign(px,py,ax,ay,bx,by),d2=_triSign(px,py,bx,by,cx,cy),d3=_triSign(px,py,cx,cy,ax,ay);
  return !((d1<0||d2<0||d3<0)&&(d1>0||d2>0||d3>0));
}}
function _ptInQuad(cs,px,py) {{
  const [tl,tr,br,bl]=cs;
  return _ptInTri(px,py,tl.x,tl.y,tr.x,tr.y,bl.x,bl.y)||_ptInTri(px,py,tr.x,tr.y,br.x,br.y,bl.x,bl.y);
}}
function _pieceHit(p,cx,cy) {{
  if (p.corners.length===4) return _ptInQuad(p.corners,cx,cy);
  return cx>=p.artX&&cx<=p.artX+p.artW&&cy>=p.artY&&cy<=p.artY+p.artH;
}}

// ── Affine warp ───────────────────────────────────────────────────────────
function _computeAffine(sx0,sy0,sx1,sy1,sx2,sy2, dx0,dy0,dx1,dy1,dx2,dy2) {{
  const det=sx0*(sy1-sy2)-sy0*(sx1-sx2)+(sx1*sy2-sy1*sx2);
  if (Math.abs(det)<1e-10) return null;
  const r0=sy1-sy2,r1=sy2-sy0,r2=sy0-sy1,r3=sx2-sx1,r4=sx0-sx2,r5=sx1-sx0;
  const r6=sx1*sy2-sy1*sx2,r7=sy0*sx2-sx0*sy2,r8=sx0*sy1-sy0*sx1;
  return {{
    a:(r0*dx0+r1*dx1+r2*dx2)/det, b:(r0*dy0+r1*dy1+r2*dy2)/det,
    c:(r3*dx0+r4*dx1+r5*dx2)/det, d:(r3*dy0+r4*dy1+r5*dy2)/det,
    e:(r6*dx0+r7*dx1+r8*dx2)/det, f:(r6*dy0+r7*dy1+r8*dy2)/det,
  }};
}}

function _drawWarpTri(img, sx0,sy0,sx1,sy1,sx2,sy2, dx0,dy0,dx1,dy1,dx2,dy2) {{
  const T=_computeAffine(sx0,sy0,sx1,sy1,sx2,sy2, dx0,dy0,dx1,dy1,dx2,dy2);
  if (!T) return;
  ctx.save();
  ctx.beginPath(); ctx.moveTo(dx0,dy0); ctx.lineTo(dx1,dy1); ctx.lineTo(dx2,dy2);
  ctx.closePath(); ctx.clip();
  ctx.setTransform(T.a,T.b,T.c,T.d,T.e,T.f);
  ctx.drawImage(img,0,0,img.naturalWidth,img.naturalHeight);
  ctx.restore();
}}

function _drawPerspArt(p) {{
  const [tl,tr,br,bl]=p.corners, w=p.img.naturalWidth, h=p.img.naturalHeight;
  _drawWarpTri(p.img, 0,0,w,0,0,h, tl.x,tl.y,tr.x,tr.y,bl.x,bl.y);
  _drawWarpTri(p.img, w,0,w,h,0,h, tr.x,tr.y,br.x,br.y,bl.x,bl.y);
}}

function _getShadowOffset(p) {{
  const a=(lightAngleDeg+180)*Math.PI/180, d=Math.max(6,Math.min(p.artW,p.artH)*0.06);
  return {{x:Math.cos(a)*d, y:Math.sin(a)*d, blur:Math.max(8,Math.min(p.artW,p.artH)*0.04)}};
}}

// ── Draw ──────────────────────────────────────────────────────────────────
let isDownloading = false;

function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(roomImg,0,0,canvas.width,canvas.height);
  pieces.forEach((p,i) => _drawPiece(p,i));
}}

function _drawPiece(p, pieceIdx) {{
  const hasWarp = p.corners.length === 4;
  if (!hasWarp && (p.artW <= 0 || p.artH <= 0)) return;

  // Shadow
  if (p.shadowOn) {{
    const sh=_getShadowOffset(p);
    ctx.save();
    if (hasWarp) {{
      ctx.filter=`blur(${{sh.blur.toFixed(1)}}px)`; ctx.fillStyle='rgba(0,0,0,0.52)';
      ctx.beginPath();
      p.corners.forEach((c,i)=> i===0?ctx.moveTo(c.x+sh.x,c.y+sh.y):ctx.lineTo(c.x+sh.x,c.y+sh.y));
      ctx.closePath(); ctx.fill();
    }} else {{
      // Use canvas shadow API so transparent-PNG artwork (oval, circle, etc.)
      // casts a shape-following shadow rather than a rectangular one.
      ctx.shadowColor='rgba(0,0,0,0.52)';
      ctx.shadowBlur=sh.blur;
      ctx.shadowOffsetX=sh.x;
      ctx.shadowOffsetY=sh.y;
      ctx.drawImage(p.img,p.artX,p.artY,p.artW,p.artH);
    }}
    ctx.restore();
  }}

  // Image
  if (hasWarp) {{ _drawPerspArt(p); }}
  else         {{ ctx.drawImage(p.img,p.artX,p.artY,p.artW,p.artH); }}

  if (isDownloading) return;

  // Corner handles (active piece in perspMode only)
  const isActive = pieceIdx === activePieceIdx;
  if (isActive && p.perspMode && hasWarp) {{
    const isTouch = navigator.maxTouchPoints > 0;
    const hr=Math.max(isTouch ? 14 : 7, canvas.width*0.009);
    p.corners.forEach((pt,i) => {{
      ctx.beginPath(); ctx.arc(pt.x,pt.y,hr,0,Math.PI*2);
      ctx.fillStyle = i===activeCorner?'#e8822a':'#fff'; ctx.fill();
      ctx.strokeStyle='rgba(0,0,0,0.35)'; ctx.lineWidth=1.5; ctx.stroke();
    }});
  }}
}}

function startLoop() {{
  requestAnimationFrame(function loop(){{ draw(); requestAnimationFrame(loop); }});
}}

// ── Toggle perspective ────────────────────────────────────────────────────
let _perspToggleLocked = false;
function togglePersp() {{
  if (_perspToggleLocked) return;
  _perspToggleLocked = true;
  setTimeout(() => {{ _perspToggleLocked = false; }}, 600);
  if (activePieceIdx < 0 && pieces.length > 0) selectPiece(0);
  const p = active(); if (!p) {{ _perspToggleLocked = false; return; }}
  p.perspMode = !p.perspMode;
  document.getElementById('perspToggle').classList.toggle('on', p.perspMode);
  document.getElementById('perspInfo').style.display = p.perspMode ? 'block' : 'none';
  if (p.perspMode && p.corners.length === 0 && p.artW > 0 && p.artH > 0) _resetCorners(p);
}}

// ── Reset adjustments ─────────────────────────────────────────────────────
function resetAdjustments() {{
  const p = active(); if (!p) return;
  p.corners=[]; p.perspMode=false;
  document.getElementById('perspToggle').classList.remove('on');
  document.getElementById('perspInfo').style.display='none';
  if (p.artW > 0 && p.artH > 0) {{
    p.artX=(canvas.width -p.artW)/2; p.artY=(canvas.height-p.artH)/2; _clampPiece(p);
  }}
}}

// ── Toggle shadow ─────────────────────────────────────────────────────────
function toggleShadow() {{
  const p = active(); if (!p) return;
  p.shadowOn = !p.shadowOn;
  document.getElementById('shadowToggle').classList.toggle('on', p.shadowOn);
}}

// ── Remove active piece ───────────────────────────────────────────────────
function removeActivePiece() {{
  if (pieces.length <= 1) return;
  pieces.splice(activePieceIdx, 1);
  activePieceIdx = Math.min(activePieceIdx, pieces.length - 1);
  renderPieceList(); syncPanelToActive();
}}

// ── Canvas coordinate helpers ─────────────────────────────────────────────
function canvasCoords(e) {{
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return [
    (src.clientX - rect.left) * (canvas.width  / rect.width),
    (src.clientY - rect.top)  * (canvas.height / rect.height),
  ];
}}
function _topPieceAt(cx,cy) {{
  for (let i = pieces.length-1; i >= 0; i--) {{
    if (_pieceHit(pieces[i],cx,cy)) return i;
  }}
  return -1;
}}

// ── Ghost mousedown suppression (1 s window after any touch on the page) ──
let _lastTouchTime = 0;
document.addEventListener('touchstart', () => {{ _lastTouchTime = Date.now(); }}, {{passive: true}});

// ── Mouse events ──────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {{
  if (dialDragging) return;
  if (Date.now() - _lastTouchTime < 1000) return; // suppress ghost mousedown on touch devices
  const [cx,cy] = canvasCoords(e);
  if (cx < 0 || cx > canvas.width || cy < 0 || cy > canvas.height) return;
  const p = active();
  if (p && p.perspMode && p.corners.length===4) {{
    const ci = _cornerHitTest(p,cx,cy);
    if (ci >= 0) {{ activeCorner=ci; dragging=true; stage.classList.add('dragging'); e.preventDefault(); return; }}
  }}
  if (p && _pieceHit(p,cx,cy)) {{
    dragging=true; activeCorner=-1;
    if (p.corners.length===4) {{ const ctr=_cornersCenter(p); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else {{ dragOffX=cx-p.artX; dragOffY=cy-p.artY; }}
    stage.classList.add('dragging'); e.preventDefault(); return;
  }}
  const idx = _topPieceAt(cx,cy);
  if (idx >= 0 && idx !== activePieceIdx) {{
    selectPiece(idx);
    const np = pieces[idx]; dragging=true; activeCorner=-1;
    if (np.corners.length===4) {{ const ctr=_cornersCenter(np); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else {{ dragOffX=cx-np.artX; dragOffY=cy-np.artY; }}
    stage.classList.add('dragging'); e.preventDefault();
  }} else if (idx < 0) {{
    activePieceIdx = -1;
    renderPieceList();
    syncPanelToActive();
  }}
}});

window.addEventListener('mousemove', e => {{
  if (dialDragging) return;
  const [cx,cy] = canvasCoords(e), p = active();
  if (dragging && p) {{
    if (p.perspMode && activeCorner >= 0) {{
      p.corners[activeCorner] = {{x:cx,y:cy}};
    }} else if (p.corners.length===4) {{
      const ctr=_cornersCenter(p), ddx=(cx-dragOffX)-ctr.x, ddy=(cy-dragOffY)-ctr.y;
      p.corners = p.corners.map(c=>({{x:c.x+ddx,y:c.y+ddy}}));
    }} else {{
      p.artX=cx-dragOffX; p.artY=cy-dragOffY; _clampPiece(p);
    }}
    return;
  }}
  hoverIdx = _topPieceAt(cx,cy);
  const onCorner = p&&p.perspMode&&p.corners.length===4&&_cornerHitTest(p,cx,cy)>=0;
  canvas.style.cursor = onCorner ? 'crosshair' : (hoverIdx >= 0 ? 'grab' : 'default');
}});

window.addEventListener('mouseup', () => {{
  if (dragging) {{ const p=active(); if (p&&p.corners.length===4&&activeCorner===-1) _syncBoundsFromCorners(p); }}
  dragging=false; activeCorner=-1; stage.classList.remove('dragging');
}});

// ── Touch events ──────────────────────────────────────────────────────────
canvas.addEventListener('touchstart', e => {{
  const [cx,cy]=canvasCoords(e), p=active();
  if (p&&p.perspMode&&p.corners.length===4) {{
    const ci=_cornerHitTest(p,cx,cy);
    if (ci>=0) {{ activeCorner=ci; dragging=true; e.preventDefault(); return; }}
  }}
  if (p&&_pieceHit(p,cx,cy)) {{
    dragging=true; activeCorner=-1;
    if (p.corners.length===4) {{ const ctr=_cornersCenter(p); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else {{ dragOffX=cx-p.artX; dragOffY=cy-p.artY; }}
    e.preventDefault(); return;
  }}
  const idx=_topPieceAt(cx,cy);
  if (idx>=0&&idx!==activePieceIdx) {{
    selectPiece(idx);
    const np=pieces[idx]; dragging=true; activeCorner=-1;
    if (np.corners.length===4) {{ const ctr=_cornersCenter(np); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else {{ dragOffX=cx-np.artX; dragOffY=cy-np.artY; }}
    e.preventDefault();
  }}
  // Empty-canvas touch on mobile does nothing — no deselect
}}, {{passive: false}});

window.addEventListener('touchmove', e => {{
  if (!dragging) return;
  const [cx,cy]=canvasCoords(e), p=active(); if (!p) return;
  if (p.perspMode&&activeCorner>=0) {{
    p.corners[activeCorner]={{x:cx,y:cy}};
  }} else if (p.corners.length===4) {{
    const ctr=_cornersCenter(p),ddx=(cx-dragOffX)-ctr.x,ddy=(cy-dragOffY)-ctr.y;
    p.corners=p.corners.map(c=>({{x:c.x+ddx,y:c.y+ddy}}));
  }} else {{ p.artX=cx-dragOffX; p.artY=cy-dragOffY; _clampPiece(p); }}
  e.preventDefault();
}}, {{passive: false}});

window.addEventListener('touchend', () => {{
  if (dragging) {{ const p=active(); if (p&&p.corners.length===4&&activeCorner===-1) _syncBoundsFromCorners(p); }}
  dragging=false; activeCorner=-1;
}});

// ── Light direction dial ──────────────────────────────────────────────────
const dialCanvas=document.getElementById('dialCanvas'), dialCtx=dialCanvas.getContext('2d');
const DIAL_R=56, DIAL_CX=60, DIAL_CY=60, HANDLE_R=8;
let dialDragging=false;

function drawDial() {{
  dialCtx.clearRect(0,0,120,120);
  const grad=dialCtx.createRadialGradient(DIAL_CX,DIAL_CY,0,DIAL_CX,DIAL_CY,DIAL_R);
  grad.addColorStop(0,'#2e2e38'); grad.addColorStop(1,'#1a1a22');
  dialCtx.beginPath(); dialCtx.arc(DIAL_CX,DIAL_CY,DIAL_R,0,Math.PI*2);
  dialCtx.fillStyle=grad; dialCtx.fill();
  dialCtx.strokeStyle='#3a3a48'; dialCtx.lineWidth=2; dialCtx.stroke();
  for(let i=0;i<8;i++){{
    const a=i*Math.PI/4, inner=i%2===0?DIAL_R-10:DIAL_R-6;
    dialCtx.beginPath();
    dialCtx.moveTo(DIAL_CX+Math.cos(a)*inner,DIAL_CY+Math.sin(a)*inner);
    dialCtx.lineTo(DIAL_CX+Math.cos(a)*(DIAL_R-2),DIAL_CY+Math.sin(a)*(DIAL_R-2));
    dialCtx.strokeStyle='#55556a'; dialCtx.lineWidth=i%2===0?1.5:1; dialCtx.stroke();
  }}
  dialCtx.beginPath(); dialCtx.arc(DIAL_CX,DIAL_CY,3,0,Math.PI*2);
  dialCtx.fillStyle='#888896'; dialCtx.fill();
  const ar=lightAngleDeg*Math.PI/180;
  const hx=DIAL_CX+Math.cos(ar)*(DIAL_R-HANDLE_R-2), hy=DIAL_CY+Math.sin(ar)*(DIAL_R-HANDLE_R-2);
  dialCtx.beginPath(); dialCtx.moveTo(DIAL_CX,DIAL_CY); dialCtx.lineTo(hx,hy);
  dialCtx.strokeStyle='#c8440a'; dialCtx.lineWidth=1.5; dialCtx.stroke();
  dialCtx.beginPath(); dialCtx.moveTo(DIAL_CX,DIAL_CY);
  dialCtx.lineTo(DIAL_CX-Math.cos(ar)*DIAL_R*0.5,DIAL_CY-Math.sin(ar)*DIAL_R*0.5);
  dialCtx.strokeStyle='rgba(100,100,120,0.5)'; dialCtx.lineWidth=1; dialCtx.stroke();
  const hg=dialCtx.createRadialGradient(hx-2,hy-2,1,hx,hy,HANDLE_R);
  hg.addColorStop(0,'#ff8855'); hg.addColorStop(1,'#c8440a');
  dialCtx.beginPath(); dialCtx.arc(hx,hy,HANDLE_R,0,Math.PI*2);
  dialCtx.fillStyle=hg; dialCtx.fill();
  dialCtx.strokeStyle='rgba(255,255,255,0.4)'; dialCtx.lineWidth=1; dialCtx.stroke();
  document.getElementById('dialAngleVal').textContent=Math.round(lightAngleDeg)+'°';
}}

function dialSetAngle(e) {{
  const rect=dialCanvas.getBoundingClientRect(), src=e.touches?e.touches[0]:e;
  const cx=(src.clientX-rect.left)*(120/rect.width);
  const cy=(src.clientY-rect.top) *(120/rect.height);
  lightAngleDeg=((Math.atan2(cy-DIAL_CY,cx-DIAL_CX)*180/Math.PI)+360)%360;
  drawDial();
}}
dialCanvas.addEventListener('mousedown', e=>{{dialDragging=true;dialSetAngle(e);e.preventDefault();}});
window.addEventListener('mousemove',    e=>{{if(dialDragging)dialSetAngle(e);}});
window.addEventListener('mouseup',      ()=>{{dialDragging=false;}});
dialCanvas.addEventListener('touchstart', e=>{{dialDragging=true;dialSetAngle(e);e.preventDefault();}},{{passive:false}});
window.addEventListener('touchmove',    e=>{{if(dialDragging){{dialSetAngle(e);e.preventDefault();}}}},{{passive:false}});
window.addEventListener('touchend',     ()=>{{dialDragging=false;}});

// ── Download ──────────────────────────────────────────────────────────────
function downloadMockup() {{
  const savedActive=activePieceIdx, savedHover=hoverIdx;
  activePieceIdx=-1; hoverIdx=-1; isDownloading=true;
  dragging=false; activeCorner=-1;
  draw();
  const link=document.createElement('a');
  link.download='wall-mockup.jpg';
  link.href=canvas.toDataURL('image/jpeg',0.94);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  activePieceIdx=savedActive; hoverIdx=savedHover; isDownloading=false;
}}

// ── Resize handling ───────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  if (!roomLoaded || pieces.length === 0) return;
  // Save normalised positions for all pieces
  const saved = pieces.map(p => ({{
    normX:    canvas.width  ? (p.artX + p.artW/2) / canvas.width  : 0.5,
    normY:    canvas.height ? (p.artY + p.artH/2) / canvas.height : 0.5,
    normCorn: p.corners.length===4
      ? p.corners.map(c=>({{x: canvas.width?c.x/canvas.width:0, y: canvas.height?c.y/canvas.height:0}}))
      : null,
  }}));
  initLayout();
  pieces.forEach((p, i) => {{
    if (p.lastSizeIn === null) return;
    _applySizeIn(p, p.lastSizeIn);
    const s = saved[i];
    p.artX = s.normX * canvas.width  - p.artW/2;
    p.artY = s.normY * canvas.height - p.artH/2;
    _clampPiece(p);
    if (s.normCorn) {{
      p.corners = s.normCorn.map(c=>({{x: c.x*canvas.width, y: c.y*canvas.height}}));
    }}
  }});
}});

// ── Email modal ───────────────────────────────────────────────────────────
function openEmailModal() {{
  document.getElementById('emailModal').style.display = 'flex';
  document.getElementById('emailInput').value = '';
  document.getElementById('emailError').style.display = 'none';
  const btn = document.getElementById('emailSendBtn');
  btn.textContent = 'Send'; btn.disabled = false; btn.style.background = '#c8440a';
  setTimeout(() => document.getElementById('emailInput').focus(), 50);
}}
function closeEmailModal() {{
  document.getElementById('emailModal').style.display = 'none';
}}
function sendEmail() {{
  const email = document.getElementById('emailInput').value.trim();
  const errEl = document.getElementById('emailError');
  if (!email || !/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email)) {{
    errEl.textContent = 'Please enter a valid email address.';
    errEl.style.display = 'block';
    return;
  }}
  errEl.style.display = 'none';
  const savedActive = activePieceIdx, savedHover = hoverIdx;
  activePieceIdx = -1; hoverIdx = -1; isDownloading = true;
  dragging = false; activeCorner = -1;
  draw();
  const imageB64 = canvas.toDataURL('image/jpeg', 0.94).replace(/^data:image\\/jpeg;base64,/, '');
  activePieceIdx = savedActive; hoverIdx = savedHover; isDownloading = false;
  const btn = document.getElementById('emailSendBtn');
  btn.textContent = 'Sending\u2026'; btn.disabled = true;
  const fd = new FormData();
  fd.append('recipient', email);
  fd.append('image_b64', imageB64);
  fd.append('gallery', GALLERY_SLUG);
  const ap = active();
  if (ap && ap.lastSizeIn) {{
    const isPort = ap.aspect < 1;
    const longIn  = ap.lastSizeIn;
    const shortIn = isPort ? Math.round((longIn * ap.aspect) * 10) / 10
                           : Math.round((longIn / ap.aspect) * 10) / 10;
    fd.append('print_size', longIn + ' x ' + shortIn);
  }}
  fetch('/mockup/email', {{ method: 'POST', body: fd }})
    .then(r => r.ok ? r.json() : r.json().then(j => Promise.reject(j.detail || 'Error')))
    .then(() => {{
      btn.textContent = '\u2713 Sent!'; btn.style.background = '#16a34a';
      setTimeout(closeEmailModal, 1800);
    }})
    .catch(err => {{
      errEl.textContent = typeof err === 'string' ? err : 'Failed to send. Please try again.';
      errEl.style.display = 'block';
      btn.textContent = 'Send'; btn.disabled = false;
    }});
}}
</script>

<!-- ═══ Email Modal ═══ -->
<div id="emailModal" style="display:none;position:fixed;inset:0;
     background:rgba(0,0,0,0.55);z-index:200;
     align-items:center;justify-content:center;">
  <div style="background:#fff;border-radius:14px;padding:32px 28px;
              max-width:380px;width:90%;box-shadow:0 8px 40px rgba(0,0,0,0.25);">
    <h3 style="font-family:Georgia,serif;font-size:1.3rem;margin-bottom:8px;">Email Mockup</h3>
    <p style="font-size:0.9rem;color:#555;margin-bottom:18px;line-height:1.5;">
      Enter your email address and we&#39;ll send your mockup as a JPEG attachment.
    </p>
    <input id="emailInput" type="email" placeholder="you@example.com"
           style="width:100%;padding:11px 14px;border:1.5px solid #d8d4cc;
                  border-radius:9px;font-size:1rem;font-family:inherit;
                  outline:none;margin-bottom:10px;box-sizing:border-box;"
           onfocus="this.style.borderColor='#c8440a'"
           onblur="this.style.borderColor='#d8d4cc'"
           onkeydown="if(event.key==='Enter')sendEmail()">
    <div id="emailError" style="display:none;color:#c8440a;
         font-size:0.85rem;margin-bottom:10px;"></div>
    <div style="display:flex;gap:10px;">
      <button onclick="closeEmailModal()"
              style="flex:1;padding:11px;border-radius:9px;border:1px solid #d8d4cc;
                     background:#f5f2ec;font-family:inherit;font-size:0.95rem;
                     font-weight:500;cursor:pointer;">Cancel</button>
      <button id="emailSendBtn" onclick="sendEmail()"
              style="flex:2;padding:11px;border-radius:9px;border:none;
                     background:#c8440a;color:#fff;font-family:inherit;
                     font-size:0.95rem;font-weight:500;cursor:pointer;
                     transition:opacity .2s,background .3s;">Send</button>
    </div>
  </div>
</div>
<script>
// ── Product size selection & Add to Cart ──────────────────────────────────
let selectedVariantId = null;
let selectedVariantPrice = '';

function selectProductSize(variantId, price, longEdge, btn) {{
  selectedVariantId = variantId;
  selectedVariantPrice = price;
  applyPreset(longEdge, btn);
  updateCartUI();
}}

function updateCartUI() {{
  if (!HAS_PRODUCT_SIZES) return;
  const cartBtn    = document.getElementById('addToCartBtn');
  const contactBtn = document.getElementById('contactSizeBtn');
  const coaPanel   = document.getElementById('coaPanel');
  if (!cartBtn) return;
  // Reset COA panel whenever variant changes
  if (coaPanel) {{
    coaPanel.style.display = 'none';
    const coaName = document.getElementById('coaName');
    const confirmBtn = document.getElementById('confirmCartBtn');
    if (coaName) coaName.value = '';
    if (confirmBtn) {{ confirmBtn.disabled = true; confirmBtn.textContent = 'Confirm \u0026 Add to Cart'; }}
  }}
  if (selectedVariantId) {{
    cartBtn.textContent = 'Add to Cart' + (selectedVariantPrice ? ' \u2014 ' + selectedVariantPrice : '');
    cartBtn.style.display = 'block';
    cartBtn.disabled = false;
    cartBtn.onclick = addToCart;
    if (contactBtn) contactBtn.style.display = 'none';
  }} else {{
    cartBtn.style.display = 'none';
    if (contactBtn) contactBtn.style.display = 'block';
  }}
}}

function addToCart() {{
  if (!selectedVariantId) return;
  if (COA_FIELD) {{
    // Show COA panel; defer cart until name is confirmed
    const coaPanel = document.getElementById('coaPanel');
    const cartBtn  = document.getElementById('addToCartBtn');
    if (coaPanel) coaPanel.style.display = 'block';
    if (cartBtn)  cartBtn.style.display  = 'none';
    const coaInput = document.getElementById('coaName');
    if (coaInput) coaInput.focus();
    return;
  }}
  _fireAddToCart('');
}}

function updateCoaBtn() {{
  const name = document.getElementById('coaName').value.trim();
  const btn  = document.getElementById('confirmCartBtn');
  if (btn) btn.disabled = !name;
}}

function confirmAddToCart() {{
  const name = (document.getElementById('coaName').value || '').trim();
  if (!name) return;
  const btn = document.getElementById('confirmCartBtn');
  if (btn) {{ btn.textContent = 'Adding\u2026'; btn.disabled = true; }}
  _fireAddToCart(name);
}}

function _fireAddToCart(coaName) {{
  const v = VARIANTS.find(x => x.id === selectedVariantId);
  if (!v || !v.sku || !v.itemId) return;
  window.parent.postMessage({{
    action:   'wallymock_addToCart',
    sku:      v.sku,
    itemId:   v.itemId,
    coaField: COA_FIELD,
    coaName:  coaName
  }}, PARENT_ORIGIN);
}}

window.addEventListener('message', function(e) {{
  if (!PARENT_ORIGIN || e.origin !== PARENT_ORIGIN) return;
  if (!e.data) return;
  const coaPanel   = document.getElementById('coaPanel');
  const coaVisible = coaPanel && coaPanel.style.display !== 'none';
  const activeBtn  = coaVisible
    ? document.getElementById('confirmCartBtn')
    : document.getElementById('addToCartBtn');
  if (!activeBtn) return;
  if (e.data.action === 'wallymock_cartSuccess') {{
    activeBtn.textContent = 'Added! \u2714 View Cart \u2192';
    activeBtn.disabled = false;
    activeBtn.onclick = function() {{ window.parent.location.href = PARENT_ORIGIN + '/cart'; }};
  }} else if (e.data.action === 'wallymock_cartError') {{
    activeBtn.textContent = 'Could not add to cart \u2014 try again';
    activeBtn.disabled = false;
    activeBtn.onclick = coaVisible ? confirmAddToCart : addToCart;
  }}
}});

if (HAS_PRODUCT_SIZES) {{
  const _origOnSizeInput = onSizeInput;
  onSizeInput = function(val) {{
    _origOnSizeInput(val);
    const num = parseFloat(val);
    if (!isNaN(num) && num > 0) {{
      const match = VARIANTS.find(v => Math.abs(v.long - num) <= 0.5);
      if (match) {{
        selectedVariantId = match.id;
        selectedVariantPrice = match.price;
      }} else {{
        selectedVariantId = null;
        selectedVariantPrice = '';
      }}
    }} else {{
      selectedVariantId = null;
      selectedVariantPrice = '';
    }}
    updateCartUI();
  }};
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Email mockup endpoint ─────────────────────────────────────────────────────

@app.post("/mockup/email")
async def email_mockup(
    recipient:  str = Form(...),
    image_b64:  str = Form(...),
    print_size: str = Form(default=""),
    gallery:    str = Form(default="kenhoehn"),
) -> Response:
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', recipient):
        raise HTTPException(400, "Invalid email address.")
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(400, "Invalid image data.")
    if not _RESEND_KEY:
        raise HTTPException(500, "Email sending is not configured on this server.")
    tenant = _get_tenant(gallery)
    size_line = f"Print size selected: {print_size} inches\n\n" if print_size else ""
    bcc = tenant.get("bcc_email") or tenant.get("email") or _FROM_EMAIL
    intro     = (tenant.get("email_intro",     "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    body      = (tenant.get("email_body",      "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    signature = (tenant.get("email_signature", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    try:
        import resend
        resend.api_key = _RESEND_KEY
        resend.Emails.send({
            "from": f"WallyMock <{_FROM_EMAIL}>",
            "to": [recipient],
            "bcc": [bcc],
            "subject": "Your WallyMock Mockup",
            "text": (
                f"Hello there,\n\n"
                f"{intro}\n\n"
                f"{size_line}"
                f"An amazing piece of art can transform your experience at home!\n\n"
                f"{body}\n\n"
                f"{signature}"
            ),
            "attachments": [{"filename": "wall-mockup.jpg", "content": list(img_bytes)}],
        })
    except Exception as exc:
        logger.error("Resend failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(500, f"Could not send email: {exc}")
    logger.info("Mockup emailed to %s (gallery=%s)", recipient, gallery)
    return Response(content='{"ok":true}', media_type="application/json")


# ── Image serving endpoint ────────────────────────────────────────────────────

@app.get("/mockup/img/{uid}")
async def serve_image(uid: str) -> Response:
    """Serve a previously-stored image by UUID."""
    data, mime = _load_image(uid)
    return Response(
        content=data,
        media_type=mime,
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Length": str(len(data)),
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)