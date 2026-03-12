from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from typing import Optional
import cv2
import numpy as np
from io import BytesIO
import base64
import math

app = FastAPI(title="Wall Mockup Tool")

MAX_FILE_SIZE      = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def detect_mime(pil_img: Image.Image) -> str:
    return Image.MIME.get(pil_img.format or "", "image/jpeg")


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wall Mockup Tool</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ink: #1a1a18; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #8a8880; --border: #d8d4cc; --blue: #2563eb;
  }
  body {
    background: var(--paper); color: var(--ink);
    font-family: 'DM Sans', sans-serif; font-weight: 300;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 56px 20px 80px;
  }
  .logo {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .tagline {
    color: var(--muted); font-size: 0.95rem;
    font-style: italic; margin-bottom: 56px;
  }

  /* ── How it works ── */
  .how-section {
    width: 100%; max-width: 860px; margin-bottom: 52px;
  }
  .how-section h2 {
    font-family: 'DM Serif Display', serif; font-size: 1.45rem;
    margin-bottom: 22px; color: var(--ink);
  }
  .steps {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 16px;
  }
  .step-card {
    background: #fff; border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 20px; position: relative;
  }
  .step-num {
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--accent); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 500; margin-bottom: 12px;
  }
  .step-card h3 { font-size: 0.92rem; font-weight: 500; margin-bottom: 6px; }
  .step-card p  { font-size: 0.78rem; color: var(--muted); line-height: 1.6; }

  /* ── Upload card ── */
  .upload-card {
    background: #fff; border: 1px solid var(--border); border-radius: 18px;
    padding: 42px 40px; width: 100%; max-width: 560px;
  }
  .upload-card h2 {
    font-family: 'DM Serif Display', serif; font-size: 1.55rem;
    margin-bottom: 6px;
  }
  .upload-card .card-sub {
    color: var(--muted); font-size: 0.82rem; font-style: italic;
    margin-bottom: 30px;
  }
  label.field-label {
    display: block; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--muted); margin-bottom: 7px; margin-top: 20px;
  }
  label.field-label:first-of-type { margin-top: 0; }

  input[type="file"], input[type="number"] {
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 9px; font-family: inherit; font-size: 0.9rem;
    background: var(--paper); color: var(--ink); outline: none;
    transition: border-color .2s;
  }
  input:focus { border-color: var(--accent); }

  /* ── Orientation toggle (radio-as-card) ── */
  .orient-row {
    display: flex; gap: 12px; margin-top: 6px;
  }
  /* Hide the native radio button completely */
  .orient-row input[type="radio"] {
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
  }
  /* The <label> IS the visible card */
  .orient-card {
    flex: 1; border: 2px solid var(--border); border-radius: 10px;
    padding: 13px 10px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s; background: var(--paper);
    user-select: none; display: block;
  }
  /* Highlight when the associated radio is checked */
  .orient-row input[type="radio"]:checked + .orient-card {
    border-color: var(--accent); background: #fff3ee;
  }
  .orient-card .icon {
    font-size: 1.6rem; display: block; margin-bottom: 5px; line-height: 1;
  }
  .orient-card .lbl { font-size: 0.8rem; font-weight: 500; color: var(--ink); }
  .orient-card .sublbl { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }

  /* ── Tip box ── */
  .tip {
    background: #f0f6ff; border: 1px solid #bdd3f5; border-radius: 9px;
    padding: 12px 16px; margin-top: 20px;
    font-size: 0.78rem; color: #1e40af; line-height: 1.6;
  }
  .tip strong { font-weight: 500; }

  button[type="submit"] {
    margin-top: 28px; width: 100%; padding: 14px;
    background: var(--accent); color: #fff; border: none;
    border-radius: 9px; font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem; font-weight: 500; cursor: pointer;
    transition: opacity .2s, transform .15s;
  }
  button[type="submit"]:hover { opacity: 0.88; transform: translateY(-1px); }
</style>
</head>
<body>

<div class="logo">Wall Mockup</div>
<p class="tagline">See any print to scale on any wall — in seconds</p>

<!-- How it works -->
<div class="how-section">
  <h2>How it works</h2>
  <div class="steps">
    <div class="step-card">
      <div class="step-num">1</div>
      <h3>Upload your photos</h3>
      <p>Choose a room/wall photo and your artwork or print file. Enter a real-world measurement of the wall space.</p>
    </div>
    <div class="step-card">
      <div class="step-num">2</div>
      <h3>Mark the measurement</h3>
      <p>Click two points on the photo that match your measurement — either wall edges (horizontal) or ceiling &amp; floor (vertical).</p>
    </div>
    <div class="step-card">
      <div class="step-num">3</div>
      <h3>Set print size</h3>
      <p>Type the longest edge of your print in inches. The mockup is generated to exact scale based on your measurement.</p>
    </div>
    <div class="step-card">
      <div class="step-num">4</div>
      <h3>Position &amp; download</h3>
      <p>Drag the print anywhere on the wall. Set the light direction for a realistic shadow, then download your mockup.</p>
    </div>
  </div>
</div>

<!-- Upload form -->
<div class="upload-card">
  <h2>Create a Mockup</h2>
  <p class="card-sub">Step 1 of 3 — Upload &amp; configure</p>

  <form action="/mockup/picker" enctype="multipart/form-data" method="post" id="mainForm">

    <label class="field-label">Room / Wall Photo</label>
    <input type="file" name="room_file" accept="image/*" required
           onchange="previewFile(this,'roomPrev')">

    <label class="field-label">Artwork / Print File</label>
    <input type="file" name="art_file" accept="image/*" required
           onchange="previewFile(this,'artPrev')">

    <label class="field-label">Measured Distance (inches)</label>
    <input type="number" name="wall_measurement" step="0.1" min="1"
           placeholder="e.g. 96" required id="wallMeasInput">

    <label class="field-label">Print Orientation</label>
    <div class="orient-row">
      <!-- Radio H (checked by default = form value "H" submitted natively) -->
      <input type="radio" name="orientation" id="orH" value="H" checked>
      <label class="orient-card" for="orH" onclick="updateTip('H')">
        <span class="icon">⬛️</span>
        <div class="lbl">Horizontal</div>
        <div class="sublbl">Landscape / wide print</div>
      </label>

      <!-- Radio V -->
      <input type="radio" name="orientation" id="orV" value="V">
      <label class="orient-card" for="orV" onclick="updateTip('V')">
        <span class="icon">▮</span>
        <div class="lbl">Vertical</div>
        <div class="sublbl">Portrait / tall print</div>
      </label>
    </div>

    <div class="tip" id="orientTip">
      <strong>Horizontal selected:</strong> In the next step you'll click the
      <strong>left edge</strong> then the <strong>right edge</strong> of the space you measured
      (e.g. two points spanning across the wall width).
    </div>

    <button type="submit">Next: Mark the Measurement →</button>
  </form>
</div>

<script>
/* Called by onclick on the label elements — reliable in all browsers */
function updateTip(v) {
  const tip = document.getElementById('orientTip');
  if (v === 'V') {
    tip.innerHTML = '<strong>Vertical selected:</strong> In the next step you\'ll click the '
      + '<strong>ceiling / top point</strong> then the <strong>floor / bottom point</strong> '
      + 'of the space you measured (e.g. two points spanning floor to ceiling).';
  } else {
    tip.innerHTML = '<strong>Horizontal selected:</strong> In the next step you\'ll click the '
      + '<strong>left edge</strong> then the <strong>right edge</strong> of the space you measured '
      + '(e.g. two points spanning across the wall width).';
  }
}
</script>
</body>
</html>"""


# ── Mockup Step 2 – Wall Picker ───────────────────────────────────────────────

@app.post("/mockup/picker", response_class=HTMLResponse)
async def mockup_picker(
    room_file:        UploadFile = File(...),
    art_file:         UploadFile = File(...),
    wall_measurement: float      = Form(...),
    orientation:      str        = Form("H"),   # "H" = horizontal, "V" = vertical
):
    if wall_measurement <= 0:
        raise HTTPException(400, "Measurement must be positive")

    orientation = orientation.strip().upper()
    if orientation not in ("H", "V"):
        orientation = "H"

    room_bytes = await room_file.read()
    art_bytes  = await art_file.read()

    room_arr = cv2.imdecode(np.frombuffer(room_bytes, np.uint8), cv2.IMREAD_COLOR)
    if room_arr is None:
        raise HTTPException(400, "Could not read room image. Please use JPG, PNG, or TIFF.")

    orig_h, orig_w = room_arr.shape[:2]

    room_pil  = Image.open(BytesIO(room_bytes))
    art_pil   = Image.open(BytesIO(art_bytes))
    room_mime = detect_mime(room_pil)
    art_mime  = detect_mime(art_pil)

    room_b64 = encode_b64(room_bytes)
    art_b64  = encode_b64(art_bytes)

    is_vertical = (orientation == "V")

    # Dynamic instruction text
    if is_vertical:
        point1_label   = "ceiling / top"
        point2_label   = "floor / bottom"
        point1_color   = "#7c3aed"
        point2_color   = "#c8440a"
        point1_badge   = "TOP"
        point2_badge   = "BOT"
        measure_axis   = "vertical"
        click_desc     = "the <strong>ceiling/top point</strong>"
        click_desc2    = "the <strong>floor/bottom point</strong>"
        orient_label   = "Vertical (portrait)"
    else:
        point1_label   = "left wall edge"
        point2_label   = "right wall edge"
        point1_color   = "#2563eb"
        point2_color   = "#c8440a"
        point1_badge   = "L"
        point2_badge   = "R"
        measure_axis   = "horizontal"
        click_desc     = "the <strong>left edge</strong> of your measured space"
        click_desc2    = "the <strong>right edge</strong> of your measured space"
        orient_label   = "Horizontal (landscape)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mark Measurement Points</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #1a1a18; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #8a8880; --border: #d8d4cc;
    --p1: {point1_color}; --p2: {point2_color};
  }}
  body {{
    font-family: 'DM Sans', sans-serif; font-weight: 300;
    background: var(--paper); color: var(--ink);
    min-height: 100vh; padding: 28px 20px 60px;
    display: flex; flex-direction: column; align-items: center;
  }}
  .page-title {{
    font-family: 'DM Serif Display', serif; font-size: 1.9rem;
    margin-bottom: 4px; text-align: center;
  }}
  .page-sub {{
    color: var(--muted); font-size: 0.85rem; font-style: italic;
    margin-bottom: 22px; text-align: center;
  }}

  /* ── Instruction bar ── */
  .instr-bar {{
    background: #fff; border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 22px; max-width: 960px; width: 100%; margin-bottom: 16px;
    display: flex; align-items: flex-start; gap: 28px; flex-wrap: wrap;
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
  .instr-text {{ font-size: 0.83rem; line-height: 1.55; padding-top: 4px; }}
  .instr-text.muted {{ color: var(--muted); }}
  .instr-meta {{
    margin-left: auto; display: flex; flex-direction: column;
    align-items: flex-end; gap: 5px; font-size: 0.76rem; color: var(--muted);
  }}
  .meta-chip {{
    background: var(--paper); border: 1px solid var(--border); border-radius: 6px;
    padding: 3px 10px; white-space: nowrap; font-size: 0.73rem;
  }}
  .ppi-chip {{
    background: #2563eb; color: #fff; border: none;
    border-radius: 6px; padding: 3px 12px; font-size: 0.73rem;
    font-weight: 500; display: none; white-space: nowrap;
  }}

  /* ── Tips box ── */
  .tips {{
    background: #f0f6ff; border: 1px solid #bdd3f5; border-radius: 10px;
    padding: 13px 18px; max-width: 960px; width: 100%; margin-bottom: 14px;
    font-size: 0.79rem; color: #1e3a8a; line-height: 1.65;
  }}
  .tips strong {{ font-weight: 500; color: #1e40af; }}
  .tips ul {{ margin: 6px 0 0 14px; }}

  /* ── Canvas ── */
  .canvas-wrap {{
    position: relative; max-width: 960px; width: 100%;
    border-radius: 12px; overflow: hidden;
    border: 2px solid var(--border);
    box-shadow: 0 6px 32px rgba(0,0,0,0.12);
    cursor: crosshair; user-select: none;
  }}
  canvas {{ display: block; width: 100%; height: auto; }}

  /* ── Controls ── */
  .controls {{
    display: flex; gap: 12px; margin-top: 14px;
    max-width: 960px; width: 100%;
  }}
  .btn {{
    padding: 12px 24px; border-radius: 9px; border: none;
    font-family: 'DM Sans', sans-serif; font-size: 0.9rem;
    font-weight: 500; cursor: pointer; transition: opacity .2s, transform .15s;
  }}
  .btn:hover {{ opacity: 0.86; transform: translateY(-1px); }}
  .btn:disabled {{ opacity: 0.32; cursor: not-allowed; transform: none; }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-ghost   {{ background: #fff; color: var(--ink); border: 1px solid var(--border); }}
</style>
</head>
<body>

<h1 class="page-title">Step 2 — Mark Your Measurement</h1>
<p class="page-sub">Click two points on the photo that span your {wall_measurement}" measurement</p>

<!-- Instruction bar -->
<div class="instr-bar">
  <div class="instr-step">
    <div class="badge badge-active" id="b1">1</div>
    <div class="instr-text" id="t1">
      Click <span id="clickDesc1">{click_desc}</span> on the photo below.
    </div>
  </div>
  <div class="instr-step">
    <div class="badge badge-waiting" id="b2">2</div>
    <div class="instr-text muted" id="t2">
      Then click <span id="clickDesc2">{click_desc2}</span>.
    </div>
  </div>
  <div class="instr-meta">
    <span class="meta-chip">📐 {wall_measurement}" · {orient_label}</span>
    <span class="ppi-chip" id="ppiChip">–</span>
  </div>
</div>

<!-- Tips -->
<div class="tips">
  <strong>Tips for accuracy:</strong>
  <ul>
    <li>Zoom in on the photo before clicking if you need more precision.</li>
    <li>Pick points at the same depth in the photo (both on the wall surface, not corners that recede).</li>
    <li>
      {"For vertical: click a point on the ceiling directly above the spot you measured, then the floor directly below it." if is_vertical else
       "For horizontal: click the leftmost point of your measured span, then the rightmost point."}
    </li>
    <li>If your click was off, press <strong>Reset</strong> and try again.</li>
  </ul>
</div>

<!-- Canvas -->
<div class="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<!-- Buttons -->
<div class="controls">
  <button class="btn btn-ghost" onclick="reset()">↺ Reset</button>
  <button class="btn btn-primary" id="nextBtn" disabled onclick="proceed()">
    Next: Position Print →
  </button>
</div>

<!-- Hidden form to POST to editor -->
<form id="rf" action="/mockup/editor" method="post" style="display:none">
  <input type="hidden" name="room_b64"       id="fR">
  <input type="hidden" name="art_b64"        id="fA">
  <input type="hidden" name="wall_measurement" value="{wall_measurement}">
  <input type="hidden" name="orientation"    value="{orientation}">
  <input type="hidden" name="pt1_x"          id="fPt1x">
  <input type="hidden" name="pt1_y"          id="fPt1y">
  <input type="hidden" name="pt2_x"          id="fPt2x">
  <input type="hidden" name="pt2_y"          id="fPt2y">
  <input type="hidden" name="orig_w"         value="{orig_w}">
  <input type="hidden" name="orig_h"         value="{orig_h}">
  <input type="hidden" name="art_mime"       value="{art_mime}">
  <input type="hidden" name="room_mime"      value="{room_mime}">
</form>

<script>
const ROOM_B64   = "{room_b64}";
const ART_B64    = "{art_b64}";
const ROOM_MIME  = "{room_mime}";
const IS_VERTICAL = {"true" if is_vertical else "false"};
const WALL_MEAS  = {wall_measurement};
const P1_COLOR   = "{point1_color}";
const P2_COLOR   = "{point2_color}";
const P1_BADGE   = "{point1_badge}";
const P2_BADGE   = "{point2_badge}";

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();

let pt1 = null, pt2 = null, phase = 1;

img.onload = () => {{
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  draw();
}};
img.src = 'data:' + ROOM_MIME + ';base64,' + ROOM_B64;

// ── Draw ──────────────────────────────────────────────────────────────────
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  if (pt1 && pt2) {{
    // Measurement line
    const midX = (pt1.x + pt2.x) / 2;
    const midY = (pt1.y + pt2.y) / 2;
    const lw   = Math.max(2, canvas.width * 0.002);

    if (IS_VERTICAL) {{
      // Vertical shaded band
      const x1 = Math.min(pt1.x, pt2.x);
      const y1 = Math.min(pt1.y, pt2.y);
      const y2 = Math.max(pt1.y, pt2.y);
      ctx.fillStyle = 'rgba(124,58,237,0.10)';
      ctx.fillRect(0, y1, canvas.width, y2 - y1);
      // Line
      ctx.strokeStyle = P1_COLOR; ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(midX, pt1.y); ctx.lineTo(midX, pt2.y); ctx.stroke();
      // Arrow heads
      arrowV(midX, pt1.y, -1, lw * 5);
      arrowV(midX, pt2.y,  1, lw * 5);
      // Label
      ctx.fillStyle = P1_COLOR;
      ctx.font = `600 ${{Math.max(12, canvas.width * 0.016)}}px DM Sans, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(WALL_MEAS + '"', midX + canvas.width * 0.025, midY);
    }} else {{
      // Horizontal shaded band
      const x1 = Math.min(pt1.x, pt2.x);
      const x2 = Math.max(pt1.x, pt2.x);
      ctx.fillStyle = 'rgba(37,99,235,0.10)';
      ctx.fillRect(x1, 0, x2 - x1, canvas.height);
      // Line
      const lineY = canvas.height * 0.07;
      ctx.strokeStyle = P1_COLOR; ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(pt1.x, lineY); ctx.lineTo(pt2.x, lineY); ctx.stroke();
      // Arrow heads
      arrowH(pt1.x, lineY, -1, lw * 5);
      arrowH(pt2.x, lineY,  1, lw * 5);
      // Label
      ctx.fillStyle = P1_COLOR;
      ctx.font = `600 ${{Math.max(12, canvas.width * 0.016)}}px DM Sans, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(WALL_MEAS + '"', midX, lineY - canvas.height * 0.012);
    }}
  }}

  if (pt1) pinPoint(pt1, P1_COLOR, P1_BADGE);
  if (pt2) pinPoint(pt2, P2_COLOR, P2_BADGE);
}}

function arrowH(x, y, dir, size) {{
  ctx.fillStyle = P1_COLOR;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - dir * size, y - size * 0.5);
  ctx.lineTo(x - dir * size, y + size * 0.5);
  ctx.closePath(); ctx.fill();
}}

function arrowV(x, y, dir, size) {{
  ctx.fillStyle = P1_COLOR;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * 0.5, y - dir * size);
  ctx.lineTo(x + size * 0.5, y - dir * size);
  ctx.closePath(); ctx.fill();
}}

function pinPoint(pt, color, label) {{
  const r  = Math.max(10, canvas.width * 0.013);
  const lw = Math.max(2, canvas.width * 0.003);
  // Crosshair line
  ctx.strokeStyle = color + 'cc'; ctx.lineWidth = lw;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  if (IS_VERTICAL) {{
    ctx.moveTo(0, pt.y); ctx.lineTo(canvas.width, pt.y);
  }} else {{
    ctx.moveTo(pt.x, 0); ctx.lineTo(pt.x, canvas.height);
  }}
  ctx.stroke(); ctx.setLineDash([]);
  // Circle
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2); ctx.fill();
  // Label
  ctx.fillStyle = '#fff';
  ctx.font = `600 ${{Math.max(9, r * 1.0)}}px DM Sans, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, pt.x, pt.y);
  ctx.textBaseline = 'alphabetic';
}}

// ── Click handling ─────────────────────────────────────────────────────────
canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const cx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
  const cy   = (e.clientY - rect.top)  * (canvas.height / rect.height);

  if (phase === 1) {{
    pt1 = {{x: cx, y: cy}};
    phase = 2;
    setStep(1, 'done');
    setStep(2, 'active2');
    document.getElementById('t2').classList.remove('muted');
  }} else if (phase === 2) {{
    pt2 = {{x: cx, y: cy}};
    phase = 3;
    setStep(2, 'done');
    // Compute pixels-per-inch
    const span = IS_VERTICAL
      ? Math.abs(pt2.y - pt1.y)
      : Math.abs(pt2.x - pt1.x);
    const ppi = (span / WALL_MEAS).toFixed(1);
    const chip = document.getElementById('ppiChip');
    chip.textContent = ppi + ' px/in · ready ✓';
    chip.style.display = 'block';
    document.getElementById('nextBtn').disabled = false;
  }}
  draw();
}});

function setStep(n, state) {{
  const el = document.getElementById('b' + n);
  if (state === 'done')    {{ el.className = 'badge badge-done';     el.textContent = '✓'; }}
  if (state === 'active')  {{ el.className = 'badge badge-active';   el.textContent = n; }}
  if (state === 'active2') {{ el.className = 'badge badge-active-2'; el.textContent = n; }}
  if (state === 'waiting') {{ el.className = 'badge badge-waiting';  el.textContent = n; }}
}}

function reset() {{
  pt1 = null; pt2 = null; phase = 1;
  setStep(1, 'active'); setStep(2, 'waiting');
  document.getElementById('t2').classList.add('muted');
  document.getElementById('ppiChip').style.display = 'none';
  document.getElementById('nextBtn').disabled = true;
  draw();
}}

function proceed() {{
  document.getElementById('fR').value    = ROOM_B64;
  document.getElementById('fA').value    = ART_B64;
  document.getElementById('fPt1x').value = pt1.x.toFixed(2);
  document.getElementById('fPt1y').value = pt1.y.toFixed(2);
  document.getElementById('fPt2x').value = pt2.x.toFixed(2);
  document.getElementById('fPt2y').value = pt2.y.toFixed(2);
  document.getElementById('rf').submit();
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Mockup Step 3 – Interactive Editor ────────────────────────────────────────

@app.post("/mockup/editor", response_class=HTMLResponse)
async def mockup_editor(
    room_b64:         str   = Form(...),
    art_b64:          str   = Form(...),
    wall_measurement: float = Form(...),
    orientation:      str   = Form("H"),
    pt1_x:            float = Form(...),
    pt1_y:            float = Form(...),
    pt2_x:            float = Form(...),
    pt2_y:            float = Form(...),
    orig_w:           int   = Form(...),
    orig_h:           int   = Form(...),
    art_mime:         str   = Form("image/jpeg"),
    room_mime:        str   = Form("image/jpeg"),
):
    orientation = orientation.strip().upper()
    is_vertical = (orientation == "V")

    # Pixels-per-inch in the original image for the measured axis
    if is_vertical:
        span_px  = abs(pt2_y - pt1_y)
    else:
        span_px  = abs(pt2_x - pt1_x)

    if span_px < 1:
        raise HTTPException(400, "The two clicked points are too close together. Please go back and re-mark.")

    ppi_orig = span_px / wall_measurement   # px / inch in the original image

    # Art aspect ratio
    art_bytes  = base64.b64decode(art_b64)
    art_pil    = Image.open(BytesIO(art_bytes)).convert("RGB")
    art_aspect = art_pil.width / art_pil.height if art_pil.height else 1.0

    # Axis label for the size input
    if is_vertical:
        long_axis_label = "Height (longest edge)"
        short_axis_note = "Width will be calculated automatically from the artwork's aspect ratio."
        size_placeholder = "e.g. 60"
        size_tip = "Because <strong>Vertical</strong> is selected, the number you enter is the <strong>height</strong> of the print in inches. The width scales automatically to match the artwork's proportions."
    else:
        long_axis_label = "Width (longest edge)"
        short_axis_note = "Height will be calculated automatically from the artwork's aspect ratio."
        size_placeholder = "e.g. 48"
        size_tip = "Because <strong>Horizontal</strong> is selected, the number you enter is the <strong>width</strong> of the print in inches. The height scales automatically to match the artwork's proportions."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Position Your Print</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #1a1a18; --paper: #f5f2ec; --accent: #c8440a;
    --muted: #8a8880; --border: #d8d4cc; --panel: #fff; --blue: #2563eb;
  }}
  body {{
    font-family: 'DM Sans', sans-serif; font-weight: 300;
    background: var(--paper); color: var(--ink);
    height: 100vh; overflow: hidden;
    display: grid; grid-template-columns: 1fr 320px;
  }}

  /* ── Stage ── */
  .stage {{ position: relative; overflow: hidden; background: #111; }}
  #mainCanvas {{ display: block; cursor: default; }}
  .stage.dragging {{ cursor: grabbing !important; }}

  /* ── Side panel ── */
  .panel {{
    background: var(--panel); border-left: 1px solid var(--border);
    display: flex; flex-direction: column; padding: 26px 22px;
    overflow-y: auto; gap: 0;
  }}
  .panel-title {{
    font-family: 'DM Serif Display', serif; font-size: 1.45rem;
    margin-bottom: 3px; line-height: 1.2;
  }}
  .panel-sub {{ color: var(--muted); font-size: 0.78rem; font-style: italic; margin-bottom: 22px; }}

  .field-label {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--muted); margin-bottom: 7px; margin-top: 20px; display: block;
  }}
  .field-label:first-of-type {{ margin-top: 0; }}

  /* ── Size input row ── */
  .size-row {{ display: flex; align-items: center; gap: 10px; }}
  .size-input {{
    width: 90px; padding: 10px 12px; border: 1px solid var(--border);
    border-radius: 8px; font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem; font-weight: 500; color: var(--ink);
    background: var(--paper); outline: none; transition: border-color .2s;
  }}
  .size-input:focus {{ border-color: var(--accent); }}
  .size-unit {{ font-size: 0.88rem; color: var(--muted); }}

  /* ── Size display ── */
  .size-display {{
    background: var(--paper); border-radius: 10px; padding: 13px 15px;
    margin-top: 10px; border: 1px solid var(--border);
  }}
  .size-inches {{
    font-family: 'DM Serif Display', serif; font-size: 1.55rem;
    line-height: 1; color: var(--ink); margin-bottom: 2px;
  }}
  .size-sub {{ font-size: 0.72rem; color: var(--muted); }}

  /* ── Tip ── */
  .tip {{
    background: #f0f6ff; border: 1px solid #bdd3f5; border-radius: 8px;
    padding: 10px 13px; font-size: 0.74rem; color: #1e3a8a; line-height: 1.6;
    margin-top: 10px;
  }}

  /* ── Light dial ── */
  .dial-wrap {{
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    padding: 14px 0 6px;
  }}
  .dial-label-row {{
    display: flex; justify-content: space-between; align-items: center;
    width: 100%; font-size: 0.72rem; color: var(--muted);
  }}
  .dial-angle-val {{
    font-size: 0.78rem; font-weight: 500; color: var(--ink);
    background: var(--paper); border: 1px solid var(--border);
    border-radius: 5px; padding: 2px 9px;
  }}
  #dialCanvas {{
    cursor: crosshair; border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
  }}
  .dial-hint {{ font-size: 0.7rem; color: var(--muted); text-align: center; font-style: italic; }}

  /* ── Shadow toggle ── */
  .toggle-row {{
    display: flex; align-items: center; justify-content: space-between; margin-top: 2px;
  }}
  .toggle-label {{ font-size: 0.85rem; }}
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

  hr {{ border: none; border-top: 1px solid var(--border); margin: 18px 0; }}

  /* ── Buttons ── */
  .btn {{
    width: 100%; padding: 13px; border-radius: 9px; border: none;
    font-family: 'DM Sans', sans-serif; font-size: 0.88rem; font-weight: 500;
    cursor: pointer; transition: opacity .2s, transform .15s; margin-top: 8px;
  }}
  .btn:hover {{ opacity: 0.88; transform: translateY(-1px); }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-ghost   {{ background: var(--paper); color: var(--ink); border: 1px solid var(--border); }}

  /* ── Hint footer ── */
  .hint {{
    margin-top: auto; padding-top: 16px;
    font-size: 0.72rem; color: var(--muted); line-height: 1.65; font-style: italic;
  }}

  /* ── Loader ── */
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

  /* ── Perspective info box ── */
  .persp-info {{
    background: #fff8f0; border: 1px solid #f0c090; border-radius: 8px;
    padding: 10px 13px; font-size: 0.73rem; color: #7a3800; line-height: 1.6;
    margin-top: 8px; display: none;
  }}
  .persp-info strong {{ font-weight: 500; }}

  /* ── Piece list ── */
  .piece-list {{
    display: flex; flex-direction: column; gap: 5px;
    margin-top: 4px; max-height: 176px; overflow-y: auto; padding-right: 2px;
  }}
  .piece-card {{
    display: flex; align-items: center; gap: 9px;
    background: var(--paper); border: 1.5px solid var(--border);
    border-radius: 9px; padding: 6px 10px; cursor: pointer;
    transition: border-color .15s, background .15s; flex-shrink: 0;
  }}
  .piece-card:hover  {{ background: #f9f6f0; }}
  .piece-card.active {{ border-color: var(--accent); background: #fff3ee; }}
  .piece-thumb {{ border-radius: 4px; background: #ddd8cc; flex-shrink: 0;
                  display: block; }}
  .piece-info  {{ flex: 1; min-width: 0; }}
  .piece-name  {{ font-size: 0.78rem; font-weight: 500; color: var(--ink);
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .piece-size  {{ font-size: 0.68rem; color: var(--muted); margin-top: 1px; }}
  .btn-add {{
    width: 100%; padding: 9px; margin-top: 7px; border-radius: 8px;
    border: 1.5px dashed var(--border); background: transparent;
    font-family: 'DM Sans', sans-serif; font-size: 0.82rem;
    color: var(--muted); cursor: pointer; transition: border-color .15s, color .15s;
  }}
  .btn-add:hover {{ border-color: var(--accent); color: var(--accent); }}
  .btn-sm  {{ padding: 8px 14px !important; font-size: 0.78rem !important; margin-top: 6px !important; }}
  .btn-danger {{ background: #fee2e2 !important; color: #991b1b !important;
                 border: 1px solid #fca5a5 !important; }}
  .btn-danger:hover {{ background: #fecaca !important; opacity: 1 !important; }}
</style>
</head>
<body>

<!-- ═══ Stage ═══ -->
<div class="stage" id="stage">
  <canvas id="mainCanvas"></canvas>
  <div id="loadOverlay"><div class="spinner"></div></div>
</div>

<!-- ═══ Side Panel ═══ -->
<div class="panel">
  <div class="panel-title">Gallery Wall</div>
  <div class="panel-sub">Step 3 — add &amp; position prints</div>

  <!-- ── Piece list ── -->
  <span class="field-label">Prints on Wall</span>
  <div class="piece-list" id="pieceList"></div>
  <input type="file" id="addFileInput" accept="image/*" style="display:none"
         onchange="onAddFile(this)">
  <button class="btn-add" onclick="document.getElementById('addFileInput').click()">
    + Add Another Print
  </button>

  <hr>

  <!-- ── Active piece controls ── -->
  <span class="field-label" id="activePieceLabel">{long_axis_label} (inches)</span>
  <div class="size-row">
    <input class="size-input" type="number" id="sizeInInput"
           min="1" max="999" step="0.5" placeholder="{size_placeholder}"
           oninput="onSizeInput(this.value)">
    <span class="size-unit">in</span>
  </div>

  <div class="tip" id="sizeTip">
    {size_tip}
    <br><br>
    <strong>Measured space:</strong> {wall_measurement}" · <strong>Orientation:</strong> {"Vertical" if is_vertical else "Horizontal"}
  </div>

  <div class="size-display">
    <div class="size-inches" id="sizeDisplay">Enter size above</div>
    <div class="size-sub"    id="sizeSubDisplay">{short_axis_note}</div>
  </div>

  <hr>

  <!-- ── Shadow (per-piece) ── -->
  <span class="field-label">Drop Shadow</span>
  <div class="toggle-row">
    <span class="toggle-label">Enable shadow</span>
    <div class="toggle on" id="shadowToggle" onclick="toggleShadow()"></div>
  </div>

  <!-- ── Perspective (per-piece) ── -->
  <span class="field-label" style="margin-top:18px;">Perspective Adjust</span>
  <div class="toggle-row">
    <span class="toggle-label">Warp corners to wall angle</span>
    <div class="toggle" id="perspToggle" onclick="togglePersp()"></div>
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

  <!-- ── Light direction dial (global) ── -->
  <span class="field-label" style="margin-top:16px;">Light Direction</span>
  <div class="dial-wrap">
    <div class="dial-label-row">
      <span>Drag to match the room's light source</span>
      <span class="dial-angle-val" id="dialAngleVal">315°</span>
    </div>
    <canvas id="dialCanvas" width="120" height="120"></canvas>
    <div class="dial-hint">☀ = light source · shadow falls opposite</div>
  </div>

  <hr>

  <button class="btn btn-primary" onclick="downloadMockup()">⬇ Download Mockup</button>
  <button class="btn btn-ghost"   onclick="history.back()">← Re-mark Wall</button>

  <p class="hint">
    • <strong>Add prints</strong> with the button above the list.<br>
    • Click a print in the list — or on the canvas — to select it.<br>
    • <strong>Perspective:</strong> toggle on, drag corners, toggle off. Warp stays until Reset.<br>
    • <strong>Reset Adjustments</strong> returns the print to flat &amp; centred.
  </p>
</div>

<script>
// ── Constants ──────────────────────────────────────────────────────────────
const ROOM_B64          = "{room_b64}";
const ROOM_MIME         = "{room_mime}";
const FIRST_ART_B64     = "{art_b64}";
const FIRST_ART_MIME    = "{art_mime}";
const FIRST_ART_ASPECT  = {art_aspect:.6f};   // W/H of the first artwork
const WALL_MEAS         = {wall_measurement};
const PPI_ORIG          = {ppi_orig:.6f};
const ORIG_W            = {orig_w};
const ORIG_H            = {orig_h};

// ── Canvas / stage ─────────────────────────────────────────────────────────
const canvas  = document.getElementById('mainCanvas');
const ctx     = canvas.getContext('2d');
const stage   = document.getElementById('stage');
const roomImg = new Image();
let displayScale = 1, ppiDisp = PPI_ORIG;
let lightAngleDeg = 315;

// ── Pieces ─────────────────────────────────────────────────────────────────
// Each piece: {{ id, name, img, aspect, artX, artY, artW, artH,
//               corners ([] or [4]), perspMode, shadowOn, lastSizeIn, initialPlaced }}
let pieces = [];
let activePieceIdx = -1;
let pieceIdCounter = 0;

function makePiece(img, aspect, name) {{
  return {{
    id: ++pieceIdCounter,
    name: name || ('Print ' + pieceIdCounter),
    img, aspect,
    artX: 0, artY: 0, artW: 0, artH: 0,
    corners: [],       // [] = flat rect;  [4] = warp baked (preserved after toggle-off)
    perspMode: false,  // true = corner handles visible and draggable
    shadowOn: true,
    lastSizeIn: null,
    initialPlaced: false,
  }};
}}

// ── Boot ───────────────────────────────────────────────────────────────────
let roomLoaded = false, firstArtLoaded = false;
roomImg.onload = () => {{ roomLoaded = true; tryStart(); }};

const firstArtImg = new Image();
firstArtImg.onload = () => {{
  firstArtLoaded = true;
  pieces.push(makePiece(firstArtImg, FIRST_ART_ASPECT, 'Print 1'));
  activePieceIdx = 0;
  renderPieceList();
  tryStart();
}};
roomImg.src     = 'data:' + ROOM_MIME      + ';base64,' + ROOM_B64;
firstArtImg.src = 'data:' + FIRST_ART_MIME + ';base64,' + FIRST_ART_B64;

function tryStart() {{
  if (!roomLoaded || !firstArtLoaded) return;
  document.getElementById('loadOverlay').style.display = 'none';
  initLayout();
  drawDial();
  syncPanelToActive();
  startLoop();
}}

// ── Drag state ─────────────────────────────────────────────────────────────
let dragging     = false;
let dragOffX     = 0, dragOffY = 0;
let activeCorner = -1;   // corner index being dragged (-1 = body drag)
let hoverIdx     = -1;   // piece index the cursor is over

// ── Layout ─────────────────────────────────────────────────────────────────
function initLayout() {{
  const stageW = stage.clientWidth, stageH = stage.clientHeight;
  const roomAsp = ORIG_W / ORIG_H;
  let dispW, dispH;
  if (roomAsp > stageW / stageH) {{ dispW = stageW; dispH = stageW / roomAsp; }}
  else                            {{ dispH = stageH; dispW = stageH * roomAsp; }}
  canvas.width = dispW; canvas.height = dispH;
  canvas.style.position = 'absolute';
  canvas.style.top  = ((stageH - dispH) / 2) + 'px';
  canvas.style.left = ((stageW - dispW) / 2) + 'px';
  canvas.style.width = dispW + 'px'; canvas.style.height = dispH + 'px';
  displayScale = dispW / ORIG_W;
  ppiDisp      = PPI_ORIG * displayScale;
}}

// ── Active piece helpers ───────────────────────────────────────────────────
function active() {{ return pieces[activePieceIdx] || null; }}

function selectPiece(idx) {{
  activePieceIdx = idx;
  renderPieceList();
  syncPanelToActive();
}}

// ── Piece list rendering ───────────────────────────────────────────────────
function renderPieceList() {{
  const ul = document.getElementById('pieceList');
  ul.innerHTML = '';
  pieces.forEach((p, i) => {{
    const card = document.createElement('div');
    card.className = 'piece-card' + (i === activePieceIdx ? ' active' : '');
    card.onclick = () => selectPiece(i);

    // Thumbnail canvas
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
  document.getElementById('removeBtn').style.display = pieces.length > 1 ? '' : 'none';
}}

// ── Sync panel to active piece ─────────────────────────────────────────────
function syncPanelToActive() {{
  const p = active();
  if (!p) return;
  const isPort = p.aspect < 1;

  document.getElementById('activePieceLabel').textContent =
    (isPort ? 'Height' : 'Width') + ' — longest edge (inches)';

  document.getElementById('sizeTip').innerHTML = `
    Enter the <strong>${{isPort ? 'height' : 'width'}}</strong> of the print
    in inches. The other dimension scales from the artwork's aspect ratio.
    <br><br>
    <strong>Measured space:</strong> ${{WALL_MEAS}}"
    &nbsp;·&nbsp; 
    <strong>Calibrated PPI:</strong> ${{(ppiDisp / displayScale).toFixed(1)}}`;

  const inp = document.getElementById('sizeInInput');
  inp.value = p.lastSizeIn != null ? p.lastSizeIn : '';
  if (p.lastSizeIn != null) {{
    const wIn = isPort ? (p.lastSizeIn * p.aspect) : p.lastSizeIn;
    const hIn = isPort ? p.lastSizeIn : (p.lastSizeIn / p.aspect);
    document.getElementById('sizeDisplay').textContent =
      wIn.toFixed(1) + '\u2009\u00d7\u2009' + hIn.toFixed(1) + ' in';
    document.getElementById('sizeSubDisplay').textContent = 'at ' + WALL_MEAS + '" measured space';
  }} else {{
    document.getElementById('sizeDisplay').textContent = 'Enter size above';
    document.getElementById('sizeSubDisplay').textContent =
      isPort ? 'Width calculated from aspect ratio' : 'Height calculated from aspect ratio';
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
    if (p.corners.length === 4) _resetCorners(p);  // re-seat existing warp
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
    const wIn = isPort ? (sizeIn * p.aspect) : sizeIn;
    const hIn = isPort ? sizeIn : (sizeIn / p.aspect);
    document.getElementById('sizeDisplay').textContent =
      wIn.toFixed(1) + '\u2009\u00d7\u2009' + hIn.toFixed(1) + ' in';
    document.getElementById('sizeSubDisplay').textContent = 'at ' + WALL_MEAS + '" measured space';
  }}
}}

// ── Clamp (flat pieces only) ───────────────────────────────────────────────
function _clampPiece(p) {{
  p.artX = p.artW >= canvas.width  ? (canvas.width  - p.artW) / 2 : Math.max(0, Math.min(p.artX, canvas.width  - p.artW));
  p.artY = p.artH >= canvas.height ? (canvas.height - p.artH) / 2 : Math.max(0, Math.min(p.artY, canvas.height - p.artH));
}}

// ── Perspective helpers ────────────────────────────────────────────────────
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
  const sx = p.artW > 0 ? newW / p.artW : 1;
  const sy = p.artH > 0 ? newH / p.artH : 1;
  p.corners = p.corners.map(c => ({{x: ctr.x + (c.x-ctr.x)*sx, y: ctr.y + (c.y-ctr.y)*sy}}));
}}

function _syncBoundsFromCorners(p) {{
  const xs = p.corners.map(c=>c.x), ys = p.corners.map(c=>c.y);
  p.artX = Math.min(...xs); p.artY = Math.min(...ys);
}}

function _cornerHitTest(p, cx, cy) {{
  const r = Math.max(14, canvas.width * 0.016);
  for (let i = 0; i < p.corners.length; i++) {{
    const dx = cx-p.corners[i].x, dy = cy-p.corners[i].y;
    if (dx*dx + dy*dy < r*r) return i;
  }}
  return -1;
}}

// ── Point-in-shape ─────────────────────────────────────────────────────────
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

// ── Affine warp (triangular decomposition) ─────────────────────────────────
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

// ── Shadow ─────────────────────────────────────────────────────────────────
function _getShadowOffset(p) {{
  const a=(lightAngleDeg+180)*Math.PI/180, d=Math.max(6,Math.min(p.artW,p.artH)*0.06);
  return {{x:Math.cos(a)*d, y:Math.sin(a)*d, blur:Math.max(8,Math.min(p.artW,p.artH)*0.04)}};
}}

// ── Draw ───────────────────────────────────────────────────────────────────
let isDownloading = false;

function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(roomImg,0,0,canvas.width,canvas.height);
  pieces.forEach((p,i) => _drawPiece(p,i));
}}

function _drawPiece(p, pieceIdx) {{
  const isActive  = pieceIdx === activePieceIdx;
  const hasWarp   = p.corners.length === 4;
  const visible   = hasWarp ? true : (p.artW>0 && p.artH>0);
  if (!visible) return;

  // Shadow
  if (p.shadowOn) {{
    const sh=_getShadowOffset(p);
    ctx.save(); ctx.filter=`blur(${{sh.blur}}px)`; ctx.fillStyle='rgba(0,0,0,0.52)';
    if (hasWarp) {{
      ctx.beginPath();
      p.corners.forEach((c,i)=> i===0?ctx.moveTo(c.x+sh.x,c.y+sh.y):ctx.lineTo(c.x+sh.x,c.y+sh.y));
      ctx.closePath(); ctx.fill();
    }} else {{ ctx.fillRect(p.artX+sh.x,p.artY+sh.y,p.artW,p.artH); }}
    ctx.restore();
  }}

  // Image
  if (hasWarp) {{ _drawPerspArt(p); }}
  else         {{ ctx.drawImage(p.img,p.artX,p.artY,p.artW,p.artH); }}

  if (isDownloading) return;

  // ── Selection / hover outline ──
  const isHover = hoverIdx === pieceIdx;
  if (isActive || isHover) {{
    ctx.strokeStyle = isActive
      ? (p.perspMode ? 'rgba(200,68,10,0.8)' : 'rgba(255,255,255,0.85)')
      : 'rgba(255,255,255,0.35)';
    ctx.lineWidth = isActive ? Math.max(1.5,canvas.width*0.002) : 1;
    ctx.setLineDash([6,4]);
    if (hasWarp) {{
      ctx.beginPath();
      p.corners.forEach((c,i)=> i===0?ctx.moveTo(c.x,c.y):ctx.lineTo(c.x,c.y));
      ctx.closePath(); ctx.stroke();
    }} else {{
      ctx.strokeRect(p.artX-1,p.artY-1,p.artW+2,p.artH+2);
    }}
    ctx.setLineDash([]);
  }}

  // Corner handles (active piece in perspMode only)
  if (isActive && p.perspMode && hasWarp) {{
    const hr=Math.max(7,canvas.width*0.009);
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

// ── Toggle perspective ─────────────────────────────────────────────────────
function togglePersp() {{
  const p=active(); if (!p) return;
  p.perspMode=!p.perspMode;
  document.getElementById('perspToggle').classList.toggle('on',p.perspMode);
  document.getElementById('perspInfo').style.display=p.perspMode?'block':'none';
  // Turn ON → seed corners from current rect if no warp exists yet
  if (p.perspMode && p.corners.length===0 && p.artW>0 && p.artH>0) _resetCorners(p);
  // Turn OFF → corners are intentionally preserved (warp stays until Reset)
}}

// ── Reset adjustments ──────────────────────────────────────────────────────
function resetAdjustments() {{
  const p=active(); if (!p) return;
  p.corners=[]; p.perspMode=false;
  document.getElementById('perspToggle').classList.remove('on');
  document.getElementById('perspInfo').style.display='none';
  if (p.artW>0 && p.artH>0) {{
    p.artX=(canvas.width -p.artW)/2; p.artY=(canvas.height-p.artH)/2; _clampPiece(p);
  }}
}}

// ── Toggle shadow ──────────────────────────────────────────────────────────
function toggleShadow() {{
  const p=active(); if (!p) return;
  p.shadowOn=!p.shadowOn;
  document.getElementById('shadowToggle').classList.toggle('on',p.shadowOn);
}}

// ── Remove active piece ────────────────────────────────────────────────────
function removeActivePiece() {{
  if (pieces.length<=1) return;
  pieces.splice(activePieceIdx,1);
  activePieceIdx=Math.min(activePieceIdx,pieces.length-1);
  renderPieceList(); syncPanelToActive();
}}

// ── Add new print from file ────────────────────────────────────────────────
function onAddFile(input) {{
  const file=input.files[0]; if (!file) return;
  const reader=new FileReader();
  reader.onload=e=>{{
    const img=new Image();
    img.onload=()=>{{
      const asp=img.naturalWidth/img.naturalHeight;
      const name=file.name.replace(/[.][^.]+$/,'').substring(0,24)||('Print '+(pieces.length+1));
      pieces.push(makePiece(img,asp,name));
      activePieceIdx=pieces.length-1;
      renderPieceList(); syncPanelToActive();
    }};
    img.src=e.target.result;
  }};
  reader.readAsDataURL(file);
  input.value='';
}}

// ── Canvas coord + hit helpers ─────────────────────────────────────────────
function canvasCoords(e) {{
  const rect=canvas.getBoundingClientRect(), src=e.touches?e.touches[0]:e;
  return [(src.clientX-rect.left)*(canvas.width/rect.width),
          (src.clientY-rect.top)*(canvas.height/rect.height)];
}}

// Returns index of topmost piece at (cx,cy); -1 if none
function _topPieceAt(cx,cy) {{
  for (let i=pieces.length-1;i>=0;i--) {{ if (_pieceHit(pieces[i],cx,cy)) return i; }}
  return -1;
}}

// ── Mouse ──────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown',e=>{{
  if (dialDragging) return;
  const [cx,cy]=canvasCoords(e), p=active();

  // Corner drag (active piece in perspMode)
  if (p && p.perspMode && p.corners.length===4) {{
    const ci=_cornerHitTest(p,cx,cy);
    if (ci>=0) {{ activeCorner=ci; dragging=true; stage.classList.add('dragging'); e.preventDefault(); return; }}
  }}

  // Body drag on active piece
  if (p && _pieceHit(p,cx,cy)) {{
    dragging=true; activeCorner=-1;
    if (p.corners.length===4) {{ const ctr=_cornersCenter(p); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else                      {{ dragOffX=cx-p.artX; dragOffY=cy-p.artY; }}
    stage.classList.add('dragging'); e.preventDefault(); return;
  }}

  // Click on a different piece → select + start drag
  const idx=_topPieceAt(cx,cy);
  if (idx>=0 && idx!==activePieceIdx) {{
    selectPiece(idx);
    const np=pieces[idx]; dragging=true; activeCorner=-1;
    if (np.corners.length===4) {{ const ctr=_cornersCenter(np); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else                       {{ dragOffX=cx-np.artX; dragOffY=cy-np.artY; }}
    stage.classList.add('dragging'); e.preventDefault();
  }}
}});

window.addEventListener('mousemove',e=>{{
  if (dialDragging) return;
  const [cx,cy]=canvasCoords(e), p=active();
  if (dragging && p) {{
    if (p.perspMode && activeCorner>=0) {{
      p.corners[activeCorner]={{x:cx,y:cy}};
    }} else if (p.corners.length===4) {{
      const ctr=_cornersCenter(p), ddx=(cx-dragOffX)-ctr.x, ddy=(cy-dragOffY)-ctr.y;
      p.corners=p.corners.map(c=>({{x:c.x+ddx,y:c.y+ddy}}));
    }} else {{
      p.artX=cx-dragOffX; p.artY=cy-dragOffY; _clampPiece(p);
    }}
    return;
  }}
  hoverIdx=_topPieceAt(cx,cy);
  const onCorner=p&&p.perspMode&&p.corners.length===4&&_cornerHitTest(p,cx,cy)>=0;
  canvas.style.cursor=onCorner?'crosshair':(hoverIdx>=0?'grab':'default');
}});

window.addEventListener('mouseup',()=>{{
  if (dragging) {{ const p=active(); if (p&&p.corners.length===4&&activeCorner===-1) _syncBoundsFromCorners(p); }}
  dragging=false; activeCorner=-1; stage.classList.remove('dragging');
}});

// ── Touch ──────────────────────────────────────────────────────────────────
canvas.addEventListener('touchstart',e=>{{
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
    selectPiece(idx); const np=pieces[idx]; dragging=true; activeCorner=-1;
    if (np.corners.length===4) {{ const ctr=_cornersCenter(np); dragOffX=cx-ctr.x; dragOffY=cy-ctr.y; }}
    else {{ dragOffX=cx-np.artX; dragOffY=cy-np.artY; }}
    e.preventDefault();
  }}
}},{{passive:false}});

window.addEventListener('touchmove',e=>{{
  if (!dragging) return;
  const [cx,cy]=canvasCoords(e), p=active(); if (!p) return;
  if (p.perspMode&&activeCorner>=0) {{
    p.corners[activeCorner]={{x:cx,y:cy}};
  }} else if (p.corners.length===4) {{
    const ctr=_cornersCenter(p), ddx=(cx-dragOffX)-ctr.x, ddy=(cy-dragOffY)-ctr.y;
    p.corners=p.corners.map(c=>({{x:c.x+ddx,y:c.y+ddy}}));
  }} else {{ p.artX=cx-dragOffX; p.artY=cy-dragOffY; _clampPiece(p); }}
  e.preventDefault();
}},{{passive:false}});

window.addEventListener('touchend',()=>{{
  if (dragging) {{ const p=active(); if (p&&p.corners.length===4&&activeCorner===-1) _syncBoundsFromCorners(p); }}
  dragging=false; activeCorner=-1;
}});

// ── Light direction dial ───────────────────────────────────────────────────
const dialCanvas=document.getElementById('dialCanvas'), dialCtx=dialCanvas.getContext('2d');
const DIAL_R=56, DIAL_CX=60, DIAL_CY=60, HANDLE_R=8;
let dialDragging=false;

function drawDial() {{
  dialCtx.clearRect(0,0,dialCanvas.width,dialCanvas.height);
  const grad=dialCtx.createRadialGradient(DIAL_CX,DIAL_CY,0,DIAL_CX,DIAL_CY,DIAL_R);
  grad.addColorStop(0,'#2e2e38'); grad.addColorStop(1,'#1a1a22');
  dialCtx.beginPath(); dialCtx.arc(DIAL_CX,DIAL_CY,DIAL_R,0,Math.PI*2);
  dialCtx.fillStyle=grad; dialCtx.fill();
  dialCtx.beginPath(); dialCtx.arc(DIAL_CX,DIAL_CY,DIAL_R,0,Math.PI*2);
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
  const cx=(src.clientX-rect.left)*(dialCanvas.width/rect.width);
  const cy=(src.clientY-rect.top)*(dialCanvas.height/rect.height);
  lightAngleDeg=((Math.atan2(cy-DIAL_CY,cx-DIAL_CX)*180/Math.PI)+360)%360;
  drawDial();
}}
dialCanvas.addEventListener('mousedown',e=>{{dialDragging=true;dialSetAngle(e);e.preventDefault();}});
window.addEventListener('mousemove',   e=>{{if(dialDragging)dialSetAngle(e);}});
window.addEventListener('mouseup',     ()=>{{dialDragging=false;}});
dialCanvas.addEventListener('touchstart',e=>{{dialDragging=true;dialSetAngle(e);e.preventDefault();}},{{passive:false}});
window.addEventListener('touchmove',   e=>{{if(dialDragging){{dialSetAngle(e);e.preventDefault();}}}},{{passive:false}});
window.addEventListener('touchend',    ()=>{{dialDragging=false;}});

// ── Download ──────────────────────────────────────────────────────────────
function downloadMockup() {{
  // Temporarily suppress all outlines and handles for a clean export
  const savedActive=activePieceIdx, savedHover=hoverIdx;
  activePieceIdx=-1; hoverIdx=-1; isDownloading=true;
  dragging=false; activeCorner=-1;
  draw();
  const link=document.createElement('a');
  link.download='wall-mockup.jpg';
  link.href=canvas.toDataURL('image/jpeg',0.94);
  link.click();
  activePieceIdx=savedActive; hoverIdx=savedHover; isDownloading=false;
}}

// ── Resize ─────────────────────────────────────────────────────────────────
window.addEventListener('resize',()=>{{
  if (!roomLoaded||pieces.length===0) return;
  // Save normalised positions + corners for all pieces
  const saved=pieces.map(p=>{{
    const hasC=p.corners.length===4;
    return {{
      normX: canvas.width  ? (p.artX+p.artW/2)/canvas.width  : 0.5,
      normY: canvas.height ? (p.artY+p.artH/2)/canvas.height : 0.5,
      normCorners: hasC ? p.corners.map(c=>({{x:c.x/canvas.width,y:c.y/canvas.height}})) : null,
    }};
  }});
  initLayout();
  pieces.forEach((p,i)=>{{
    if (p.lastSizeIn===null) return;
    _applySizeIn(p,p.lastSizeIn);
    const s=saved[i];
    p.artX=s.normX*canvas.width -p.artW/2;
    p.artY=s.normY*canvas.height-p.artH/2;
    _clampPiece(p);
    if (s.normCorners) p.corners=s.normCorners.map(c=>({{x:c.x*canvas.width,y:c.y*canvas.height}}));
  }});
}});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)