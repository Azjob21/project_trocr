import os
import numpy as np
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv("TROCR_MODEL_PATH", "./model")
PROCESSOR_PATH = os.getenv("TROCR_PROCESSOR_PATH", "microsoft/trocr-base-printed")
MAX_NEW_TOKENS = int(os.getenv("TROCR_MAX_NEW_TOKENS", "64"))
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading processor from : {PROCESSOR_PATH}")
print(f"Loading model from     : {MODEL_PATH}")
print(f"Device                 : {DEVICE}")

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Model ready.")


# ── Robust line detection ─────────────────────────────────────────────────────
def detect_text_bands(img_pil, min_height=10, merge_gap=8):
    """
    Robust detection for low-contrast receipts.
    Uses percentile contrast stretch + adaptive threshold (mean + 0.3*std).
    """
    gray = np.array(img_pil.convert("L"), dtype=np.float32)

    # contrast stretch to 2nd–98th percentile
    p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
    if p98 > p2:
        gray = np.clip((gray - p2) / (p98 - p2) * 255, 0, 255)

    # invert: dark text → high values
    inv = 255 - gray

    # mild blur to reduce noise
    inv_img = Image.fromarray(inv.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1))
    inv     = np.array(inv_img, dtype=np.float32)

    # row projection
    proj   = inv.mean(axis=1)
    thresh = proj.mean() + 0.3 * proj.std()
    thresh = max(thresh, 5.0)

    # find bands
    in_band, bands, start = False, [], 0
    for i, val in enumerate(proj):
        if not in_band and val > thresh:
            in_band, start = True, i
        elif in_band and val <= thresh:
            in_band = False
            if i - start >= min_height:
                bands.append([start, i])
    if in_band and len(proj) - start >= min_height:
        bands.append([start, int(len(proj))])

    # merge nearby bands
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))

    return merged


# ── OCR ───────────────────────────────────────────────────────────────────────
def run_ocr(img_pil):
    pv = processor(images=img_pil.convert("RGB"), return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        gen = model.generate(pv, max_new_tokens=MAX_NEW_TOKENS)
    return processor.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


# ── Draw annotated image ──────────────────────────────────────────────────────
COLORS = [
    "#00e5ff","#ff6b6b","#ffd93d","#6bcb77",
    "#a78bfa","#f97316","#ec4899","#14b8a6",
    "#f43f5e","#84cc16","#0ea5e9","#d946ef",
]

def draw_boxes(img_pil, bands, predictions):
    annotated = img_pil.copy().convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0,0,0,0))
    draw      = ImageDraw.Draw(overlay)
    w         = img_pil.width

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, ((y0, y1), text) in enumerate(zip(bands, predictions)):
        hx = COLORS[i % len(COLORS)]
        r,g,b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
        draw.rectangle([(0,y0),(w,y1)], fill=(r,g,b,30))
        draw.rectangle([(0,y0),(w-1,y1)], outline=(r,g,b,200), width=2)
        label = f"[{i+1:02d}] {text[:55]}{'…' if len(text)>55 else ''}"
        tb    = draw.textbbox((4, y0+2), label, font=font)
        draw.rectangle([tb[0]-2,tb[1]-1,tb[2]+4,tb[3]+2], fill=(r,g,b,210))
        draw.text((4, y0+2), label, fill=(255,255,255,255), font=font)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


# ── Main ──────────────────────────────────────────────────────────────────────
def process_receipt(image, max_width, min_band_h, merge_gap):
    if image is None:
        return None, "No image uploaded.", ""

    img  = Image.fromarray(image).convert("RGB")
    w, h = img.size

    if w > int(max_width):
        scale = int(max_width) / w
        img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h  = img.size

    bands = detect_text_bands(img, min_height=int(min_band_h), merge_gap=int(merge_gap))

    if not bands:
        return np.array(img), "No text bands detected.", \
               f"0 bands | {w}×{h}px | {str(DEVICE).upper()}"

    predictions = []
    for (y0, y1) in bands:
        crop = img.crop((0, max(0,y0-4), w, min(h,y1+4)))
        predictions.append(run_ocr(crop) or "[—]")

    annotated = draw_boxes(img, bands, predictions)
    full_text = "\n".join(f"[{i+1:02d}] {t}" for i,t in enumerate(predictions))
    stats     = f"Detected {len(bands)} text bands  |  Image: {w}×{h}px  |  Device: {str(DEVICE).upper()}"

    return np.array(annotated), full_text, stats


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0d0f14;--surface:#161920;--surf2:#1e222d;--accent:#00e5ff;--text:#e2e8f0;--muted:#64748b;--border:rgba(255,255,255,0.07);--r:8px}
body,.gradio-container{background:var(--bg)!important;font-family:'DM Sans',sans-serif!important;color:var(--text)!important}
.app-header{border-left:3px solid var(--accent);padding:14px 20px;margin-bottom:20px;background:var(--surface);border-radius:var(--r)}
.app-header h1{font-family:'Space Mono',monospace!important;font-size:20px!important;font-weight:700!important;color:#fff!important;margin:0 0 4px 0!important}
.app-header p{color:var(--muted)!important;font-size:12px!important;margin:0!important}
label span{font-family:'Space Mono',monospace!important;font-size:11px!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--accent)!important}
input[type=range]{accent-color:var(--accent)!important}
button.primary{background:var(--accent)!important;color:#000!important;font-family:'Space Mono',monospace!important;font-weight:700!important;font-size:13px!important;border:none!important;border-radius:4px!important;transition:opacity .15s!important}
button.primary:hover{opacity:.8!important}
textarea{background:var(--surf2)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:'Space Mono',monospace!important;font-size:12px!important;border-radius:var(--r)!important}
.stats-box textarea{background:rgba(0,229,255,.05)!important;border-color:rgba(0,229,255,.2)!important;color:var(--accent)!important;font-size:11px!important}
.accordion{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important}
.footer-bar{margin-top:14px;padding:10px 16px;background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.2);border-radius:var(--r);font-size:11px;color:#94a3b8;font-family:'Space Mono',monospace}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as app:
    gr.HTML('<div class="app-header"><h1>⬡ RECEIPT OCR — TrOCR</h1><p>Upload receipt → auto-detect text lines → extract with TrOCR → view annotated results</p></div>')

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="RECEIPT IMAGE", type="numpy")
            with gr.Accordion("DETECTION SETTINGS", open=True, elem_classes=["accordion"]):
                gr.HTML("<p style='font-size:11px;color:#64748b;margin:0 0 10px;font-family:Space Mono,monospace'>Default values work for most receipts.</p>")
                max_width  = gr.Slider(400, 2000, value=1200, step=100, label="MAX IMAGE WIDTH (px)")
                min_band_h = gr.Slider(5, 60, value=10, step=1,  label="MIN BAND HEIGHT (px)", info="Lower = detect smaller text")
                merge_gap  = gr.Slider(2, 40,  value=8,  step=1,  label="MERGE GAP (px)",       info="Higher = merge more lines together")
            run_btn = gr.Button("EXTRACT TEXT", variant="primary")

        with gr.Column(scale=1):
            annotated_out = gr.Image(label="ANNOTATED RECEIPT", type="numpy", interactive=False)
            stats_out     = gr.Textbox(label="STATS", interactive=False, lines=1, max_lines=1, elem_classes=["stats-box"])
            text_out      = gr.Textbox(label="EXTRACTED TEXT (LINE BY LINE)", interactive=False, lines=22, max_lines=50)

    run_btn.click(
        fn=process_receipt,
        inputs=[image_input, max_width, min_band_h, merge_gap],
        outputs=[annotated_out, text_out, stats_out]
    )

    gr.HTML('<div class="footer-bar">MODEL: local ./model &nbsp;|&nbsp; ARCH: TrOCR-base-printed fine-tuned on SROIE 2019 &nbsp;|&nbsp; QUANTIX receipt scanner prototype</div>')

if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE","false").strip().lower() == "true"
    app.launch(share=share)
