import base64
import io
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError
from pydantic import BaseModel

MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "./model")
PROCESSOR_PATH = os.getenv("TROCR_PROCESSOR_PATH", "microsoft/trocr-base-printed")
MAX_NEW_TOKENS = int(os.getenv("TROCR_MAX_NEW_TOKENS", "64"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_ASPECT = 2.5

BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "templates" / "index.html"

print(f"[OCR] Loading processor : {PROCESSOR_PATH}")
print(f"[OCR] Loading model     : {MODEL_PATH}")
print(f"[OCR] Device            : {DEVICE}")

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("[OCR] Model ready.\n")

app = FastAPI(title="Receipt OCR API", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalise_receipt(img_pil: Image.Image) -> Image.Image:
    """
    Normalize contrast so dark ink stays dark across white/yellow receipts.
    """
    gray = img_pil.convert("L")
    arr = np.array(gray, dtype=np.float32)
    paper = np.percentile(arr, 95)
    if paper < 200:
        arr = arr * (255.0 / max(paper, 1))
        arr = np.clip(arr, 0, 255)
        gray = Image.fromarray(arr.astype(np.uint8))
    return gray


def _bbox_from_mask(mask: np.ndarray, min_density: float = 0.02):
    row_density = mask.mean(axis=1)
    col_density = mask.mean(axis=0)
    ys = np.where(row_density > min_density)[0]
    xs = np.where(col_density > min_density)[0]
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _valid_crop_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> bool:
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < int(w * 0.2) or crop_h < int(h * 0.2):
        return False
    area_ratio = (crop_w * crop_h) / float(w * h)
    shrunk = crop_w < int(w * 0.98) or crop_h < int(h * 0.98)
    return shrunk and 0.06 <= area_ratio <= 0.95


def auto_crop_receipt(img_pil: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Crop the image around the dense text region before line-level OCR.

    This helps when users upload a full-scene image while the model was
    trained on tighter receipt crops.
    """
    w, h = img_pil.size
    if w < 80 or h < 80:
        return img_pil, False

    # Downscale for stable and fast mask analysis.
    scale = max(w, h) / 900.0 if max(w, h) > 900 else 1.0
    if scale > 1.0:
        ws = max(1, int(round(w / scale)))
        hs = max(1, int(round(h / scale)))
        work = img_pil.resize((ws, hs), Image.LANCZOS)
    else:
        work = img_pil
        ws, hs = w, h

    rgb = np.array(work.convert("RGB"), dtype=np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    luminance = rgb.mean(axis=2)

    # Bright low-chroma regions correspond to receipt paper in table/desk photos.
    whiteness = luminance - 0.80 * (mx - mn)
    white_thr = np.percentile(whiteness, 60)
    paper_mask = (whiteness >= white_thr).astype(np.uint8) * 255

    # Fill text holes and smooth small gaps in the paper mask.
    mask_img = (
        Image.fromarray(paper_mask)
        .filter(ImageFilter.MaxFilter(9))
        .filter(ImageFilter.MinFilter(9))
    )
    mask = np.array(mask_img) > 0
    bbox = _bbox_from_mask(mask, min_density=0.02)
    if bbox is None:
        return img_pil, False

    xs0, ys0, xs1, ys1 = bbox
    x0 = int(round(xs0 * scale))
    y0 = int(round(ys0 * scale))
    x1 = int(round(xs1 * scale))
    y1 = int(round(ys1 * scale))

    # Slight padding to keep margins around the receipt.
    pad_x = max(8, int((x1 - x0) * 0.03))
    pad_y = max(10, int((y1 - y0) * 0.04))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    if not _valid_crop_bbox(x0, y0, x1, y1, w, h):
        return img_pil, False
    return img_pil.crop((x0, y0, x1, y1)), True


def detect_text_bands(
    img_pil: Image.Image,
    min_height: int = 8,
    merge_gap: int = 6,
    pad: int = 3,
    threshold_scale: float = 0.60,
) -> list:
    """
    Detect horizontal text line bands and return (x0, y0, x1, y1) boxes.
    """
    w, h = img_pil.size
    gray = normalise_receipt(img_pil)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(blurred, dtype=np.float32)
    inv = 255.0 - arr

    proj = inv.mean(axis=1)
    nonzero = proj[proj > 1.0]
    if len(nonzero) == 0:
        return []

    thresh = np.percentile(nonzero, 75) * threshold_scale
    thresh = max(thresh, 3.0)

    in_band = False
    start = 0
    bands = []
    for i, val in enumerate(proj):
        if not in_band and val > thresh:
            in_band = True
            start = i
        elif in_band and val <= thresh:
            in_band = False
            if i - start >= min_height:
                bands.append([start, i])
    if in_band and h - start >= min_height:
        bands.append([start, int(h)])

    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))

    bboxes = []
    for (y0, y1) in merged:
        band_arr = inv[y0:y1, :]
        if band_arr.size == 0:
            continue
        col_proj = band_arr.max(axis=0)
        ink_cols = np.where(col_proj > max(thresh * 0.5, 2.0))[0]
        if len(ink_cols) == 0:
            continue
        x0 = max(0, int(ink_cols[0]) - pad * 2)
        x1 = min(w, int(ink_cols[-1]) + pad * 2)
        yp0 = max(0, y0 - pad)
        yp1 = min(h, y1 + pad)
        bw = x1 - x0
        bh = yp1 - yp0
        if bw < 10 or bh < 4:
            continue
        bboxes.append((x0, yp0, x1, yp1))

    return bboxes


def _is_band_layout_degenerate(bboxes: list, w: int, h: int) -> bool:
    if not bboxes:
        return True
    if len(bboxes) == 1:
        x0, y0, x1, y1 = bboxes[0]
        area_ratio = ((x1 - x0) * (y1 - y0)) / float(max(w * h, 1))
        return area_ratio > 0.70
    return False


def preprocess_crop(crop: Image.Image) -> Image.Image:
    """
    Keep a line-like aspect ratio before TrOCR resize.
    """
    crop = crop.convert("RGB")
    w, h = crop.size
    if h == 0:
        return crop
    aspect = w / h
    if aspect < MIN_ASPECT:
        target_w = int(h * MIN_ASPECT)
        pad_total = target_w - w
        pad_left = pad_total // 2
        padded = Image.new("RGB", (target_w, h), (255, 255, 255))
        padded.paste(crop, (pad_left, 0))
        return padded
    return crop


def run_ocr(crop: Image.Image) -> str:
    crop = preprocess_crop(crop)
    pv = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        gen = model.generate(pv, max_new_tokens=MAX_NEW_TOKENS)
    return processor.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


PALETTE = [
    (0, 229, 255),
    (255, 107, 107),
    (255, 217, 61),
    (107, 203, 119),
    (167, 139, 250),
    (249, 115, 22),
    (236, 72, 153),
    (20, 184, 166),
]


def draw_annotated(img_pil: Image.Image, bboxes: list, predictions: list) -> Image.Image:
    out = img_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

    for i, ((x0, y0, x1, y1), text) in enumerate(zip(bboxes, predictions)):
        r, g, b = PALETTE[i % len(PALETTE)]
        draw.rectangle([(x0, y0), (x1, y1)], fill=(r, g, b, 28))
        draw.rectangle([(x0, y0), (x1, y1)], outline=(r, g, b, 200), width=2)
        label = f"[{i + 1:02d}] {text[:55]}{'...' if len(text) > 55 else ''}"
        tb = draw.textbbox((x0 + 4, y0 + 2), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 4, tb[3] + 2], fill=(r, g, b, 210))
        draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(out, overlay).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class LineResult(BaseModel):
    index: int
    text: str
    x_left: int
    y_top: int
    x_right: int
    y_bottom: int
    confidence: str


class OCRResponse(BaseModel):
    lines: List[LineResult]
    annotated_image: str
    cropped_image: str
    receipt_cropped: bool
    total_lines: int
    image_size: str
    device: str
    full_text: str


@app.get("/")
async def serve_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=500, detail=f"UI file not found: {UI_FILE}")
    return FileResponse(str(UI_FILE))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": str(DEVICE),
        "ui_found": UI_FILE.exists(),
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    max_width: int = 1400,
    min_band_height: int = 8,
    merge_gap: int = 6,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    w, h = img.size
    if w > max_width:
        scale = max_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    receipt_img, was_cropped = auto_crop_receipt(img)

    bboxes = detect_text_bands(
        receipt_img,
        min_height=min_band_height,
        merge_gap=merge_gap,
        threshold_scale=0.60,
    )
    if _is_band_layout_degenerate(bboxes, receipt_img.width, receipt_img.height):
        bboxes = detect_text_bands(
            receipt_img,
            min_height=max(5, min_band_height - 2),
            merge_gap=merge_gap + 2,
            threshold_scale=0.48,
            pad=4,
        )
    if _is_band_layout_degenerate(bboxes, receipt_img.width, receipt_img.height):
        bboxes = detect_text_bands(
            receipt_img,
            min_height=max(4, min_band_height - 3),
            merge_gap=merge_gap + 4,
            threshold_scale=0.36,
            pad=4,
        )
    if not bboxes:
        bboxes = [(0, 0, receipt_img.width, receipt_img.height)]

    predictions = []
    for (x0, y0, x1, y1) in bboxes:
        crop = receipt_img.crop((x0, y0, x1, y1))
        predictions.append(run_ocr(crop) or "")

    if not any(text.strip() for text in predictions):
        bboxes = [(0, 0, receipt_img.width, receipt_img.height)]
        predictions = [run_ocr(receipt_img) or ""]

    annotated = draw_annotated(receipt_img, bboxes, predictions)

    lines = []
    for i, ((x0, y0, x1, y1), text) in enumerate(zip(bboxes, predictions)):
        conf = "high" if len(text) > 4 else ("medium" if len(text) > 1 else "low")
        lines.append(
            LineResult(
                index=i + 1,
                text=text,
                x_left=x0,
                y_top=y0,
                x_right=x1,
                y_bottom=y1,
                confidence=conf,
            )
        )

    rw, rh = receipt_img.size
    full_text = "\n".join(l.text for l in lines if l.text)
    return OCRResponse(
        lines=lines,
        annotated_image=pil_to_b64(annotated),
        cropped_image=pil_to_b64(receipt_img),
        receipt_cropped=was_cropped,
        total_lines=len(lines),
        image_size=f"{rw}x{rh}px",
        device=str(DEVICE).upper(),
        full_text=full_text,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
