import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from pydantic import BaseModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "./model")
PROCESSOR_PATH = os.getenv("TROCR_PROCESSOR_PATH", "microsoft/trocr-base-printed")
MAX_NEW_TOKENS = int(os.getenv("TROCR_MAX_NEW_TOKENS", "64"))
NUM_BEAMS = int(os.getenv("TROCR_NUM_BEAMS", "4"))

CRAFT_TEXT_THRESHOLD = float(os.getenv("CRAFT_TEXT_THRESHOLD", "0.4"))
CRAFT_LINK_THRESHOLD = float(os.getenv("CRAFT_LINK_THRESHOLD", "0.2"))
CRAFT_LOW_TEXT = float(os.getenv("CRAFT_LOW_TEXT", "0.3"))
CRAFT_LONG_SIZE = int(os.getenv("CRAFT_LONG_SIZE", "1600"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "templates" / "index.html"
CRAFT_OUTPUT_DIR = BASE_DIR / ".craft_output"


def _patch_torchvision_vgg_urls() -> None:
    import torchvision.models.vgg as vgg

    if not hasattr(vgg, "model_urls"):
        vgg.model_urls = {
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        }


print(f"[OCR] Loading processor : {PROCESSOR_PATH}")
print(f"[OCR] Loading model     : {MODEL_PATH}")
print(f"[OCR] Device            : {DEVICE}")
print(f"[OCR] Beams             : {NUM_BEAMS}")

processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

_patch_torchvision_vgg_urls()
try:
    from craft_text_detector import Craft
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'craft-text-detector'. Run: pip install craft-text-detector --no-deps"
    ) from exc

craft = Craft(
    output_dir=str(CRAFT_OUTPUT_DIR),
    crop_type="box",
    cuda=torch.cuda.is_available(),
    text_threshold=CRAFT_TEXT_THRESHOLD,
    link_threshold=CRAFT_LINK_THRESHOLD,
    low_text=CRAFT_LOW_TEXT,
    long_size=CRAFT_LONG_SIZE,
)

print("[OCR] TrOCR + CRAFT pipeline ready.\n")

app = FastAPI(title="Receipt OCR API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BBox = Tuple[int, int, int, int]


def quad_to_bbox(quad: np.ndarray, img_w: int, img_h: int, pad: int = 4) -> BBox:
    pts = np.asarray(quad, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("Invalid quad shape.")
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(img_w, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(img_h, int(np.ceil(pts[:, 1].max())) + pad)
    return x0, y0, x1, y1


def filter_noise_boxes(
    bboxes: List[BBox],
    img_w: int,
    min_width: int = 20,
    min_height: int = 8,
    min_area: int = 200,
) -> List[BBox]:
    clean: List[BBox] = []
    for (x0, y0, x1, y1) in bboxes:
        width = x1 - x0
        height = y1 - y0
        area = width * height
        aspect = width / max(height, 1)
        if width < min_width:
            continue
        if height < min_height:
            continue
        if area < min_area:
            continue
        if aspect > 30:
            continue
        if width > img_w * 0.98:
            continue
        clean.append((x0, y0, x1, y1))
    return clean


def group_boxes_into_lines(bboxes: List[BBox], tolerance: float = 0.3) -> List[Dict[str, List[BBox] | BBox]]:
    if not bboxes:
        return []

    sorted_boxes = sorted(bboxes, key=lambda b: (b[1] + b[3]) / 2.0)
    grouped: List[Dict[str, List[BBox] | BBox]] = []

    for box in sorted_boxes:
        bx0, by0, bx1, by1 = box
        b_center = (by0 + by1) / 2.0
        b_height = by1 - by0
        placed = False

        for line in grouped:
            lx0, ly0, lx1, ly1 = line["merged"]  # type: ignore[index]
            l_center = (ly0 + ly1) / 2.0
            l_height = ly1 - ly0
            if abs(b_center - l_center) < max(b_height, l_height) * tolerance:
                words = line["words"]  # type: ignore[index]
                words.append(box)
                line["merged"] = (
                    min(lx0, bx0),
                    min(ly0, by0),
                    max(lx1, bx1),
                    max(ly1, by1),
                )
                placed = True
                break

        if not placed:
            grouped.append({"words": [box], "merged": box})

    grouped.sort(key=lambda line: line["merged"][1])  # type: ignore[index]
    normalized: List[Dict[str, List[BBox] | BBox]] = []
    for line in grouped:
        words = sorted(line["words"], key=lambda b: b[0])  # type: ignore[index]
        merged = (
            min(b[0] for b in words),
            min(b[1] for b in words),
            max(b[2] for b in words),
            max(b[3] for b in words),
        )
        normalized.append({"words": words, "merged": merged})
    return normalized


def detect_word_boxes(
    img_pil: Image.Image,
    min_word_width: int = 20,
    min_word_height: int = 8,
    min_word_area: int = 200,
) -> List[BBox]:
    img_w, img_h = img_pil.size
    img_arr = np.array(img_pil.convert("RGB"))
    try:
        detection = craft.detect_text(image=img_arr)
    except TypeError:
        detection = craft.detect_text(img_arr)

    quads = detection.get("boxes", []) if isinstance(detection, dict) else []
    bboxes: List[BBox] = []
    for quad in quads:
        try:
            bboxes.append(quad_to_bbox(quad, img_w, img_h, pad=4))
        except ValueError:
            continue

    return filter_noise_boxes(
        bboxes,
        img_w=img_w,
        min_width=min_word_width,
        min_height=min_word_height,
        min_area=min_word_area,
    )


def run_ocr_crop(img_pil: Image.Image, bbox: BBox) -> str:
    x0, y0, x1, y1 = bbox
    width, height = img_pil.size
    left = max(0, int(x0))
    top = max(0, int(y0))
    right = min(width, int(x1))
    bottom = min(height, int(y1))

    if right <= left or bottom <= top:
        return ""

    crop = img_pil.crop((left, top, right, bottom)).convert("RGB")
    if crop.width < 8 or crop.height < 8:
        return ""

    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_ocr_line(img_pil: Image.Image, word_bboxes: List[BBox]) -> str:
    words: List[str] = []
    for bbox in sorted(word_bboxes, key=lambda b: b[0]):
        text = run_ocr_crop(img_pil, bbox)
        if text:
            words.append(text)
    return " ".join(words).strip()


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


def draw_annotated(img_pil: Image.Image, bboxes: List[BBox], predictions: List[str]) -> Image.Image:
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
        "pipeline": "craft+trocr-v3",
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    max_width: int = 1400,
    line_tolerance: float = 0.30,
    min_word_width: int = 20,
    min_word_height: int = 8,
    min_word_area: int = 200,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    if max_width < 200:
        raise HTTPException(status_code=400, detail="max_width must be >= 200.")
    if not 0.05 <= line_tolerance <= 1.0:
        raise HTTPException(status_code=400, detail="line_tolerance must be in [0.05, 1.0].")

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

    try:
        word_boxes = detect_word_boxes(
            img,
            min_word_width=min_word_width,
            min_word_height=min_word_height,
            min_word_area=min_word_area,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CRAFT detection failed: {exc}") from exc

    line_groups = group_boxes_into_lines(word_boxes, tolerance=line_tolerance)
    bboxes: List[BBox] = []
    predictions: List[str] = []
    for group in line_groups:
        words = group["words"]  # type: ignore[assignment]
        merged = group["merged"]  # type: ignore[assignment]
        text = run_ocr_line(img, words)  # type: ignore[arg-type]
        bboxes.append(merged)  # type: ignore[arg-type]
        predictions.append(text)

    if not bboxes:
        bboxes = [(0, 0, img.width, img.height)]
        predictions = [run_ocr_crop(img, bboxes[0])]

    if not any(text.strip() for text in predictions):
        bboxes = [(0, 0, img.width, img.height)]
        predictions = [run_ocr_crop(img, bboxes[0])]

    annotated = draw_annotated(img, bboxes, predictions)

    lines: List[LineResult] = []
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

    rw, rh = img.size
    full_text = "\n".join(line.text for line in lines if line.text)
    return OCRResponse(
        lines=lines,
        annotated_image=pil_to_b64(annotated),
        cropped_image=pil_to_b64(img),
        receipt_cropped=False,
        total_lines=len(lines),
        image_size=f"{rw}x{rh}px",
        device=str(DEVICE).upper(),
        full_text=full_text,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
