import os

import gradio as gr
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = os.getenv("TROCR_MODEL_ID", "microsoft/trocr-base-printed")
PROCESSOR_ID = os.getenv("TROCR_PROCESSOR_ID", MODEL_ID)
MAX_NEW_TOKENS = int(os.getenv("TROCR_MAX_NEW_TOKENS", "64"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained(PROCESSOR_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()


def predict(image):
    if image is None:
        return "No image provided."
    img = Image.fromarray(image).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated = model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload receipt image"),
    outputs=gr.Textbox(label="Extracted text"),
    title="Receipt OCR — TrOCR",
    description="Upload a receipt image to extract text with TrOCR.",
)


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() == "true"
    app.launch(share=share)
