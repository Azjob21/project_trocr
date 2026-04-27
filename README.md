# TrOCR Receipt OCR

Lightweight Gradio app for OCR on receipt images using TrOCR.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

By default, the app loads `microsoft/trocr-base-printed`.

## Use a local fine-tuned model

Set environment variables before running:

```bash
set TROCR_MODEL_ID=.\model
set TROCR_PROCESSOR_ID=microsoft/trocr-base-printed
python app.py
```

`TROCR_MODEL_ID` can be a local path or a Hugging Face model ID.
