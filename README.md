# TrOCR Receipt OCR (SROIE 2019)

Fine-tuned Transformer OCR pipeline for printed receipt text recognition using **TrOCR** (`microsoft/trocr-base-printed`), with a simple **Gradio** demo app.

## Highlights

- OCR model: VisionEncoderDecoder (ViT encoder + RoBERTa decoder via TrOCR)
- Dataset: SROIE 2019 (receipt OCR benchmark)
- Demo: upload a receipt image and get extracted text
- Report: `docs/trocr-report.html`

## Repository Structure

```text
project_trocr/
├─ app.py                         # Gradio inference app
├─ requirements.txt               # Python dependencies
├─ model/                         # Optional local fine-tuned model (not tracked)
├─ notebooks/                     # EDA / training / evaluation notebooks
└─ docs/
   ├─ trocr-report.html           # Final project report
   └─ output/                     # Figures used in the report
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with default Hugging Face model

```bash
python app.py
```

### 3. Run with your local fine-tuned model

`TROCR_MODEL_ID` accepts either a local path or a Hugging Face model ID.

**PowerShell**
```powershell
$env:TROCR_MODEL_ID = (Resolve-Path .\model).Path
$env:TROCR_PROCESSOR_ID = "microsoft/trocr-base-printed"
python app.py
```

**CMD**
```cmd
set TROCR_MODEL_ID=.\model
set TROCR_PROCESSOR_ID=microsoft/trocr-base-printed
python app.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TROCR_MODEL_ID` | `microsoft/trocr-base-printed` | Model path or Hugging Face model ID |
| `TROCR_PROCESSOR_ID` | same as `TROCR_MODEL_ID` | Processor/tokenizer source |
| `TROCR_MAX_NEW_TOKENS` | `64` | Max generated tokens per prediction |
| `GRADIO_SHARE` | `false` | Set `true` to create a public Gradio link |

## Notes

- First run with a remote model can download large weights (~1.3 GB).
- Local runtime artifacts are ignored by git (`.gradio/`, local `model/`).
