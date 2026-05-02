# TrOCR Receipt OCR (SROIE 2019)

Fine-tuned receipt OCR project based on **TrOCR** (`microsoft/trocr-base-printed`) with:
1. a **Gradio** demo (`app.py`)
2. a **FastAPI V3 pipeline** (`api.py`) using **CRAFT + TrOCR** (detect words -> group lines -> recognize)

## Highlights

- OCR model: VisionEncoderDecoder (ViT encoder + RoBERTa decoder via TrOCR)
- Dataset: SROIE 2019 (receipt OCR benchmark)
- Demo: upload a receipt image and get extracted text
- Report: `docs/trocr-report.html`

## Repository Structure

```text
project_trocr/
├─ app.py                         # Gradio inference app
├─ api.py                         # FastAPI V3 CRAFT + TrOCR pipeline
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

If you run Python 3.13, install CRAFT explicitly:

```bash
pip install craft-text-detector --no-deps
```

### 2. Run with default Hugging Face model

```bash
python app.py
```

### 3. Run the FastAPI pipeline (CRAFT + TrOCR)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Run with your local fine-tuned model

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
| `TROCR_NUM_BEAMS` | `4` | Beam size used during generation in `api.py` |
| `TROCR_MODEL_PATH` | `./model` | Local TrOCR model path used by `api.py` |
| `TROCR_PROCESSOR_PATH` | `microsoft/trocr-base-printed` | Processor source used by `api.py` |
| `CRAFT_TEXT_THRESHOLD` | `0.4` | CRAFT text threshold (`api.py`) |
| `CRAFT_LINK_THRESHOLD` | `0.2` | CRAFT link threshold (`api.py`) |
| `CRAFT_LOW_TEXT` | `0.3` | CRAFT low-text threshold (`api.py`) |
| `CRAFT_LONG_SIZE` | `1600` | CRAFT resize long side (`api.py`) |
| `GRADIO_SHARE` | `false` | Set `true` to create a public Gradio link |

## Notes

- First run with a remote model can download large weights (~1.3 GB).
- Local runtime artifacts are ignored by git (`.gradio/`, local `model/`).
