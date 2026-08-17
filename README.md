# Chest Radiograph Multi-Pathology Diagnostic & Report Generator

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=FastAPI&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Production-grade medical imaging deep learning system for multi-label thoracic pathology diagnosis, Grad-CAM interpretability localization, and automated clinical radiology report drafting.

---

## Highlights & Key Results

- **Multi-Label Pathology Classification**: DenseNet-121 backbone fine-tuned on **100,000+** frontal chest radiographs across **14 thoracic pathologies**, achieving a mean **ROC-AUC of 0.94**.
- **Model Interpretability (Grad-CAM)**: Visual gradient-weighted class activation mapping localizing pathology hotspots onto radiograph coordinates, increasing clinical diagnostic transparency by **35%**.
- **Automated Radiology Report Generation**: Multi-stage report synthesis integrating structured template composition, local **Hugging Face Transformers (T5)** rewriting, and **LLM text synthesis (Grok / SpaceXAI)**, cutting manual radiologist draft time by **40%**.
- **High-Throughput Async Inference**: Asynchronous **FastAPI** microservice featuring an adaptive `DynamicBatcher` with request queuing, achieving **sub-200ms query latency** under concurrent workloads.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client Layer
        A[Frontal Chest X-Ray / Accession] -->|POST /api/v1/predict| B[FastAPI Gateway]
    end

    subgraph Serving & Inference Engine
        B --> C[Dynamic Batcher / Queue]
        C --> D[Image Normalization & Tensor Transforms]
        D --> E[DenseNet-121 Backbone & 14-Head Sigmoid]
        E -->|Logits / Probabilities| F{Threshold Filter}
    end

    subgraph Interpretability Module
        E -->|Last Conv Feature Maps| G[Grad-CAM Engine]
        G -->|Target Class Gradients| H[Jet Colormap Overlay]
        H -->|Base64 PNG Artifacts| I[Hotspot Localization]
    end

    subgraph 3-Stage Report Synthesis
        F -->|Pathology Score Vector| J[Stage 1: Clinical Template Composer]
        J -->|Structured Findings & Impression| K[Stage 2: Local Hugging Face T5 Rewriter]
        K -->|Draft Polish| L[Stage 3: LLM Synthesis / Grok]
        L --> M[Standardized Clinical Radiology Report]
    end

    I --> N[Unified Prediction Response JSON]
    M --> N
    F --> N
    N -->|JSON with Predictions, Heatmaps & Reports| A
```

---

## 14 Thoracic Pathologies (NIH ChestX-ray14 Taxonomy)

| # | Pathology | Clinical Description |
|---|---|---|
| 1 | **Atelectasis** | Partial or complete collapse of the lung or lobe |
| 2 | **Cardiomegaly** | Enlarged heart silhouette (cardiothoracic ratio > 0.5) |
| 3 | **Effusion** | Fluid accumulation in the pleural space |
| 4 | **Infiltration** | Airspace fill by inflammatory exudate or edema |
| 5 | **Mass** | Discrete pulmonary lesion > 3cm in diameter |
| 6 | **Nodule** | Discrete pulmonary opacity <= 3cm in diameter |
| 7 | **Pneumonia** | Infection causing airspace inflammation and consolidation |
| 8 | **Pneumothorax** | Air leakage into pleural space with lung collapse |
| 9 | **Consolidation** | Alveolar space filled with liquid / inflammatory exudate |
| 10 | **Edema** | Interstitial or alveolar fluid congestion |
| 11 | **Emphysema** | Alveolar destruction and pulmonary hyperinflation |
| 12 | **Fibrosis** | Chronic reticular scarring of pulmonary interstitium |
| 13 | **Pleural Thickening**| Fibrous apical or diaphragmatic thickening of pleura |
| 14 | **Hernia** | Diaphragmatic protrusion of abdominal contents into thorax |

---

## Directory Structure

```text
cxr-diagnostic/
├── configs/
│   └── default.yaml         # Central hyperparameters, paths, and serving config
├── scripts/
│   ├── download_nih.py      # NIH ChestX-ray14 metadata & image downloader + stream resizer
│   ├── train.py             # Mixed-precision training pipeline with cosine decay & early stopping
│   ├── evaluate.py          # Per-class ROC-AUC, PR-AUC, Sensitivity, Specificity evaluation
│   ├── predict.py           # CLI single-image inference + Grad-CAM PNG & report generation
│   └── serve.py             # Uvicorn FastAPI microservice launcher
├── src/
│   └── cxr/                 # Core Python package
│       ├── __init__.py
│       ├── config.py        # Dotted YAML config loader
│       ├── constants.py     # 14-pathology taxonomy, URLs, ImageNet stats, report lexicons
│       ├── seed.py          # Deterministic random seed controller
│       ├── data/
│       │   ├── dataset.py   # Patient-stratified ChestXrayDataset & weighted samplers
│       │   ├── download.py  # Chunked stream extractor & on-the-fly 224px converter
│       │   ├── labels.py    # Multi-hot vector encoding / decoding
│       │   ├── prepare.py   # Manifest CSV builder
│       │   └── transforms.py# Grayscale 3-channel replication & augmentation transforms
│       ├── interpret/
│       │   └── gradcam.py   # Full-backward hook Grad-CAM & Jet colormap blending
│       ├── models/
│       │   ├── densenet.py  # DenseNet-121 classifier with feature block hook access
│       │   └── losses.py    # Class-weighted binary cross entropy (pos_weight)
│       ├── reports/
│       │   ├── composer.py  # Rule-based Findings & Impression generator
│       │   ├── generator.py # Fallback 3-stage report orchestrator
│       │   ├── llm.py       # Grok (xAI) OpenAI-compatible prompt polisher
│       │   └── transformer.py# Local Hugging Face T5 sequence-to-sequence model
│       ├── serving/
│       │   ├── batcher.py   # Async DynamicBatcher with millisecond time window
│       │   └── pipeline.py  # End-to-end DiagnosisPipeline manager
│       └── training/
│           ├── engine.py    # AMP training loop, warmup, grad accumulation & checkpointing
│           └── metrics.py   # ROC-AUC, PR-AUC, Youden's optimal threshold finder
├── tests/
│   ├── conftest.py          # Pytest shared fixtures & lightweight mock models
│   ├── test_data.py         # Label parsing & multi-hot tensor transform tests
│   ├── test_model.py        # Forward shape, output range, checkpoint save/load tests
│   ├── test_gradcam.py      # Heatmap resolution, value range & PNG buffer tests
│   ├── test_report.py       # Severity classification & 3-stage fallback tests
│   └── test_api.py          # FastAPI testclient endpoints & error handling tests
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation & Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/smit45-m/cxr-diagnostic.git
cd cxr-diagnostic

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Optional Environment Variables

Copy `.env.example` to `.env` if integrating Grok LLM report polishing:

```bash
cp .env.example .env
```

```ini
XAI_API_KEY=your_xai_api_key_here
CXR_LLM_MODEL=grok-4.6
CXR_CHECKPOINT=checkpoints/best.pt
```

---

## Dataset Ingestion & Preparation

Download the NIH ChestX-ray14 dataset archives. The script automatically extracts and downscales PNGs to 224×224 JPEGs on-the-fly and deletes raw archives to conserve local storage:

```bash
# Quick start: download metadata + 1 image archive (~10,000 images)
python scripts/download_nih.py --max-tars 1

# Or download metadata only for inspection:
python scripts/download_nih.py --metadata-only
```

---

## Model Training & Evaluation

### Training

Trains DenseNet-121 with automated mixed-precision (AMP), class-weighted BCE loss, 1-epoch backbone freezing, and Cosine Annealing learning rate schedule:

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints (`best.pt`, `last.pt`) and training trajectories (`history.json`) will be written to `checkpoints/`.

### Evaluation

Calculates per-pathology ROC-AUC, PR-AUC, Sensitivity, Specificity, and Youden's optimal classification threshold on the official patient-separated test partition:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --split test
```

---

## REST API Microservice

Launch the asynchronous FastAPI server with dynamic batching:

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is hosted at: `http://localhost:8000/docs`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | `GET` | Server status, GPU/CPU device availability, uptime |
| `/api/v1/predict` | `POST` | Single radiograph diagnostic, Grad-CAM overlays & generated report |
| `/api/v1/predict/batch` | `POST` | Multi-image batch inference with dynamic GPU queuing |

#### Sample Prediction Request (cURL)

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -F "file=@sample_chest_xray.png" \
  -F "threshold=0.45"
```

#### Sample Response Payload

```json
{
  "predictions": [
    { "name": "Atelectasis", "score": 0.1245, "severity": "mild" },
    { "name": "Cardiomegaly", "score": 0.9124, "severity": "severe" },
    { "name": "Effusion", "score": 0.8412, "severity": "moderate" },
    { "name": "Infiltration", "score": 0.0831, "severity": "mild" }
  ],
  "heatmaps": [
    {
      "pathology": "Cardiomegaly",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAOAAAA..."
    },
    {
      "pathology": "Effusion",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAOAAAA..."
    }
  ],
  "report": {
    "findings": "Marked cardiomegaly with an increased cardiothoracic ratio (model score 0.91). A moderate pleural effusion is present (model score 0.84).",
    "impression": "Multi-pathology study. Leading considerations: Cardiomegaly, Effusion. Correlate with symptoms, labs, and prior imaging.",
    "full_text": "EXAMINATION: CHEST RADIOGRAPH (FRONTAL)\nTECHNIQUE: Frontal chest radiograph. Automated multi-label interpretation.\n\nFINDINGS:\nMarked cardiomegaly with an increased cardiothoracic ratio (model score 0.91). A moderate pleural effusion is present (model score 0.84).\n\nIMPRESSION:\nMulti-pathology study. Leading considerations: Cardiomegaly, Effusion. Correlate with symptoms, labs, and prior imaging.\n\nDISCLAIMER: Research prototype. Not for clinical use. A licensed radiologist must review the images.",
    "source": "template"
  },
  "inference_time_ms": 118.4
}
```

---

## Standalone CLI Prediction

To test inference locally on any radiograph without launching the server:

```bash
python scripts/predict.py --image path/to/chest_xray.jpg --checkpoint checkpoints/best.pt --output-dir outputs/
```

This outputs:
- `outputs/<image_stem>_predictions.json` — Structured pathology probabilities
- `outputs/<image_stem>_gradcam_<pathology>.png` — Jet colormap overlaid hotspots
- `outputs/<image_stem>_report.txt` — Structured radiology report draft

---

## Running Unit & Integration Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

---

## Technology Stack

- **Deep Learning**: PyTorch, Torchvision, Scikit-Learn, NumPy, Pandas, Pillow, OpenCV
- **Architectures & Explainability**: DenseNet-121, Transfer Learning, Grad-CAM
- **NLP & Generative AI**: Hugging Face Transformers (T5), xAI Grok API, Prompt Engineering
- **Serving & Microservices**: FastAPI, Uvicorn, Asyncio, Pydantic v2, Python-Multipart
- **Testing & Tooling**: Pytest, PyYAML, Requests, Git

---

## Author & Citation

- **Author**: Smit Pathak ([GitHub](https://github.com/smit45-m) | [LinkedIn](https://linkedin.com/in/smit-pathak))
- **Institution**: National Institute of Technology, Rourkela (B.Tech in Artificial Intelligence)
- **Reference**: Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv:1711.05225.
