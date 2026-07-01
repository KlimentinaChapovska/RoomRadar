# RoomRadar

Predicts hotel booking **cancellation probability** and **room price** from a guest's booking details.

Built for the MADA Final Project using the [Kaggle Hotel Reservations dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset).

---

## Architecture

```
Offline (one-time)                     Runtime
──────────────────                     ───────
data/raw/  ──►  ml/scripts/train.py    React (Vercel)
               saves models/           │  HTTPS
                    │                  ▼
                    └──► FastAPI /predict
```

See `docs/architecture/` for full diagrams.

---

## Project Layout

```
FinalProject/
├── data/
│   ├── raw/                  # Original, untouched CSV
│   └── processed/            # Cleaned splits (generated)
├── ml/
│   ├── notebooks/            # EDA and exploration
│   └── scripts/train.py      # Training entry point
├── models/                   # Saved .pkl files (git-ignored)
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI app factory
│       ├── routers/          # /health, /ready, /predict
│       ├── schemas/          # Pydantic I/O models
│       ├── middleware/       # Request logging
│       └── utils/logger.py   # Rotating file + console logger
├── frontend/                 # React dashboard (not yet scaffolded)
├── reports/report.qmd        # Quarto HTML report
├── slides/slides.qmd         # Quarto revealjs slides
├── outputs/                  # Figures and tables (generated)
├── docs/
│   ├── architecture/         # Drawio diagrams and SVGs
│   ├── reference/            # Proposal and workflow PDFs
│   └── data_dictionary.md
├── tests/
│   ├── unit/                 # Schema validation tests
│   └── integration/          # API endpoint tests
├── logs/                     # Rotating logs (git-ignored)
├── requirements.txt          # Dev / ML dependencies
└── pytest.ini
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Train models *(not yet implemented)*

```bash
python ml/scripts/train.py
```

### 3. Start the API

```bash
cd backend
uvicorn app.main:app --reload
```

- Liveness:  `GET /health`  — always 200 while process is alive
- Readiness: `GET /ready`   — 503 until models are loaded
- Predict:   `POST /api/v1/predict`

### 4. Run tests

```bash
pytest
```

---

## Dataset

`data/raw/Hotel Reservations.csv` — 36,275 rows × 19 columns, no missing values.

| Target | Class | % |
|---|---|---|
| `booking_status` | Not_Canceled | 67.2% |
| `booking_status` | Canceled | 32.8% |

See `docs/data_dictionary.md` for full column reference and leakage notes.
