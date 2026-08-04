# SensorSync DataOps Studio

Web-based multimodal data cleaning, alignment and auto-labeling toolchain for robot skill fine-tuning.

Interview demo (~3 min): select episode → spot ~50 ms RGB/joint offset → Auto Align → compare Raw/Aligned timelines → detect blur/drop/force spike → edit skill labels → export Dataset Card.

## Quick start (Docker Compose)

```bash
docker compose up --build
```

- UI: http://localhost:3000
- API: http://localhost:8000
- Health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

First backend start generates synthetic episode `EP_0042` into `sample_data/` if missing.

Stop:

```bash
docker compose down
```

## Local development (optional)

```bash
# Backend
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
python scripts/generate_episode.py
export SAMPLE_DATA_DIR=$PWD/sample_data
export PYTHONPATH=$PWD/backend
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Import Hugging Face LeRobot datasets

In the left panel **Import LeRobot (HF)**:

1. Enter a Hub repo id, e.g. `lerobot/pusht`
2. Set max episodes (start with 1–2)
3. Click **Import & Analyze**

The backend downloads LeRobot v3 parquet/MP4 shards via `huggingface_hub`, converts each episode into the local SensorSync format, then runs automatic quality analysis.

```bash
# Optional token for gated datasets
export HF_TOKEN=hf_xxx
docker compose up --build
```

API:

- `POST /datasets/lerobot/preview` `{"repo_id":"lerobot/pusht"}`
- `POST /datasets/lerobot/import` `{"repo_id":"lerobot/pusht","max_episodes":2}`

Imported episodes appear as `HF_<repo>_<index>` in the episode list.

## Auto quality analysis (on load)

Opening an episode (or the dataset list) runs [`backend/app/services/quality.py`](backend/app/services/quality.py):

- Measures sync offset / jitter / drop rate from sensor timestamps
- Detects blur, depth holes, force spikes, TCP jumps
- Scores 0–100 and decides **Pass / Review / Reject** with explicit reasons

See results in the left episode list and right **Overview** panel (`Auto Decision` + `Why this status` + sensor diagnostics).

## Demo script

1. Open EP_0042 (pre-selected) → Overview shows auto decision **Review** and reasons.
2. Sync tab → **Estimate Offset** (RGB ~+48 ms).
3. **Apply Alignment** → switch timeline **Raw | Aligned** (48 ms → ~5 ms).
4. Quality tab → **Detect / Clean Issues** → click an issue to seek.
5. Labels tab → adjust a segment boundary.
6. Export → **Create Dataset v1.2** → show Dataset Card lineage.

## Pitch (EN)

> This tool is not only a data viewer. It closes the loop between sensor synchronization, data quality control and AI skill fine-tuning. It automatically detects synchronization errors, repairs or rejects corrupted samples, segments robot operations into skill phases, and exports traceable datasets for VLA or policy fine-tuning.

## Stack

- Frontend: Next.js 14, Tailwind, ECharts, Zustand
- Backend: FastAPI, NumPy/SciPy/Pandas/OpenCV
- Data: synthetic Episode (Parquet + JPEG/MP4 + JSON); MCAP importer stub reserved
