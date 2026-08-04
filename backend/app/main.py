from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import datasets, episodes

app = FastAPI(
    title="SensorSync DataOps Studio API",
    description="Multimodal robot data cleaning, alignment and auto-labeling demo API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(episodes.router)


@app.get("/health")
def health():
    sample = Path(__file__).resolve().parents[2] / "sample_data" / "episodes" / "EP_0042"
    return {
        "status": "ok",
        "sample_data_ready": (sample / "metadata.json").exists(),
    }
