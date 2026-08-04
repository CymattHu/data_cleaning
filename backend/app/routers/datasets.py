import os

from fastapi import APIRouter, HTTPException

from ..importers.lerobot_importer import import_lerobot_dataset, preview_lerobot_dataset
from ..models.schemas import ExportRequest, LeRobotImportRequest, LeRobotPreviewRequest
from ..services import store
from ..services.pipeline import export_dataset

router = APIRouter(tags=["datasets"])


@router.get("/datasets")
def get_datasets():
    return store.datasets()


@router.post("/datasets/export")
def create_export(body: ExportRequest):
    return export_dataset(body.model_dump())


@router.post("/datasets/lerobot/preview")
def lerobot_preview(body: LeRobotPreviewRequest):
    token = body.token or os.environ.get("HF_TOKEN")
    try:
        return preview_lerobot_dataset(body.repo_id, revision=body.revision, token=token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Failed to preview dataset: {exc}") from exc


@router.post("/datasets/lerobot/import")
def lerobot_import(body: LeRobotImportRequest):
    token = body.token or os.environ.get("HF_TOKEN")
    try:
        result = import_lerobot_dataset(
            repo_id=body.repo_id,
            max_episodes=body.max_episodes,
            episode_indices=body.episode_indices,
            revision=body.revision,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Failed to import dataset: {exc}") from exc

    # Invalidate analysis cache for newly imported episodes
    for ep in result.get("episodes", []):
        store.analysis_cache.pop(ep["episode_id"], None)
        store.issues_override.pop(ep["episode_id"], None)
        store.quality_override.pop(ep["episode_id"], None)
        store.alignment_applied.pop(ep["episode_id"], None)

    return result
