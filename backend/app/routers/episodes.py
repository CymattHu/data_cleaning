from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..models.schemas import AlignRequest, EstimateOffsetRequest, LabelsUpdate
from ..services import store
from ..services.pipeline import (
    align_episode,
    build_timeline,
    clean_episode,
    estimate_offsets,
)
from ..services.quality import analyze_episode

router = APIRouter(tags=["episodes"])


@router.get("/episodes/{episode_id}")
def get_episode(episode_id: str):
    try:
        meta = store.load_metadata(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Episode {episode_id} not found") from exc

    # Auto quality analysis on load
    analysis = analyze_episode(episode_id)

    sync_settings = store.get_sync_settings(episode_id)
    offsets = estimate_offsets(
        episode_id, reference_clock=sync_settings.get("reference_clock", "joint_state")
    )
    if store.alignment_applied.get(episode_id):
        # After alignment, residual offsets are near-zero in the aligned frame
        offsets = {
            **offsets,
            "rgb_ms": 0.0,
            "depth_ms": 0.0,
            "ft_ms": 0.0,
            "joint_ms": 0.0,
            "average_sync_error_ms": analysis["sync_error_ms"],
        }

    return {
        "id": episode_id,
        "status": analysis["status"],
        "quality_score": analysis["quality_score"],
        "sync_error_ms": analysis["sync_error_ms"],
        "dropped_frames_pct": analysis["dropped_frames_pct"],
        "label_confidence": analysis["label_confidence"],
        "success": analysis.get("success", meta.get("success", True)),
        "duration_s": meta.get("duration_s", 12.0),
        "offsets": offsets,
        "issues": analysis.get("issues") or store.get_issues(episode_id),
        "labels": store.get_labels(episode_id),
        "quality_report": {
            "analyzed": analysis.get("analyzed", False),
            "source": analysis.get("source", "unknown"),
            "decision_reasons": analysis.get("decision_reasons", []),
            "score_breakdown": analysis.get("score_breakdown"),
            "sensor_stats": analysis.get("sensor_stats", {}),
            "blur_stats": analysis.get("blur_stats"),
            "thresholds": analysis.get("thresholds", {}),
            "measured_offsets_ms": analysis.get("measured_offsets_ms"),
        },
        "metadata": {
            "task": meta.get("task", "connector_insertion"),
            "alignment_applied": store.alignment_applied.get(episode_id, False),
            "cleaned": store.cleaned.get(episode_id, False),
            "has_data": store.has_full_data(episode_id),
        },
        "sync_settings": sync_settings,
        "alignment_report": store.get_alignment_report(episode_id),
        "clean_report": store.get_clean_report(episode_id),
    }


@router.post("/episodes/{episode_id}/analyze")
def post_analyze(episode_id: str):
    if not store.has_full_data(episode_id):
        raise HTTPException(400, f"No playable sensor data for {episode_id}")
    # Clear cache and recompute from tables
    store.analysis_cache.pop(episode_id, None)
    return analyze_episode(episode_id, force_refresh=True)


@router.get("/episodes/{episode_id}/timeline")
def get_timeline(
    episode_id: str,
    mode: str = Query("raw", pattern="^(raw|aligned)$"),
):
    try:
        return build_timeline(episode_id, mode)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@router.get("/episodes/{episode_id}/media/{stream}")
def get_media(episode_id: str, stream: str, frame: int = 0):
    base = store.episode_path(episode_id)
    if stream == "rgb":
        path = base / "rgb" / f"{frame:05d}.jpg"
    elif stream == "depth":
        path = base / "depth" / f"{frame:05d}.jpg"
    elif stream == "rgb_video":
        path = base / "rgb.mp4"
    else:
        raise HTTPException(400, "Unknown stream")
    if not path.exists():
        raise HTTPException(404, f"Media not found: {path.name}")
    return FileResponse(path)


@router.post("/episodes/{episode_id}/align")
def post_align(episode_id: str, body: AlignRequest | None = None):
    body = body or AlignRequest()
    if not store.has_full_data(episode_id):
        raise HTTPException(400, f"No playable sensor data for {episode_id}")
    result = align_episode(episode_id, body.model_dump())
    # Invalidate analysis so aligned sync is reflected
    store.analysis_cache.pop(episode_id, None)
    analyze_episode(episode_id)
    return result


@router.post("/episodes/{episode_id}/estimate_offset")
def post_estimate(episode_id: str, body: EstimateOffsetRequest | None = None):
    body = body or EstimateOffsetRequest()
    store.set_sync_settings(episode_id, reference_clock=body.reference_clock)
    return estimate_offsets(episode_id, reference_clock=body.reference_clock)


@router.post("/episodes/{episode_id}/clean")
def post_clean(episode_id: str):
    return clean_episode(episode_id)


@router.get("/episodes/{episode_id}/labels")
def get_labels(episode_id: str):
    return {"episode_id": episode_id, "labels": store.get_labels(episode_id)}


@router.put("/episodes/{episode_id}/labels")
def put_labels(episode_id: str, body: LabelsUpdate):
    labels = [seg.model_dump() for seg in body.labels]
    return {"episode_id": episode_id, "labels": store.set_labels(episode_id, labels)}
