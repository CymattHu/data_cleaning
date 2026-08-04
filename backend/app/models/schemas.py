from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SkillSegment(BaseModel):
    name: str
    start: float
    end: float
    confidence: float = 0.9


class Issue(BaseModel):
    t: float
    type: str
    message: str
    severity: Literal["low", "medium", "high"] = "medium"
    action: Literal["Keep", "Repair", "Interpolate", "Trim", "Reject", "Needs Review"] = (
        "Needs Review"
    )


class SyncOffsets(BaseModel):
    reference_clock: str = "joint_state"
    rgb_ms: float
    depth_ms: float
    ft_ms: float
    average_sync_error_ms: float


class EpisodeSummary(BaseModel):
    id: str
    status: Literal["pass", "review", "reject"]
    quality_score: float
    sync_error_ms: float
    issue_count: int
    success: Optional[bool] = None
    has_data: bool = False


class EpisodeDetail(BaseModel):
    id: str
    status: Literal["pass", "review", "reject"]
    quality_score: float
    sync_error_ms: float
    dropped_frames_pct: float
    label_confidence: float
    success: Optional[bool] = None
    duration_s: float
    offsets: SyncOffsets
    issues: list[Issue]
    labels: list[SkillSegment]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlignRequest(BaseModel):
    reference_clock: Literal[
        "joint_state",
        "camera_rgb",
        "camera_depth",
        "force_torque",
        "ptp_grandmaster",
    ] = "joint_state"
    target_rate_hz: float = Field(20.0, ge=1.0, le=1000.0)
    rgb_method: str = "nearest"
    joint_method: str = "linear"
    tcp_method: str = "slerp"
    force_method: str = "lowpass"
    event_method: str = "zoh"


class EstimateOffsetRequest(BaseModel):
    reference_clock: Literal[
        "joint_state",
        "camera_rgb",
        "camera_depth",
        "force_torque",
        "ptp_grandmaster",
    ] = "joint_state"


class AlignResponse(BaseModel):
    episode_id: str
    before_sync_error_ms: float
    after_sync_error_ms: float
    offsets: SyncOffsets
    target_rate_hz: float
    methods: dict[str, str]


class CleanResponse(BaseModel):
    episode_id: str
    issues: list[Issue]
    quality_score: float
    dropped_frames_pct: float


class LabelsUpdate(BaseModel):
    labels: list[SkillSegment]


class ExportRequest(BaseModel):
    format: Literal["lerobot", "rlds", "hdf5", "parquet"] = "lerobot"
    target_rate_hz: float = 20.0
    include_raw: bool = True
    include_failed: bool = True
    episode_ids: list[str] = Field(default_factory=lambda: ["EP_0042"])
    version: str = "v1.2"


class LeRobotPreviewRequest(BaseModel):
    repo_id: str = Field(..., examples=["lerobot/pusht"])
    revision: str | None = "main"
    token: str | None = None


class LeRobotImportRequest(BaseModel):
    repo_id: str = Field(..., examples=["lerobot/pusht"])
    max_episodes: int = Field(3, ge=1, le=20)
    episode_indices: list[int] | None = None
    revision: str | None = "main"
    token: str | None = None



class DatasetCard(BaseModel):
    version: str
    episodes: int
    accepted: int
    rejected: int
    manual_review: int
    success_episodes: int
    failure_episodes: int
    average_sync_error_ms: float
    camera_drop_rate_pct: float
    format: str
    lineage: list[str]
    output_path: str


class TimelinePoint(BaseModel):
    t: float
    present: bool = True


class TimelineResponse(BaseModel):
    episode_id: str
    mode: Literal["raw", "aligned"]
    duration_s: float
    current_sync_error_ms: float
    sensors: dict[str, list[dict[str, Any]]]
    force_series: list[dict[str, float]]
    tcp_series: list[dict[str, float]]
    gripper_series: list[dict[str, Any]]
    joint_series: list[dict[str, Any]]
    skill_segments: list[SkillSegment]
    anomaly_regions: list[dict[str, Any]]
    drop_regions: list[dict[str, Any]]
    offset_regions: list[dict[str, Any]]
