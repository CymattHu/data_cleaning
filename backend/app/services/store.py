from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Prefer SAMPLE_DATA_DIR for Docker; fall back to repo-root/sample_data
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DATA = Path(os.environ.get("SAMPLE_DATA_DIR", _DEFAULT_ROOT / "sample_data"))
EPISODE_DIR = SAMPLE_DATA / "episodes"


# Fallback catalog; live values are refreshed from on-disk metadata when available.
EPISODE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "EP_0038",
        "status": "pass",
        "quality_score": 92,
        "sync_error_ms": 4.2,
        "issue_count": 0,
        "success": True,
        "has_data": True,
    },
    {
        "id": "EP_0039",
        "status": "review",
        "quality_score": 71,
        "sync_error_ms": 22.0,
        "issue_count": 2,
        "success": True,
        "has_data": True,
    },
    {
        "id": "EP_0040",
        "status": "reject",
        "quality_score": 41,
        "sync_error_ms": 86.0,
        "issue_count": 7,
        "success": False,
        "has_data": True,
    },
    {
        "id": "EP_0041",
        "status": "pass",
        "quality_score": 88,
        "sync_error_ms": 6.1,
        "issue_count": 1,
        "success": True,
        "has_data": True,
    },
    {
        "id": "EP_0042",
        "status": "review",
        "quality_score": 76,
        "sync_error_ms": 48.0,
        "issue_count": 4,
        "success": True,
        "has_data": True,
    },
]


class EpisodeStore:
    """In-memory demo state layered on top of on-disk synthetic episode."""

    def __init__(self) -> None:
        self.alignment_applied: dict[str, bool] = {}
        self.alignment_reports: dict[str, dict[str, Any]] = {}
        self.cleaned: dict[str, bool] = {}
        self.clean_reports: dict[str, dict[str, Any]] = {}
        self.labels_override: dict[str, list[dict[str, Any]]] = {}
        self.issues_override: dict[str, list[dict[str, Any]]] = {}
        self.issues_detected: dict[str, list[dict[str, Any]]] = {}
        self.quality_override: dict[str, float] = {}
        self.analysis_cache: dict[str, dict[str, Any]] = {}
        self.sync_settings: dict[str, dict[str, Any]] = {}
        self.exports: list[dict[str, Any]] = []
        self.dataset_version = "v1.1"
        self._cache: dict[str, Any] = {}

    def get_alignment_report(self, episode_id: str) -> dict[str, Any] | None:
        return self.alignment_reports.get(episode_id)

    def get_clean_report(self, episode_id: str) -> dict[str, Any] | None:
        return self.clean_reports.get(episode_id)

    def get_sync_settings(self, episode_id: str) -> dict[str, Any]:
        return self.sync_settings.get(
            episode_id,
            {"reference_clock": "joint_state", "target_rate_hz": 20.0},
        )

    def set_sync_settings(
        self,
        episode_id: str,
        *,
        reference_clock: str | None = None,
        target_rate_hz: float | None = None,
    ) -> dict[str, Any]:
        cur = dict(self.get_sync_settings(episode_id))
        if reference_clock is not None:
            cur["reference_clock"] = reference_clock
        if target_rate_hz is not None:
            cur["target_rate_hz"] = float(target_rate_hz)
        self.sync_settings[episode_id] = cur
        return cur

    def datasets_raw_catalog(self) -> list[dict[str, Any]]:
        return [dict(item) for item in EPISODE_CATALOG]

    def discover_episode_ids(self) -> list[str]:
        """Union of built-in catalog + any episode folders on disk (e.g. HF imports)."""
        ids: list[str] = [item["id"] for item in EPISODE_CATALOG]
        if EPISODE_DIR.exists():
            for path in sorted(EPISODE_DIR.iterdir()):
                if path.is_dir() and (path / "metadata.json").exists():
                    if path.name not in ids:
                        ids.append(path.name)
        return ids

    def datasets(self) -> dict[str, Any]:
        # Lazy import to avoid circular dependency with quality.py
        from .quality import analyze_episode

        catalog_map = {item["id"]: dict(item) for item in EPISODE_CATALOG}
        episodes = []
        hf_count = 0
        for ep_id in self.discover_episode_ids():
            ep = catalog_map.get(
                ep_id,
                {
                    "id": ep_id,
                    "status": "review",
                    "quality_score": 70,
                    "sync_error_ms": 0,
                    "issue_count": 0,
                    "success": None,
                    "has_data": True,
                },
            )
            meta_path = self.episode_path(ep_id) / "metadata.json"
            ep["has_data"] = meta_path.exists()
            if meta_path.exists():
                with meta_path.open() as f:
                    meta = json.load(f)
                if meta.get("source", {}).get("type") == "huggingface_lerobot":
                    hf_count += 1
                    ep["source"] = "huggingface"
                analysis = analyze_episode(ep_id)
                ep["status"] = analysis["status"]
                ep["quality_score"] = analysis["quality_score"]
                ep["sync_error_ms"] = analysis["sync_error_ms"]
                ep["issue_count"] = analysis.get("issue_count", len(analysis.get("issues", [])))
                ep["success"] = analysis.get("success", ep.get("success"))
            episodes.append(ep)

        # Show HuggingFace imports first so newly imported episodes are visible
        episodes.sort(key=lambda e: (0 if str(e["id"]).startswith("HF_") else 1, e["id"]))

        review_count = sum(1 for e in episodes if e["status"] == "review")
        dataset_name = "Trial_2026_08"
        if hf_count:
            dataset_name = f"Trial_2026_08 + HF({hf_count})"
        return {
            "project": "Connector Insertion PoC",
            "dataset": dataset_name,
            "trees": [
                {
                    "name": "Datasets",
                    "children": [
                        {"id": "raw", "name": "Raw Data", "count": len(episodes)},
                        {
                            "id": "hf",
                            "name": "HuggingFace LeRobot",
                            "count": hf_count,
                        },
                        {
                            "id": "clean_v11",
                            "name": "Cleaned / Accepted",
                            "count": sum(1 for e in episodes if e["status"] == "pass"),
                        },
                        {"id": "review", "name": "Review Queue", "count": review_count},
                    ],
                }
            ],
            "episodes": episodes,
            "dataset_version": self.dataset_version,
        }

    def episode_path(self, episode_id: str) -> Path:
        return EPISODE_DIR / episode_id

    def has_full_data(self, episode_id: str) -> bool:
        meta = self.episode_path(episode_id) / "metadata.json"
        return meta.exists()

    def load_metadata(self, episode_id: str) -> dict[str, Any]:
        path = self.episode_path(episode_id) / "metadata.json"
        if not path.exists():
            # Fall back to catalog stub
            for item in EPISODE_CATALOG:
                if item["id"] == episode_id:
                    return {
                        "id": episode_id,
                        "duration_s": 12.0,
                        "success": item.get("success"),
                        "quality_score": item["quality_score"],
                        "sync_error_ms": item["sync_error_ms"],
                        "dropped_frames_pct": 3.2 if episode_id == "EP_0042" else 0.5,
                        "label_confidence": 0.88,
                        "offsets": {
                            "reference_clock": "joint_state",
                            "rgb_ms": 48.0,
                            "depth_ms": 55.0,
                            "ft_ms": -8.0,
                            "average_sync_error_ms": item["sync_error_ms"],
                        },
                        "issues": [],
                        "labels": [],
                        "injected": {},
                    }
            raise FileNotFoundError(episode_id)
        with path.open() as f:
            return json.load(f)

    def load_table(self, episode_id: str, name: str) -> pd.DataFrame:
        key = f"{episode_id}:{name}"
        if key in self._cache:
            return self._cache[key]
        path = self.episode_path(episode_id) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        self._cache[key] = df
        return df

    def get_labels(self, episode_id: str) -> list[dict[str, Any]]:
        if episode_id in self.labels_override:
            return self.labels_override[episode_id]
        meta = self.load_metadata(episode_id)
        return meta.get("labels", [])

    def set_labels(self, episode_id: str, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.labels_override[episode_id] = labels
        return labels

    def get_issues(self, episode_id: str) -> list[dict[str, Any]]:
        if episode_id in self.issues_override:
            return self.issues_override[episode_id]
        meta = self.load_metadata(episode_id)
        return meta.get("issues", [])

    def sync_error(self, episode_id: str) -> float:
        if self.alignment_applied.get(episode_id):
            return 5.0
        meta = self.load_metadata(episode_id)
        return float(meta.get("sync_error_ms", 48.0))

    def quality_score(self, episode_id: str) -> float:
        if episode_id in self.quality_override:
            return self.quality_override[episode_id]
        meta = self.load_metadata(episode_id)
        score = float(meta.get("quality_score", 76))
        if self.alignment_applied.get(episode_id):
            score = min(100.0, score + 8)
        if self.cleaned.get(episode_id):
            score = min(100.0, score + 6)
        return score


store = EpisodeStore()
