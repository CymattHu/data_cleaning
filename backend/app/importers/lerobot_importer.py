"""Import Hugging Face LeRobot datasets (v3.0 / v2.1-ish) into SensorSync episodes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from ..services.store import EPISODE_DIR, SAMPLE_DATA


def _safe_repo_slug(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id.replace("/", "_"))


def _as_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def _feature_video_keys(info: dict[str, Any]) -> list[str]:
    keys = []
    for name, feat in (info.get("features") or {}).items():
        dtype = feat.get("dtype")
        if dtype in {"video", "image"} or name.startswith("observation.image"):
            keys.append(name)
    return keys


def _load_info(repo_id: str, revision: str | None, token: str | None) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id,
        "meta/info.json",
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    with open(path) as f:
        return json.load(f)


def _load_episodes_meta(repo_id: str, revision: str | None, token: str | None) -> pd.DataFrame:
    files = list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token)
    ep_files = sorted(
        f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    if not ep_files:
        # v2.1 fallback
        jsonl = "meta/episodes.jsonl"
        if jsonl in files:
            path = hf_hub_download(
                repo_id, jsonl, repo_type="dataset", revision=revision, token=token
            )
            rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            return pd.DataFrame(rows)
        raise FileNotFoundError("No meta/episodes found in dataset")

    frames = []
    for rel in ep_files:
        path = hf_hub_download(
            repo_id, rel, repo_type="dataset", revision=revision, token=token
        )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _download_data_file(
    repo_id: str,
    info: dict[str, Any],
    chunk_index: int,
    file_index: int,
    revision: str | None,
    token: str | None,
) -> Path:
    template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    rel = template.format(chunk_index=chunk_index, file_index=file_index)
    return Path(
        hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision, token=token)
    )


def _download_video_file(
    repo_id: str,
    info: dict[str, Any],
    video_key: str,
    chunk_index: int,
    file_index: int,
    revision: str | None,
    token: str | None,
) -> Path:
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    rel = template.format(
        video_key=video_key, chunk_index=chunk_index, file_index=file_index
    )
    return Path(
        hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision, token=token)
    )


def _extract_video_segment(
    video_path: Path,
    t0: float,
    t1: float,
    out_rgb_dir: Path,
    out_depth_dir: Path,
    out_mp4: Path,
) -> pd.DataFrame:
    """Extract frames in [t0, t1) using PyAV (supports AV1)."""
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 10.0)

    # Seek near start (AV_TIME_BASE microseconds)
    if t0 > 0:
        container.seek(int(t0 * 1_000_000), any_frame=False, backward=True)

    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        None,  # set after first frame
    )
    size_set = False
    rows = []
    frame_idx = 0

    for frame in container.decode(video=0):
        ts = float(frame.time) if frame.time is not None else frame_idx / fps
        if ts < t0 - 1e-3:
            continue
        if ts >= t1 - 1e-6:
            break
        img = frame.to_ndarray(format="bgr24")
        # Upscale tiny policy frames for UI readability
        if img.shape[0] < 160 or img.shape[1] < 160:
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        if not size_set:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(
                str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            size_set = True

        local_t = max(0.0, ts - t0)
        cv2.imwrite(str(out_rgb_dir / f"{frame_idx:05d}.jpg"), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_depth_dir / f"{frame_idx:05d}.jpg"), depth)
        writer.write(img)

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        rows.append(
            {
                "t": local_t,
                "frame_idx": frame_idx,
                "blur_score": blur,
                "brightness": float(gray.mean()),
                "motion_proxy": float(gray.mean()),
                "dropped": False,
            }
        )
        frame_idx += 1

    container.close()
    writer.release()
    if not rows:
        raise RuntimeError(f"No frames extracted from {video_path} in [{t0}, {t1})")
    return pd.DataFrame(rows)


def _heuristic_labels(duration: float, success: bool | None) -> list[dict[str, Any]]:
    # Lightweight phase split for imported episodes
    cuts = [0.0, duration * 0.2, duration * 0.45, duration * 0.7, duration * 0.9, duration]
    names = ["Approach", "Manipulate", "Transport", "Place", "Retreat"]
    labels = []
    for i, name in enumerate(names):
        labels.append(
            {
                "name": name,
                "start": round(cuts[i], 3),
                "end": round(cuts[i + 1], 3),
                "confidence": 0.85 if success else 0.8,
            }
        )
    return labels


def convert_episode(
    repo_id: str,
    ep_row: pd.Series,
    info: dict[str, Any],
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    episode_index = int(ep_row["episode_index"])
    slug = _safe_repo_slug(repo_id)
    episode_id = f"HF_{slug}_{episode_index:04d}"
    out = EPISODE_DIR / episode_id
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    out.mkdir(parents=True, exist_ok=True)

    chunk = int(ep_row.get("data/chunk_index", ep_row.get("chunk_index", 0)))
    file_idx = int(ep_row.get("data/file_index", ep_row.get("file_index", 0)))
    data_path = _download_data_file(repo_id, info, chunk, file_idx, revision, token)
    df = pd.read_parquet(data_path)
    ep_df = df[df["episode_index"] == episode_index].copy()
    if ep_df.empty and "dataset_from_index" in ep_row:
        # fallback by global index range
        a, b = int(ep_row["dataset_from_index"]), int(ep_row["dataset_to_index"])
        ep_df = df.iloc[a:b].copy()
    if ep_df.empty:
        raise ValueError(f"No frames for episode_index={episode_index}")

    ep_df = ep_df.sort_values("frame_index" if "frame_index" in ep_df.columns else "timestamp")
    timestamps = ep_df["timestamp"].to_numpy(dtype=np.float64)
    # Normalize to start at 0
    t0_local = float(timestamps[0])
    t = timestamps - t0_local
    duration = float(t[-1]) if len(t) else float(len(ep_df)) / float(info.get("fps", 10))

    # Joint / TCP from observation.state
    states = np.stack([_as_float_array(v) for v in ep_df["observation.state"].to_list()])
    n, d = states.shape
    # Pad to 6 joints for UI compatibility
    q = np.zeros((n, 6), dtype=np.float64)
    q[:, : min(6, d)] = states[:, : min(6, d)]
    dq = np.gradient(q, t, axis=0) if n > 1 else np.zeros_like(q)

    joint_df = pd.DataFrame(
        {
            "t": t,
            "q0": q[:, 0],
            "q1": q[:, 1],
            "q2": q[:, 2],
            "q3": q[:, 3],
            "q4": q[:, 4],
            "q5": q[:, 5],
            "dq0": dq[:, 0],
            "dq1": dq[:, 1],
            "dq2": dq[:, 2],
        }
    )
    joint_df.to_parquet(out / "joint_state.parquet", index=False)

    # TCP proxy from first state dims (smooth-normalized to avoid fake jumps)
    def _smooth_unit(col: np.ndarray) -> np.ndarray:
        cmin, cmax = float(np.min(col)), float(np.max(col))
        if cmax - cmin < 1e-6:
            return np.zeros_like(col)
        return (col - cmin) / (cmax - cmin)

    tcp = pd.DataFrame(
        {
            "t": t,
            "x": 0.2 + 0.15 * _smooth_unit(q[:, 0]),
            "y": 0.0 + 0.10 * _smooth_unit(q[:, 1]) if d > 1 else np.zeros(n),
            "z": 0.18 + 0.04 * _smooth_unit(q[:, 2] if d > 2 else q[:, 0]),
            "qx": np.zeros(n),
            "qy": np.zeros(n),
            "qz": np.zeros(n),
            "qw": np.ones(n),
        }
    )
    tcp.to_parquet(out / "tcp_pose.parquet", index=False)

    # Action deltas as low-magnitude wrench proxy (avoid raw action units as Newtons)
    actions = np.stack([_as_float_array(v) for v in ep_df["action"].to_list()])
    act_delta = np.linalg.norm(np.diff(actions, axis=0, prepend=actions[:1]), axis=1)
    act_delta_n = 25.0 * act_delta / (np.percentile(act_delta, 95) + 1e-6)
    force = pd.DataFrame(
        {
            "t": t,
            "fx": np.zeros(n),
            "fy": np.zeros(n),
            "fz": act_delta_n,
            "tx": np.zeros(n),
            "ty": np.zeros(n),
            "tz": np.zeros(n),
        }
    )
    force.to_parquet(out / "force_torque.parquet", index=False)

    width = 0.08 - 0.04 * (act_delta - act_delta.min()) / (np.ptp(act_delta) + 1e-6)
    grip = pd.DataFrame({"t": t, "width": width, "closed": width < 0.05})
    grip.to_parquet(out / "gripper_state.parquet", index=False)

    # LeRobot next.success is often all-False when success is not annotated.
    # Only mark True when observed; otherwise leave unknown (None).
    success = None
    if "next.success" in ep_df.columns and bool(ep_df["next.success"].astype(bool).any()):
        success = True

    events = [{"t": 0.0, "event": "episode_start"}]
    if success is True:
        events.append({"t": duration * 0.9, "event": "task_success"})
    elif success is False:
        events.append({"t": duration * 0.9, "event": "task_fail"})
    events.append({"t": duration, "event": "episode_end"})
    pd.DataFrame(events).to_parquet(out / "task_event.parquet", index=False)

    # Video
    video_keys = _feature_video_keys(info)
    rgb_df = None
    if video_keys:
        vkey = video_keys[0]
        v_chunk_col = f"videos/{vkey}/chunk_index"
        v_file_col = f"videos/{vkey}/file_index"
        v_from_col = f"videos/{vkey}/from_timestamp"
        v_to_col = f"videos/{vkey}/to_timestamp"
        if v_chunk_col in ep_row.index:
            v_chunk = int(ep_row[v_chunk_col])
            v_file = int(ep_row[v_file_col])
            vt0 = float(ep_row[v_from_col])
            vt1 = float(ep_row[v_to_col])
            video_path = _download_video_file(
                repo_id, info, vkey, v_chunk, v_file, revision, token
            )
            rgb_df = _extract_video_segment(
                video_path, vt0, vt1, rgb_dir, depth_dir, out / "rgb.mp4"
            )
            # Align length to state timestamps if needed
            if len(rgb_df) != len(t):
                # resample blur scores onto state timeline via nearest frame
                pass

    if rgb_df is None:
        # No video: generate placeholder frames from state
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        writer = None
        for i, ti in enumerate(t):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            img[:] = (30, 34, 40)
            cv2.putText(
                img,
                f"{episode_id} t={ti:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 220, 255),
                1,
            )
            x = int(40 + (q[i, 0] % 200))
            y = int(80 + (q[i, 1] % 120) if d > 1 else 120)
            cv2.circle(img, (x, y), 12, (80, 180, 230), -1)
            cv2.imwrite(str(rgb_dir / f"{i:05d}.jpg"), img)
            depth = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(depth_dir / f"{i:05d}.jpg"), depth)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(info.get("fps", 10)), (320, 240)
                )
            writer.write(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows.append(
                {
                    "t": float(ti),
                    "frame_idx": i,
                    "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    "brightness": float(gray.mean()),
                    "motion_proxy": float(x),
                    "dropped": False,
                }
            )
        if writer:
            writer.release()
        rgb_df = pd.DataFrame(rows)

    rgb_df.to_parquet(out / "camera_rgb.parquet", index=False)
    depth_df = rgb_df[["t", "frame_idx", "dropped"]].copy()
    depth_df["valid_ratio"] = 0.98
    depth_df.to_parquet(out / "camera_depth.parquet", index=False)

    labels = _heuristic_labels(duration, success)
    tasks = ep_row["tasks"] if "tasks" in ep_row.index else []
    if isinstance(tasks, np.ndarray):
        task_list = [str(x) for x in tasks.tolist()]
    elif isinstance(tasks, list):
        task_list = [str(x) for x in tasks]
    else:
        task_list = [str(tasks)] if tasks is not None else []

    fps = float(info.get("fps", 10))
    metadata = {
        "id": episode_id,
        "task": task_list[0] if task_list else info.get("robot_type", "lerobot"),
        "tasks": task_list,
        "duration_s": duration,
        "success": success,
        "status": "review",
        "quality_score": 70,
        "sync_error_ms": 0.0,
        "dropped_frames_pct": 0.0,
        "label_confidence": 0.8,
        "offsets": {
            "reference_clock": "joint_state",
            "rgb_ms": 0.0,
            "depth_ms": 0.0,
            "ft_ms": 0.0,
            "average_sync_error_ms": 0.0,
        },
        "injected": {"offsets_ms": {"rgb": 0.0, "depth": 0.0, "ft": 0.0}},
        "labels": labels,
        "issues": [],
        "sensors": {
            "rgb_hz": fps,
            "depth_hz": fps,
            "joint_hz": fps,
            "ft_hz": fps,
        },
        "source": {
            "type": "huggingface_lerobot",
            "repo_id": repo_id,
            "revision": revision or "main",
            "episode_index": episode_index,
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
        },
    }
    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "episode_id": episode_id,
        "episode_index": episode_index,
        "duration_s": duration,
        "frames": int(len(ep_df)),
        "success": success,
        "task": metadata["task"],
        "path": str(out),
    }


def import_lerobot_dataset(
    repo_id: str,
    max_episodes: int = 3,
    episode_indices: list[int] | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Download selected episodes from a HF LeRobot dataset and convert them."""
    info = _load_info(repo_id, revision, token)
    ep_meta = _load_episodes_meta(repo_id, revision, token)
    ep_meta = ep_meta.sort_values("episode_index")

    if episode_indices:
        selected = ep_meta[ep_meta["episode_index"].isin(episode_indices)]
    else:
        selected = ep_meta.head(max_episodes)

    imported = []
    errors = []
    for _, row in selected.iterrows():
        try:
            imported.append(
                convert_episode(repo_id, row, info, revision=revision, token=token)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"episode_index": int(row.get("episode_index", -1)), "error": str(exc)}
            )

    catalog_path = SAMPLE_DATA / "hf_imports.json"
    record = {
        "repo_id": repo_id,
        "revision": revision or "main",
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes_in_repo": int(info.get("total_episodes", len(ep_meta))),
        "imported": imported,
        "errors": errors,
    }
    history = []
    if catalog_path.exists():
        history = json.loads(catalog_path.read_text())
        if not isinstance(history, list):
            history = [history]
    history.append(record)
    catalog_path.write_text(json.dumps(history, indent=2))

    return {
        "repo_id": repo_id,
        "revision": revision or "main",
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "imported_count": len(imported),
        "episodes": imported,
        "errors": errors,
    }


def preview_lerobot_dataset(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    info = _load_info(repo_id, revision, token)
    ep_meta = _load_episodes_meta(repo_id, revision, token)
    return {
        "repo_id": repo_id,
        "revision": revision or "main",
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes": int(info.get("total_episodes", len(ep_meta))),
        "total_frames": int(info.get("total_frames", 0)),
        "features": list((info.get("features") or {}).keys()),
        "video_keys": _feature_video_keys(info),
        "sample_episode_indices": [
            int(x) for x in ep_meta["episode_index"].head(10).tolist()
        ],
    }
