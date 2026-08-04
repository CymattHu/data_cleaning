#!/usr/bin/env python3
"""Generate synthetic connector-insertion episodes for the interview demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

DURATION = 12.0
RGB_HZ = 30.0
DEPTH_HZ = 30.0
JOINT_HZ = 500.0
FT_HZ = 1000.0
GRIP_HZ = 50.0

SKILL_SEGMENTS = [
    {"name": "Approach", "start": 0.0, "end": 3.2, "confidence": 0.97},
    {"name": "Grasp", "start": 3.2, "end": 4.8, "confidence": 0.96},
    {"name": "Lift", "start": 4.8, "end": 6.3, "confidence": 0.94},
    {"name": "Pre-insert", "start": 6.3, "end": 8.1, "confidence": 0.91},
    {"name": "Contact", "start": 8.1, "end": 9.0, "confidence": 0.88},
    {"name": "Insert", "start": 9.0, "end": 11.4, "confidence": 0.91},
    {"name": "Release", "start": 11.4, "end": 11.7, "confidence": 0.93},
    {"name": "Retreat", "start": 11.7, "end": 12.0, "confidence": 0.95},
]

# Quality profiles: status matches data quality, all are fully playable
EPISODE_PROFILES: dict[str, dict[str, Any]] = {
    "EP_0038": {
        "status": "pass",
        "quality_score": 92,
        "sync_error_ms": 4.2,
        "dropped_frames_pct": 0.3,
        "label_confidence": 0.96,
        "success": True,
        "offsets_ms": {"rgb": 4.0, "depth": 5.0, "ft": -1.0},
        "inject_blur": False,
        "inject_drop": False,
        "inject_depth_missing": False,
        "inject_force_spike": False,
        "inject_tcp_jump": False,
        "issues": [],
    },
    "EP_0039": {
        "status": "review",
        "quality_score": 71,
        "sync_error_ms": 22.0,
        "dropped_frames_pct": 1.5,
        "label_confidence": 0.82,
        "success": True,
        "offsets_ms": {"rgb": 22.0, "depth": 28.0, "ft": -4.0},
        "inject_blur": True,
        "inject_drop": False,
        "inject_depth_missing": False,
        "inject_force_spike": True,
        "inject_tcp_jump": False,
        "issues": [
            {
                "t": 4.32,
                "type": "blur",
                "message": "RGB blur detected",
                "severity": "medium",
                "action": "Repair",
            },
            {
                "t": 8.74,
                "type": "force_spike",
                "message": "Force spike: 55 N",
                "severity": "medium",
                "action": "Needs Review",
            },
        ],
    },
    "EP_0040": {
        "status": "reject",
        "quality_score": 41,
        "sync_error_ms": 86.0,
        "dropped_frames_pct": 8.5,
        "label_confidence": 0.55,
        "success": False,
        "offsets_ms": {"rgb": 86.0, "depth": 92.0, "ft": -18.0},
        "inject_blur": True,
        "inject_drop": True,
        "inject_depth_missing": True,
        "inject_force_spike": True,
        "inject_tcp_jump": True,
        "issues": [
            {
                "t": 2.10,
                "type": "drop",
                "message": "RGB frame drop burst",
                "severity": "high",
                "action": "Reject",
            },
            {
                "t": 4.32,
                "type": "blur",
                "message": "RGB blur detected",
                "severity": "medium",
                "action": "Repair",
            },
            {
                "t": 5.95,
                "type": "drop",
                "message": "Sustained RGB drop",
                "severity": "high",
                "action": "Reject",
            },
            {
                "t": 6.18,
                "type": "depth_missing",
                "message": "Depth missing region > 35%",
                "severity": "high",
                "action": "Reject",
            },
            {
                "t": 8.74,
                "type": "force_spike",
                "message": "Force spike: 95 N",
                "severity": "high",
                "action": "Reject",
            },
            {
                "t": 9.50,
                "type": "force_spike",
                "message": "Repeated force saturation",
                "severity": "high",
                "action": "Reject",
            },
            {
                "t": 11.42,
                "type": "tcp_jump",
                "message": "TCP discontinuity",
                "severity": "high",
                "action": "Reject",
            },
        ],
    },
    "EP_0041": {
        "status": "pass",
        "quality_score": 88,
        "sync_error_ms": 6.1,
        "dropped_frames_pct": 0.6,
        "label_confidence": 0.93,
        "success": True,
        "offsets_ms": {"rgb": 6.0, "depth": 7.0, "ft": -2.0},
        "inject_blur": False,
        "inject_drop": False,
        "inject_depth_missing": False,
        "inject_force_spike": False,
        "inject_tcp_jump": False,
        "issues": [
            {
                "t": 6.18,
                "type": "depth_missing",
                "message": "Minor depth hole (~12%)",
                "severity": "low",
                "action": "Keep",
            }
        ],
    },
    "EP_0042": {
        "status": "review",
        "quality_score": 76,
        "sync_error_ms": 48.0,
        "dropped_frames_pct": 3.2,
        "label_confidence": 0.88,
        "success": True,
        "offsets_ms": {"rgb": 48.0, "depth": 55.0, "ft": -8.0},
        "inject_blur": True,
        "inject_drop": True,
        "inject_depth_missing": True,
        "inject_force_spike": True,
        "inject_tcp_jump": True,
        "issues": [
            {
                "t": 4.32,
                "type": "blur",
                "message": "RGB blur detected",
                "severity": "medium",
                "action": "Repair",
            },
            {
                "t": 6.18,
                "type": "depth_missing",
                "message": "Depth missing region > 35%",
                "severity": "medium",
                "action": "Interpolate",
            },
            {
                "t": 8.74,
                "type": "force_spike",
                "message": "Force spike: 72 N",
                "severity": "high",
                "action": "Needs Review",
            },
            {
                "t": 11.42,
                "type": "tcp_jump",
                "message": "TCP discontinuity",
                "severity": "high",
                "action": "Trim",
            },
        ],
    },
}


def skill_at(t: float) -> str:
    for seg in SKILL_SEGMENTS:
        if seg["start"] <= t < seg["end"] or (t == DURATION and t <= seg["end"]):
            return seg["name"]
    return SKILL_SEGMENTS[-1]["name"]


def tcp_trajectory(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 3.2:
            alpha = ti / 3.2
            x[i], y[i], z[i] = 0.35 - 0.10 * alpha, 0.05 * (1 - alpha), 0.25 - 0.08 * alpha
        elif ti < 4.8:
            x[i], y[i], z[i] = 0.25, 0.0, 0.17
        elif ti < 6.3:
            alpha = (ti - 4.8) / 1.5
            x[i], y[i], z[i] = 0.25, 0.0, 0.17 + 0.08 * alpha
        elif ti < 8.1:
            alpha = (ti - 6.3) / 1.8
            x[i] = 0.25 + 0.08 * alpha
            y[i] = 0.0
            z[i] = 0.25 - 0.02 * alpha
        elif ti < 9.0:
            alpha = (ti - 8.1) / 0.9
            x[i], y[i], z[i] = 0.33, 0.0, 0.23 - 0.01 * alpha
        elif ti < 11.4:
            alpha = (ti - 9.0) / 2.4
            x[i], y[i], z[i] = 0.33, 0.0, 0.22 - 0.06 * alpha
        else:
            alpha = (ti - 11.4) / 0.6
            x[i], y[i], z[i] = 0.33 - 0.04 * alpha, 0.0, 0.16 + 0.06 * alpha
    return x, y, z


def force_profile(t: np.ndarray, spike_amp: float = 72.0, extra_spike: bool = False) -> np.ndarray:
    fz = np.zeros_like(t)
    for i, ti in enumerate(t):
        if 3.4 <= ti < 4.6:
            fz[i] = 12 + 4 * np.sin((ti - 3.4) * 8)
        elif 8.1 <= ti < 9.0:
            fz[i] = 8 + 20 * ((ti - 8.1) / 0.9)
        elif 9.0 <= ti < 11.4:
            fz[i] = 28 + 6 * np.sin((ti - 9.0) * 3)
        else:
            fz[i] = 1.5 * np.random.randn()
    if spike_amp > 0:
        fz = fz + np.exp(-0.5 * ((t - 8.74) / 0.02) ** 2) * spike_amp
    if extra_spike:
        fz = fz + np.exp(-0.5 * ((t - 9.50) / 0.02) ** 2) * 90
    return fz


def make_rgb_frame(t: float, episode_id: str, blur: bool = False) -> np.ndarray:
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (28, 32, 38)
    cv2.rectangle(img, (0, 0), (w, 40), (18, 22, 28), -1)
    skill = skill_at(t)
    cv2.putText(
        img,
        f"{episode_id}  t={t:05.2f}s  {skill}",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
    )
    obj_x = int(80 + 120 * min(1.0, t / 8.1))
    obj_y = int(140 - 30 * np.sin(t * 0.7))
    cv2.circle(img, (obj_x, obj_y), 18, (70, 160, 220), -1)
    cv2.circle(img, (230, 150), 22, (90, 90, 90), 2)
    gw = 40 if t < 3.5 or t > 11.4 else 12
    cv2.rectangle(img, (obj_x - gw, obj_y - 28), (obj_x - gw + 8, obj_y + 28), (200, 180, 80), -1)
    cv2.rectangle(img, (obj_x + gw - 8, obj_y - 28), (obj_x + gw, obj_y + 28), (200, 180, 80), -1)
    if blur:
        img = cv2.GaussianBlur(img, (21, 21), 8)
    else:
        noise = np.random.randint(0, 12, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
    return img


def make_depth_frame(t: float, missing: bool = False) -> np.ndarray:
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (80 + 0.2 * xx + 0.15 * yy + 20 * np.sin(t)).astype(np.uint8)
    color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    cv2.putText(color, f"DEPTH t={t:05.2f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if missing:
        color[60:200, 80:240] = (20, 20, 20)
        cv2.putText(color, "MISSING", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return color


def generate(out_dir: Path, episode_id: str) -> None:
    if episode_id not in EPISODE_PROFILES:
        raise ValueError(f"Unknown episode profile: {episode_id}")
    profile = EPISODE_PROFILES[episode_id]
    offsets = profile["offsets_ms"]

    ep = out_dir / "episodes" / episode_id
    rgb_dir = ep / "rgb"
    depth_dir = ep / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    seed = int(episode_id.split("_")[-1])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    joint_t = np.arange(0, DURATION, 1.0 / JOINT_HZ)
    x, y, z = tcp_trajectory(joint_t)
    q = np.stack(
        [
            0.2 + 0.3 * x,
            -0.4 + 0.5 * y,
            0.8 - 0.6 * z,
            0.1 * np.sin(joint_t),
            0.2 * np.cos(joint_t * 0.5),
            0.05 * joint_t / DURATION,
        ],
        axis=1,
    )
    dq = np.gradient(q, joint_t, axis=0)
    pd.DataFrame(
        {
            "t": joint_t,
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
    ).to_parquet(ep / "joint_state.parquet", index=False)

    tcp_t = joint_t[::10]
    tx, ty, tz = tcp_trajectory(tcp_t)
    tx = tx.copy()
    if profile["inject_tcp_jump"]:
        jump_mask = (tcp_t >= 11.42) & (tcp_t < 11.45)
        tx[jump_mask] += 0.07
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(tcp_t), 1))
    pd.DataFrame(
        {
            "t": tcp_t,
            "x": tx,
            "y": ty,
            "z": tz,
            "qx": quat[:, 0],
            "qy": quat[:, 1],
            "qz": quat[:, 2],
            "qw": quat[:, 3],
        }
    ).to_parquet(ep / "tcp_pose.parquet", index=False)

    ft_t_true = np.arange(0, DURATION, 1.0 / FT_HZ)
    ft_t = ft_t_true + offsets["ft"] / 1000.0
    spike = 0.0
    if profile["inject_force_spike"]:
        spike = 95.0 if episode_id == "EP_0040" else (55.0 if episode_id == "EP_0039" else 72.0)
    fz = force_profile(
        ft_t_true,
        spike_amp=spike,
        extra_spike=(episode_id == "EP_0040"),
    )
    pd.DataFrame(
        {
            "t": ft_t,
            "fx": rng.normal(0, 0.5, size=len(ft_t)),
            "fy": rng.normal(0, 0.5, size=len(ft_t)),
            "fz": fz,
            "tx": rng.normal(0, 0.05, size=len(ft_t)),
            "ty": rng.normal(0, 0.05, size=len(ft_t)),
            "tz": rng.normal(0, 0.05, size=len(ft_t)),
        }
    ).to_parquet(ep / "force_torque.parquet", index=False)

    grip_t = np.arange(0, DURATION, 1.0 / GRIP_HZ)
    # Failed episode: gripper never firmly closes
    if not profile["success"]:
        width = np.where(grip_t < 3.5, 0.08, np.where(grip_t < 11.4, 0.045, 0.08))
    else:
        width = np.where(grip_t < 3.5, 0.08, np.where(grip_t < 11.4, 0.015, 0.08))
    closed = width < 0.03
    pd.DataFrame({"t": grip_t, "width": width, "closed": closed}).to_parquet(
        ep / "gripper_state.parquet", index=False
    )

    rgb_true = np.arange(0, DURATION, 1.0 / RGB_HZ)
    keep = np.ones(len(rgb_true), dtype=bool)
    if profile["inject_drop"]:
        keep[(rgb_true >= 5.95) & (rgb_true <= 6.15)] = False
        keep[int(2.1 * RGB_HZ)] = False
        keep[int(7.3 * RGB_HZ)] = False
        if episode_id == "EP_0040":
            keep[(rgb_true >= 3.0) & (rgb_true <= 3.35)] = False

    rgb_rows = []
    frame_idx = 0
    writer = cv2.VideoWriter(str(ep / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), RGB_HZ, (320, 240))
    for i, t_true in enumerate(rgb_true):
        if not keep[i]:
            continue
        t_stamp = t_true + offsets["rgb"] / 1000.0
        blur = bool(profile["inject_blur"] and abs(t_true - 4.32) < 0.08)
        img = make_rgb_frame(t_true, episode_id, blur=blur)
        cv2.imwrite(str(rgb_dir / f"{frame_idx:05d}.jpg"), img)
        writer.write(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_rows.append(
            {
                "t": t_stamp,
                "frame_idx": frame_idx,
                "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "brightness": float(gray.mean()),
                "motion_proxy": float(80 + 120 * min(1.0, t_true / 8.1)),
                "dropped": False,
            }
        )
        frame_idx += 1
    writer.release()
    pd.DataFrame(rgb_rows).to_parquet(ep / "camera_rgb.parquet", index=False)

    depth_true = np.arange(0, DURATION, 1.0 / DEPTH_HZ)
    depth_rows = []
    for d_idx, t_true in enumerate(depth_true):
        t_stamp = t_true + offsets["depth"] / 1000.0
        heavy_missing = bool(profile["inject_depth_missing"] and abs(t_true - 6.18) < 0.12)
        mild_missing = bool(episode_id == "EP_0041" and abs(t_true - 6.18) < 0.04)
        img = make_depth_frame(t_true, missing=heavy_missing or mild_missing)
        cv2.imwrite(str(depth_dir / f"{d_idx:05d}.jpg"), img)
        if heavy_missing:
            valid_ratio = 0.55
        elif mild_missing:
            valid_ratio = 0.88  # minor hole → low severity / Pass-compatible
        else:
            valid_ratio = 0.98
        depth_rows.append(
            {
                "t": t_stamp,
                "frame_idx": d_idx,
                "valid_ratio": valid_ratio,
                "dropped": False,
            }
        )
    pd.DataFrame(depth_rows).to_parquet(ep / "camera_depth.parquet", index=False)

    events = [
        {"t": 0.0, "event": "episode_start"},
        {"t": 3.2, "event": "grasp_start"},
        {"t": 4.5, "event": "grasp_success" if profile["success"] else "grasp_fail"},
        {"t": 8.1, "event": "contact"},
        {"t": 11.2, "event": "insert_success" if profile["success"] else "insert_fail"},
        {"t": 11.4, "event": "release"},
        {"t": 12.0, "event": "episode_end"},
    ]
    pd.DataFrame(events).to_parquet(ep / "task_event.parquet", index=False)

    labels = [
        {**seg, "confidence": max(0.5, seg["confidence"] - (0.0 if profile["success"] else 0.2))}
        for seg in SKILL_SEGMENTS
    ]

    metadata = {
        "id": episode_id,
        "task": "connector_insertion",
        "duration_s": DURATION,
        "success": profile["success"],
        "status": profile["status"],
        "quality_score": profile["quality_score"],
        "sync_error_ms": profile["sync_error_ms"],
        "dropped_frames_pct": profile["dropped_frames_pct"],
        "label_confidence": profile["label_confidence"],
        "offsets": {
            "reference_clock": "joint_state",
            "rgb_ms": offsets["rgb"],
            "depth_ms": offsets["depth"],
            "ft_ms": offsets["ft"],
            "average_sync_error_ms": profile["sync_error_ms"],
        },
        "injected": {"offsets_ms": offsets},
        "labels": labels,
        "issues": profile["issues"],
        "sensors": {
            "rgb_hz": RGB_HZ,
            "depth_hz": DEPTH_HZ,
            "joint_hz": JOINT_HZ,
            "ft_hz": FT_HZ,
        },
    }
    with (ep / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Generated {episode_id}: status={profile['status']} "
        f"sync={profile['sync_error_ms']}ms issues={len(profile['issues'])} frames={frame_idx}"
    )


def generate_all(out_dir: Path) -> None:
    for episode_id in EPISODE_PROFILES:
        generate(out_dir, episode_id)
    catalog = {
        "project": "Connector Insertion PoC",
        "dataset": "Trial_2026_08",
        "episodes": list(EPISODE_PROFILES.keys()),
        "primary_episode": "EP_0042",
    }
    with (out_dir / "catalog.json").open("w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog written: {out_dir / 'catalog.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_data",
    )
    parser.add_argument("--episode-id", default=None, help="Single episode, or omit for all")
    parser.add_argument("--all", action="store_true", help="Generate all demo episodes")
    args = parser.parse_args()
    if args.all or args.episode_id is None:
        generate_all(args.out)
    else:
        generate(args.out, args.episode_id)


if __name__ == "__main__":
    main()
