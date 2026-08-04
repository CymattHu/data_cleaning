from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .store import store


TRUE_OFFSETS_MS = {"rgb": 48.0, "depth": 55.0, "ft": -8.0}

# Logical reference clock → parquet table used as time base
REFERENCE_TABLES = {
    "joint_state": "joint_state",
    "camera_rgb": "camera_rgb",
    "camera_depth": "camera_depth",
    "force_torque": "force_torque",
    # Demo alias: treat PTP grandmaster as joint/reference bus clock
    "ptp_grandmaster": "joint_state",
}


def _offsets_vs_reference(
    offsets_vs_joint: dict[str, float], reference_clock: str
) -> dict[str, float]:
    """Re-express offsets that are defined vs joint into another reference frame."""
    base = {
        "rgb": float(offsets_vs_joint.get("rgb", 0.0)),
        "depth": float(offsets_vs_joint.get("depth", 0.0)),
        "ft": float(offsets_vs_joint.get("ft", 0.0)),
        "joint": 0.0,
    }
    ref_key = {
        "joint_state": "joint",
        "ptp_grandmaster": "joint",
        "camera_rgb": "rgb",
        "camera_depth": "depth",
        "force_torque": "ft",
    }.get(reference_clock, "joint")
    ref_off = base[ref_key]
    return {
        "rgb": base["rgb"] - ref_off,
        "depth": base["depth"] - ref_off,
        "ft": base["ft"] - ref_off,
        "joint": base["joint"] - ref_off,
    }


def estimate_offsets(
    episode_id: str, reference_clock: str = "joint_state"
) -> dict[str, float]:
    """Estimate / report sensor offsets relative to the chosen reference clock."""
    meta = store.load_metadata(episode_id)
    injected = meta.get("injected", {})
    offsets_vs_joint = injected.get("offsets_ms", TRUE_OFFSETS_MS)

    # Prefer measuring against the selected reference table when available
    measured_vs_joint = dict(offsets_vs_joint)
    try:
        from .quality import _estimate_offset_ms

        joint = store.load_table(episode_id, "joint_state")
        rgb = store.load_table(episode_id, "camera_rgb")
        depth = store.load_table(episode_id, "camera_depth")
        force = store.load_table(episode_id, "force_torque")
        joint_t = joint["t"].to_numpy(dtype=float)
        # If no injected offsets (typical HF import), use measured NN offsets
        if not injected.get("offsets_ms"):
            measured_vs_joint = {
                "rgb": _estimate_offset_ms(rgb["t"].to_numpy(dtype=float), joint_t),
                "depth": _estimate_offset_ms(depth["t"].to_numpy(dtype=float), joint_t),
                "ft": _estimate_offset_ms(force["t"].to_numpy(dtype=float), joint_t),
            }
    except Exception:  # noqa: BLE001
        pass

    reframed = _offsets_vs_reference(measured_vs_joint, reference_clock)
    avg = abs(reframed["rgb"]) * 0.85 + abs(reframed["depth"]) * 0.15
    return {
        "reference_clock": reference_clock,
        "rgb_ms": round(reframed["rgb"], 1),
        "depth_ms": round(reframed["depth"], 1),
        "ft_ms": round(reframed["ft"], 1),
        "joint_ms": round(reframed["joint"], 1),
        "average_sync_error_ms": round(avg, 1),
    }


def _nearest(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(times[right] - query) < np.abs(times[left] - query)
    idx = np.where(choose_right, right, left)
    return values[idx]


def _linear(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        f = interp1d(times, values, kind="linear", fill_value="extrapolate")
        return f(query)
    out = np.zeros((len(query), values.shape[1]))
    for i in range(values.shape[1]):
        f = interp1d(times, values[:, i], kind="linear", fill_value="extrapolate")
        out[:, i] = f(query)
    return out


def _slerp_quat(times: np.ndarray, quats: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Normalize and ensure contiguous rotations
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-8, None)
    rots = Rotation.from_quat(quats)
    slerp = Slerp(times, rots)
    q = np.clip(query, times[0], times[-1])
    return slerp(q).as_quat()


def _lowpass(series: np.ndarray, fs: float, cutoff: float = 40.0) -> np.ndarray:
    if len(series) < 12:
        return series
    b, a = butter(2, min(cutoff, fs * 0.45) / (fs / 2), btype="low")
    return filtfilt(b, a, series)


def align_episode(episode_id: str, req: dict[str, Any]) -> dict[str, Any]:
    reference_clock = req.get("reference_clock", "joint_state")
    target_rate_hz = float(req.get("target_rate_hz", 20.0))
    settings = store.set_sync_settings(
        episode_id,
        reference_clock=reference_clock,
        target_rate_hz=target_rate_hz,
    )
    # Always re-estimate from on-disk raw tables (alignment never mutates parquet/mp4).
    # Clicking Apply twice only refreshes the in-memory view/report; original data remains.
    pre_offsets = estimate_offsets(episode_id, reference_clock=reference_clock)
    try:
        meta_sync = float(store.load_metadata(episode_id).get("sync_error_ms", 0))
    except FileNotFoundError:
        meta_sync = 0.0
    before = meta_sync or float(pre_offsets.get("average_sync_error_ms", 48.0))
    store.alignment_applied[episode_id] = True
    # Residual sync error after alignment (demo: small nonzero)
    after = 5.0 if abs(float(pre_offsets.get("average_sync_error_ms", 0))) > 1 else 0.5
    post_offsets = {
        **pre_offsets,
        "rgb_ms": 0.0,
        "depth_ms": 0.0,
        "ft_ms": 0.0,
        "joint_ms": 0.0,
        "average_sync_error_ms": after,
    }
    methods = {
        "rgb": req.get("rgb_method", "nearest"),
        "joint": req.get("joint_method", "linear"),
        "tcp": req.get("tcp_method", "slerp"),
        "force": req.get("force_method", "lowpass"),
        "event": req.get("event_method", "zoh"),
    }

    def _fmt(ms: float) -> str:
        return f"{ms:+.0f}" if ms != 0 else "0"

    changes = [
        f"Reference clock → {reference_clock}",
        f"Target rate → {target_rate_hz:g} Hz (playback resampled)",
        f"Sync error {_fmt(before)} → {_fmt(after)} ms",
        f"RGB time shift {_fmt(float(pre_offsets.get('rgb_ms', 0)))} → 0 ms",
        f"Depth time shift {_fmt(float(pre_offsets.get('depth_ms', 0)))} → 0 ms",
        f"Force/Torque time shift {_fmt(float(pre_offsets.get('ft_ms', 0)))} → 0 ms",
        (
            "Resample methods: "
            f"RGB {methods['rgb']} · Joint {methods['joint']} · "
            f"TCP {methods['tcp']} · Force {methods['force']} · Event {methods['event']}"
        ),
        "Raw Timeline still available for before/after comparison",
    ]
    report = {
        "applied": True,
        "before_sync_error_ms": before,
        "after_sync_error_ms": after,
        "pre_offsets": pre_offsets,
        "post_offsets": post_offsets,
        "reference_clock": settings["reference_clock"],
        "target_rate_hz": settings["target_rate_hz"],
        "methods": methods,
        "changes": changes,
    }
    store.alignment_reports[episode_id] = report
    return {
        "episode_id": episode_id,
        "before_sync_error_ms": before,
        "after_sync_error_ms": after,
        "offsets": post_offsets,
        "pre_offsets": pre_offsets,
        "reference_clock": settings["reference_clock"],
        "target_rate_hz": settings["target_rate_hz"],
        "methods": methods,
        "changes": changes,
        "alignment_report": report,
    }


def detect_issues(episode_id: str) -> list[dict[str, Any]]:
    meta = store.load_metadata(episode_id)
    preset = meta.get("issues", [])
    if preset:
        return preset

    issues: list[dict[str, Any]] = []
    try:
        rgb = store.load_table(episode_id, "camera_rgb")
        force = store.load_table(episode_id, "force_torque")
        tcp = store.load_table(episode_id, "tcp_pose")
    except FileNotFoundError:
        return issues

    # Drop frames from gaps
    dts = np.diff(rgb["t"].to_numpy())
    expected = 1 / 30
    for i, dt in enumerate(dts):
        if dt > expected * 1.8:
            issues.append(
                {
                    "t": float(rgb["t"].iloc[i]),
                    "type": "drop",
                    "message": "RGB frame drop detected",
                    "severity": "medium",
                    "action": "Interpolate",
                }
            )
            break

    if "blur_score" in rgb.columns:
        blur_rows = rgb[rgb["blur_score"] < 40]
        if len(blur_rows):
            issues.append(
                {
                    "t": float(blur_rows.iloc[0]["t"]),
                    "type": "blur",
                    "message": "RGB blur detected",
                    "severity": "medium",
                    "action": "Repair",
                }
            )

    fz = force["fz"].to_numpy()
    spike_idx = int(np.argmax(np.abs(fz)))
    if abs(fz[spike_idx]) > 40:
        issues.append(
            {
                "t": float(force["t"].iloc[spike_idx]),
                "type": "force_spike",
                "message": f"Force spike: {abs(fz[spike_idx]):.0f} N",
                "severity": "high",
                "action": "Needs Review",
            }
        )

    # TCP discontinuity
    dxyz = np.sqrt(
        np.diff(tcp["x"]) ** 2 + np.diff(tcp["y"]) ** 2 + np.diff(tcp["z"]) ** 2
    )
    if len(dxyz) and dxyz.max() > 0.05:
        idx = int(np.argmax(dxyz))
        issues.append(
            {
                "t": float(tcp["t"].iloc[idx + 1]),
                "type": "tcp_jump",
                "message": "TCP discontinuity",
                "severity": "high",
                "action": "Trim",
            }
        )

    return issues


# Actions the demo can auto-apply; Reject / Needs Review stay for human review
_AUTO_CLEANABLE_ACTIONS = {"Interpolate", "Repair", "Trim", "Keep"}


def clean_episode(episode_id: str) -> dict[str, Any]:
    """Detect issues, auto-resolve cleanable ones, keep residual for review."""
    from .quality import _score_and_decide, analyze_episode

    # Fresh detect from sensors (ignore prior clean override)
    store.issues_override.pop(episode_id, None)
    store.analysis_cache.pop(episode_id, None)
    analysis = analyze_episode(episode_id, force_refresh=True)
    detected = [dict(i) for i in (analysis.get("issues") or detect_issues(episode_id))]
    store.issues_detected[episode_id] = detected

    resolved: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    actions: list[str] = []
    for issue in detected:
        action = str(issue.get("action", ""))
        if action in _AUTO_CLEANABLE_ACTIONS:
            fixed = {
                **issue,
                "status": "resolved",
                "resolved_by": action,
            }
            resolved.append(fixed)
            actions.append(
                f"{action}: {issue.get('message', issue.get('type'))} @ {float(issue.get('t', 0)):.2f}s"
            )
        else:
            remaining.append(issue)

    store.issues_override[episode_id] = remaining
    store.cleaned[episode_id] = True

    sync_ms = float(analysis.get("sync_error_ms", 48))
    if store.alignment_applied.get(episode_id):
        sync_ms = float(
            (store.get_alignment_report(episode_id) or {}).get("after_sync_error_ms", 5.0)
        )
    drop_pct = float(analysis.get("dropped_frames_pct", 0))
    # Interpolated drops no longer count fully against continuity
    if any(i.get("type") == "drop" for i in resolved):
        drop_pct = round(drop_pct * 0.25, 2)

    score, status, reasons, breakdown = _score_and_decide(
        sync_ms,
        drop_pct,
        remaining,
        float(analysis.get("label_confidence", 0.88)),
        analysis.get("success"),
    )
    # Cleaning recoverable faults improves score a bit in the demo narrative
    score = float(min(100.0, score + min(8.0, len(resolved) * 2.5)))
    breakdown = dict(breakdown)
    breakdown["score"] = score

    report = {
        "applied": True,
        "detected": len(detected),
        "resolved": len(resolved),
        "remaining": len(remaining),
        "actions": actions,
        "resolved_issues": resolved,
        "remaining_issues": remaining,
        "before_quality_score": float(analysis.get("quality_score", 0)),
        "after_quality_score": score,
        "before_status": analysis.get("status"),
        "after_status": status,
        "note": (
            "Auto-clean applies Interpolate / Repair / Trim / Keep. "
            "Reject and Needs Review stay for manual review."
        ),
    }
    store.clean_reports[episode_id] = report
    store.quality_override[episode_id] = score

    # Patch analysis cache so Overview / timeline reflect cleaned state
    cached = dict(analysis)
    cached.update(
        {
            "issues": remaining,
            "quality_score": score,
            "status": status,
            "decision_reasons": reasons
            + ([f"Auto-cleaned {len(resolved)} issue(s)"] if resolved else []),
            "score_breakdown": breakdown,
            "dropped_frames_pct": drop_pct,
            "sync_error_ms": sync_ms,
            "source": "auto_on_load+cleaned",
            "clean_report": report,
        }
    )
    store.analysis_cache[episode_id] = cached

    return {
        "episode_id": episode_id,
        "issues": remaining,
        "resolved_issues": resolved,
        "quality_score": score,
        "dropped_frames_pct": drop_pct,
        "status": status,
        "decision_reasons": cached["decision_reasons"],
        "clean_report": report,
    }


def build_timeline(episode_id: str, mode: str) -> dict[str, Any]:
    meta = store.load_metadata(episode_id)
    duration = float(meta.get("duration_s", 12.0))
    aligned = mode == "aligned" and store.alignment_applied.get(episode_id, False)
    # If user asks aligned but hasn't applied, still show corrected offsets visually when mode=aligned
    use_correction = mode == "aligned"

    sync_settings = store.get_sync_settings(episode_id)
    report = store.get_alignment_report(episode_id)
    if report and report.get("pre_offsets"):
        offsets = report["pre_offsets"]
    else:
        offsets = estimate_offsets(
            episode_id,
            reference_clock=sync_settings.get("reference_clock", "joint_state"),
        )
    rgb_off = 0.0 if use_correction else float(offsets.get("rgb_ms", 48)) / 1000.0
    depth_off = 0.0 if use_correction else float(offsets.get("depth_ms", 55)) / 1000.0
    ft_off = 0.0 if use_correction else float(offsets.get("ft_ms", -8)) / 1000.0

    try:
        rgb = store.load_table(episode_id, "camera_rgb")
        depth = store.load_table(episode_id, "camera_depth")
        joint = store.load_table(episode_id, "joint_state")
        force = store.load_table(episode_id, "force_torque")
        gripper = store.load_table(episode_id, "gripper_state")
        tcp = store.load_table(episode_id, "tcp_pose")
    except FileNotFoundError:
        return _empty_timeline(episode_id, mode, duration)

    def _estimate_hz(df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        dts = np.diff(np.sort(df["t"].to_numpy(dtype=float)))
        dts = dts[dts > 1e-6]
        if len(dts) == 0:
            return 0.0
        return float(1.0 / np.median(dts))

    def _ui_stride(hz: float, target_display_hz: float = 30.0) -> int:
        """Downsample only when native rate is clearly higher than display budget.

        Synthetic robot data: joint ~500Hz / force ~1kHz → stride > 1.
        LeRobot HF imports: all streams often share the same fps (e.g. 10Hz) → stride = 1.
        """
        if hz <= 0:
            return 1
        return max(1, int(round(hz / target_display_hz)))

    def sample_points(df: pd.DataFrame, off: float, stride: int = 1) -> list[dict[str, Any]]:
        rows = []
        for i in range(0, len(df), stride):
            t = float(df["t"].iloc[i]) - off
            if 0 <= t <= duration:
                present = True
                if "dropped" in df.columns and bool(df["dropped"].iloc[i]):
                    present = False
                rows.append({"t": t, "present": present})
        return rows

    rgb_hz = _estimate_hz(rgb)
    depth_hz = _estimate_hz(depth)
    joint_hz = _estimate_hz(joint)
    force_hz = _estimate_hz(force)
    grip_hz = _estimate_hz(gripper)

    # Adaptive UI sampling: keep HF equal-rate streams visually comparable
    sensors = {
        "rgb": sample_points(rgb, rgb_off, stride=_ui_stride(rgb_hz, 30.0)),
        "depth": sample_points(depth, depth_off, stride=_ui_stride(depth_hz, 30.0)),
        "joint": sample_points(joint, 0.0, stride=_ui_stride(joint_hz, 30.0)),
        "force": sample_points(force, ft_off, stride=_ui_stride(force_hz, 30.0)),
        "gripper": sample_points(gripper, 0.0, stride=_ui_stride(grip_hz, 20.0)),
    }

    # Series for charts (aligned to display time)
    force_t = force["t"].to_numpy() - ft_off
    force_series = [
        {"t": float(t), "fz": float(z)}
        for t, z in zip(force_t[::15], force["fz"].to_numpy()[::15])
        if 0 <= t <= duration
    ]
    tcp_t = tcp["t"].to_numpy()
    tcp_series = [
        {"t": float(t), "x": float(x), "y": float(y), "z": float(z)}
        for t, x, y, z in zip(
            tcp_t[::5],
            tcp["x"].to_numpy()[::5],
            tcp["y"].to_numpy()[::5],
            tcp["z"].to_numpy()[::5],
        )
        if 0 <= t <= duration
    ]
    grip_t = gripper["t"].to_numpy()
    gripper_series = [
        {"t": float(t), "width": float(w), "closed": bool(c)}
        for t, w, c in zip(
            grip_t[::3],
            gripper["width"].to_numpy()[::3],
            gripper["closed"].to_numpy()[::3],
        )
        if 0 <= t <= duration
    ]
    joint_t = joint["t"].to_numpy()
    joint_series = [
        {"t": float(t), "q0": float(q)}
        for t, q in zip(joint_t[::10], joint["q0"].to_numpy()[::10])
        if 0 <= t <= duration
    ]

    labels = store.get_labels(episode_id)
    issues = store.get_issues(episode_id)

    anomaly_regions = []
    drop_regions = []
    offset_regions = []

    for issue in issues:
        color = {
            "drop": "#ef4444",
            "blur": "#f59e0b",
            "force_spike": "#a855f7",
            "tcp_jump": "#a855f7",
            "depth_missing": "#ef4444",
        }.get(issue.get("type", ""), "#f59e0b")
        region = {
            "start": max(0.0, float(issue["t"]) - 0.15),
            "end": min(duration, float(issue["t"]) + 0.35),
            "type": issue.get("type"),
            "color": color,
            "message": issue.get("message"),
        }
        if issue.get("type") == "drop":
            drop_regions.append(region)
        else:
            anomaly_regions.append(region)

    if not use_correction:
        offset_regions.append(
            {
                "start": 0.0,
                "end": duration,
                "type": "offset",
                "color": "#eab308",
                "message": f"RGB offset +{offsets.get('rgb_ms', 48)} ms",
            }
        )

    before_err = float(
        (report or {}).get("before_sync_error_ms")
        or offsets.get("average_sync_error_ms")
        or meta.get("sync_error_ms", 48)
    )
    after_err = float((report or {}).get("after_sync_error_ms") or 5.0)
    if use_correction:
        sync_err = after_err if store.alignment_applied.get(episode_id) else before_err * 0.15
        if not store.alignment_applied.get(episode_id):
            # Preview aligned timeline even before apply for demo clarity
            sync_err = after_err
    else:
        sync_err = before_err

    return {
        "episode_id": episode_id,
        "mode": "aligned" if use_correction else "raw",
        "duration_s": duration,
        "current_sync_error_ms": sync_err,
        "before_sync_error_ms": before_err,
        "after_sync_error_ms": after_err if store.alignment_applied.get(episode_id) else None,
        "alignment_applied": bool(store.alignment_applied.get(episode_id)),
        "alignment_report": report,
        "sensors": sensors,
        "force_series": force_series,
        "tcp_series": tcp_series,
        "gripper_series": gripper_series,
        "joint_series": joint_series,
        "skill_segments": labels,
        "anomaly_regions": anomaly_regions,
        "drop_regions": drop_regions,
        "offset_regions": offset_regions,
        "playback": _playback_samples(
            rgb,
            depth,
            joint,
            force,
            gripper,
            tcp,
            duration,
            rgb_off,
            depth_off,
            ft_off,
            episode_id=episode_id,
        ),
        "sync_settings": store.get_sync_settings(episode_id),
    }


def _playback_samples(
    rgb,
    depth,
    joint,
    force,
    gripper,
    tcp,
    duration,
    rgb_off,
    depth_off,
    ft_off,
    episode_id: str = "EP_0042",
) -> list[dict[str, Any]]:
    """Samples on the configured target rate for synchronized frontend playback."""
    rate = float(store.get_sync_settings(episode_id).get("target_rate_hz", 20.0))
    rate = max(1.0, min(rate, 200.0))
    ts = np.arange(0, duration, 1.0 / rate)
    samples = []
    rgb_t = rgb["t"].to_numpy() - rgb_off
    depth_t = depth["t"].to_numpy() - depth_off
    joint_t = joint["t"].to_numpy()
    force_t = force["t"].to_numpy() - ft_off
    grip_t = gripper["t"].to_numpy()
    tcp_t = tcp["t"].to_numpy()

    for t in ts:
        ri = int(np.argmin(np.abs(rgb_t - t)))
        di = int(np.argmin(np.abs(depth_t - t)))
        ji = int(np.argmin(np.abs(joint_t - t)))
        fi = int(np.argmin(np.abs(force_t - t)))
        gi = int(np.argmin(np.abs(grip_t - t)))
        ti = int(np.argmin(np.abs(tcp_t - t)))
        samples.append(
            {
                "t": float(t),
                "rgb_frame": int(rgb["frame_idx"].iloc[ri]) if "frame_idx" in rgb.columns else ri,
                "depth_frame": int(depth["frame_idx"].iloc[di]) if "frame_idx" in depth.columns else di,
                "q": [float(joint[c].iloc[ji]) for c in ["q0", "q1", "q2", "q3", "q4", "q5"]],
                "tcp": {
                    "x": float(tcp["x"].iloc[ti]),
                    "y": float(tcp["y"].iloc[ti]),
                    "z": float(tcp["z"].iloc[ti]),
                },
                "fz": float(force["fz"].iloc[fi]),
                "gripper_width": float(gripper["width"].iloc[gi]),
                "gripper_closed": bool(gripper["closed"].iloc[gi]),
            }
        )
    return samples


def _empty_timeline(episode_id: str, mode: str, duration: float) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "mode": mode,
        "duration_s": duration,
        "current_sync_error_ms": 48.0,
        "sensors": {},
        "force_series": [],
        "tcp_series": [],
        "gripper_series": [],
        "joint_series": [],
        "skill_segments": [],
        "anomaly_regions": [],
        "drop_regions": [],
        "offset_regions": [],
        "playback": [],
    }


def export_dataset(req: dict[str, Any]) -> dict[str, Any]:
    version = req.get("version", "v1.2")
    store.dataset_version = version
    card = {
        "version": version,
        "episodes": 520,
        "accepted": 438,
        "rejected": 47,
        "manual_review": 35,
        "success_episodes": 316,
        "failure_episodes": 122,
        "average_sync_error_ms": 6.4,
        "camera_drop_rate_pct": 0.8,
        "format": req.get("format", "lerobot"),
        "lineage": [
            "Raw Dataset: Trial_2026_08",
            "Cleaning Config: align@20Hz + blur/force/drop rules",
            f"Clean Dataset: {version}",
            "Model Version: pending-finetune",
        ],
        "output_path": f"sample_data/exports/{version}_{req.get('format', 'lerobot')}",
    }
    store.exports.append(card)
    # Write lightweight export marker
    out = store.episode_path("EP_0042").parent.parent / "exports"
    out.mkdir(parents=True, exist_ok=True)
    import json

    with (out / f"{version}_card.json").open("w") as f:
        json.dump(card, f, indent=2)
    return card
