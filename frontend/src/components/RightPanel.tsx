"use client";

import { ReferenceClock, ScoreBreakdown } from "@/lib/api";
import { useStudio } from "@/store/studio";

const FALLBACK_DIMENSIONS: ScoreBreakdown["dimensions"] = [
  {
    id: "sync",
    label: "Sync Alignment",
    weight_pct: 35,
    max_penalty: 35,
    formula: "min(35, |sync_ms| × 0.35)",
    explain: "Cross-sensor clock offset vs reference. Larger lag/jitter burns score fastest.",
    penalty: 0,
    kept: 35,
    detail: "—",
  },
  {
    id: "drop",
    label: "Frame Continuity",
    weight_pct: 20,
    max_penalty: 20,
    formula: "min(20, drop% × 2.5)",
    explain: "RGB drop / missing-frame rate. Gaps break temporal training samples.",
    penalty: 0,
    kept: 20,
    detail: "—",
  },
  {
    id: "faults",
    label: "Sensor Faults",
    weight_pct: 37,
    max_penalty: 37,
    formula: "high×6 + medium×3 + (fail ? 15 : 0), capped at 37",
    explain: "Blur, depth holes, force spikes, TCP jumps, and failed episodes.",
    penalty: 0,
    kept: 37,
    detail: "—",
  },
  {
    id: "label",
    label: "Label Confidence",
    weight_pct: 8,
    max_penalty: 8,
    formula: "−8 if confidence < 0.7, else 0",
    explain: "Skill-segment auto-label trust. Low confidence needs manual review.",
    penalty: 0,
    kept: 8,
    detail: "—",
  },
];

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "sync", label: "Sync" },
  { id: "quality", label: "Quality" },
  { id: "labels", label: "Labels" },
  { id: "export", label: "Export" },
] as const;

const REFERENCE_CLOCKS: { id: ReferenceClock; label: string }[] = [
  { id: "joint_state", label: "Joint State" },
  { id: "camera_rgb", label: "Camera RGB" },
  { id: "camera_depth", label: "Camera Depth" },
  { id: "force_torque", label: "Force / Torque" },
  { id: "ptp_grandmaster", label: "PTP Grandmaster" },
];

const RATE_PRESETS = [10, 20, 30, 50] as const;

export function RightPanel() {
  const {
    activeTab,
    setActiveTab,
    episode,
    estimateOffset,
    applyAlignment,
    runClean,
    setCurrentTime,
    updateLabelBoundary,
    exportFormat,
    exportDataset,
    exportCard,
    alignmentApplied,
    referenceClock,
    targetRateHz,
    setReferenceClock,
    setTargetRateHz,
  } = useStudio();

  return (
    <aside className="flex w-[25%] min-w-[260px] flex-col border-l border-line bg-panel">
      <div className="flex border-b border-line">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-1 py-2 text-[11px] ${
              activeTab === tab.id
                ? "border-b-2 border-accent text-accent"
                : "text-mute hover:text-ink"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-auto p-3 text-sm">
        {activeTab === "overview" && episode && (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <Metric label="Data Quality" value={`${episode.quality_score.toFixed(0)} / 100`} />
              <Metric
                label="Average Sync Error"
                value={`${episode.sync_error_ms.toFixed(0)} ms`}
                warn={episode.sync_error_ms > 20}
              />
              <Metric label="Dropped Frames" value={`${episode.dropped_frames_pct.toFixed(1)}%`} />
              <Metric
                label="Label Confidence"
                value={`${(episode.label_confidence * 100).toFixed(0)}%`}
              />
            </div>

            <ScoreDimensionsCard breakdown={episode.quality_report?.score_breakdown} />

            <div
              className={`rounded border px-2 py-2 text-xs ${
                episode.status === "pass"
                  ? "border-ok/40 bg-ok/10 text-ok"
                  : episode.status === "reject"
                    ? "border-bad/40 bg-bad/10 text-bad"
                    : "border-warn/40 bg-warn/10 text-warn"
              }`}
            >
              <div className="mb-1 font-semibold uppercase tracking-wide">
                Auto Decision: {episode.status}
              </div>
              <div className="text-[10px] opacity-80">
                {episode.quality_report?.analyzed
                  ? `Analyzed on load (${episode.quality_report.source})`
                  : "Metadata fallback"}
              </div>
            </div>

            {episode.quality_report?.decision_reasons?.length ? (
              <div className="rounded border border-line bg-bg p-2">
                <div className="mb-1 text-[10px] uppercase tracking-wide text-mute">
                  Why this status
                </div>
                <ul className="space-y-1 text-[11px] text-ink">
                  {episode.quality_report.decision_reasons.map((r) => (
                    <li key={r}>• {r}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {episode.quality_report?.blur_stats?.available && (
              <div className="rounded border border-line bg-bg p-2">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wide text-mute">
                    RGB Blurriness (Laplacian)
                  </span>
                  <button
                    type="button"
                    className="text-[10px] text-accent hover:underline"
                    onClick={() => {
                      const t = episode.quality_report?.blur_stats?.worst_t;
                      if (t != null) setCurrentTime(t);
                    }}
                  >
                    Jump to worst
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px] text-ink">
                  <span className="text-mute">min / mean / max</span>
                  <span>
                    {episode.quality_report.blur_stats.min} / {episode.quality_report.blur_stats.mean}{" "}
                    / {episode.quality_report.blur_stats.max}
                  </span>
                  <span className="text-mute">p10</span>
                  <span>{episode.quality_report.blur_stats.p10}</span>
                  <span className="text-mute">threshold</span>
                  <span>&lt; {episode.quality_report.blur_stats.threshold} = blurry</span>
                  <span className="text-mute">blurry frames</span>
                  <span
                    className={
                      episode.quality_report.blur_stats.blurry_frames > 0
                        ? "text-warn"
                        : "text-ok"
                    }
                  >
                    {episode.quality_report.blur_stats.blurry_frames} /{" "}
                    {episode.quality_report.blur_stats.frame_count} (
                    {episode.quality_report.blur_stats.blurry_pct}%)
                  </span>
                </div>
                <div className="mt-1 text-[10px] text-mute">
                  Higher Laplacian = sharper. Values below threshold are flagged as blur issues.
                </div>
              </div>
            )}

            {episode.quality_report?.sensor_stats &&
              Object.keys(episode.quality_report.sensor_stats).length > 0 && (
                <div className="rounded border border-line bg-bg p-2">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-mute">
                    Sensor diagnostics
                  </div>
                  <div className="space-y-1 font-mono text-[10px] text-ink">
                    {Object.entries(episode.quality_report.sensor_stats).map(([name, s]) => (
                      <div key={name} className="flex justify-between gap-2">
                        <span className="uppercase text-mute">{name}</span>
                        <span>
                          {s.actual_hz}Hz · jitter {s.jitter_ms}ms · off {s.offset_ms >= 0 ? "+" : ""}
                          {s.offset_ms}ms
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            <div className="rounded border border-line bg-bg p-2 text-[10px] text-mute">
              Rules: Pass if sync≤10ms & drop≤1% & no blocking faults; Review if 10–50ms or
              repairable issues; Reject if sync&gt;50ms / drop&gt;5% / ≥3 high issues / failed+critical.
              {alignmentApplied ? " Alignment applied." : ""}
            </div>
          </div>
        )}

        {activeTab === "sync" && episode && (
          <div className="space-y-3">
            <label className="block space-y-1">
              <span className="text-[10px] uppercase tracking-wide text-mute">
                Reference Clock
              </span>
              <select
                value={referenceClock}
                onChange={(e) => setReferenceClock(e.target.value as ReferenceClock)}
                className="w-full rounded border border-line bg-bg px-2 py-1.5 text-xs text-ink"
              >
                {REFERENCE_CLOCKS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="space-y-1">
              <span className="text-[10px] uppercase tracking-wide text-mute">
                Target Rate (Hz)
              </span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  max={1000}
                  step={1}
                  value={targetRateHz}
                  onChange={(e) => setTargetRateHz(Number(e.target.value))}
                  className="w-full rounded border border-line bg-bg px-2 py-1.5 text-xs text-ink"
                />
                <span className="shrink-0 text-xs text-mute">Hz</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {RATE_PRESETS.map((hz) => (
                  <button
                    key={hz}
                    type="button"
                    onClick={() => setTargetRateHz(hz)}
                    className={`rounded border px-2 py-0.5 text-[10px] ${
                      targetRateHz === hz
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-line text-mute hover:border-accent"
                    }`}
                  >
                    {hz}
                  </button>
                ))}
              </div>
            </div>

            <Row
              label="RGB Offset"
              value={`${episode.offsets.rgb_ms >= 0 ? "+" : ""}${episode.offsets.rgb_ms.toFixed(0)} ms`}
            />
            <Row
              label="Depth Offset"
              value={`${episode.offsets.depth_ms >= 0 ? "+" : ""}${episode.offsets.depth_ms.toFixed(0)} ms`}
            />
            <Row label="FT Offset" value={`${episode.offsets.ft_ms.toFixed(0)} ms`} />
            <div className="rounded border border-line bg-bg p-2 text-[11px] text-mute">
              Methods: RGB Nearest · Joint Linear · TCP SLERP · Force Lowpass · Event ZOH
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => void estimateOffset()}
                className="flex-1 rounded border border-line px-2 py-1.5 text-xs hover:border-accent"
              >
                Estimate Offset
              </button>
              <button
                type="button"
                onClick={() => void applyAlignment()}
                className="flex-1 rounded bg-accent px-2 py-1.5 text-xs font-semibold text-bg"
              >
                Apply Alignment
              </button>
            </div>
          </div>
        )}

        {activeTab === "quality" && episode && (
          <div className="space-y-2">
            {episode.quality_report?.blur_stats?.available && (
              <div className="rounded border border-line bg-bg p-2 text-[11px] text-ink">
                <div className="mb-1 text-[10px] uppercase text-mute">Blurriness summary</div>
                <div>
                  mean {episode.quality_report.blur_stats.mean} · min{" "}
                  {episode.quality_report.blur_stats.min} · blurry{" "}
                  <span
                    className={
                      episode.quality_report.blur_stats.blurry_frames > 0
                        ? "text-warn"
                        : "text-ok"
                    }
                  >
                    {episode.quality_report.blur_stats.blurry_frames}
                  </span>{" "}
                  frames
                </div>
              </div>
            )}
            <button
              type="button"
              onClick={() => void runClean()}
              className="mb-2 w-full rounded bg-accent px-2 py-1.5 text-xs font-semibold text-bg"
            >
              Detect / Clean Issues
            </button>
            {episode.issues.length === 0 && (
              <p className="text-xs text-mute">No issues flagged. Blurriness stats are still shown above.</p>
            )}
            {episode.issues.map((issue) => (
              <button
                key={`${issue.t}-${issue.type}`}
                type="button"
                onClick={() => setCurrentTime(issue.t)}
                className="block w-full rounded border border-line bg-bg px-2 py-2 text-left hover:border-accent"
              >
                <div className="flex justify-between text-xs">
                  <span className="font-mono text-accent">
                    {formatTs(issue.t)}
                  </span>
                  <span className="text-mute">{issue.action}</span>
                </div>
                <div className="mt-1 text-xs text-ink">{issue.message}</div>
              </button>
            ))}
          </div>
        )}

        {activeTab === "labels" && episode && (
          <div className="space-y-2">
            {episode.labels.map((seg, idx) => (
              <div key={seg.name} className="rounded border border-line bg-bg p-2">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-xs font-medium text-ink">{seg.name}</span>
                  <span className="text-[10px] text-mute">
                    conf {(seg.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <label className="text-mute">
                    Start
                    <input
                      type="number"
                      step={0.1}
                      value={seg.start}
                      onChange={(e) =>
                        void updateLabelBoundary(idx, "start", Number(e.target.value))
                      }
                      className="mt-0.5 w-full rounded border border-line bg-panel px-1 py-0.5 text-ink"
                    />
                  </label>
                  <label className="text-mute">
                    End
                    <input
                      type="number"
                      step={0.1}
                      value={seg.end}
                      onChange={(e) =>
                        void updateLabelBoundary(idx, "end", Number(e.target.value))
                      }
                      className="mt-0.5 w-full rounded border border-line bg-panel px-1 py-0.5 text-ink"
                    />
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === "export" && (
          <div className="space-y-3">
            <div className="space-y-1 text-xs">
              {(["lerobot", "rlds", "hdf5", "parquet"] as const).map((fmt) => (
                <label key={fmt} className="flex items-center gap-2 text-ink">
                  <input
                    type="radio"
                    name="fmt"
                    checked={exportFormat === fmt}
                    onChange={() => useStudio.setState({ exportFormat: fmt })}
                  />
                  {fmt === "lerobot" ? "LeRobot" : fmt.toUpperCase()}
                </label>
              ))}
            </div>
            <Row label="Target Rate" value="20 Hz" />
            <Row label="Include Raw Data" value="Yes" />
            <Row label="Include Failed Episodes" value="Yes" />
            <button
              type="button"
              onClick={() => void exportDataset()}
              className="w-full rounded bg-accent px-2 py-2 text-xs font-semibold text-bg"
            >
              Create Dataset v1.2
            </button>
            {exportCard && (
              <div className="rounded border border-ok/40 bg-bg p-2 text-[11px] text-ink">
                <div className="mb-1 font-semibold text-ok">Dataset Card</div>
                <div>Version: {exportCard.version}</div>
                <div>
                  Episodes: {exportCard.episodes} · Accepted: {exportCard.accepted} · Rejected:{" "}
                  {exportCard.rejected}
                </div>
                <div>Manual Review: {exportCard.manual_review}</div>
                <div>
                  Success / Failure: {exportCard.success_episodes} / {exportCard.failure_episodes}
                </div>
                <div>Avg Sync Error: {exportCard.average_sync_error_ms} ms</div>
                <div>Camera Drop Rate: {exportCard.camera_drop_rate_pct}%</div>
                <div className="mt-2 text-mute">
                  {exportCard.lineage.map((l) => (
                    <div key={l}>→ {l}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

function Metric({
  label,
  value,
  warn,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div className="rounded border border-line bg-bg p-2">
      <div className="text-[10px] uppercase tracking-wide text-mute">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${warn ? "text-warn" : "text-ink"}`}>
        {value}
      </div>
    </div>
  );
}

function ScoreDimensionsCard({ breakdown }: { breakdown?: ScoreBreakdown }) {
  const dims = breakdown?.dimensions?.length ? breakdown.dimensions : FALLBACK_DIMENSIONS;
  const note =
    breakdown?.weights_note ||
    "Weights are max penalty shares of the 100-point score (35+20+37+8).";

  return (
    <div className="rounded border border-line bg-bg p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-mute">
        Quality score · 4 dimensions & weights
      </div>
      <p className="mb-2 text-[10px] text-mute">
        Score starts at 100, then subtracts penalties. {note}
      </p>
      <div className="space-y-2">
        {dims.map((d) => {
          const keptPct = d.max_penalty > 0 ? (d.kept / d.max_penalty) * 100 : 100;
          return (
            <div key={d.id} className="rounded border border-line/70 px-2 py-1.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[11px] font-medium text-ink">{d.label}</span>
                <span className="font-mono text-[10px] text-accent">{d.weight_pct}% weight</span>
              </div>
              <div className="mt-0.5 text-[10px] text-mute">{d.explain}</div>
              <div className="mt-1 h-1.5 overflow-hidden rounded bg-line/40">
                <div
                  className={`h-full ${d.penalty > 0 ? "bg-warn" : "bg-ok"}`}
                  style={{ width: `${Math.max(4, keptPct)}%` }}
                />
              </div>
              <div className="mt-1 flex flex-wrap justify-between gap-x-2 font-mono text-[10px] text-ink">
                <span>
                  −{d.penalty.toFixed(1)} / max −{d.max_penalty}
                </span>
                <span className="text-mute">{d.detail}</span>
              </div>
              <div className="mt-0.5 font-mono text-[9px] text-mute">{d.formula}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-line/60 py-1.5 text-xs">
      <span className="text-mute">{label}</span>
      <span className="font-mono text-ink">{value}</span>
    </div>
  );
}

function formatTs(t: number) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}
