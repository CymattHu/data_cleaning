"use client";

import { useMemo } from "react";
import { api } from "@/lib/api";
import { useStudio } from "@/store/studio";

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex min-h-0 flex-col overflow-hidden rounded border border-line bg-bg">
      <div className="border-b border-line px-2 py-1 text-[11px] uppercase tracking-wide text-mute">
        {title}
      </div>
      <div className="relative flex-1">{children}</div>
    </div>
  );
}

function padRange(min: number, max: number, fallback = 0.1): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-fallback, fallback];
  if (max - min < 1e-6) {
    const c = (min + max) / 2;
    return [c - fallback, c + fallback];
  }
  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

/** Top-down-ish side view: X horizontal, Z vertical — real TCP trail. */
function TcpPathView({
  tcp,
  series,
  t,
}: {
  tcp: { x: number; y: number; z: number };
  series: { t: number; x: number; y: number; z: number }[];
  t: number;
}) {
  const w = 320;
  const h = 240;
  const padL = 42;
  const padR = 14;
  const padT = 28;
  const padB = 32;

  const visible = series.filter((p) => p.t <= t + 1e-6);
  const trail = visible.length ? visible : series.slice(0, 1);

  const xs = series.map((p) => p.x);
  const zs = series.map((p) => p.z);
  const [xMin, xMax] = padRange(
    Math.min(...xs, tcp.x),
    Math.max(...xs, tcp.x),
    0.05
  );
  const [zMin, zMax] = padRange(
    Math.min(...zs, tcp.z),
    Math.max(...zs, tcp.z),
    0.05
  );

  const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * (w - padL - padR);
  const sy = (z: number) => padT + (1 - (z - zMin) / (zMax - zMin)) * (h - padT - padB);

  const past = trail.map((p) => `${sx(p.x)},${sy(p.z)}`).join(" ");
  const future = series
    .filter((p) => p.t > t)
    .map((p) => `${sx(p.x)},${sy(p.z)}`)
    .join(" ");

  const cx = sx(tcp.x);
  const cy = sy(tcp.z);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full">
      <rect width={w} height={h} fill="#0f141a" />
      <text x="8" y="16" fill="#94a3b8" fontSize="11">
        TCP X–Z path · now ({tcp.x.toFixed(3)}, {tcp.z.toFixed(3)}) m
      </text>

      {/* Axes */}
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="#334155" />
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke="#334155" />
      <text x={w / 2} y={h - 8} fill="#64748b" fontSize="10" textAnchor="middle">
        X (m)
      </text>
      <text
        x="12"
        y={h / 2}
        fill="#64748b"
        fontSize="10"
        textAnchor="middle"
        transform={`rotate(-90 12 ${h / 2})`}
      >
        Z (m)
      </text>
      <text x={padL} y={h - padB + 12} fill="#475569" fontSize="9">
        {xMin.toFixed(2)}
      </text>
      <text x={w - padR} y={h - padB + 12} fill="#475569" fontSize="9" textAnchor="end">
        {xMax.toFixed(2)}
      </text>
      <text x={padL - 4} y={h - padB} fill="#475569" fontSize="9" textAnchor="end">
        {zMin.toFixed(2)}
      </text>
      <text x={padL - 4} y={padT + 8} fill="#475569" fontSize="9" textAnchor="end">
        {zMax.toFixed(2)}
      </text>

      {future && (
        <polyline points={future} fill="none" stroke="#334155" strokeWidth="1.5" strokeDasharray="3 3" />
      )}
      {past && (
        <polyline points={past} fill="none" stroke="#38bdf8" strokeWidth="2" />
      )}
      <circle cx={cx} cy={cy} r="5" fill="#fbbf24" stroke="#0f141a" strokeWidth="1" />
      <text x={cx + 8} y={cy - 6} fill="#fbbf24" fontSize="10">
        EE
      </text>
    </svg>
  );
}

/** Force Z vs time — signed force, cursor synced with playback. */
function ForceChart({
  fz,
  series,
  t,
  duration,
}: {
  fz: number;
  series: { t: number; fz: number }[];
  t: number;
  duration: number;
}) {
  const w = 320;
  const h = 240;
  const padL = 42;
  const padR = 14;
  const padT = 28;
  const padB = 32;

  const vals = series.map((p) => p.fz);
  const absMax = Math.max(20, ...vals.map((v) => Math.abs(v)), Math.abs(fz));
  const yMin = -absMax;
  const yMax = absMax;
  const dur = duration > 0 ? duration : series.at(-1)?.t || 12;

  const sx = (time: number) => padL + (time / dur) * (w - padL - padR);
  const sy = (v: number) =>
    padT + (1 - (v - yMin) / (yMax - yMin)) * (h - padT - padB);

  const points = series.map((p) => `${sx(p.t)},${sy(p.fz)}`).join(" ");
  const zeroY = sy(0);
  const cursorX = sx(Math.min(Math.max(t, 0), dur));
  const nowY = sy(fz);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full">
      <rect width={w} height={h} fill="#0f141a" />
      <text x="8" y="16" fill="#94a3b8" fontSize="11">
        Force Z · now {fz.toFixed(1)} N
      </text>

      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="#334155" />
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke="#334155" />
      <line
        x1={padL}
        y1={zeroY}
        x2={w - padR}
        y2={zeroY}
        stroke="#475569"
        strokeDasharray="4 3"
      />
      <text x={w / 2} y={h - 8} fill="#64748b" fontSize="10" textAnchor="middle">
        time (s)
      </text>
      <text
        x="12"
        y={h / 2}
        fill="#64748b"
        fontSize="10"
        textAnchor="middle"
        transform={`rotate(-90 12 ${h / 2})`}
      >
        Fz (N)
      </text>
      <text x={padL - 4} y={padT + 8} fill="#475569" fontSize="9" textAnchor="end">
        {yMax.toFixed(0)}
      </text>
      <text x={padL - 4} y={zeroY + 3} fill="#64748b" fontSize="9" textAnchor="end">
        0
      </text>
      <text x={padL - 4} y={h - padB} fill="#475569" fontSize="9" textAnchor="end">
        {yMin.toFixed(0)}
      </text>

      {points && (
        <polyline points={points} fill="none" stroke="#c084fc" strokeWidth="1.8" />
      )}
      <line x1={cursorX} y1={padT} x2={cursorX} y2={h - padB} stroke="#38bdf8" strokeWidth="1" />
      <circle cx={cursorX} cy={nowY} r="4" fill="#c084fc" stroke="#e2e8f0" strokeWidth="1" />
    </svg>
  );
}

export function PlaybackGrid() {
  const { selectedEpisodeId, timeline, currentTime, episode, targetRateHz } = useStudio();

  const sample = useMemo(() => {
    const pb = timeline?.playback || [];
    if (!pb.length) {
      return {
        rgb_frame: 0,
        depth_frame: 0,
        q: [0, 0.5, 0, 0, 0, 0],
        tcp: { x: 0.3, y: 0, z: 0.2 },
        fz: 0,
        gripper_width: 0.08,
        gripper_closed: false,
      };
    }
    const rate =
      timeline?.sync_settings?.target_rate_hz ||
      targetRateHz ||
      (pb.length > 1 ? 1 / Math.max(pb[1].t - pb[0].t, 1e-6) : 20);
    const idx = Math.min(pb.length - 1, Math.max(0, Math.round(currentTime * rate)));
    return pb[idx];
  }, [timeline, currentTime, targetRateHz]);

  const hasData = Boolean(episode?.metadata.has_data);
  const duration = timeline?.duration_s ?? 12;

  return (
    <div className="grid h-full grid-cols-2 grid-rows-2 gap-2">
      <Panel title="RGB Camera">
        {hasData ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={api.mediaUrl(selectedEpisodeId, "rgb", sample.rgb_frame)}
            alt="RGB"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-mute">
            No RGB stream for this episode
          </div>
        )}
      </Panel>
      <Panel title="Depth Camera">
        {hasData ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={api.mediaUrl(selectedEpisodeId, "depth", sample.depth_frame)}
            alt="Depth"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-mute">
            No depth stream
          </div>
        )}
      </Panel>
      <Panel title="TCP Path (X–Z)">
        <TcpPathView
          tcp={sample.tcp}
          series={timeline?.tcp_series || []}
          t={currentTime}
        />
      </Panel>
      <Panel title="Force Z (N)">
        <ForceChart
          fz={sample.fz}
          series={timeline?.force_series || []}
          t={currentTime}
          duration={duration}
        />
      </Panel>
    </div>
  );
}
