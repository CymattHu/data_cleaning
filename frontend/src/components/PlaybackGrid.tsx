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

function RobotView({
  tcp,
  q,
}: {
  tcp: { x: number; y: number; z: number };
  q: number[];
}) {
  const base = { x: 40, y: 160 };
  const j1 = { x: base.x + 50, y: base.y - 20 };
  const j2 = {
    x: j1.x + 70 * Math.cos(q[1] || 0),
    y: j1.y - 70 * Math.sin(q[1] || 0.5),
  };
  const ee = {
    x: 40 + tcp.x * 400,
    y: 180 - tcp.z * 400,
  };
  return (
    <svg viewBox="0 0 320 240" className="h-full w-full">
      <defs>
        <linearGradient id="floor" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#1a222c" />
          <stop offset="100%" stopColor="#121820" />
        </linearGradient>
      </defs>
      <rect width="320" height="240" fill="url(#floor)" />
      <line x1="20" y1="190" x2="300" y2="190" stroke="#334155" strokeWidth="1" />
      {/* Target hole */}
      <ellipse cx="230" cy="170" rx="16" ry="6" fill="#243041" stroke="#64748b" />
      {/* Arm links */}
      <line x1={base.x} y1={base.y} x2={j1.x} y2={j1.y} stroke="#38bdf8" strokeWidth="5" />
      <line x1={j1.x} y1={j1.y} x2={j2.x} y2={j2.y} stroke="#38bdf8" strokeWidth="4" />
      <line x1={j2.x} y1={j2.y} x2={ee.x} y2={ee.y} stroke="#7dd3fc" strokeWidth="3" />
      <circle cx={ee.x} cy={ee.y} r="5" fill="#fbbf24" />
      {/* TCP trail hint */}
      <polyline
        points={`${ee.x - 40},${ee.y + 10} ${ee.x - 20},${ee.y + 4} ${ee.x},${ee.y}`}
        fill="none"
        stroke="#fbbf2488"
        strokeWidth="1.5"
      />
      <text x="8" y="18" fill="#94a3b8" fontSize="11">
        TCP ({tcp.x.toFixed(2)}, {tcp.y.toFixed(2)}, {tcp.z.toFixed(2)})
      </text>
    </svg>
  );
}

function ForceTcpView({
  fz,
  series,
  t,
}: {
  fz: number;
  series: { t: number; fz: number }[];
  t: number;
}) {
  const w = 320;
  const h = 240;
  const maxF = 80;
  const duration = series.length ? series[series.length - 1].t : 12;
  const points = series
    .map((p) => {
      const x = (p.t / duration) * (w - 40) + 30;
      const y = h - 30 - (Math.min(maxF, Math.abs(p.fz)) / maxF) * (h - 60);
      return `${x},${y}`;
    })
    .join(" ");
  const cursorX = (t / duration) * (w - 40) + 30;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full">
      <rect width={w} height={h} fill="#0f141a" />
      <text x="8" y="18" fill="#94a3b8" fontSize="11">
        Force Z: {fz.toFixed(1)} N
      </text>
      <polyline points={points} fill="none" stroke="#c084fc" strokeWidth="1.5" />
      <line x1={cursorX} y1="30" x2={cursorX} y2={h - 20} stroke="#38bdf8" strokeWidth="1" />
      <line x1="30" y1={h - 30} x2={w - 10} y2={h - 30} stroke="#334155" />
    </svg>
  );
}

export function PlaybackGrid() {
  const { selectedEpisodeId, timeline, currentTime, episode } = useStudio();

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
    const idx = Math.min(pb.length - 1, Math.max(0, Math.round(currentTime * 20)));
    return pb[idx];
  }, [timeline, currentTime]);

  const hasData = Boolean(episode?.metadata.has_data);

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
      <Panel title="Robot 3D View">
        <RobotView tcp={sample.tcp} q={sample.q} />
      </Panel>
      <Panel title="Force / TCP View">
        <ForceTcpView fz={sample.fz} series={timeline?.force_series || []} t={currentTime} />
      </Panel>
    </div>
  );
}
