"use client";

import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import { useStudio } from "@/store/studio";

const ROW_ORDER = ["task", "gripper", "force", "joint", "depth", "rgb"] as const;
const ROW_LABEL: Record<(typeof ROW_ORDER)[number], string> = {
  rgb: "RGB",
  depth: "Depth",
  joint: "Joint",
  force: "Force",
  gripper: "Gripper",
  task: "Task",
};

export function SensorTimeline() {
  const {
    timeline,
    timelineMode,
    currentTime,
    setCurrentTime,
    setTimelineMode,
    alignmentApplied,
  } = useStudio();

  const option = useMemo(() => {
    const duration = timeline?.duration_s ?? 12;
    const categories = ROW_ORDER.map((k) => ROW_LABEL[k]);

    const scatterData: [number, number][] = [];
    ROW_ORDER.forEach((key, row) => {
      if (key === "task" || key === "gripper") return;
      const points = timeline?.sensors?.[key === "rgb" ? "rgb" : key] || [];
      // Backend already adapts stride; only light-thin ultra-dense rows for chart perf
      const thin = points.length > 400 ? Math.ceil(points.length / 300) : 1;
      points.forEach((p, i) => {
        if (i % thin !== 0) return;
        if (p.present) scatterData.push([p.t, row]);
      });
    });

    const markAreas: {
      name: string;
      itemStyle: { color: string };
      data: [{ xAxis: number }, { xAxis: number }][];
    }[] = [];

    (timeline?.drop_regions || []).forEach((r) => {
      markAreas.push({
        name: "drop",
        itemStyle: { color: "rgba(239,68,68,0.22)" },
        data: [[{ xAxis: r.start }, { xAxis: r.end }]],
      });
    });
    (timeline?.anomaly_regions || []).forEach((r) => {
      markAreas.push({
        name: r.type,
        itemStyle: {
          color:
            r.type === "force_spike" || r.type === "tcp_jump"
              ? "rgba(168,85,247,0.25)"
              : "rgba(245,158,11,0.22)",
        },
        data: [[{ xAxis: r.start }, { xAxis: r.end }]],
      });
    });
    if (timelineMode === "raw") {
      (timeline?.offset_regions || []).forEach((r) => {
        markAreas.push({
          name: "offset",
          itemStyle: { color: "rgba(234,179,8,0.12)" },
          data: [[{ xAxis: r.start }, { xAxis: r.end }]],
        });
      });
    }

    const skillCustom: {
      name: string;
      value: [number, number, number, string];
    }[] = (timeline?.skill_segments || []).map((seg) => ({
      name: seg.name,
      value: [seg.start, 0, seg.end - seg.start, seg.name],
    }));

    const gripperSegs = (() => {
      const g = timeline?.gripper_series || [];
      if (!g.length) return [] as { start: number; end: number }[];
      const segs: { start: number; end: number }[] = [];
      let start: number | null = null;
      g.forEach((p, i) => {
        if (p.closed && start === null) start = p.t;
        if ((!p.closed || i === g.length - 1) && start !== null) {
          segs.push({ start, end: p.t });
          start = null;
        }
      });
      return segs;
    })();

    return {
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 70, right: 20, top: 28, bottom: 28 },
      tooltip: {
        trigger: "item",
        backgroundColor: "#1a222c",
        borderColor: "#334155",
        textStyle: { color: "#e2e8f0", fontSize: 11 },
      },
      xAxis: {
        type: "value",
        min: 0,
        max: duration,
        axisLabel: { color: "#94a3b8", fontSize: 10, formatter: (v: number) => `${v.toFixed(1)}s` },
        splitLine: { lineStyle: { color: "#1f2937" } },
        axisLine: { lineStyle: { color: "#334155" } },
      },
      yAxis: {
        type: "category",
        data: categories,
        axisLabel: { color: "#cbd5e1", fontSize: 11 },
        axisLine: { lineStyle: { color: "#334155" } },
        splitLine: { show: true, lineStyle: { color: "#1f2937" } },
      },
      series: [
        {
          type: "scatter",
          symbolSize: 5,
          itemStyle: { color: "#38bdf8" },
          data: scatterData,
          markLine: {
            symbol: "none",
            lineStyle: { color: "#fbbf24", width: 1.5 },
            data: [{ xAxis: currentTime }],
            label: { show: false },
          },
          markArea: {
            silent: true,
            data: markAreas.flatMap((m) =>
              m.data.map((pair) => [
                { xAxis: pair[0].xAxis, itemStyle: m.itemStyle },
                { xAxis: pair[1].xAxis },
              ])
            ),
          },
        },
        {
          type: "custom",
          name: "skills",
          renderItem: (params: any, api: any) => {
            const start = api.value(0);
            const row = api.value(1);
            const span = api.value(2);
            const label = api.value(3);
            const yIndex = ROW_ORDER.indexOf("task");
            const startCoord = api.coord([start, yIndex]);
            const endCoord = api.coord([start + span, yIndex]);
            const height = 14;
            return {
              type: "group",
              children: [
                {
                  type: "rect",
                  shape: {
                    x: startCoord[0],
                    y: startCoord[1] - height / 2,
                    width: Math.max(endCoord[0] - startCoord[0], 2),
                    height,
                  },
                  style: { fill: "rgba(34,197,94,0.45)", stroke: "#22c55e" },
                },
                {
                  type: "text",
                  style: {
                    x: startCoord[0] + 4,
                    y: startCoord[1],
                    text: label,
                    fill: "#dcfce7",
                    font: "10px sans-serif",
                    textVerticalAlign: "middle",
                  },
                },
              ],
            };
          },
          data: skillCustom,
          encode: { x: [0, 2], y: 1 },
        },
        {
          type: "custom",
          name: "gripper",
          renderItem: (_params: any, api: any) => {
            const start = api.value(0);
            const span = api.value(1);
            const yIndex = ROW_ORDER.indexOf("gripper");
            const startCoord = api.coord([start, yIndex]);
            const endCoord = api.coord([start + span, yIndex]);
            return {
              type: "rect",
              shape: {
                x: startCoord[0],
                y: startCoord[1] - 6,
                width: Math.max(endCoord[0] - startCoord[0], 2),
                height: 12,
              },
              style: { fill: "rgba(251,191,36,0.5)", stroke: "#fbbf24" },
            };
          },
          data: gripperSegs.map((s) => [s.start, s.end - s.start]),
        },
      ],
    };
  }, [timeline, timelineMode, currentTime]);

  return (
    <div className="mt-2 rounded border border-line bg-panel">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <div className="flex items-center gap-3 text-xs">
          <span className="text-mute">Multi-sensor Timeline</span>
          <span className="rounded bg-bg px-2 py-0.5 text-warn">
            Sync Error: {(timeline?.current_sync_error_ms ?? 0).toFixed(1)} ms
          </span>
          {alignmentApplied && (
            <span className="text-[11px] text-mute">
              {timelineMode === "aligned" ? (
                <span className="text-ok">
                  Aligned view · {(timeline?.after_sync_error_ms ?? timeline?.current_sync_error_ms ?? 5).toFixed(0)}{" "}
                  ms
                </span>
              ) : (
                <span className="text-warn">
                  Raw (pre-align) · {(timeline?.before_sync_error_ms ?? 48).toFixed(0)} ms
                </span>
              )}
            </span>
          )}
          {!alignmentApplied && timelineMode === "raw" && (
            <span className="text-warn">Unaligned</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {alignmentApplied && (
            <span className="hidden text-[10px] text-mute sm:inline">对比未对齐 ↔ 已对齐</span>
          )}
          <div className="flex rounded border border-line text-xs">
            <button
              type="button"
              onClick={() => void setTimelineMode("raw")}
              className={`px-2.5 py-1 ${timelineMode === "raw" ? "bg-accent text-bg" : "text-mute"}`}
            >
              Raw Timeline
            </button>
            <button
              type="button"
              onClick={() => void setTimelineMode("aligned")}
              className={`px-2.5 py-1 ${
                timelineMode === "aligned" ? "bg-accent text-bg" : "text-mute"
              }`}
            >
              Aligned Timeline
            </button>
          </div>
        </div>
      </div>
      <div className="legend flex flex-wrap gap-3 px-3 pt-2 text-[10px] text-mute">
        <span className="inline-flex items-center gap-1">
          <i className="inline-block h-2 w-3 rounded-sm bg-bad/70" /> Drop
        </span>
        <span className="inline-flex items-center gap-1">
          <i className="inline-block h-2 w-3 rounded-sm bg-warn/70" /> Offset / Blur
        </span>
        <span className="inline-flex items-center gap-1">
          <i className="inline-block h-2 w-3 rounded-sm bg-purple-500/70" /> Force / TCP
        </span>
        <span className="inline-flex items-center gap-1">
          <i className="inline-block h-2 w-3 rounded-sm bg-ok/70" /> Skill
        </span>
      </div>
      <ReactECharts
        option={option}
        style={{ height: 220, width: "100%" }}
        onEvents={{
          click: (params: any) => {
            if (typeof params?.value === "number") setCurrentTime(params.value);
            else if (Array.isArray(params?.value)) setCurrentTime(Number(params.value[0]));
          },
        }}
      />
    </div>
  );
}
