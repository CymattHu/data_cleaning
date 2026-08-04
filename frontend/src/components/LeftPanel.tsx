"use client";

import { useStudio } from "@/store/studio";

const statusIcon = {
  pass: "✓",
  review: "⚠",
  reject: "✕",
} as const;

const statusColor = {
  pass: "text-ok",
  review: "text-warn",
  reject: "text-bad",
} as const;

function shortId(id: string) {
  if (!id.startsWith("HF_")) return id;
  // HF_lerobot_svla_so100_stacking_0000 -> stacking_0000
  const parts = id.replace(/^HF_/, "").split("_");
  if (parts.length <= 3) return id.replace(/^HF_/, "");
  return parts.slice(-2).join("_");
}

export function LeftPanel() {
  const {
    datasetTree,
    episodes,
    selectedEpisodeId,
    selectEpisode,
    hfRepoId,
    setHfRepoId,
    hfMaxEpisodes,
    setHfMaxEpisodes,
    importLeRobot,
    importing,
    refreshDatasets,
  } = useStudio();

  const hfCount = episodes.filter((e) => e.id.startsWith("HF_")).length;

  return (
    <aside className="flex w-[20%] min-w-[240px] flex-col border-r border-line bg-panel">
      <section className="border-b border-line p-3">
        <h2 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-mute">
          Import LeRobot (HF)
        </h2>
        <input
          value={hfRepoId}
          onChange={(e) => setHfRepoId(e.target.value)}
          placeholder="lerobot/pusht"
          className="mb-2 w-full rounded border border-line bg-bg px-2 py-1.5 text-xs text-ink"
        />
        <div className="mb-2 flex items-center gap-2">
          <input
            type="number"
            min={1}
            max={20}
            value={hfMaxEpisodes}
            onChange={(e) => setHfMaxEpisodes(Number(e.target.value) || 1)}
            className="w-16 rounded border border-line bg-bg px-2 py-1.5 text-xs text-ink"
            title="Max episodes"
          />
          <button
            type="button"
            disabled={importing || !hfRepoId.trim()}
            onClick={() => void importLeRobot(hfRepoId.trim(), hfMaxEpisodes)}
            className="flex-1 rounded bg-accent px-2 py-1.5 text-xs font-semibold text-bg disabled:opacity-50"
          >
            {importing ? "Importing…" : "Import"}
          </button>
        </div>
        <p className="text-[10px] text-mute">
          Tip: after import, HF episodes appear at the top of the list.
        </p>
      </section>

      <section className="border-b border-line p-3">
        <h2 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-mute">
          Datasets
        </h2>
        <ul className="space-y-1 text-sm">
          {datasetTree.map((d) => (
            <li
              key={d.id}
              className="flex items-center justify-between rounded px-2 py-1.5 text-ink hover:bg-bg"
            >
              <span>{d.name}</span>
              <span className="text-xs text-mute">{d.count}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="flex min-h-0 flex-1 flex-col p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-mute">
            Episodes ({episodes.length})
            {hfCount > 0 ? ` · HF ${hfCount}` : ""}
          </h2>
          <button
            type="button"
            onClick={() => void refreshDatasets()}
            className="rounded border border-line px-2 py-0.5 text-[10px] text-mute hover:border-accent hover:text-accent"
            title="Reload episode list from disk"
          >
            Refresh
          </button>
        </div>
        <ul className="min-h-0 flex-1 space-y-2 overflow-auto pr-1">
          {episodes.map((ep) => {
            const active = ep.id === selectedEpisodeId;
            const isHf = ep.id.startsWith("HF_");
            return (
              <li key={ep.id}>
                <button
                  type="button"
                  onClick={() => void selectEpisode(ep.id)}
                  className={`w-full rounded border px-2.5 py-2 text-left transition ${
                    active
                      ? "border-accent bg-bg"
                      : "border-line bg-transparent hover:border-mute"
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between gap-1">
                    <span className="truncate text-sm font-medium text-ink" title={ep.id}>
                      {isHf ? `HF/${shortId(ep.id)}` : ep.id}
                    </span>
                    <span className={`shrink-0 text-sm ${statusColor[ep.status]}`}>
                      {statusIcon[ep.status]}{" "}
                      {ep.status === "pass"
                        ? "Pass"
                        : ep.status === "review"
                          ? "Review"
                          : "Reject"}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 text-[10px] text-mute">
                    <span>Q: {Number(ep.quality_score).toFixed(0)}</span>
                    <span>Sync: {Number(ep.sync_error_ms).toFixed(0)} ms</span>
                    <span>Issues: {ep.issue_count}</span>
                  </div>
                  {isHf && (
                    <div className="mt-1 truncate text-[10px] text-accent" title={ep.id}>
                      {ep.id}
                    </div>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </section>
    </aside>
  );
}
