"use client";

import { useStudio } from "@/store/studio";

export function TopBar() {
  const {
    project,
    dataset,
    selectedEpisodeId,
    datasetVersion,
    status,
    runPipeline,
    exportDataset,
  } = useStudio();

  return (
    <header className="flex h-12 items-center justify-between border-b border-line bg-panel px-4">
      <div className="flex items-center gap-6">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-lg tracking-wide text-accent">SensorSync</span>
          <span className="text-sm text-mute">DataOps Studio</span>
        </div>
        <div className="hidden items-center gap-4 text-xs text-mute md:flex">
          <span>
            Project: <em className="not-italic text-ink">{project}</em>
          </span>
          <span>
            Dataset: <em className="not-italic text-ink">{dataset}</em>
          </span>
          <span>
            Episode: <em className="not-italic text-accent">{selectedEpisodeId}</em>
          </span>
          <span>
            Version: <em className="not-italic text-ink">{datasetVersion}</em>
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="max-w-[220px] truncate rounded border border-line bg-bg px-2 py-1 text-xs text-mute">
          {status}
        </span>
        <button
          type="button"
          onClick={() => void runPipeline()}
          className="rounded bg-accent px-3 py-1.5 text-xs font-semibold text-bg hover:bg-accent2"
        >
          Run Pipeline
        </button>
        <button
          type="button"
          onClick={() => void exportDataset()}
          className="rounded border border-line px-3 py-1.5 text-xs text-ink hover:border-accent"
        >
          Export Dataset
        </button>
      </div>
    </header>
  );
}
