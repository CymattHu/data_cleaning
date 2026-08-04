"use client";

import { create } from "zustand";
import {
  api,
  DatasetCard,
  EpisodeDetail,
  EpisodeSummary,
  ReferenceClock,
  SkillSegment,
  TimelineResponse,
} from "@/lib/api";

type TabId = "overview" | "sync" | "quality" | "labels" | "export";

type StudioState = {
  project: string;
  dataset: string;
  datasetVersion: string;
  episodes: EpisodeSummary[];
  datasetTree: { id: string; name: string; count: number }[];
  selectedEpisodeId: string;
  episode: EpisodeDetail | null;
  timeline: TimelineResponse | null;
  timelineMode: "raw" | "aligned";
  currentTime: number;
  playing: boolean;
  playbackRate: number;
  activeTab: TabId;
  status: string;
  alignmentApplied: boolean;
  cleaned: boolean;
  exportCard: DatasetCard | null;
  exportFormat: "lerobot" | "rlds" | "hdf5" | "parquet";
  loading: boolean;
  error: string | null;
  referenceClock: ReferenceClock;
  targetRateHz: number;
  setReferenceClock: (clock: ReferenceClock) => void;
  setTargetRateHz: (hz: number) => void;
  init: () => Promise<void>;
  selectEpisode: (id: string) => Promise<void>;
  setTimelineMode: (mode: "raw" | "aligned") => Promise<void>;
  setCurrentTime: (t: number) => void;
  setPlaying: (v: boolean) => void;
  setPlaybackRate: (r: number) => void;
  setActiveTab: (tab: TabId) => void;
  estimateOffset: () => Promise<void>;
  applyAlignment: () => Promise<void>;
  runClean: () => Promise<void>;
  runPipeline: () => Promise<void>;
  updateLabelBoundary: (index: number, field: "start" | "end", value: number) => Promise<void>;
  exportDataset: () => Promise<void>;
  refreshEpisode: () => Promise<void>;
  refreshDatasets: () => Promise<void>;
  importLeRobot: (repoId: string, maxEpisodes: number) => Promise<string | null>;
  hfRepoId: string;
  setHfRepoId: (v: string) => void;
  hfMaxEpisodes: number;
  setHfMaxEpisodes: (n: number) => void;
  importing: boolean;
};

function syncFromEpisode(episode: EpisodeDetail): {
  referenceClock: ReferenceClock;
  targetRateHz: number;
} {
  const clock = (episode.sync_settings?.reference_clock ||
    episode.offsets?.reference_clock ||
    "joint_state") as ReferenceClock;
  const hz = Number(episode.sync_settings?.target_rate_hz ?? 20);
  return {
    referenceClock: clock,
    targetRateHz: Number.isFinite(hz) ? hz : 20,
  };
}

export const useStudio = create<StudioState>((set, get) => ({
  project: "Connector Insertion PoC",
  dataset: "Trial_2026_08",
  datasetVersion: "v1.1",
  episodes: [],
  datasetTree: [],
  selectedEpisodeId: "EP_0042",
  episode: null,
  timeline: null,
  timelineMode: "raw",
  currentTime: 0,
  playing: false,
  playbackRate: 1,
  activeTab: "overview",
  status: "Idle",
  alignmentApplied: false,
  cleaned: false,
  exportCard: null,
  exportFormat: "lerobot",
  loading: false,
  error: null,
  hfRepoId: "lerobot/pusht",
  hfMaxEpisodes: 2,
  importing: false,
  referenceClock: "joint_state",
  targetRateHz: 20,

  setReferenceClock: (clock) => set({ referenceClock: clock }),
  setTargetRateHz: (hz) => {
    const n = Number(hz);
    if (!Number.isFinite(n)) return;
    set({ targetRateHz: Math.max(1, Math.min(1000, n)) });
  },

  init: async () => {
    set({ loading: true, error: null, status: "Loading datasets…" });
    try {
      await get().refreshDatasets();
      const preferred =
        get().episodes.find((e) => e.id === "EP_0042")?.id || get().episodes[0]?.id;
      if (preferred) await get().selectEpisode(preferred);
      set({ status: "Ready", loading: false });
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : "Failed to load",
        loading: false,
        status: "Error",
      });
    }
  },

  refreshDatasets: async () => {
    const data = await api.getDatasets();
    set({
      project: data.project,
      dataset: data.dataset,
      datasetVersion: data.dataset_version,
      episodes: data.episodes,
      datasetTree: data.trees[0]?.children || [],
    });
  },

  setHfRepoId: (v) => set({ hfRepoId: v }),
  setHfMaxEpisodes: (n) => set({ hfMaxEpisodes: n }),

  importLeRobot: async (repoId, maxEpisodes) => {
    set({ importing: true, status: `Importing ${repoId}…`, error: null });
    try {
      const preview = await api.previewLeRobot({ repo_id: repoId });
      set({
        status: `Found ${preview.total_episodes} eps @ ${preview.fps || "?"}Hz, importing ${maxEpisodes}…`,
      });
      const result = await api.importLeRobot({
        repo_id: repoId,
        max_episodes: maxEpisodes,
      });
      await get().refreshDatasets();
      const first = result.episodes[0]?.episode_id || null;
      if (first) await get().selectEpisode(first);
      set({
        importing: false,
        status: `Imported ${result.imported_count} episode(s) from ${repoId}`,
        activeTab: "overview",
      });
      return first;
    } catch (e) {
      // Import may have written files even if the HTTP call timed out — refresh list
      try {
        await get().refreshDatasets();
      } catch {
        /* ignore */
      }
      const hf = get().episodes.filter((ep) => ep.id.startsWith("HF_"));
      set({
        importing: false,
        error: e instanceof Error ? e.message : "Import failed",
        status: hf.length
          ? `Import request ended, but ${hf.length} HF episode(s) found locally — refreshed list`
          : "Import failed",
      });
      if (hf[0]) {
        await get().selectEpisode(hf[0].id);
        return hf[0].id;
      }
      return null;
    }
  },

  selectEpisode: async (id: string) => {
    set({
      selectedEpisodeId: id,
      loading: true,
      status: `Loading ${id}…`,
      currentTime: 0,
      playing: false,
    });
    try {
      const [episode, timeline] = await Promise.all([
        api.getEpisode(id),
        api.getTimeline(id, get().timelineMode),
      ]);
      set({
        episode,
        timeline,
        ...syncFromEpisode(episode),
        alignmentApplied: Boolean(episode.metadata.alignment_applied),
        cleaned: Boolean(episode.metadata.cleaned),
        loading: false,
        status: `${id} loaded`,
      });
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : "Failed to load episode",
        loading: false,
        status: "Error",
      });
    }
  },

  setTimelineMode: async (mode) => {
    set({ timelineMode: mode, status: `Timeline: ${mode}` });
    const id = get().selectedEpisodeId;
    try {
      const timeline = await api.getTimeline(id, mode);
      set({ timeline });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "Timeline failed" });
    }
  },

  setCurrentTime: (t) => set({ currentTime: Math.max(0, t) }),
  setPlaying: (v) => set({ playing: v }),
  setPlaybackRate: (r) => set({ playbackRate: r }),
  setActiveTab: (tab) => set({ activeTab: tab }),

  refreshEpisode: async () => {
    const id = get().selectedEpisodeId;
    const [episode, timeline] = await Promise.all([
      api.getEpisode(id),
      api.getTimeline(id, get().timelineMode),
    ]);
    set({
      episode,
      timeline,
      ...syncFromEpisode(episode),
      alignmentApplied: Boolean(episode.metadata.alignment_applied),
      cleaned: Boolean(episode.metadata.cleaned),
    });
  },

  estimateOffset: async () => {
    set({ status: "Estimating offsets…" });
    const { selectedEpisodeId, referenceClock } = get();
    const offsets = await api.estimateOffset(selectedEpisodeId, {
      reference_clock: referenceClock,
    });
    const episode = get().episode;
    if (episode) {
      set({
        episode: {
          ...episode,
          offsets,
          sync_settings: {
            reference_clock: referenceClock,
            target_rate_hz: get().targetRateHz,
          },
        },
        status: `RGB offset ${offsets.rgb_ms >= 0 ? "+" : ""}${offsets.rgb_ms.toFixed(0)} ms`,
        activeTab: "sync",
      });
    }
  },

  applyAlignment: async () => {
    set({ status: "Applying alignment…", activeTab: "sync" });
    const { selectedEpisodeId, referenceClock, targetRateHz } = get();
    const result = await api.align(selectedEpisodeId, {
      reference_clock: referenceClock,
      target_rate_hz: targetRateHz,
    });
    set({ alignmentApplied: true, timelineMode: "aligned" });
    await get().refreshEpisode();
    // Prefer Aligned view after apply; Raw Timeline remains for before/after compare
    await get().setTimelineMode("aligned");
    set({
      status: `Aligned @ ${result.target_rate_hz} Hz: ${result.before_sync_error_ms.toFixed(0)} → ${result.after_sync_error_ms.toFixed(0)} ms · toggle Raw to compare`,
      activeTab: "sync",
    });
  },

  runClean: async () => {
    set({ status: "Detecting & cleaning issues…", activeTab: "quality" });
    const result = await api.clean(get().selectedEpisodeId);
    set({ cleaned: true });
    await get().refreshEpisode();
    const report = result.clean_report;
    set({
      status: report
        ? `Cleaned ${report.resolved}/${report.detected} · score ${report.before_quality_score.toFixed(0)} → ${report.after_quality_score.toFixed(0)} · ${report.remaining} left`
        : "Quality clean complete",
      activeTab: "quality",
    });
  },

  runPipeline: async () => {
    set({ status: "Running pipeline…" });
    await get().applyAlignment();
    await get().runClean();
    set({ status: "Pipeline complete", activeTab: "labels" });
  },

  updateLabelBoundary: async (index, field, value) => {
    const episode = get().episode;
    if (!episode) return;
    const labels: SkillSegment[] = episode.labels.map((l, i) =>
      i === index ? { ...l, [field]: value } : l
    );
    // Keep monotonic-ish boundaries
    if (field === "start" && index > 0) {
      labels[index - 1] = { ...labels[index - 1], end: value };
    }
    if (field === "end" && index < labels.length - 1) {
      labels[index + 1] = { ...labels[index + 1], start: value };
    }
    const res = await api.updateLabels(get().selectedEpisodeId, labels);
    set({
      episode: { ...episode, labels: res.labels },
      timeline: get().timeline
        ? { ...get().timeline!, skill_segments: res.labels }
        : null,
      status: "Labels updated",
    });
  },

  exportDataset: async () => {
    set({ status: "Exporting dataset…", activeTab: "export" });
    const card = await api.exportDataset({
      format: get().exportFormat,
      target_rate_hz: 20,
      include_raw: true,
      include_failed: true,
      version: "v1.2",
    });
    set({
      exportCard: card,
      datasetVersion: card.version,
      status: `Exported ${card.version}`,
    });
  },
}));
