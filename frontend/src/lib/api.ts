const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs, ...rest } = init || {};
  const controller = new AbortController();
  const timer =
    timeoutMs && timeoutMs > 0
      ? window.setTimeout(() => controller.abort(), timeoutMs)
      : null;
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...rest,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(rest.headers || {}),
      },
      cache: "no-store",
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Request failed: ${res.status}`);
    }
    return res.json() as Promise<T>;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs}ms: ${path}`);
    }
    throw err;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export type EpisodeSummary = {
  id: string;
  status: "pass" | "review" | "reject";
  quality_score: number;
  sync_error_ms: number;
  issue_count: number;
  success?: boolean;
  has_data?: boolean;
};

export type SkillSegment = {
  name: string;
  start: number;
  end: number;
  confidence: number;
};

export type Issue = {
  t: number;
  type: string;
  message: string;
  severity: string;
  action: string;
};

export type BlurStats = {
  available: boolean;
  threshold: number;
  frame_count: number;
  blurry_frames: number;
  blurry_pct: number;
  min: number | null;
  mean: number | null;
  max: number | null;
  p10: number | null;
  worst_t: number | null;
};

export type ScoreDimension = {
  id: string;
  label: string;
  weight_pct: number;
  max_penalty: number;
  formula: string;
  explain: string;
  penalty: number;
  kept: number;
  detail: string;
};

export type ScoreBreakdown = {
  score: number;
  base: number;
  dimensions: ScoreDimension[];
  weights_note: string;
  high_issues: number;
  medium_issues: number;
};

export type QualityReport = {
  analyzed: boolean;
  source: string;
  decision_reasons: string[];
  score_breakdown?: ScoreBreakdown;
  sensor_stats: Record<
    string,
    {
      expected_hz: number;
      actual_hz: number;
      jitter_ms: number;
      drop_rate_pct: number;
      offset_ms: number;
    }
  >;
  blur_stats?: BlurStats;
  thresholds: Record<string, number>;
  measured_offsets_ms?: Record<string, number>;
};

export type EpisodeDetail = {
  id: string;
  status: string;
  quality_score: number;
  sync_error_ms: number;
  dropped_frames_pct: number;
  label_confidence: number;
  success?: boolean;
  duration_s: number;
  offsets: {
    reference_clock: string;
    rgb_ms: number;
    depth_ms: number;
    ft_ms: number;
    average_sync_error_ms: number;
  };
  issues: Issue[];
  labels: SkillSegment[];
  quality_report?: QualityReport;
  metadata: Record<string, unknown>;
  sync_settings?: {
    reference_clock: string;
    target_rate_hz: number;
  };
};

export type ReferenceClock =
  | "joint_state"
  | "camera_rgb"
  | "camera_depth"
  | "force_torque"
  | "ptp_grandmaster";

export type SyncSettings = {
  reference_clock: ReferenceClock;
  target_rate_hz: number;
};

export type TimelineResponse = {
  episode_id: string;
  mode: "raw" | "aligned";
  duration_s: number;
  current_sync_error_ms: number;
  sensors: Record<string, { t: number; present: boolean }[]>;
  force_series: { t: number; fz: number }[];
  tcp_series: { t: number; x: number; y: number; z: number }[];
  gripper_series: { t: number; width: number; closed: boolean }[];
  joint_series: { t: number; q0: number }[];
  skill_segments: SkillSegment[];
  anomaly_regions: { start: number; end: number; type: string; color: string; message: string }[];
  drop_regions: { start: number; end: number; type: string; color: string; message: string }[];
  offset_regions: { start: number; end: number; type: string; color: string; message: string }[];
  playback: {
    t: number;
    rgb_frame: number;
    depth_frame: number;
    q: number[];
    tcp: { x: number; y: number; z: number };
    fz: number;
    gripper_width: number;
    gripper_closed: boolean;
  }[];
  sync_settings?: SyncSettings;
};

export type DatasetCard = {
  version: string;
  episodes: number;
  accepted: number;
  rejected: number;
  manual_review: number;
  success_episodes: number;
  failure_episodes: number;
  average_sync_error_ms: number;
  camera_drop_rate_pct: number;
  format: string;
  lineage: string[];
  output_path: string;
};

export const api = {
  baseUrl: API_URL,
  getDatasets: () =>
    request<{
      project: string;
      dataset: string;
      trees: { name: string; children: { id: string; name: string; count: number }[] }[];
      episodes: EpisodeSummary[];
      dataset_version: string;
    }>("/datasets"),
  getEpisode: (id: string) => request<EpisodeDetail>(`/episodes/${id}`),
  getTimeline: (id: string, mode: "raw" | "aligned") =>
    request<TimelineResponse>(`/episodes/${id}/timeline?mode=${mode}`),
  align: (id: string, settings?: Partial<SyncSettings>) =>
    request<{
      before_sync_error_ms: number;
      after_sync_error_ms: number;
      offsets: EpisodeDetail["offsets"];
      reference_clock: string;
      target_rate_hz: number;
    }>(`/episodes/${id}/align`, {
      method: "POST",
      body: JSON.stringify({
        reference_clock: settings?.reference_clock ?? "joint_state",
        target_rate_hz: settings?.target_rate_hz ?? 20,
      }),
    }),
  estimateOffset: (id: string, settings?: Partial<SyncSettings>) =>
    request<EpisodeDetail["offsets"]>(`/episodes/${id}/estimate_offset`, {
      method: "POST",
      body: JSON.stringify({
        reference_clock: settings?.reference_clock ?? "joint_state",
      }),
    }),
  clean: (id: string) =>
    request<{ issues: Issue[]; quality_score: number; dropped_frames_pct: number }>(
      `/episodes/${id}/clean`,
      { method: "POST", body: "{}" }
    ),
  analyze: (id: string) =>
    request<Record<string, unknown>>(`/episodes/${id}/analyze`, {
      method: "POST",
      body: "{}",
    }),
  updateLabels: (id: string, labels: SkillSegment[]) =>
    request<{ labels: SkillSegment[] }>(`/episodes/${id}/labels`, {
      method: "PUT",
      body: JSON.stringify({ labels }),
    }),
  exportDataset: (payload: {
    format: string;
    target_rate_hz: number;
    include_raw: boolean;
    include_failed: boolean;
    version: string;
  }) =>
    request<DatasetCard>("/datasets/export", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  mediaUrl: (id: string, stream: "rgb" | "depth", frame: number) =>
    `${API_URL}/episodes/${id}/media/${stream}?frame=${frame}`,
  previewLeRobot: (payload: { repo_id: string; revision?: string; token?: string }) =>
    request<{
      repo_id: string;
      revision: string;
      codebase_version?: string;
      fps?: number;
      robot_type?: string;
      total_episodes: number;
      total_frames: number;
      features: string[];
      video_keys: string[];
      sample_episode_indices: number[];
    }>("/datasets/lerobot/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  importLeRobot: (payload: {
    repo_id: string;
    max_episodes: number;
    episode_indices?: number[];
    revision?: string;
    token?: string;
  }) =>
    request<{
      repo_id: string;
      imported_count: number;
      episodes: { episode_id: string; episode_index: number; duration_s: number; task: string }[];
      errors: { episode_index: number; error: string }[];
    }>("/datasets/lerobot/import", {
      method: "POST",
      body: JSON.stringify(payload),
      // HF download + video extract can take several minutes
      timeoutMs: 10 * 60 * 1000,
    }),
};
