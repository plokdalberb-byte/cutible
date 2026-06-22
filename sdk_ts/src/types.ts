// Cutible TypeScript SDK — Type definitions

export interface ProjectSummary {
  id: string;
  fps: number;
  resolution: string;
  duration: number;
  n_assets: number;
  tracks: TrackSummary[];
}

export interface TrackSummary {
  id: string;
  kind: string;
  n_clips: number;
  n_texts: number;
  duration: number;
}

export interface Diff {
  verb: string;
  changed: string[];
  details: Record<string, any>;
}

export interface VerbError {
  error: string;
  hint: string;
  context: Record<string, any>;
}

export interface RenderResult {
  ok: boolean;
  output?: string;
  duration?: number;
  has_audio?: boolean;
  content_hash?: string;
  size_bytes?: number;
  qc?: QCReport;
  error?: string;
}

export interface QCReport {
  path: string;
  passed: boolean;
  duration: number;
  has_video: boolean;
  has_audio: boolean;
  integrated_lufs?: number;
  violations: Violation[];
}

export interface Violation {
  code: string;
  severity: string;
  message: string;
  at?: number;
}

export interface IngestResult {
  asset_id: string;
  uri: string;
  success: boolean;
  proxy_path?: string;
  index_summary?: Record<string, any>;
  error?: string;
}

export interface NarrativeSummary {
  project_id: string;
  total_duration: number;
  n_assets: number;
  n_speakers: number;
  synopsis_preview?: string;
}

export interface AgentResult {
  brief: string;
  iterations: any[];
  final_project: Record<string, any> | null;
  passed: boolean;
}

export interface OTIOResult {
  ok: boolean;
  output?: string;
  imported?: string;
  format?: string;
  summary?: ProjectSummary;
  warning?: string;
}

export interface RenderFarmResult {
  ok: boolean;
  output?: string;
  segments?: number;
  total_segments?: number;
  duration?: number;
  size_bytes?: number;
  progress?: {
    total: number;
    completed: number;
    failed: number;
    running: number;
    pending: number;
    percent: number;
  };
  error?: string;
}

export type AssetType = "video" | "audio" | "image" | "color";
export type TrackKind = "video" | "audio" | "caption";
export type TransitionKind = "in" | "out";
export type ZoomLevel = "summary" | "outline" | "detail";
export type AspectRatio = "16:9" | "9:16" | "1:1" | "4:3";
export type EditStyle = "informative" | "energetic" | "calm" | "professional" | "humorous";
