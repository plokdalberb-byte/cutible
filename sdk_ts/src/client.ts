/**
 * Cutible TypeScript SDK — Client
 *
 * Full-featured client for the Cutible agent-native montage engine.
 * Supports both in-process (direct import) and HTTP (REST API) modes.
 *
 * Usage:
 *   import { CutibleClient } from 'cutible-sdk';
 *
 *   const client = new CutibleClient({ apiUrl: 'http://localhost:8000' });
 *   await client.createProject('demo', { fps: 30 });
 *   await client.addAsset('speaker', 'video', { uri: 'speaker.mp4', duration: 60 });
 *   await client.addTrack('v_main', 'video');
 *   await client.addClip('v_main', 'speaker', { srcIn: 0, srcOut: 10 });
 *   const result = await client.render('output.mp4');
 */

import axios, { AxiosInstance } from 'axios';
import type {
  ProjectSummary,
  Diff,
  RenderResult,
  QCReport,
  IngestResult,
  NarrativeSummary,
  AgentResult,
  OTIOResult,
  RenderFarmResult,
  AssetType,
  TrackKind,
  TransitionKind,
  ZoomLevel,
  AspectRatio,
  EditStyle,
} from './types';

export interface CutibleClientConfig {
  apiUrl?: string;
  timeout?: number;
}

export class CutibleClient {
  private apiUrl: string | null;
  private http: AxiosInstance | null;
  private projectId: string | null;

  constructor(config: CutibleClientConfig = {}) {
    this.apiUrl = config.apiUrl || null;
    this.projectId = null;
    if (this.apiUrl) {
      this.http = axios.create({
        baseURL: this.apiUrl,
        timeout: config.timeout || 120000,
        headers: { 'Content-Type': 'application/json' },
      });
    } else {
      this.http = null;
    }
  }

  // ---- Project Management ----

  async createProject(
    projectId: string,
    opts: { fps?: number; width?: number; height?: number; prompt?: string } = {}
  ): Promise<{ created: string; summary: ProjectSummary }> {
    if (this.http) {
      const res = await this.http.post('/projects', {
        id: projectId,
        fps: opts.fps || 30,
        width: opts.width || 1920,
        height: opts.height || 1080,
        prompt: opts.prompt || '',
      });
      this.projectId = projectId;
      return res.data;
    }
    throw new Error('HTTP mode required for createProject');
  }

  async loadProject(projectId: string, path: string): Promise<{ loaded: string; summary: ProjectSummary }> {
    if (this.http) {
      const res = await this.http.post(`/projects/${projectId}/load`, { path });
      this.projectId = projectId;
      return res.data;
    }
    throw new Error('HTTP mode required for loadProject');
  }

  async read(zoom: ZoomLevel = 'outline'): Promise<any> {
    if (this.http && this.projectId) {
      const res = await this.http.get(`/projects/${this.projectId}`, { params: { zoom } });
      return res.data;
    }
    throw new Error('HTTP mode required for read');
  }

  // ---- Asset Management ----

  async addAsset(
    assetId: string,
    type: AssetType,
    opts: { uri?: string; duration?: number; color?: string } = {}
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_asset',
        args: { asset_id: assetId, type, ...opts },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addAsset');
  }

  // ---- Track Management ----

  async addTrack(trackId: string, kind: TrackKind): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_track',
        args: { track_id: trackId, kind },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addTrack');
  }

  // ---- Clip Operations ----

  async addClip(
    trackId: string,
    asset: string,
    opts: {
      srcIn?: number;
      srcOut?: number;
      timelineIn?: number;
      speed?: number;
      volume?: number;
      rationale?: string;
    } = {}
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_clip',
        args: {
          track_id: trackId,
          asset,
          src_in: opts.srcIn,
          src_out: opts.srcOut,
          timeline_in: opts.timelineIn,
          speed: opts.speed,
          volume: opts.volume,
          rationale: opts.rationale,
        },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addClip');
  }

  async trim(
    clipId: string,
    opts: { srcIn?: number; srcOut?: number } = {}
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'trim',
        args: { clip_id: clipId, src_in: opts.srcIn, src_out: opts.srcOut },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for trim');
  }

  async move(clipId: string, timelineIn: number): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'move',
        args: { clip_id: clipId, timeline_in: timelineIn },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for move');
  }

  async split(clipId: string, t: number): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'split',
        args: { clip_id: clipId, t },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for split');
  }

  async rippleDelete(clipId: string): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'ripple_delete',
        args: { clip_id: clipId },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for rippleDelete');
  }

  async setSpeed(clipId: string, speed: number): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'set_speed',
        args: { clip_id: clipId, speed },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for setSpeed');
  }

  async setVolume(clipId: string, volume: number): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'set_volume',
        args: { clip_id: clipId, volume },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for setVolume');
  }

  // ---- Transitions ----

  async addTransition(
    clipId: string,
    kind: TransitionKind = 'in',
    duration: number = 0.5
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_transition',
        args: { clip_id: clipId, kind, duration },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addTransition');
  }

  // ---- Text Layers ----

  async addTextLayer(
    trackId: string,
    text: string,
    timelineIn: number,
    timelineOut: number,
    opts: { fontSize?: number; fontColor?: string } = {}
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_text_layer',
        args: {
          track_id: trackId,
          text,
          timeline_in: timelineIn,
          timeline_out: timelineOut,
          font_size: opts.fontSize,
          font_color: opts.fontColor,
        },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addTextLayer');
  }

  // ---- Audio ----

  async addAudio(
    asset: string,
    opts: {
      srcIn?: number;
      srcOut?: number;
      timelineIn?: number;
      volume?: number;
      trackId?: string;
    } = {}
  ): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'add_audio',
        args: {
          asset,
          src_in: opts.srcIn,
          src_out: opts.srcOut,
          timeline_in: opts.timelineIn,
          volume: opts.volume,
          track_id: opts.trackId || 'music',
        },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for addAudio');
  }

  // ---- Checkpoint / Undo ----

  async checkpoint(label: string = ''): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'checkpoint',
        args: { label },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for checkpoint');
  }

  async undo(): Promise<Diff> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/verbs`, {
        verb: 'undo',
        args: {},
      });
      return res.data;
    }
    throw new Error('HTTP mode required for undo');
  }

  // ---- Render ----

  async render(
    output: string,
    opts: { runQc?: boolean; dryRun?: boolean } = {}
  ): Promise<RenderResult> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/render`, {
        output,
        run_qc: opts.runQc !== false,
        dry_run: opts.dryRun || false,
      });
      return res.data;
    }
    throw new Error('HTTP mode required for render');
  }

  // ---- QC ----

  async qc(file: string, expectedDuration?: number): Promise<QCReport> {
    if (this.http) {
      const res = await this.http.post('/qc', {
        file,
        expected_duration: expectedDuration,
      });
      return res.data;
    }
    throw new Error('HTTP mode required for qc');
  }

  // ---- Save / Load ----

  async save(path: string): Promise<{ saved: string; hash: string }> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/save`, { path });
      return res.data;
    }
    throw new Error('HTTP mode required for save');
  }

  // ---- Ingest Pipeline ----

  async ingestAsset(assetId: string, uri: string): Promise<IngestResult> {
    if (this.http) {
      const res = await this.http.post('/ingest', {
        asset_id: assetId,
        uri,
      });
      return res.data;
    }
    throw new Error('HTTP mode required for ingestAsset');
  }

  async buildNarrative(projectId: string = 'default'): Promise<NarrativeSummary> {
    if (this.http) {
      const res = await this.http.get(`/index/${projectId}/narrative`);
      return res.data;
    }
    throw new Error('HTTP mode required for buildNarrative');
  }

  async searchIndex(query: string): Promise<any[]> {
    if (this.http) {
      const res = await this.http.get('/index/search', { params: { q: query } });
      return res.data.results || [];
    }
    throw new Error('HTTP mode required for searchIndex');
  }

  // ---- Agent Swarm ----

  async runAgent(
    brief: string,
    opts: {
      targetDuration?: number;
      style?: EditStyle;
      maxIterations?: number;
      indexDir?: string;
    } = {}
  ): Promise<AgentResult> {
    if (this.http) {
      const res = await this.http.post('/agent/run', {
        brief,
        target_duration: opts.targetDuration || 60,
        style: opts.style || 'informative',
        max_iterations: opts.maxIterations || 3,
        index_dir: opts.indexDir || '.cutible/index',
      });
      return res.data;
    }
    throw new Error('HTTP mode required for runAgent');
  }

  // ---- OTIO Bridge ----

  async exportOtio(outputPath: string): Promise<OTIOResult> {
    if (this.http && this.projectId) {
      const res = await this.http.get(`/projects/${this.projectId}/otio`, {
        params: { output_path: outputPath },
      });
      return res.data;
    }
    throw new Error('HTTP mode required for exportOtio');
  }

  async importOtio(otioPath: string, projectId?: string): Promise<OTIOResult> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/otio/import`, {
        otio_path: otioPath,
      });
      return res.data;
    }
    throw new Error('HTTP mode required for importOtio');
  }

  // ---- Render Farm ----

  async renderFarm(
    output: string,
    opts: { nWorkers?: number } = {}
  ): Promise<RenderFarmResult> {
    if (this.http && this.projectId) {
      const res = await this.http.post(`/projects/${this.projectId}/render-farm`, {
        output,
        n_workers: opts.nWorkers || 2,
      });
      return res.data;
    }
    throw new Error('HTTP mode required for renderFarm');
  }
}
