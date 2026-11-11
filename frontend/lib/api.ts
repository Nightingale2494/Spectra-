import {
  AnalysisJob,
  AnalysisResults,
  AudioAsset,
  DetectorInfo,
  DspPreset,
  GeneratedReport,
  HealthStatus,
  SpectrogramData,
  TemporalAnalysisData,
  TemporalComparisonData,
  WaveformData,
} from "@/types/spectra";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000/api";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errText = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      errText = body.detail || body.message || errText;
    } catch {
      errText = await res.text();
    }
    throw new Error(errText);
  }
  return res.json() as Promise<T>;
}

export function getAudioStreamUrl(assetId: string): string {
  return `${API_BASE}/audio/${assetId}/stream`;
}

export function getReportDownloadUrl(
  jobId: string,
  format: "json" | "pdf",
  experimentId?: string
): string {
  const q = new URLSearchParams({ format });
  if (experimentId) q.set("experiment_id", experimentId);
  return `${API_BASE}/jobs/${jobId}/report?${q.toString()}`;
}

export function getJobArtifactDownloadUrl(
  jobId: string,
  target: "original" | "processed" | "report_json" | "report_pdf",
  experimentId?: string
): string {
  const q = new URLSearchParams({ target });
  if (experimentId) q.set("experiment_id", experimentId);
  return `${API_BASE}/jobs/${jobId}/download?${q.toString()}`;
}

export function getStoredReportDownloadUrl(reportId: string): string {
  return `${API_BASE}/reports/${reportId}/download`;
}

export const spectraApi = {
  getHealth: () => fetch(`${API_BASE}/health`).then((r) => handleResponse<HealthStatus>(r)),

  getDetectors: () =>
    fetch(`${API_BASE}/detectors`).then((r) => handleResponse<DetectorInfo[]>(r)),

  getDspPresets: () =>
    fetch(`${API_BASE}/dsp/presets`).then((r) => handleResponse<DspPreset[]>(r)),

  getReferenceSamples: () =>
    fetch(`${API_BASE}/audio/samples`).then((r) =>
      handleResponse<
        Array<{
          sample_id: string;
          filename: string;
          title: string;
          description: string;
        }>
      >(r)
    ),

  loadReferenceSample: (sampleId: string, runSync = true) =>
    fetch(`${API_BASE}/audio/load-sample/${sampleId}?run_sync=${runSync}`, {
      method: "POST",
    }).then((r) =>
      handleResponse<{
        asset: AudioAsset;
        initial_job_id: string;
        message: string;
      }>(r)
    ),

  uploadAudio: async (file: File, runSync = true) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/audio/upload?run_sync=${runSync}`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<{
      asset: AudioAsset;
      initial_job_id: string;
      message: string;
    }>(res);
  },

  listJobs: (limit = 30) =>
    fetch(`${API_BASE}/jobs?limit=${limit}`).then((r) =>
      handleResponse<
        Array<{
          job: AnalysisJob;
          asset: AudioAsset | null;
          c2pa_status: string | null;
          detectors: any[];
          experiments_count: number;
        }>
      >(r)
    ),

  getJobStatus: (jobId: string) =>
    fetch(`${API_BASE}/jobs/${jobId}`).then((r) => handleResponse<AnalysisJob>(r)),

  getJobResults: (jobId: string) =>
    fetch(`${API_BASE}/jobs/${jobId}/results`).then((r) =>
      handleResponse<AnalysisResults>(r)
    ),

  reanalyzeJob: (jobId: string, runSync = true) =>
    fetch(`${API_BASE}/jobs/${jobId}/analyze?run_sync=${runSync}`, {
      method: "POST",
    }).then((r) => handleResponse<AnalysisJob>(r)),

  getWaveform: (assetId: string, points = 600) =>
    fetch(`${API_BASE}/audio/${assetId}/waveform?points=${points}`).then((r) =>
      handleResponse<WaveformData>(r)
    ),

  getSpectrogram: (assetId: string) =>
    fetch(`${API_BASE}/audio/${assetId}/spectrogram`).then((r) =>
      handleResponse<SpectrogramData>(r)
    ),

  runDspProcess: (
    jobId: string,
    payload: {
      operation_type: string;
      preset_name?: string | null;
      engine?: string;
      parameters: Record<string, any>;
      export_format?: string;
      run_detectors_on_output?: boolean;
    }
  ) =>
    fetch(`${API_BASE}/jobs/${jobId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) =>
      handleResponse<{
        experiment: any;
        processed_asset: AudioAsset;
        comparison: any;
      }>(r)
    ),

  exportCleanMetadata: (
    jobId: string,
    payload: {
      strip_conventional_metadata: boolean;
      strip_c2pa_manifest: boolean;
      custom_tags?: Record<string, string>;
    }
  ) =>
    fetch(`${API_BASE}/jobs/${jobId}/metadata-export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) =>
      handleResponse<{
        experiment: any;
        processed_asset: AudioAsset;
      }>(r)
    ),

  getComparisons: (jobId: string) =>
    fetch(`${API_BASE}/jobs/${jobId}/compare`).then((r) =>
      handleResponse<{
        job_id: string;
        source_asset: AudioAsset;
        items: Array<{
          experiment: any;
          processed_asset: AudioAsset | null;
          comparison: any | null;
        }>;
      }>(r)
    ),

  createReportRecord: (
    jobId: string,
    payload: {
      report_format: "json" | "pdf";
      experiment_id?: string | null;
      include_raw_manifest?: boolean;
    }
  ) =>
    fetch(`${API_BASE}/jobs/${jobId}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => handleResponse<GeneratedReport>(r)),

  listReports: () =>
    fetch(`${API_BASE}/reports`).then((r) => handleResponse<GeneratedReport[]>(r)),

  getSettingsInfo: () =>
    fetch(`${API_BASE}/settings/info`).then((r) => handleResponse<Record<string, any>>(r)),

  getLatestBenchmarks: () =>
    fetch(`${API_BASE}/benchmarks/latest`).then((r) =>
      handleResponse<{ available: boolean; data: any }>(r)
    ),

  getTemporalAnalysis: (
    jobId: string,
    params?: {
      hop_sec?: number;
      fakeprint_window_sec?: number;
      include_exploratory_fakeprint?: boolean;
      force_recompute?: boolean;
    }
  ) => {
    const q = new URLSearchParams();
    if (params?.hop_sec !== undefined) q.set("hop_sec", String(params.hop_sec));
    if (params?.fakeprint_window_sec !== undefined)
      q.set("fakeprint_window_sec", String(params.fakeprint_window_sec));
    if (params?.include_exploratory_fakeprint !== undefined)
      q.set("include_exploratory_fakeprint", String(params.include_exploratory_fakeprint));
    if (params?.force_recompute !== undefined)
      q.set("force_recompute", String(params.force_recompute));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return fetch(`${API_BASE}/jobs/${jobId}/temporal${qs}`).then((r) =>
      handleResponse<TemporalAnalysisData>(r)
    );
  },

  postTemporalAnalysis: (
    jobId: string,
    payload: {
      hop_sec: number;
      fakeprint_window_sec?: number;
      include_exploratory_fakeprint?: boolean;
      force_recompute?: boolean;
    }
  ) =>
    fetch(`${API_BASE}/jobs/${jobId}/temporal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => handleResponse<TemporalAnalysisData>(r)),

  cancelTemporalAnalysis: (jobId: string) =>
    fetch(`${API_BASE}/jobs/${jobId}/temporal/cancel`, {
      method: "POST",
    }).then((r) => handleResponse<{ job_id: string; status: string }>(r)),

  getTemporalComparison: (jobId: string, experimentId?: string, hopSec = 2.5) => {
    const q = new URLSearchParams({ hop_sec: String(hopSec) });
    if (experimentId) q.set("experiment_id", experimentId);
    return fetch(`${API_BASE}/jobs/${jobId}/temporal-compare?${q.toString()}`).then((r) =>
      handleResponse<TemporalComparisonData>(r)
    );
  },
};
