"use client";

import React, { useState } from "react";
import {
  Sliders,
  Play,
  GitCompare,
  FileText,
  Download,
  Settings as SettingsIcon,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  HardDrive,
  Shield,
  Sparkles,
  RefreshCw,
  Layers,
} from "lucide-react";
import {
  AnalysisResults,
  ComparisonResult,
  DetectorInfo,
  DspPreset,
  GeneratedReport,
  HealthStatus,
  ProcessingExperiment,
  WaveformData,
} from "@/types/spectra";
import {
  getAudioStreamUrl,
  getJobArtifactDownloadUrl,
  getReportDownloadUrl,
  getStoredReportDownloadUrl,
  spectraApi,
} from "@/lib/api";
import AudioWorkstationPlayer from "./AudioWorkstationPlayer";
import { ComparisonSpectraChart } from "./ForensicCharts";

/* ============================================================================
 * PAGE 4: DSP LABORATORY VIEW
 * ============================================================================ */
export function DspLaboratoryView({
  results,
  waveform,
  presets,
  onExperimentCompleted,
  onNavigateComparison,
}: {
  results: AnalysisResults | null;
  waveform: WaveformData | null;
  presets: DspPreset[];
  onExperimentCompleted: () => Promise<void>;
  onNavigateComparison: () => void;
}) {
  const [selectedPreset, setSelectedPreset] = useState<string>(
    presets[0]?.id || "vocoder_band_smoothing"
  );
  const [operationType, setOperationType] = useState<string>(
    presets[0]?.operation_type || "spectral_smoothing"
  );
  const [engine, setEngine] = useState<string>(presets[0]?.engine || "scipy_dsp");
  const [params, setParams] = useState<Record<string, any>>(
    presets[0]?.parameters || {
      freq_min_hz: 1000.0,
      freq_max_hz: 8000.0,
      smoothing_bins: 9,
      mix: 0.65,
      n_fft: 4096,
    }
  );
  const [processing, setProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{
    experiment: ProcessingExperiment;
    comparison: ComparisonResult;
  } | null>(null);

  const defaultOperationParams: Record<string, Record<string, any>> = {
    spectral_smoothing: { freq_min_hz: 1000, freq_max_hz: 8000, smoothing_bins: 9, mix: 0.65, n_fft: 4096 },
    frequency_selective: { freq_min_hz: 1000, freq_max_hz: 8000, spike_threshold_db: 2.2, suppression_ratio: 0.6, n_fft: 4096 },
    notch_comb: { fundamental_hz: 400.0, min_freq_hz: 1600.0, max_freq_hz: 7600.0, q: 30.0 },
    lowpass: { cutoff_hz: 7500.0, order: 6 },
    highpass: { cutoff_hz: 80.0, order: 4 },
    bandpass: { low_hz: 300.0, high_hz: 5000.0, order: 4 },
    parametric_eq: { center_hz: 4000.0, q_factor: 2.5, gain_db: -4.5 },
    eq_3band: { low_gain_db: 1.5, low_freq_hz: 120.0, mid_gain_db: -0.5, mid_freq_hz: 1500.0, mid_q: 0.8, high_gain_db: -2.0, high_freq_hz: 6500.0 },
    noise_reduction: { reduction_db: 10.0, sensitivity: 1.2, n_fft: 2048 },
    dynamic_compression: { threshold_db: -16.0, ratio: 3.0, attack_ms: 12.0, release_ms: 120.0, makeup_gain_db: 2.0 },
    gain_limiter: { gain_db: 1.5, enable_limiter: true, ceiling_db: -0.5 },
    resample: { target_sr: 16000, roundtrip_to_original_sr: true },
    codec_simulation: { codec: "mp3", bitrate_kbps: 128 },
    pedalboard_chain: { highpass_hz: 80.0, peak_freq_hz: 4500.0, peak_gain_db: -2.5, threshold_db: -16.0, ratio: 2.5, ceiling_db: -0.5 },
  };

  if (!results) {
    return (
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-8 text-center">
        <Sliders className="mx-auto h-10 w-10 text-slate-500 mb-3" />
        <h3 className="text-base font-semibold text-slate-200">
          No Active Audio Job Selected
        </h3>
        <p className="mt-1 text-xs text-slate-400">
          Upload an audio track or load a reference sample from the Dashboard first to run
          controlled DSP experiments.
        </p>
      </div>
    );
  }

  const applyPreset = (presetId: string) => {
    setSelectedPreset(presetId);
    const found = presets.find((p) => p.id === presetId);
    if (found) {
      setOperationType(found.operation_type);
      setEngine(found.engine);
      setParams({ ...found.parameters });
    }
  };

  const handleRunDsp = async () => {
    setProcessing(true);
    setErrorMsg(null);
    try {
      const res = await spectraApi.runDspProcess(results.job.id, {
        operation_type: operationType,
        preset_name: selectedPreset || null,
        engine,
        parameters: params,
        export_format: "wav",
        run_detectors_on_output: true,
      });
      setLastResult({
        experiment: res.experiment,
        comparison: res.comparison,
      });
      await onExperimentCompleted();
    } catch (err: any) {
      setErrorMsg(err.message || "DSP processing failed.");
    } finally {
      setProcessing(false);
    }
  };

  const latestExp = lastResult?.experiment || results.experiments[0] || null;
  const latestComp = lastResult?.comparison || results.comparisons[0] || null;

  return (
    <div className="space-y-6">
      {/* Scientific Guardrail Banner */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-200/90 leading-relaxed">
          <span className="font-semibold text-amber-300">
            Controlled Signal Processing Laboratory:
          </span>{" "}
          Experiments modify physical time/frequency signal characteristics while preserving the
          immutable original asset ({results.asset.sha256_checksum.slice(0, 12)}…). Experimental
          artifact reduction evaluates detector sensitivity and fidelity trade-offs; it does not
          guarantee that processed audio becomes universally untraceable.
        </div>
      </div>

      {/* Audio Player with A/B Toggle if processed variant exists */}
      <AudioWorkstationPlayer
        title={results.asset.original_filename}
        subtitle={`Immutable Source SHA-256: ${results.asset.sha256_checksum.slice(0, 16)}…`}
        audioUrl={getAudioStreamUrl(results.asset.id)}
        secondaryAudioUrl={
          latestExp?.output_asset_id ? getAudioStreamUrl(latestExp.output_asset_id) : null
        }
        secondaryLabel={latestExp ? `${latestExp.operation_type}` : "Processed Variant"}
        waveform={waveform}
        downloadUrl={
          latestExp?.output_asset_id
            ? getJobArtifactDownloadUrl(results.job.id, "processed", latestExp.id)
            : getJobArtifactDownloadUrl(results.job.id, "original")
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Preset Selector & DSP Rack */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                <Sliders className="h-4 w-4 text-cyan-400" />
                DSP Processing Rack & Filter Chain
              </h3>
              <p className="text-xs text-slate-400">
                Select a laboratory preset or customize filter topology and parameters
              </p>
            </div>
            <span className="rounded bg-cyan-500/10 border border-cyan-500/30 px-2.5 py-1 font-mono text-xs text-cyan-300">
              Engine: {engine}
            </span>
          </div>

          {/* Preset Cards */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Laboratory Presets
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => applyPreset(p.id)}
                  className={`text-left rounded-lg border p-3 transition ${
                    selectedPreset === p.id
                      ? "border-cyan-500 bg-cyan-950/30 text-slate-100"
                      : "border-slate-800 bg-slate-900/50 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold">{p.name}</span>
                    <span className="font-mono text-[10px] text-cyan-400">{p.operation_type}</span>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-400 line-clamp-2">{p.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Operation & Engine Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                DSP Operation Topology
              </label>
              <select
                value={operationType}
                onChange={(e) => {
                  const newOp = e.target.value;
                  setOperationType(newOp);
                  setSelectedPreset("");
                  if (defaultOperationParams[newOp]) {
                    setParams({ ...defaultOperationParams[newOp] });
                  }
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-100 font-mono focus:border-cyan-500 focus:outline-none"
              >
                <option value="frequency_selective">frequency_selective (1–8 kHz Band Attenuation)</option>
                <option value="notch_comb">notch_comb (Harmonic Comb Notch Filter)</option>
                <option value="spectral_smoothing">spectral_smoothing (STFT Magnitude Smoothing)</option>
                <option value="lowpass">lowpass (Butterworth Low-Pass Filter)</option>
                <option value="highpass">highpass (Butterworth High-Pass Filter)</option>
                <option value="bandpass">bandpass (Butterworth Band-Pass Filter)</option>
                <option value="parametric_eq">parametric_eq (Biquad Peaking EQ)</option>
                <option value="eq_3band">eq_3band (3-Band Shelving/Mid EQ)</option>
                <option value="noise_reduction">noise_reduction (Spectral Gate Denoiser)</option>
                <option value="dynamic_compression">dynamic_compression (RMS Envelope Compressor)</option>
                <option value="gain_limiter">gain_limiter (Gain Stage + Soft Ceiling Limiter)</option>
                <option value="resample">resample (SoXR Sample Rate Round-Trip)</option>
                <option value="codec_simulation">codec_simulation (Lossy MP3/AAC Transcode)</option>
                <option value="pedalboard_chain">pedalboard_chain (Spotify Pedalboard Studio Chain)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Execution Engine
              </label>
              <select
                value={engine}
                onChange={(e) => setEngine(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-100 font-mono focus:border-cyan-500 focus:outline-none"
              >
                <option value="scipy_dsp">scipy_dsp (Zero-Phase SciPy / Librosa STFT)</option>
                <option value="pedalboard">pedalboard (Spotify Pedalboard C++ Audio Unit)</option>
              </select>
            </div>
          </div>

          {/* Dynamic Parameter Sliders / Inputs */}
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Active Filter Parameters
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                JSON-editable & slider-controlled
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {Object.entries(params).map(([k, v]) => (
                <div key={k} className="rounded border border-slate-800 bg-slate-900/60 p-2.5">
                  <label className="block font-mono text-[11px] text-cyan-300 mb-1">{k}</label>
                  {typeof v === "number" ? (
                    <input
                      type="number"
                      step="any"
                      value={v}
                      onChange={(e) =>
                        setParams({ ...params, [k]: parseFloat(e.target.value) || 0 })
                      }
                      className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-100"
                    />
                  ) : (
                    <input
                      type="text"
                      value={String(v)}
                      onChange={(e) => setParams({ ...params, [k]: e.target.value })}
                      className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-100"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {errorMsg && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-3 text-xs text-rose-200">
              {errorMsg}
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <button
              type="button"
              disabled={processing}
              onClick={handleRunDsp}
              className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-5 py-2.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50 transition shadow-md"
            >
              {processing ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Executing DSP & Re-evaluating Detectors...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Execute DSP Transform & Run Before/After Comparison
                </>
              )}
            </button>

            {latestComp && (
              <button
                type="button"
                onClick={onNavigateComparison}
                className="inline-flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-xs font-semibold text-amber-300 hover:bg-amber-500/20 transition"
              >
                <GitCompare className="h-4 w-4" />
                Inspect Full Before/After Comparison →
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Latest Experiment Telemetry & Audit Trail */}
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-100 border-b border-slate-800 pb-2.5">
            Latest Experiment Telemetry
          </h3>

          {latestExp ? (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                  <span className="text-slate-400 block text-[10px]">Operation</span>
                  <span className="font-mono font-semibold text-cyan-300">
                    {latestExp.operation_type}
                  </span>
                </div>
                <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                  <span className="text-slate-400 block text-[10px]">Execution Time</span>
                  <span className="font-mono font-semibold text-slate-100">
                    {latestExp.processing_time_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                  <span className="text-slate-400 block text-[10px]">Duration / Channels</span>
                  <span className="font-mono font-semibold text-emerald-400">
                    {latestExp.duration_preserved && latestExp.channels_preserved
                      ? "Preserved ✓"
                      : "Modified"}
                  </span>
                </div>
                <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                  <span className="text-slate-400 block text-[10px]">Signal Stability</span>
                  <span className="font-mono font-semibold text-emerald-400">
                    {!latestExp.nan_inf_detected && latestExp.reconstruction_valid
                      ? "Finite PCM ✓"
                      : "Warning"}
                  </span>
                </div>
              </div>

              {latestComp && (
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 space-y-2">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Detector Score Shifts
                  </div>
                  {latestComp.detector_deltas.map((d) => {
                    const bProb = d.original_probability ?? d.before_probability ?? null;
                    const aProb = d.processed_probability ?? d.after_probability ?? null;
                    return (
                      <div
                        key={d.detector_id}
                        className="flex items-center justify-between border-b border-slate-800/70 pb-1.5 last:border-none last:pb-0 font-mono text-[11px]"
                      >
                        <span className="text-slate-300 truncate max-w-[140px]">
                          {d.detector_id}
                        </span>
                        <span className="text-slate-200">
                          {bProb !== null ? (bProb * 100).toFixed(1) + "%" : "N/A"} →{" "}
                          <strong className="text-cyan-300">
                            {aProb !== null ? (aProb * 100).toFixed(1) + "%" : "N/A"}
                          </strong>
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                  Audit Log
                </div>
                <div className="max-h-40 overflow-y-auto rounded border border-slate-800 bg-slate-950 p-2.5 font-mono text-[11px] text-slate-300 space-y-1">
                  {latestExp.audit_log.map((entry, i) => (
                    <div key={i}>• {entry}</div>
                  ))}
                </div>
              </div>

              {latestExp.output_asset_id && (
                <a
                  href={getJobArtifactDownloadUrl(results.job.id, "processed", latestExp.id)}
                  download
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 transition"
                >
                  <Download className="h-4 w-4" />
                  Download Processed WAV ({latestExp.processed_sha256?.slice(0, 10)}…)
                </a>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-400">
              No DSP experiment executed on this track yet. Choose a preset on the left and click{" "}
              <strong>Execute DSP Transform</strong>.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
 * PAGE 5: COMPARISON VIEW
 * ============================================================================ */
export function ComparisonView({
  results,
  onNavigateDsp,
}: {
  results: AnalysisResults | null;
  onNavigateDsp: () => void;
}) {
  const [selectedExpIdx, setSelectedExpIdx] = useState(0);

  if (!results || results.comparisons.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-8 text-center">
        <GitCompare className="mx-auto h-10 w-10 text-slate-500 mb-3" />
        <h3 className="text-base font-semibold text-slate-200">
          No Before/After Comparisons Available Yet
        </h3>
        <p className="mt-1 text-xs text-slate-400 max-w-md mx-auto">
          Run a controlled DSP transformation in the DSP Laboratory first to compare original and
          processed spectra, 1–8 kHz fakeprint residuals, fidelity metrics, and detector deltas.
        </p>
        <button
          type="button"
          onClick={onNavigateDsp}
          className="mt-4 inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition"
        >
          <Sliders className="h-4 w-4" />
          Open DSP Laboratory
        </button>
      </div>
    );
  }

  const comp = results.comparisons[selectedExpIdx] || results.comparisons[0];
  const exp =
    results.experiments.find((e) => e.id === comp.experiment_id) || results.experiments[0];

  return (
    <div className="space-y-6">
      {/* Experiment Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#0d111a] p-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Before vs. After Forensic & Fidelity Comparison
          </h3>
          <p className="text-xs text-slate-400 font-mono">
            Source: {results.asset.original_filename} ({results.asset.sha256_checksum.slice(0, 12)}
            …) → Processed ({exp?.processed_sha256?.slice(0, 12)}…)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Experiment:</span>
          <select
            value={selectedExpIdx}
            onChange={(e) => setSelectedExpIdx(Number(e.target.value))}
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-mono text-slate-100"
          >
            {results.comparisons.map((c, idx) => {
              const matchingExp = results.experiments.find((e) => e.id === c.experiment_id);
              return (
                <option key={c.id} value={idx}>
                  #{idx + 1} • {matchingExp?.operation_type || "DSP"} (
                  {matchingExp?.preset_name || matchingExp?.engine_used})
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Numerical Fidelity & Distortion Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">SNR (Signal-to-Noise)</span>
          <span className="mt-1 block font-mono text-lg font-bold text-cyan-300">
            {comp.snr_db !== null ? `${comp.snr_db.toFixed(2)} dB` : "∞ dB"}
          </span>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">PSNR</span>
          <span className="mt-1 block font-mono text-lg font-bold text-slate-100">
            {comp.psnr_db !== null ? `${comp.psnr_db.toFixed(2)} dB` : "∞ dB"}
          </span>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">Log-Spectral Dist (LSD)</span>
          <span className="mt-1 block font-mono text-lg font-bold text-amber-300">
            {comp.spectral_distance_l2.toFixed(3)} dB
          </span>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">Δ RMS Level</span>
          <span className="mt-1 block font-mono text-lg font-bold text-slate-100">
            {comp.delta_rms_db >= 0 ? "+" : ""}
            {comp.delta_rms_db.toFixed(2)} dB
          </span>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">Δ Peak Level</span>
          <span className="mt-1 block font-mono text-lg font-bold text-slate-100">
            {comp.delta_peak_db >= 0 ? "+" : ""}
            {comp.delta_peak_db.toFixed(2)} dB
          </span>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-3.5">
          <span className="text-[11px] text-slate-400 block">Phase Alignment Offset</span>
          <span className="mt-1 block font-mono text-lg font-bold text-emerald-300">
            {comp.alignment_offset_samples} smp
          </span>
        </div>
      </div>

      {/* Spectra & Fakeprint Residual Overlay Charts */}
      <ComparisonSpectraChart comparison={comp} />

      {/* Detector Score Deltas Table */}
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
        <h4 className="text-sm font-semibold text-slate-100 mb-3">
          Before vs. After AI-Music Detector Score Deltas
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Detector Model</th>
                <th className="py-2.5 px-3">Original Probability</th>
                <th className="py-2.5 px-3">Processed Probability</th>
                <th className="py-2.5 px-3">Delta (ΔP)</th>
                <th className="py-2.5 px-3">Original Label</th>
                <th className="py-2.5 px-3">Processed Label</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {comp.detector_deltas.map((d) => {
                const origProb = d.original_probability ?? d.before_probability ?? null;
                const procProb = d.processed_probability ?? d.after_probability ?? null;
                const origLbl = d.original_label ?? d.before_label ?? "";
                const procLbl = d.processed_label ?? d.after_label ?? "";
                const delta = d.delta_probability ?? 0;
                return (
                  <tr key={d.detector_id} className="hover:bg-slate-900/40">
                    <td className="py-2.5 px-3 font-semibold text-slate-100">
                      {d.detector_name}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {origProb !== null
                        ? `${(origProb * 100).toFixed(2)}%`
                        : "N/A"}
                    </td>
                    <td className="py-2.5 px-3 text-cyan-300 font-semibold">
                      {procProb !== null
                        ? `${(procProb * 100).toFixed(2)}%`
                        : "N/A"}
                    </td>
                    <td
                      className={`py-2.5 px-3 font-semibold ${
                        delta < 0
                          ? "text-emerald-400"
                          : delta > 0
                          ? "text-rose-400"
                          : "text-slate-400"
                      }`}
                    >
                      {d.delta_probability !== null
                        ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}%`
                        : "0.00%"}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{origLbl}</td>
                    <td className="py-2.5 px-3 text-slate-200">{procLbl}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Temporal Segment-by-Segment DSP Comparison (Phase 5 Part A.5) */}
      {comp.residual_comparison?.temporal_comparison && (
        <div
          data-testid="temporal-dsp-comparison-panel"
          className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <h4 className="text-sm font-semibold text-slate-100">
                Temporal Segment-by-Segment Before/After Comparison (5.0s Windows)
              </h4>
              <p className="text-xs text-slate-400 font-mono">
                Time Alignment Offset: {comp.residual_comparison.temporal_comparison.alignment_offset_samples} samples (
                {comp.residual_comparison.temporal_comparison.alignment_offset_ms.toFixed(2)} ms) •{" "}
                {comp.residual_comparison.temporal_comparison.segment_count} aligned windows
              </p>
            </div>
            <div className="flex items-center gap-3 font-mono text-xs">
              <span className="rounded bg-slate-900 border border-slate-800 px-2.5 py-1 text-cyan-300">
                Mean Δ SONICS-γ:{" "}
                {comp.residual_comparison.temporal_comparison.summary.mean_delta_sonics_probability !== null
                  ? `${(
                      comp.residual_comparison.temporal_comparison.summary.mean_delta_sonics_probability * 100
                    ).toFixed(2)}%`
                  : "N/A"}
              </span>
              <span className="rounded bg-slate-900 border border-slate-800 px-2.5 py-1 text-amber-300">
                Mean Segment LSD:{" "}
                {comp.residual_comparison.temporal_comparison.summary.mean_segment_lsd_db?.toFixed(2) ?? "0.00"} dB
              </span>
            </div>
          </div>

          <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-xs text-amber-200">
            <strong className="text-amber-300">Scientific Disclaimer:</strong>{" "}
            {comp.residual_comparison.temporal_comparison.scientific_disclaimer}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse font-mono text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                  <th className="py-2 px-2.5">Segment Span</th>
                  <th className="py-2 px-2.5">Orig SONICS-γ</th>
                  <th className="py-2 px-2.5">Proc SONICS-γ</th>
                  <th className="py-2 px-2.5">Δ SONICS-γ</th>
                  <th className="py-2 px-2.5">Orig / Proc Fakeprint (Expl.)</th>
                  <th className="py-2 px-2.5">Seg SNR / LSD</th>
                  <th className="py-2 px-2.5">Δ RMS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {comp.residual_comparison.temporal_comparison.segments.map((seg) => {
                  const dSonics = seg.delta_sonics_probability ?? 0;
                  return (
                    <tr key={seg.window_index} className="hover:bg-slate-900/40">
                      <td className="py-2 px-2.5 text-slate-200 font-semibold">
                        [{seg.start_sec.toFixed(1)}s – {seg.end_sec.toFixed(1)}s]
                      </td>
                      <td className="py-2 px-2.5 text-slate-300">
                        {seg.original_sonics_probability !== null
                          ? `${(seg.original_sonics_probability * 100).toFixed(2)}%`
                          : "N/A"}
                      </td>
                      <td className="py-2 px-2.5 text-cyan-300 font-semibold">
                        {seg.processed_sonics_probability !== null
                          ? `${(seg.processed_sonics_probability * 100).toFixed(2)}%`
                          : "N/A"}
                      </td>
                      <td
                        className={`py-2 px-2.5 font-semibold ${
                          dSonics < 0
                            ? "text-emerald-400"
                            : dSonics > 0
                            ? "text-rose-400"
                            : "text-slate-400"
                        }`}
                      >
                        {seg.delta_sonics_probability !== null
                          ? `${dSonics >= 0 ? "+" : ""}${(dSonics * 100).toFixed(2)}%`
                          : "0.00%"}
                      </td>
                      <td className="py-2 px-2.5 text-purple-300">
                        {seg.original_fakeprint_probability !== null &&
                        seg.processed_fakeprint_probability !== null
                          ? `${(seg.original_fakeprint_probability * 100).toFixed(1)}% → ${(
                              seg.processed_fakeprint_probability * 100
                            ).toFixed(1)}%`
                          : "N/A"}
                      </td>
                      <td className="py-2 px-2.5 text-slate-300">
                        {seg.segment_snr_db.toFixed(1)} dB / {seg.segment_lsd_db.toFixed(2)} dB
                      </td>
                      <td className="py-2 px-2.5 text-slate-400">
                        {seg.segment_rms_delta_db >= 0 ? "+" : ""}
                        {seg.segment_rms_delta_db.toFixed(2)} dB
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {comp.residual_comparison.temporal_comparison.quality_warnings.length > 0 && (
            <div className="rounded border border-rose-500/30 bg-rose-950/20 p-2.5 font-mono text-[11px] text-rose-200 space-y-1">
              {comp.residual_comparison.temporal_comparison.quality_warnings.map((qw, idx) => (
                <div key={idx}>• {qw}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Technical Observations & Warnings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-2">
            Forensic Observations
          </h4>
          <ul className="space-y-1.5 text-xs text-slate-300">
            {comp.observations.map((obs, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-cyan-400">•</span>
                <span>{obs}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-amber-400 mb-2">
            Fidelity & Provenance Warnings
          </h4>
          {comp.warnings.length > 0 ? (
            <ul className="space-y-1.5 text-xs text-amber-200">
              {comp.warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-400">▲</span>
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-emerald-400">
              No clipping, phase misalignment, or severe distortion warnings triggered.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
 * PAGE 6: REPORTS VIEW
 * ============================================================================ */
export function ReportsView({
  results,
  reports,
  onRefreshReports,
}: {
  results: AnalysisResults | null;
  reports: GeneratedReport[];
  onRefreshReports: () => Promise<void>;
}) {
  const [generating, setGenerating] = useState(false);

  const handleCreateReport = async (format: "json" | "pdf") => {
    if (!results) return;
    setGenerating(true);
    try {
      await spectraApi.createReportRecord(results.job.id, {
        report_format: format,
        experiment_id: results.experiments[0]?.id || null,
        include_raw_manifest: true,
      });
      await onRefreshReports();
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Active Job Report Generator */}
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <FileText className="h-4 w-4 text-cyan-400" />
              Reproducible Forensic Report Generator
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Includes SHA-256 asset checksums, C2PA verification, spectral measurements, pinned
              model revisions, DSP audit logs, and scientific caveats.
            </p>
          </div>

          {results && (
            <div className="flex flex-wrap items-center gap-2.5">
              <a
                href={getReportDownloadUrl(
                  results.job.id,
                  "json",
                  results.experiments[0]?.id
                )}
                download
                onClick={() => setTimeout(onRefreshReports, 1000)}
                className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition"
              >
                <Download className="h-4 w-4" />
                Export JSON Report
              </a>

              <a
                href={getReportDownloadUrl(
                  results.job.id,
                  "pdf",
                  results.experiments[0]?.id
                )}
                download
                onClick={() => setTimeout(onRefreshReports, 1000)}
                className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/50 bg-cyan-500/15 px-4 py-2 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/25 transition"
              >
                <Download className="h-4 w-4" />
                Export PDF Technical Report
              </a>
            </div>
          )}
        </div>

        {results && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 border-t border-slate-800 pt-4 font-mono text-xs">
            <div className="rounded border border-slate-800 bg-slate-950 p-3">
              <span className="text-slate-400 block text-[10px]">Active Asset</span>
              <span className="text-slate-100 font-semibold">
                {results.asset.original_filename}
              </span>
              <span className="block text-[10px] text-cyan-400 mt-0.5">
                SHA-256: {results.asset.sha256_checksum.slice(0, 20)}…
              </span>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950 p-3">
              <span className="text-slate-400 block text-[10px]">Pinned Detector Revisions</span>
              {results.detectors.map((d) => (
                <div key={d.id} className="text-[11px] text-slate-300 truncate">
                  {d.detector_id}: {d.model_revision.slice(0, 12)}
                </div>
              ))}
            </div>
            <div className="rounded border border-slate-800 bg-slate-950 p-3">
              <span className="text-slate-400 block text-[10px]">Software Stack</span>
              <span className="text-[11px] text-slate-300 block">
                SPECTRA v{results.software_versions?.spectra || "1.0.0"} • Python{" "}
                {results.software_versions?.python || "3.14"}
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                c2pa {results.software_versions?.c2pa_python} • ONNX{" "}
                {results.software_versions?.onnxruntime}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Stored Reports Table */}
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-slate-100">
            Generated Report Archive ({reports.length})
          </h4>
          {results && (
            <div className="flex gap-2">
              <button
                type="button"
                disabled={generating}
                onClick={() => handleCreateReport("json")}
                className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-700"
              >
                + Snapshot JSON
              </button>
              <button
                type="button"
                disabled={generating}
                onClick={() => handleCreateReport("pdf")}
                className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-700"
              >
                + Snapshot PDF
              </button>
            </div>
          )}
        </div>

        {reports.length === 0 ? (
          <p className="text-xs text-slate-400 py-6 text-center">
            No archived reports yet. Click <strong>Export JSON Report</strong> or{" "}
            <strong>Export PDF Technical Report</strong> above.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse font-mono text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-3">Format</th>
                  <th className="py-2.5 px-3">Report Title</th>
                  <th className="py-2.5 px-3">Report SHA-256</th>
                  <th className="py-2.5 px-3">Size</th>
                  <th className="py-2.5 px-3">Created</th>
                  <th className="py-2.5 px-3 text-right">Download</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {reports.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-900/40">
                    <td className="py-2.5 px-3">
                      <span
                        className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
                          r.report_format === "pdf"
                            ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                            : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                        }`}
                      >
                        {r.report_format}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-200">{r.title}</td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {r.sha256_checksum.slice(0, 16)}…
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {(r.file_size_bytes / 1024).toFixed(1)} KB
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {new Date(r.created_at).toLocaleString()}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <a
                        href={getStoredReportDownloadUrl(r.id)}
                        download
                        className="inline-flex items-center gap-1 rounded border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1 text-[11px] text-cyan-300 hover:bg-cyan-500/20"
                      >
                        <Download className="h-3 w-3" />
                        Download
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================================
 * PAGE 7: SETTINGS & SYSTEM DIAGNOSTICS VIEW
 * ============================================================================ */
export function SettingsView({
  health,
  detectors,
  settingsInfo,
  benchmarkData,
}: {
  health: HealthStatus | null;
  detectors: DetectorInfo[];
  settingsInfo: Record<string, any> | null;
  benchmarkData: any;
}) {
  const comfy = settingsInfo?.comfyui || {};

  return (
    <div className="space-y-6">
      {/* System Hardware & Runtime Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase mb-2">
            <Cpu className="h-4 w-4" />
            Compute & GPU Hardware
          </div>
          <div className="space-y-1 font-mono text-xs text-slate-300">
            <div>CPU Cores: {health?.cpu_count || 16} Logical Processors</div>
            <div>
              GPU: {comfy.gpu_name || "NVIDIA GeForce RTX 4070 Laptop GPU"} (
              {comfy.gpu_vram_mb || 8188} MB VRAM)
            </div>
            <div>
              Default Inference Device:{" "}
              <span className="text-cyan-300 font-semibold">
                {settingsInfo?.default_device || "cpu"}
              </span>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase mb-2">
            <HardDrive className="h-4 w-4" />
            Local Storage & Database
          </div>
          <div className="space-y-1 font-mono text-xs text-slate-300">
            <div>SQLite WAL Status: {health?.database_ok ? "Healthy ✓" : "Error"}</div>
            <div>
              Stored Assets: {settingsInfo?.storage_stats?.total_assets ?? 0} • Jobs:{" "}
              {settingsInfo?.storage_stats?.total_jobs ?? 0}
            </div>
            <div>
              DSP Experiments: {settingsInfo?.storage_stats?.total_experiments ?? 0} • Reports:{" "}
              {settingsInfo?.storage_stats?.total_reports ?? 0}
            </div>
            <div>Max Upload Limit: {settingsInfo?.max_upload_mb || 100} MB</div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
          <div className="flex items-center gap-2 text-violet-400 text-xs font-semibold uppercase mb-2">
            <Shield className="h-4 w-4" />
            Media & C2PA Engine
          </div>
          <div className="space-y-1 font-mono text-xs text-slate-300">
            <div>C2PA SDK: {health?.c2pa_sdk_version || "c2pa-rs 0.91.0"}</div>
            <div>FFmpeg Binary: {health?.ffmpeg_available ? "Available ✓" : "Missing"}</div>
            <div>Formats: WAV, FLAC, MP3, M4A/AAC</div>
            <div>SPECTRA Version: v{health?.version || "1.0.0"}</div>
          </div>
        </div>
      </div>

      {/* Installed Detector Registry & Checksum Verification */}
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Installed AI-Music Detectors & Pinned Revisions
        </h3>
        <div className="space-y-3">
          {detectors.map((d) => (
            <div
              key={d.detector_id}
              className="rounded-lg border border-slate-800 bg-slate-950 p-4 flex flex-wrap items-start justify-between gap-4"
            >
              <div className="space-y-1 max-w-2xl">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-slate-100">{d.name}</span>
                  <span
                    className={`rounded px-2 py-0.5 font-mono text-[10px] font-semibold ${
                      d.available
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        : "bg-slate-800 text-slate-400 border border-slate-700"
                    }`}
                  >
                    {d.available ? "AVAILABLE & VERIFIED" : "OPTIONAL / NOT INSTALLED"}
                  </span>
                </div>
                <p className="text-xs text-slate-400">{d.description}</p>
                <div className="pt-1 flex flex-wrap gap-4 font-mono text-[11px] text-slate-400">
                  <span>ID: {d.detector_id}</span>
                  <span>Rev: {d.model_revision.slice(0, 16)}</span>
                  <span>Requirements: {d.input_duration_requirements}</span>
                  <span>License: {d.license}</span>
                </div>
              </div>
              <div className="font-mono text-xs text-right">
                <div className="text-cyan-300">Device: {d.active_device}</div>
                <div className="text-[11px] text-slate-400 mt-1">{d.status_message}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Model Weight File Checksums */}
        {settingsInfo?.model_files && (
          <div className="mt-4 border-t border-slate-800 pt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              Local Model Weight Files & SHA-256 Integrity
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-1.5 px-2">Relative Path</th>
                    <th className="py-1.5 px-2">Status</th>
                    <th className="py-1.5 px-2">Size</th>
                    <th className="py-1.5 px-2">SHA-256 Checksum</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {settingsInfo.model_files.map((mf: any) => (
                    <tr key={mf.path}>
                      <td className="py-1.5 px-2 text-slate-200">{mf.path}</td>
                      <td className="py-1.5 px-2">
                        {mf.exists ? (
                          <span className="text-emerald-400">Present ✓</span>
                        ) : (
                          <span className="text-rose-400">Missing</span>
                        )}
                      </td>
                      <td className="py-1.5 px-2 text-slate-400">
                        {(mf.size_bytes / 1024).toFixed(1)} KB
                      </td>
                      <td className="py-1.5 px-2 text-cyan-300">
                        {mf.sha256 ? `${mf.sha256.slice(0, 28)}…` : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* ComfyUI Local Integration Status */}
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Layers className="h-4 w-4 text-amber-400" />
              Optional ComfyUI Desktop Integration (Instance &ldquo;Nyx&rdquo;)
            </h3>
            <p className="text-xs text-slate-400">
              Strictly read-only discovery & loopback API adapter — existing ComfyUI installation is
              never modified
            </p>
          </div>
          <span
            className={`rounded px-2.5 py-1 font-mono text-xs font-semibold ${
              comfy.service_reachable
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                : "bg-slate-800 text-slate-300 border border-slate-700"
            }`}
          >
            {comfy.service_reachable
              ? `ONLINE (${comfy.api_url})`
              : `OFFLINE STANDBY (${comfy.api_url || "http://127.0.0.1:8188"})`}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950 p-3 space-y-1 text-slate-300">
            <div>Desktop App Version: v{comfy.desktop_version || "1.0.39.0"}</div>
            <div>ComfyUI Core Version: v{comfy.core_version || "0.33.1"}</div>
            <div>Discovered Audio Nodes: {(comfy.available_audio_nodes || []).length}</div>
            <div className="text-[11px] text-amber-300/90 pt-1">{comfy.notes}</div>
          </div>

          <div className="rounded border border-slate-800 bg-slate-950 p-3">
            <div className="text-slate-400 text-[11px] mb-1.5">
              Discovered Local Audio Checkpoints ({(comfy.audio_models || []).length}):
            </div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {(comfy.audio_models || []).map((m: any, i: number) => (
                <div key={i} className="flex justify-between text-[11px]">
                  <span className="text-slate-200 truncate">{m.name}</span>
                  <span className="text-cyan-400 shrink-0 ml-2">
                    {m.size_mb} MB ({m.category})
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Offline Benchmark Summary */}
      {benchmarkData?.available && benchmarkData.data && (
        <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-cyan-400" />
            Offline Scientific Benchmark Cohort Summary ({benchmarkData.data.benchmark_suite})
          </h3>
          <p className="text-xs text-slate-400">
            Evaluates both a 36-track synthetic DSP/spectral verification cohort (with disjoint
            calibration/test generators) and an external real-world music recording cohort.
          </p>
          {benchmarkData.data.scientific_validity_notice && (
            <div className="rounded border border-amber-500/30 bg-amber-950/20 p-3 text-[11px] text-amber-200">
              <strong>External Validity & Scope Notice:</strong>{" "}
              {benchmarkData.data.scientific_validity_notice}
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
            {Object.entries(
              benchmarkData.data.models || benchmarkData.data.detectors || {}
            ).map(([detId, dObj]: [string, any]) => (
              <div key={detId} className="rounded border border-slate-800 bg-slate-950 p-3.5">
                <div className="font-semibold text-cyan-300 mb-1.5">{detId}</div>
                <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-300">
                  <span>Synthetic AUROC: {dObj.sealed_test_calibrated?.auroc}</span>
                  <span>Calibrated F1: {dObj.sealed_test_calibrated?.f1_score}</span>
                  <span>Calibrated TPR: {dObj.sealed_test_calibrated?.recall_tpr}</span>
                  <span>Calibrated FPR: {dObj.sealed_test_calibrated?.fpr}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
