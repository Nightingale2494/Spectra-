"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  Upload,
  Microscope,
  Sliders,
  GitCompare,
  FileText,
  Settings,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  CheckCircle2,
  AlertCircle,
  Play,
  Download,
  RefreshCw,
  FileAudio,
  Sparkles,
  Lock,
  Cpu,
  Radio,
} from "lucide-react";
import {
  AnalysisResults,
  DetectorInfo,
  DspPreset,
  GeneratedReport,
  HealthStatus,
  SpectrogramData,
  WaveformData,
} from "@/types/spectra";
import {
  getAudioStreamUrl,
  getJobArtifactDownloadUrl,
  getReportDownloadUrl,
  spectraApi,
} from "@/lib/api";
import AudioWorkstationPlayer, { SeekRangeRequest } from "@/components/AudioWorkstationPlayer";
import SpectrogramCanvas from "@/components/SpectrogramCanvas";
import { FakeprintInspectorChart } from "@/components/ForensicCharts";
import TemporalForensicTimeline from "@/components/TemporalForensicTimeline";
import {
  ComparisonView,
  DspLaboratoryView,
  ReportsView,
  SettingsView,
} from "@/components/WorkstationViews";

type NavPage =
  | "dashboard"
  | "upload"
  | "forensics"
  | "dsp"
  | "comparison"
  | "reports"
  | "settings";

export default function SpectraWorkstationApp() {
  const [activePage, setActivePage] = useState<NavPage>("dashboard");

  // Global System State
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [playerCurrentTime, setPlayerCurrentTime] = useState<number>(0);
  const [playerIsPlaying, setPlayerIsPlaying] = useState<boolean>(false);
  const [seekRequest, setSeekRequest] = useState<SeekRangeRequest | null>(null);
  const [presets, setPresets] = useState<DspPreset[]>([]);
  const [samples, setSamples] = useState<
    Array<{ sample_id: string; filename: string; title: string; description: string }>
  >([]);
  const [recentJobs, setRecentJobs] = useState<any[]>([]);
  const [reports, setReports] = useState<GeneratedReport[]>([]);
  const [settingsInfo, setSettingsInfo] = useState<Record<string, any> | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);

  // Active Job & Forensic Payload State
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [waveform, setWaveform] = useState<WaveformData | null>(null);
  const [spectrogram, setSpectrogram] = useState<SpectrogramData | null>(null);

  // Loading & Error States
  const [loadingJob, setLoadingJob] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [metaExporting, setMetaExporting] = useState(false);
  const [metaExportResult, setMetaExportResult] = useState<any>(null);

  const refreshGlobalData = useCallback(async () => {
    try {
      const [h, dets, pres, smps, jobs, reps, sInfo, bench] = await Promise.all([
        spectraApi.getHealth(),
        spectraApi.getDetectors(),
        spectraApi.getDspPresets(),
        spectraApi.getReferenceSamples(),
        spectraApi.listJobs(25),
        spectraApi.listReports(),
        spectraApi.getSettingsInfo(),
        spectraApi.getLatestBenchmarks(),
      ]);
      setHealth(h);
      setDetectors(dets);
      setPresets(pres);
      setSamples(smps);
      setRecentJobs(jobs);
      setReports(reps);
      setSettingsInfo(sInfo);
      setBenchmarkData(bench);
    } catch (err) {
      console.error("Failed to fetch initial SPECTRA system telemetry:", err);
    }
  }, []);

  const loadJobFullDetails = useCallback(async (jobId: string) => {
    setLoadingJob(true);
    setActiveJobId(jobId);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem("spectra_active_job_id", jobId);
      } catch {}
    }
    setMetaExportResult(null);
    try {
      const res = await spectraApi.getJobResults(jobId);
      setResults(res);
      const [wf, spec] = await Promise.all([
        spectraApi.getWaveform(res.asset.id, 650),
        spectraApi.getSpectrogram(res.asset.id),
      ]);
      setWaveform(wf);
      setSpectrogram(spec);
    } catch (err) {
      console.error("Error loading job details:", err);
    } finally {
      setLoadingJob(false);
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        const savedPage = window.localStorage.getItem("spectra_active_page") as NavPage | null;
        if (
          savedPage &&
          ["dashboard", "upload", "forensics", "dsp", "comparison", "reports", "settings"].includes(savedPage)
        ) {
          setActivePage(savedPage);
        }
      } catch {}
    }

    refreshGlobalData().then(() => {
      spectraApi
        .listJobs(25)
        .then((jobs) => {
          let targetJobId: string | null = null;
          if (typeof window !== "undefined") {
            try {
              const savedJobId = window.localStorage.getItem("spectra_active_job_id");
              if (savedJobId && jobs.some((j: any) => j.job?.id === savedJobId)) {
                targetJobId = savedJobId;
              }
            } catch {}
          }
          if (!targetJobId && jobs.length > 0 && jobs[0].job?.id) {
            targetJobId = jobs[0].job.id;
          }
          if (targetJobId) {
            loadJobFullDetails(targetJobId);
          }
        })
        .catch(() => {});
    });
  }, [refreshGlobalData, loadJobFullDetails]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem("spectra_active_page", activePage);
      } catch {}
    }
  }, [activePage]);

  const handleLoadReferenceSample = async (sampleId: string) => {
    setUploading(true);
    setUploadError(null);
    try {
      const resp = await spectraApi.loadReferenceSample(sampleId, true);
      await refreshGlobalData();
      await loadJobFullDetails(resp.initial_job_id);
      setActivePage("forensics");
    } catch (err: any) {
      setUploadError(err.message || "Failed to load reference sample.");
    } finally {
      setUploading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const resp = await spectraApi.uploadAudio(file, true);
      await refreshGlobalData();
      await loadJobFullDetails(resp.initial_job_id);
      setActivePage("forensics");
    } catch (err: any) {
      setUploadError(err.message || "Upload failed validation.");
    } finally {
      setUploading(false);
    }
  };

  const handleMetadataCleanExport = async (stripC2pa: boolean) => {
    if (!results) return;
    setMetaExporting(true);
    try {
      const out = await spectraApi.exportCleanMetadata(results.job.id, {
        strip_conventional_metadata: true,
        strip_c2pa_manifest: stripC2pa,
      });
      await loadJobFullDetails(results.job.id);
      setMetaExportResult(out);
    } catch (err: any) {
      alert(err.message || "Metadata clean export failed.");
    } finally {
      setMetaExporting(false);
    }
  };

  const renderC2paBadge = (status?: string | null) => {
    switch (status) {
      case "manifest_valid":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-emerald-300">
            <ShieldCheck className="h-3.5 w-3.5" />
            C2PA Manifest Valid
          </span>
        );
      case "manifest_invalid":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-rose-300">
            <ShieldAlert className="h-3.5 w-3.5" />
            C2PA Tampered / Invalid
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-800/80 px-2.5 py-0.5 font-mono text-[11px] text-slate-300">
            <ShieldQuestion className="h-3.5 w-3.5 text-slate-400" />
            No C2PA Manifest
          </span>
        );
    }
  };

  const navItems: Array<{ id: NavPage; label: string; icon: any; badge?: string }> = [
    { id: "dashboard", label: "1. Laboratory Dashboard", icon: Activity },
    { id: "upload", label: "2. Audio Ingestion", icon: Upload },
    {
      id: "forensics",
      label: "3. Forensic Analysis",
      icon: Microscope,
      badge: results ? "ACTIVE" : undefined,
    },
    { id: "dsp", label: "4. DSP Laboratory", icon: Sliders },
    {
      id: "comparison",
      label: "5. A/B Comparison",
      icon: GitCompare,
      badge:
        results && results.comparisons.length > 0
          ? String(results.comparisons.length)
          : undefined,
    },
    {
      id: "reports",
      label: "6. Technical Reports",
      icon: FileText,
      badge: reports.length > 0 ? String(reports.length) : undefined,
    },
    { id: "settings", label: "7. System & Models", icon: Settings },
  ];

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-[#07090e] text-slate-100 workstation-grid-bg">
      {/* Left Navigation Sidebar */}
      <aside className="w-full lg:w-64 shrink-0 border-b lg:border-b-0 lg:border-r border-slate-800/90 bg-[#0a0d14]/95 backdrop-blur flex flex-col justify-between">
        <div>
          {/* Brand Header */}
          <div className="p-5 border-b border-slate-800/90">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 text-slate-950 shadow-md shadow-cyan-500/20">
                <Radio className="h-5 w-5" />
              </div>
              <div>
                <h1 className="font-mono text-base font-bold tracking-wider text-white">
                  SPECTRA
                </h1>
                <p className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                  Audio Forensics Lab
                </p>
              </div>
            </div>
          </div>

          {/* Navigation Menu */}
          <nav className="p-3 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActivePage(item.id)}
                  className={`w-full flex items-center justify-between rounded-lg px-3 py-2.5 text-xs font-medium transition ${
                    isActive
                      ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-sm"
                      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200 border border-transparent"
                  }`}
                >
                  <span className="flex items-center gap-2.5">
                    <Icon
                      className={`h-4 w-4 ${
                        isActive ? "text-cyan-400" : "text-slate-400"
                      }`}
                    />
                    {item.label}
                  </span>
                  {item.badge && (
                    <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-cyan-300">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Active Track Mini-Card & Telemetry */}
        <div className="p-4 border-t border-slate-800/90 space-y-3 text-xs">
          {results ? (
            <div className="rounded-lg border border-slate-800 bg-slate-950/90 p-3">
              <div className="text-[10px] font-mono uppercase text-cyan-400 mb-1">
                Active Specimen
              </div>
              <div className="font-semibold text-slate-200 truncate">
                {results.asset.original_filename}
              </div>
              <div className="mt-1 font-mono text-[11px] text-slate-400">
                {results.asset.original_format} • {results.asset.sample_rate} Hz •{" "}
                {results.asset.duration_seconds.toFixed(2)}s
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 p-3 text-slate-400 text-[11px]">
              No active specimen loaded.
            </div>
          )}

          <div className="flex items-center justify-between font-mono text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <span
                className={`h-2 w-2 rounded-full ${
                  health?.status === "healthy" ? "bg-emerald-400" : "bg-amber-400"
                }`}
              />
              Core API {health?.status || "connecting"}
            </span>
            <span>
              Detectors: {health?.detectors_ready ?? 2}/{health?.detectors_total ?? 3}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top Workstation Status Bar */}
        <header className="border-b border-slate-800/90 bg-[#0a0d14]/90 backdrop-blur px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-100 uppercase tracking-wider font-mono">
              {navItems.find((n) => n.id === activePage)?.label}
            </h2>
            {loadingJob && (
              <span className="inline-flex items-center gap-1.5 rounded bg-cyan-500/10 px-2 py-0.5 font-mono text-xs text-cyan-300">
                <RefreshCw className="h-3 w-3 animate-spin" />
                Computing STFT & Forensics...
              </span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            {samples.map((s) => (
              <button
                key={s.sample_id}
                type="button"
                disabled={uploading}
                onClick={() => handleLoadReferenceSample(s.sample_id)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900/90 px-2.5 py-1.5 text-xs font-mono text-slate-200 hover:border-cyan-500/50 hover:text-cyan-300 transition"
                title={s.description}
              >
                <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
                Load: {s.title.split("(")[0].trim()}
              </button>
            ))}
          </div>
        </header>

        {/* Page Content Container */}
        <div className="p-6 max-w-7xl w-full mx-auto space-y-6">
          {/* ================================================================
           * PAGE 1: DASHBOARD VIEW
           * ================================================================ */}
          {activePage === "dashboard" && (
            <div className="space-y-6">
              {/* Five Pillars Scientific Distinction Banner */}
              <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                    <Microscope className="h-4 w-4 text-cyan-400" />
                    SPECTRA Multi-Layer Forensic Architecture (Five Distinct Mechanisms)
                  </h3>
                  <span className="font-mono text-[11px] text-slate-400">
                    No single detector output is presented as definitive proof of content origin
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
                  <div className="rounded-lg border border-slate-800/90 bg-slate-950/70 p-3">
                    <div className="font-mono font-semibold text-emerald-400 mb-1">
                      1. C2PA Provenance
                    </div>
                    <p className="text-slate-400 text-[11px]">
                      Cryptographically signed JUMBF manifests verifying content creation history.
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-800/90 bg-slate-950/70 p-3">
                    <div className="font-mono font-semibold text-cyan-400 mb-1">
                      2. Conventional Tags
                    </div>
                    <p className="text-slate-400 text-[11px]">
                      Mutable ID3, Vorbis, RIFF INFO, and MP4 container metadata fields.
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-800/90 bg-slate-950/70 p-3">
                    <div className="font-mono font-semibold text-violet-400 mb-1">
                      3. Audio Watermarks
                    </div>
                    <p className="text-slate-400 text-[11px]">
                      Intentionally embedded psychoacoustic or neural identification signals.
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-800/90 bg-slate-950/70 p-3">
                    <div className="font-mono font-semibold text-amber-400 mb-1">
                      4. Spectral Artifacts
                    </div>
                    <p className="text-slate-400 text-[11px]">
                      Physical 1–8 kHz neural-vocoder comb peaks & lower-hull residuals.
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-800/90 bg-slate-950/70 p-3">
                    <div className="font-mono font-semibold text-rose-400 mb-1">
                      5. AI Classifiers
                    </div>
                    <p className="text-slate-400 text-[11px]">
                      Statistical representations from ONNX LR & SONICS SpecTTTra-γ models.
                    </p>
                  </div>
                </div>
              </div>

              {/* Quick Reference Sample Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {samples.map((s) => (
                  <div
                    key={s.sample_id}
                    className="rounded-xl border border-slate-800 bg-[#0d111a] p-4 flex flex-col justify-between hover:border-cyan-500/40 transition"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="rounded bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 font-mono text-[10px] text-cyan-300">
                          Reference Specimen
                        </span>
                        <span className="font-mono text-[11px] text-slate-400">
                          {s.filename}
                        </span>
                      </div>
                      <h4 className="text-sm font-semibold text-slate-100">{s.title}</h4>
                      <p className="mt-1 text-xs text-slate-400 leading-relaxed">
                        {s.description}
                      </p>
                    </div>
                    <button
                      type="button"
                      disabled={uploading}
                      onClick={() => handleLoadReferenceSample(s.sample_id)}
                      className="mt-4 w-full inline-flex items-center justify-center gap-2 rounded-lg bg-slate-800 hover:bg-cyan-500 hover:text-slate-950 px-3 py-2 text-xs font-semibold text-slate-100 transition"
                    >
                      <Play className="h-3.5 w-3.5" />
                      Load & Run Full Forensic Analysis
                    </button>
                  </div>
                ))}
              </div>

              {/* Recent Forensic Jobs Table */}
              <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Recent Forensic Analysis Jobs ({recentJobs.length})
                    </h3>
                    <p className="text-xs text-slate-400">
                      Click any job to open its interactive spectrogram, fakeprint curve, C2PA
                      manifest, and DSP lab
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActivePage("upload")}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition"
                  >
                    <Upload className="h-3.5 w-3.5" />
                    Upload New Audio
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse font-mono text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2.5 px-3">Specimen Filename</th>
                        <th className="py-2.5 px-3">Format / SR</th>
                        <th className="py-2.5 px-3">C2PA Provenance</th>
                        <th className="py-2.5 px-3">Fakeprint ONNX</th>
                        <th className="py-2.5 px-3">SONICS SpecTTTra-γ</th>
                        <th className="py-2.5 px-3">DSP Runs</th>
                        <th className="py-2.5 px-3 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70">
                      {recentJobs.map((item) => {
                        const j = item.job;
                        const a = item.asset;
                        const fp = (item.detectors || []).find(
                          (d: any) => d.detector_id === "fakeprint_onnx"
                        );
                        const sn = (item.detectors || []).find(
                          (d: any) => d.detector_id === "sonics_gamma_5s"
                        );
                        const isSelected = activeJobId === j.id;
                        return (
                          <tr
                            key={j.id}
                            className={`hover:bg-slate-900/60 transition ${
                              isSelected ? "bg-cyan-950/20" : ""
                            }`}
                          >
                            <td className="py-2.5 px-3 font-semibold text-slate-100">
                              {a?.original_filename || "Unknown"}
                            </td>
                            <td className="py-2.5 px-3 text-slate-400">
                              {a?.original_format} • {a?.sample_rate} Hz •{" "}
                              {a?.duration_seconds?.toFixed(1)}s
                            </td>
                            <td className="py-2.5 px-3">
                              {renderC2paBadge(item.c2pa_status)}
                            </td>
                            <td className="py-2.5 px-3">
                              {fp && fp.probability !== null ? (
                                <span
                                  className={
                                    fp.probability >= 0.5
                                      ? "text-amber-300 font-semibold"
                                      : "text-emerald-300"
                                  }
                                >
                                  {(fp.probability * 100).toFixed(1)}%
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>
                            <td className="py-2.5 px-3">
                              {sn && sn.probability !== null ? (
                                <span className="text-cyan-300">
                                  {(sn.probability * 100).toFixed(1)}%
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {item.experiments_count || 0}
                            </td>
                            <td className="py-2.5 px-3 text-right">
                              <button
                                type="button"
                                onClick={async () => {
                                  await loadJobFullDetails(j.id);
                                  setActivePage("forensics");
                                }}
                                className="rounded border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1 text-[11px] font-semibold text-cyan-300 hover:bg-cyan-500 hover:text-slate-950 transition"
                              >
                                Inspect →
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================
           * PAGE 2: AUDIO UPLOAD & INGESTION VIEW
           * ================================================================ */}
          {activePage === "upload" && (
            <div className="space-y-6">
              <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-6">
                <h3 className="text-base font-semibold text-slate-100 mb-1">
                  Secure Forensic Audio Ingestion
                </h3>
                <p className="text-xs text-slate-400 mb-5">
                  Validates magic container signatures (RIFF/WAV, FLAC, ID3/MP3, ftyp/M4A), computes
                  immutable SHA-256 checksums, extracts C2PA manifests and Mutagen tags, and launches
                  multi-detector spectral analysis.
                </p>

                <label
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                      handleFileUpload(e.dataTransfer.files[0]);
                    }
                  }}
                  className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950/60 p-10 text-center cursor-pointer hover:border-cyan-500/60 hover:bg-slate-900/40 transition"
                >
                  <FileAudio className="h-12 w-12 text-cyan-400 mb-3" />
                  <span className="text-sm font-semibold text-slate-200">
                    {uploading
                      ? "Validating Container Signature & Running Forensic Pipeline..."
                      : "Drag & Drop Audio Specimen Here, or Click to Browse"}
                  </span>
                  <span className="mt-1 font-mono text-xs text-slate-400">
                    Supported formats: .WAV, .FLAC, .MP3, .M4A / .AAC (Up to{" "}
                    {settingsInfo?.max_upload_mb || 100} MB)
                  </span>
                  <input
                    type="file"
                    accept=".wav,.flac,.mp3,.m4a,.aac,audio/*"
                    disabled={uploading}
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFileUpload(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                  />
                </label>

                {uploadError && (
                  <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-950/30 p-3.5 text-xs text-rose-200 flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-rose-400 shrink-0" />
                    <span>{uploadError}</span>
                  </div>
                )}
              </div>

              {results && (
                <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-slate-100">
                      Currently Ingested Specimen Technical Profile
                    </h4>
                    <button
                      type="button"
                      onClick={() => setActivePage("forensics")}
                      className="rounded-lg bg-cyan-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400"
                    >
                      Open Full Forensic Analysis →
                    </button>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                    <div className="rounded border border-slate-800 bg-slate-950 p-3">
                      <span className="text-slate-400 block text-[10px]">Format / Codec</span>
                      <span className="text-slate-100 font-semibold">
                        {results.asset.original_format} ({results.asset.codec})
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-3">
                      <span className="text-slate-400 block text-[10px]">Sample Rate / Ch</span>
                      <span className="text-slate-100 font-semibold">
                        {results.asset.sample_rate} Hz • {results.asset.channels}ch
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-3">
                      <span className="text-slate-400 block text-[10px]">Duration / Size</span>
                      <span className="text-slate-100 font-semibold">
                        {results.asset.duration_seconds.toFixed(2)}s •{" "}
                        {(results.asset.file_size_bytes / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-3">
                      <span className="text-slate-400 block text-[10px]">SHA-256 Checksum</span>
                      <span className="text-cyan-300 font-semibold truncate block">
                        {results.asset.sha256_checksum.slice(0, 18)}…
                      </span>
                    </div>
                  </div>

                  <AudioWorkstationPlayer
                    title={results.asset.original_filename}
                    subtitle={`SHA-256: ${results.asset.sha256_checksum}`}
                    audioUrl={getAudioStreamUrl(results.asset.id)}
                    waveform={waveform}
                    downloadUrl={getJobArtifactDownloadUrl(results.job.id, "original")}
                  />
                </div>
              )}
            </div>
          )}

          {/* ================================================================
           * PAGE 3: FORENSIC ANALYSIS VIEW
           * ================================================================ */}
          {activePage === "forensics" && results && (
            <div className="space-y-6">
              {/* Interactive Workstation Player */}
              <AudioWorkstationPlayer
                title={results.asset.original_filename}
                subtitle={`${results.asset.original_format} (${results.asset.codec}) • ${results.asset.sample_rate} Hz • ${results.asset.channels}ch • SHA-256: ${results.asset.sha256_checksum.slice(0, 24)}…`}
                audioUrl={getAudioStreamUrl(results.asset.id)}
                waveform={waveform}
                downloadUrl={getJobArtifactDownloadUrl(results.job.id, "original")}
                externalSeekRequest={seekRequest}
                onTimeChange={(t, playing) => {
                  setPlayerCurrentTime(t);
                  setPlayerIsPlaying(playing);
                }}
              />

              {/* Section A: C2PA Content Credentials & Conventional Metadata */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* C2PA Provenance Panel */}
                <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">
                        1. C2PA Content Credentials Inspection
                      </h3>
                      <p className="text-xs text-slate-400 font-mono">
                        Official c2pa-python SDK ({health?.c2pa_sdk_version || "0.91.0"})
                      </p>
                    </div>
                    {renderC2paBadge(results.provenance?.c2pa_status)}
                  </div>

                  <div className="rounded-lg border border-slate-800/90 bg-slate-950 p-3 text-xs text-slate-300">
                    <strong className="text-cyan-300">Scientific Provenance Notice:</strong>{" "}
                    {results.provenance?.scientific_notice}
                  </div>

                  {results.provenance?.credential_present ? (
                    <div className="space-y-2.5 font-mono text-xs">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                          <span className="text-[10px] text-slate-400 block">
                            Active Manifest ID
                          </span>
                          <span className="text-slate-100 truncate block">
                            {results.provenance.active_manifest_id}
                          </span>
                        </div>
                        <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                          <span className="text-[10px] text-slate-400 block">
                            Claim Generator
                          </span>
                          <span className="text-cyan-300 truncate block">
                            {results.provenance.claim_generator || "Unknown"}
                          </span>
                        </div>
                      </div>

                      {results.provenance.certificate_info && (
                        <div className="rounded border border-slate-800 bg-slate-900/50 p-2.5">
                          <span className="text-[10px] text-slate-400 block">
                            Signing Certificate & Algorithm
                          </span>
                          <span className="text-slate-200">
                            Issuer: {results.provenance.certificate_info.issuer || "Self/Test"} •
                            Alg: {results.provenance.certificate_info.alg || "ES256"}
                          </span>
                        </div>
                      )}

                      {results.provenance.validation_errors.length > 0 && (
                        <div className="rounded border border-rose-500/40 bg-rose-950/30 p-2.5 text-rose-200">
                          <div className="font-semibold text-[11px] mb-1">
                            Cryptographic Validation Failures:
                          </div>
                          {results.provenance.validation_errors.map((e, i) => (
                            <div key={i}>• {e}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 text-xs text-slate-400">
                      No embedded JUMBF C2PA manifest was found in this audio container. Most
                      standard DAW exports and consumer audio recordings do not embed C2PA manifests;
                      therefore, manifest absence is neutral and never treated as evidence of human or
                      AI origin.
                    </div>
                  )}
                </div>

                {/* Conventional Metadata & Authorized Clean Export Panel */}
                <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 flex flex-col justify-between space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-100">
                          2. Conventional Container Metadata (ID3 / RIFF / Vorbis / MP4)
                        </h3>
                        <p className="text-xs text-slate-400 font-mono">
                          Inspected independently via Mutagen • Format:{" "}
                          {results.provenance?.conventional_metadata?.format_family ||
                            results.asset.original_format}
                        </p>
                      </div>
                      <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-300">
                        {results.provenance?.conventional_metadata?.tag_count ?? 0} tags
                      </span>
                    </div>

                    {(() => {
                      const cm = results.provenance?.conventional_metadata || {};
                      const rawTags =
                        cm.tags && typeof cm.tags === "object"
                          ? cm.tags
                          : Object.fromEntries(
                              Object.entries(cm).filter(
                                ([k]) => !["format_family", "tag_count", "tags"].includes(k)
                              )
                            );
                      const entries = Object.entries(rawTags);
                      return entries.length > 0 ? (
                        <div className="max-h-40 overflow-y-auto rounded border border-slate-800 bg-slate-950 p-3 font-mono text-xs space-y-1.5">
                          {entries.map(([k, v]) => (
                            <div key={k} className="flex justify-between gap-4">
                              <span className="text-cyan-400 shrink-0">{k}:</span>
                              <span className="text-slate-200 truncate">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="rounded border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400 font-mono">
                          No mutable ID3/RIFF/Vorbis tags present in container header.
                        </div>
                      );
                    })()}
                  </div>

                  {/* Authorized Metadata-Clean Export Controls */}
                  <div className="border-t border-slate-800 pt-3 space-y-2.5">
                    <div className="text-xs font-semibold text-slate-200">
                      Authorized Metadata-Clean Export (Preserves Immutable Original)
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        disabled={metaExporting}
                        onClick={() => handleMetadataCleanExport(false)}
                        className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 transition"
                      >
                        Strip Conventional Tags Only
                      </button>
                      <button
                        type="button"
                        disabled={metaExporting}
                        onClick={() => handleMetadataCleanExport(true)}
                        className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-300 hover:bg-amber-500/20 transition"
                      >
                        Clean Export (Strip Tags & Container Chunks)
                      </button>
                    </div>
                    {metaExportResult && (
                      <div className="rounded border border-emerald-500/40 bg-emerald-950/30 p-2.5 font-mono text-[11px] text-emerald-200 flex items-center justify-between">
                        <span>
                          Created Clean Export:{" "}
                          {metaExportResult.processed_asset.sanitized_filename}
                        </span>
                        <a
                          href={getAudioStreamUrl(metaExportResult.processed_asset.id)}
                          download
                          className="underline text-emerald-300 font-semibold"
                        >
                          Download Clean File
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Section B: Multi-Detector AI Classification Panel */}
              <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Multi-Detector AI-Music Classification & Spectral Artifact Scoring
                    </h3>
                    <p className="text-xs text-slate-400">
                      Independent execution of frequency-domain residual and spectro-temporal
                      transformer detectors
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActivePage("dsp")}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500/15 border border-cyan-500/40 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/25"
                  >
                    <Sliders className="h-3.5 w-3.5" />
                    Test Sensitivity in DSP Lab →
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.detectors.map((det) => {
                    const probPct =
                      det.probability !== null ? (det.probability * 100).toFixed(2) : "N/A";
                    const isAbstained = Boolean(det.details?.abstained || det.details?.ood_detected);
                    const isHigh = !isAbstained && (det.probability ?? 0) >= det.threshold;
                    return (
                      <div
                        key={det.id}
                        className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <h4 className="text-sm font-semibold text-slate-100">
                              {det.detector_name}
                            </h4>
                            <p className="font-mono text-[11px] text-slate-400">
                              Rev: {det.model_revision.slice(0, 14)} • Device: {det.device_used} •{" "}
                              {det.processing_time_ms.toFixed(1)} ms
                            </p>
                          </div>
                          <span
                            className={`rounded-full px-2.5 py-0.5 font-mono text-xs font-bold ${
                              isAbstained
                                ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                                : isHigh
                                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                            }`}
                          >
                            {isAbstained ? `ABSTAINED (Raw: ${probPct}%)` : `${probPct}%`}
                          </span>
                        </div>

                        {/* Probability Progress Bar */}
                        <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full transition-all ${
                              isAbstained
                                ? "bg-rose-400/70"
                                : isHigh
                                ? "bg-amber-400"
                                : "bg-emerald-400"
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(2, (det.probability ?? 0) * 100))}%`,
                            }}
                          />
                        </div>

                        <div className="flex items-center justify-between font-mono text-xs">
                          <span className={isAbstained ? "text-rose-300 font-semibold" : "text-slate-300"}>
                            Label: {det.predicted_label}
                          </span>
                          <span className="text-slate-400">
                            Threshold: {(det.threshold * 100).toFixed(0)}%
                          </span>
                        </div>

                        {det.details && (
                          <div className="rounded border border-slate-800/80 bg-slate-900/50 p-2 font-mono text-[11px] text-slate-400 flex flex-wrap gap-x-4 gap-y-1">
                            {det.details.calibrated_probability !== undefined && (
                              <span>
                                Calibrated:{" "}
                                <strong className="text-cyan-300">
                                  {(Number(det.details.calibrated_probability) * 100).toFixed(1)}%
                                </strong>
                              </span>
                            )}
                            {det.details.prediction_entropy_bits !== undefined && (
                              <span>
                                Entropy H:{" "}
                                <strong className="text-slate-200">
                                  {Number(det.details.prediction_entropy_bits).toFixed(3)} bits
                                </strong>
                              </span>
                            )}
                            {det.details.crop_policy && (
                              <span>
                                Policy:{" "}
                                <strong className="text-slate-200">
                                  {String(det.details.crop_policy)} ({det.details.windows_evaluated ?? 1} win)
                                </strong>
                              </span>
                            )}
                            {det.details.center_window_prob !== undefined && (
                              <span>
                                Center: {(Number(det.details.center_window_prob) * 100).toFixed(1)}%
                              </span>
                            )}
                            {det.details.multi_window_iqr_prob !== undefined &&
                              Number(det.details.windows_evaluated ?? 1) > 1 && (
                                <span>
                                  Window IQR: ±{(Number(det.details.multi_window_iqr_prob) * 50).toFixed(1)}%
                                </span>
                              )}
                            {det.details.temporal_stationarity !== undefined && (
                              <span>
                                Stationarity: {Number(det.details.temporal_stationarity).toFixed(2)}
                              </span>
                            )}
                          </div>
                        )}

                        {det.details?.ood_warning && (
                          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-2.5 text-[11px] text-rose-200 leading-relaxed">
                            {det.details.ood_warning}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Section B.2: Interactive Temporal AI-Detection Timeline (Phase 5 Part A) */}
              <TemporalForensicTimeline
                jobId={results.job.id}
                waveform={waveform}
                currentTime={playerCurrentTime}
                isPlaying={playerIsPlaying}
                onSeekAndPlayRange={(startSec, endSec, autoPlay) => {
                  setSeekRequest({
                    startSec,
                    endSec,
                    autoPlay: autoPlay ?? true,
                    nonce: Date.now(),
                  });
                }}
              />

              {/* Section C: 2D STFT Spectrogram Canvas */}
              <SpectrogramCanvas spectrogram={spectrogram} />

              {/* Section D: 1-8 kHz Neural-Vocoder Fakeprint & Hull Visualization */}
              <FakeprintInspectorChart spectrogram={spectrogram} />

              {/* Section E: Time & Frequency Domain Signal Metrics + Forensic Observations */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-[#0d111a] p-5">
                  <h4 className="text-sm font-semibold text-slate-100 mb-3">
                    Time-Domain & Frequency-Domain Signal Forensics
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">RMS Level</span>
                      <span className="text-slate-100 font-semibold">
                        {results.forensics.time_domain?.rms_db ??
                          results.forensics.time_domain?.rms_dbfs}{" "}
                        dBFS
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Peak Level</span>
                      <span className="text-slate-100 font-semibold">
                        {results.forensics.time_domain?.peak_db ??
                          results.forensics.time_domain?.peak_dbfs}{" "}
                        dBFS
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Crest Factor</span>
                      <span className="text-slate-100 font-semibold">
                        {results.forensics.time_domain?.crest_factor_db} dB
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Zero-Crossing Rate</span>
                      <span className="text-slate-100 font-semibold">
                        {results.forensics.time_domain?.zero_crossing_rate ??
                          results.forensics.time_domain?.dc_offset}
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Spectral Centroid</span>
                      <span className="text-cyan-300 font-semibold">
                        {results.forensics.frequency_domain?.spectral_centroid_hz ??
                          results.forensics.frequency_domain?.spectral_centroid_hz_mean}{" "}
                        Hz
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Spectral Roll-off (95%)</span>
                      <span className="text-cyan-300 font-semibold">
                        {results.forensics.frequency_domain?.spectral_rolloff_95_hz} Hz
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Effective Cutoff</span>
                      <span className="text-amber-300 font-semibold">
                        {results.forensics.frequency_domain?.effective_bandwidth_cutoff_hz ??
                          results.forensics.frequency_domain?.spectral_rolloff_95_hz}{" "}
                        Hz
                      </span>
                    </div>
                    <div className="rounded border border-slate-800 bg-slate-950 p-2.5">
                      <span className="text-slate-400 block text-[10px]">Comb Periodicity</span>
                      <span className="text-rose-300 font-semibold">
                        {results.forensics.fakeprint?.comb_periodicity_score ??
                          results.forensics.fakeprint?.periodicity_score}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-5">
                  <h4 className="text-sm font-semibold text-slate-100 mb-3">
                    Automated Forensic Observations
                  </h4>
                  <ul className="space-y-2 text-xs text-slate-300 leading-relaxed">
                    {(results.forensics.observations || []).map((obs, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-cyan-400 font-bold">•</span>
                        <span>{obs}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================
           * PAGES 4, 5, 6, 7
           * ================================================================ */}
          {activePage === "dsp" && (
            <DspLaboratoryView
              results={results}
              waveform={waveform}
              presets={presets}
              onExperimentCompleted={async () => {
                if (activeJobId) {
                  await loadJobFullDetails(activeJobId);
                  await refreshGlobalData();
                }
              }}
              onNavigateComparison={() => setActivePage("comparison")}
            />
          )}

          {activePage === "comparison" && (
            <ComparisonView
              results={results}
              onNavigateDsp={() => setActivePage("dsp")}
            />
          )}

          {activePage === "reports" && (
            <ReportsView
              results={results}
              reports={reports}
              onRefreshReports={refreshGlobalData}
            />
          )}

          {activePage === "settings" && (
            <SettingsView
              health={health}
              detectors={detectors}
              settingsInfo={settingsInfo}
              benchmarkData={benchmarkData}
            />
          )}
        </div>
      </main>
    </div>
  );
}
