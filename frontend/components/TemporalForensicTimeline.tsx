"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Eye,
  GitBranch,
  Info,
  Layers,
  Play,
  RefreshCw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  TemporalAnalysisData,
  TemporalRegionSummary,
  TemporalWindowStep,
  WaveformData,
} from "@/types/spectra";
import { spectraApi } from "@/lib/api";

interface TemporalForensicTimelineProps {
  jobId: string;
  waveform: WaveformData | null;
  currentTime: number;
  isPlaying: boolean;
  onSeekAndPlayRange: (startSec: number, endSec?: number | null, autoPlay?: boolean) => void;
}

export default function TemporalForensicTimeline({
  jobId,
  waveform,
  currentTime,
  isPlaying,
  onSeekAndPlayRange,
}: TemporalForensicTimelineProps) {
  const [temporalData, setTemporalData] = useState<TemporalAnalysisData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [cancelNotice, setCancelNotice] = useState<string | null>(null);

  // Configuration controls
  const [hopSec, setHopSec] = useState<number>(2.5);
  const [fakeprintWindowSec, setFakeprintWindowSec] = useState<number>(5.0);
  const [showExploratoryFakeprint, setShowExploratoryFakeprint] = useState<boolean>(true);

  // Interactive zoom, pan, hover, and range selection state
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [panRatio, setPanRatio] = useState<number>(0); // 0 to 1
  const [hoveredStepIdx, setHoveredStepIdx] = useState<number | null>(null);
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);
  const [selectedRange, setSelectedRange] = useState<[number, number] | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const loadTemporal = async (
    targetHop = hopSec,
    targetFpWin = fakeprintWindowSec,
    includeFp = showExploratoryFakeprint,
    forceRecompute = false
  ) => {
    if (!jobId) return;
    setLoading(true);
    setErrorMsg(null);
    setCancelNotice(null);
    try {
      const data = await spectraApi.getTemporalAnalysis(jobId, {
        hop_sec: targetHop,
        fakeprint_window_sec: targetFpWin,
        include_exploratory_fakeprint: includeFp,
        force_recompute: forceRecompute,
      });
      setTemporalData(data);
      setSelectedStepIdx(0);
      if (data.timeline_steps.length > 0 && !selectedRange) {
        const first = data.timeline_steps[0];
        setSelectedRange([first.start_sec, first.end_sec]);
      }
    } catch (err: any) {
      const msg = err.message || "Failed to load temporal forensic analysis.";
      if (msg.toLowerCase().includes("cancel")) {
        setCancelNotice("Temporal inference run was cancelled cleanly; existing cache preserved.");
      } else {
        setErrorMsg(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCancelTemporal = async () => {
    if (!jobId) return;
    try {
      await spectraApi.cancelTemporalAnalysis(jobId);
      setCancelNotice("Temporal cancellation requested for active job; existing cache preserved.");
      setLoading(false);
    } catch (err: any) {
      setCancelNotice("Temporal cancellation signal dispatched.");
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTemporal(hopSec, fakeprintWindowSec, showExploratoryFakeprint, false);
  }, [jobId]);

  const durationSec = useMemo(() => {
    return (
      temporalData?.recording_duration_sec ||
      waveform?.duration_seconds ||
      0
    );
  }, [temporalData, waveform]);

  // Visible time window based on zoomLevel and panRatio
  const visibleTimeSpan = useMemo(() => {
    if (durationSec <= 0) return { start: 0, end: 10, span: 10 };
    const span = durationSec / zoomLevel;
    const maxStart = Math.max(0, durationSec - span);
    const start = panRatio * maxStart;
    return { start, end: start + span, span };
  }, [durationSec, zoomLevel, panRatio]);

  // Active inspected step (hovered or selected)
  const inspectedStep: TemporalWindowStep | null = useMemo(() => {
    if (!temporalData || temporalData.timeline_steps.length === 0) return null;
    if (hoveredStepIdx !== null && temporalData.timeline_steps[hoveredStepIdx]) {
      return temporalData.timeline_steps[hoveredStepIdx];
    }
    return (
      temporalData.timeline_steps[selectedStepIdx] ||
      temporalData.timeline_steps[0]
    );
  }, [temporalData, hoveredStepIdx, selectedStepIdx]);

  // Draw synchronized Waveform + Region Bands + Per-Model Probability Traces + Playhead
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !temporalData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const { start: vStart, end: vEnd, span: vSpan } = visibleTimeSpan;
    if (vSpan <= 0) return;

    const timeToX = (t: number) => ((t - vStart) / vSpan) * w;

    ctx.clearRect(0, 0, w, h);

    // Layout bands:
    // Top 36% (0 to 0.36h): Aligned Waveform Envelope + Selected Range Highlight
    // Bottom 64% (0.40h to 0.92h): Probability Trace Plot (0.0 to 1.0) + Region Background
    const waveTop = 6;
    const waveHeight = h * 0.32;
    const waveMid = waveTop + waveHeight / 2;

    const plotTop = h * 0.42;
    const plotBottom = h * 0.90;
    const plotHeight = plotBottom - plotTop;
    const probToY = (p: number) => plotBottom - Math.max(0, Math.min(1, p)) * plotHeight;

    // 1. Draw Region Summary background shading across the probability plot
    for (const reg of temporalData.region_summaries) {
      const x1 = Math.max(0, timeToX(reg.start_sec));
      const x2 = Math.min(w, timeToX(reg.end_sec));
      if (x2 <= x1) continue;

      if (reg.region_label === "Elevated AI-associated features") {
        ctx.fillStyle = "rgba(245, 158, 11, 0.12)";
      } else if (reg.region_label === "Model disagreement") {
        ctx.fillStyle = "rgba(168, 85, 247, 0.15)";
      } else if (reg.region_label === "Insufficient evidence") {
        ctx.fillStyle = "rgba(244, 63, 94, 0.12)";
      } else {
        ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
      }
      ctx.fillRect(x1, plotTop, x2 - x1, plotHeight);
    }

    // 2. Draw Selected Range highlight across both waveform and plot
    if (selectedRange) {
      const sx1 = Math.max(0, timeToX(selectedRange[0]));
      const sx2 = Math.min(w, timeToX(selectedRange[1]));
      if (sx2 > sx1) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.14)";
        ctx.fillRect(sx1, 0, sx2 - sx1, h);
        ctx.strokeStyle = "rgba(6, 182, 212, 0.65)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(sx1, 1, sx2 - sx1, h - 2);
      }
    }

    // 3. Draw Waveform Envelope in top band
    ctx.strokeStyle = "rgba(30, 41, 59, 0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, waveMid);
    ctx.lineTo(w, waveMid);
    ctx.stroke();

    if (waveform && waveform.peaks_max.length > 0 && durationSec > 0) {
      const n = waveform.peaks_max.length;
      for (let i = 0; i < n; i++) {
        const t = (i / n) * durationSec;
        if (t < vStart || t > vEnd) continue;
        const x = timeToX(t);
        const pMax = Math.min(1, Math.max(-1, waveform.peaks_max[i]));
        const pMin = Math.min(1, Math.max(-1, waveform.peaks_min[i]));
        const y1 = waveMid - pMax * (waveHeight * 0.46);
        const y2 = waveMid - pMin * (waveHeight * 0.46);
        ctx.strokeStyle = t <= currentTime ? "#06b6d4" : "rgba(100, 116, 139, 0.45)";
        ctx.lineWidth = Math.max(1.1, (w / n) * zoomLevel * 0.8);
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, Math.max(y1 + 1.2, y2));
        ctx.stroke();
      }
    }

    // 4. Draw Probability Plot Grid & Threshold Line (0.50)
    for (const pVal of [0.0, 0.25, 0.5, 0.75, 1.0]) {
      const y = probToY(pVal);
      ctx.strokeStyle = pVal === 0.5 ? "rgba(245, 158, 11, 0.55)" : "rgba(30, 41, 59, 0.65)";
      ctx.lineWidth = pVal === 0.5 ? 1.2 : 1;
      if (pVal === 0.5) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = pVal === 0.5 ? "#fbbf24" : "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.fillText(`${Math.round(pVal * 100)}%`, 6, y - 3);
    }

    // 5. Draw Section Transition vertical markers
    for (const tr of temporalData.section_transitions) {
      if (tr.timestamp_sec < vStart || tr.timestamp_sec > vEnd) continue;
      const tx = timeToX(tr.timestamp_sec);
      ctx.strokeStyle = "rgba(244, 63, 94, 0.65)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(tx, plotTop);
      ctx.lineTo(tx, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const steps = temporalData.timeline_steps;

    // 6. Draw Exploratory Fakeprint ONNX Segment Trace (if enabled)
    if (showExploratoryFakeprint && steps.length > 0) {
      ctx.strokeStyle = "rgba(168, 85, 247, 0.85)";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      let started = false;
      for (const s of steps) {
        if (s.fakeprint_exploratory_probability === null) continue;
        const x = timeToX(s.center_sec);
        const y = probToY(s.fakeprint_exploratory_probability);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);

      for (const s of steps) {
        if (s.fakeprint_exploratory_probability === null) continue;
        if (s.center_sec < vStart || s.center_sec > vEnd) continue;
        const x = timeToX(s.center_sec);
        const y = probToY(s.fakeprint_exploratory_probability);
        ctx.fillStyle = "#c084fc";
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 7. Draw Secondary Cross-Generator Detector Trace (if available)
    if (steps.some((s) => s.cross_generator_probability !== null && s.cross_generator_probability !== undefined)) {
      ctx.strokeStyle = "rgba(52, 211, 153, 0.85)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      for (const s of steps) {
        if (s.cross_generator_probability === null || s.cross_generator_probability === undefined) continue;
        const x = timeToX(s.center_sec);
        const y = probToY(s.cross_generator_probability);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // 8. Draw Primary SONICS SpecTTTra-gamma (5s) Sliding-Window Trace & Window Span Bars
    if (steps.length > 0) {
      // Draw horizontal window span bars showing exact 5s window coverage
      for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        if (s.sonics_gamma_probability === null) continue;
        const wx1 = Math.max(0, timeToX(s.start_sec));
        const wx2 = Math.min(w, timeToX(s.end_sec));
        if (wx2 <= wx1) continue;
        const wy = probToY(s.sonics_gamma_probability);
        const isActive = i === hoveredStepIdx || i === selectedStepIdx;
        ctx.strokeStyle = s.abstained
          ? "rgba(244, 63, 94, 0.55)"
          : isActive
          ? "rgba(34, 211, 238, 0.9)"
          : "rgba(6, 182, 212, 0.35)";
        ctx.lineWidth = isActive ? 3.0 : 1.8;
        ctx.beginPath();
        ctx.moveTo(wx1, wy);
        ctx.lineTo(wx2, wy);
        ctx.stroke();
      }

      // Connect window centers for trajectory curve
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      let started = false;
      for (const s of steps) {
        if (s.sonics_gamma_probability === null) continue;
        const x = timeToX(s.center_sec);
        const y = probToY(s.sonics_gamma_probability);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw window center nodes
      for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        if (s.sonics_gamma_probability === null) continue;
        if (s.center_sec < vStart || s.center_sec > vEnd) continue;
        const x = timeToX(s.center_sec);
        const y = probToY(s.sonics_gamma_probability);
        const isActive = i === hoveredStepIdx || i === selectedStepIdx;

        ctx.fillStyle = s.abstained
          ? "#f43f5e"
          : s.disagreement
          ? "#c084fc"
          : s.sonics_gamma_probability >= 0.5
          ? "#f59e0b"
          : "#10b981";
        ctx.beginPath();
        ctx.arc(x, y, isActive ? 5.5 : 4.0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    // 9. Draw Time Axis Markers along bottom
    const tickCount = 8;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px JetBrains Mono, monospace";
    for (let tIdx = 0; tIdx <= tickCount; tIdx++) {
      const frac = tIdx / tickCount;
      const tSec = vStart + frac * vSpan;
      const tx = frac * w;
      ctx.fillText(`${tSec.toFixed(1)}s`, Math.min(w - 34, Math.max(4, tx - 12)), h - 4);
    }

    // 10. Draw Synchronized Playhead Cursor
    if (currentTime >= vStart && currentTime <= vEnd) {
      const px = timeToX(currentTime);
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
    }
  }, [
    temporalData,
    waveform,
    durationSec,
    visibleTimeSpan,
    currentTime,
    showExploratoryFakeprint,
    hoveredStepIdx,
    selectedStepIdx,
    selectedRange,
    zoomLevel,
  ]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !temporalData || temporalData.timeline_steps.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetSec = visibleTimeSpan.start + ratio * visibleTimeSpan.span;

    // Find closest timeline step
    let bestIdx = 0;
    let bestDist = Infinity;
    temporalData.timeline_steps.forEach((s, idx) => {
      const d = Math.abs(s.center_sec - targetSec);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = idx;
      }
    });
    setHoveredStepIdx(bestIdx);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !temporalData || temporalData.timeline_steps.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetSec = visibleTimeSpan.start + ratio * visibleTimeSpan.span;

    let bestIdx = 0;
    let bestDist = Infinity;
    temporalData.timeline_steps.forEach((s, idx) => {
      const d = Math.abs(s.center_sec - targetSec);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = idx;
      }
    });
    const step = temporalData.timeline_steps[bestIdx];
    setSelectedStepIdx(bestIdx);
    setSelectedRange([step.start_sec, step.end_sec]);
    onSeekAndPlayRange(step.start_sec, step.end_sec, true);
  };

  const handleRegionSelectAndPlay = (reg: TemporalRegionSummary, autoPlay = true) => {
    setSelectedRange([reg.start_sec, reg.end_sec]);
    if (temporalData) {
      const idx = temporalData.timeline_steps.findIndex(
        (s) => s.start_sec >= reg.start_sec - 0.05
      );
      if (idx >= 0) setSelectedStepIdx(idx);
    }
    onSeekAndPlayRange(reg.start_sec, reg.end_sec, autoPlay);
  };

  const renderRegionBadge = (label: string) => {
    if (label === "Elevated AI-associated features") {
      return (
        <span className="rounded-full border border-amber-500/40 bg-amber-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-amber-300">
          {label}
        </span>
      );
    }
    if (label === "Model disagreement") {
      return (
        <span className="rounded-full border border-purple-500/40 bg-purple-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-purple-300">
          {label}
        </span>
      );
    }
    if (label === "Insufficient evidence") {
      return (
        <span className="rounded-full border border-rose-500/40 bg-rose-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-rose-300">
          {label}
        </span>
      );
    }
    return (
      <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-emerald-300">
        {label}
      </span>
    );
  };

  return (
    <div
      data-testid="temporal-forensic-timeline"
      className="rounded-xl border border-slate-800 bg-[#0d111a] p-5 space-y-5"
    >
      {/* Header & Configuration Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-slate-100">
              Temporal AI-Detection Timeline & Sliding-Window Forensics
            </h3>
            {temporalData?.cache_hit && (
              <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[10px] text-cyan-300">
                SHA-256 Cached ({temporalData.processing_time_ms.toFixed(1)} ms)
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Independent per-window (1, T) inference • Window: 5.0s • Hop:{" "}
            {temporalData?.configuration.hop_sec ?? hopSec}s • Neutral region attribution
          </p>
        </div>

        {/* Interactive Resolution & Overlay Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1 text-xs">
            <span className="text-slate-400 font-mono text-[11px]">Hop:</span>
            {[1.0, 2.5, 5.0].map((hVal) => (
              <button
                key={hVal}
                type="button"
                data-testid={`temporal-hop-${hVal.toFixed(1)}`}
                disabled={loading}
                onClick={() => {
                  setHopSec(hVal);
                  loadTemporal(hVal, fakeprintWindowSec, showExploratoryFakeprint, false);
                }}
                className={`rounded px-2 py-0.5 font-mono text-[11px] transition ${
                  Math.abs(hopSec - hVal) < 0.05
                    ? "bg-cyan-500 text-slate-950 font-semibold"
                    : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                {hVal.toFixed(1)}s
              </button>
            ))}
          </div>

          <label className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              data-testid="temporal-toggle-fakeprint"
              checked={showExploratoryFakeprint}
              onChange={(e) => setShowExploratoryFakeprint(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-cyan-500"
            />
            <span className="font-mono text-[11px]">
              Exploratory Fakeprint Trace ({fakeprintWindowSec}s)
            </span>
          </label>

          {/* Zoom Controls */}
          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 px-2 py-1">
            <button
              type="button"
              data-testid="temporal-zoom-out"
              onClick={() => setZoomLevel((z) => Math.max(1, z / 2))}
              className="p-1 text-slate-400 hover:text-slate-100"
              title="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span
              data-testid="temporal-zoom-level"
              className="font-mono text-[11px] text-cyan-300 px-1"
            >
              {zoomLevel}x
            </span>
            <button
              type="button"
              data-testid="temporal-zoom-in"
              onClick={() => setZoomLevel((z) => Math.min(8, z * 2))}
              className="p-1 text-slate-400 hover:text-slate-100"
              title="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
          </div>

          <button
            type="button"
            data-testid="temporal-recompute-btn"
            disabled={loading}
            onClick={() =>
              loadTemporal(hopSec, fakeprintWindowSec, showExploratoryFakeprint, true)
            }
            className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50 transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            {loading ? "Scoring Windows..." : "Recompute"}
          </button>

          <button
            type="button"
            data-testid="temporal-cancel-btn"
            onClick={handleCancelTemporal}
            className="inline-flex items-center gap-1 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1.5 text-xs font-semibold text-rose-300 hover:bg-rose-500/20 transition"
          >
            Cancel
          </button>
        </div>
      </div>

      {cancelNotice && (
        <div
          data-testid="temporal-cancel-notice"
          className="rounded-lg border border-amber-500/40 bg-amber-950/30 p-2.5 text-xs font-mono text-amber-200 flex items-center justify-between"
        >
          <span>{cancelNotice}</span>
          <button
            type="button"
            onClick={() => setCancelNotice(null)}
            className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] text-amber-200 hover:bg-amber-500/30"
          >
            Dismiss
          </button>
        </div>
      )}

      {errorMsg && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-3 text-xs text-rose-200 flex items-center justify-between">
          <span>{errorMsg}</span>
          <button
            type="button"
            onClick={() => loadTemporal(hopSec, fakeprintWindowSec, showExploratoryFakeprint, true)}
            className="rounded bg-rose-500/20 px-2.5 py-1 font-mono text-[11px] text-rose-200 hover:bg-rose-500/30"
          >
            Retry
          </button>
        </div>
      )}

      {temporalData && (
        <>
          {/* Legend & Selected Range Audition Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-800/90 bg-slate-950 p-3 text-xs">
            <div className="flex flex-wrap items-center gap-4 font-mono text-[11px]">
              <span className="flex items-center gap-1.5 text-cyan-300">
                <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 inline-block" />
                SONICS SpecTTTra-γ (5.0s window, {temporalData.configuration.hop_sec}s hop)
              </span>
              {showExploratoryFakeprint && (
                <span className="flex items-center gap-1.5 text-purple-300">
                  <span className="h-2.5 w-2.5 rounded-full bg-purple-400 inline-block" />
                  Fakeprint ONNX Segment (Exploratory, {temporalData.configuration.fakeprint_segment_window_sec}s)
                </span>
              )}
              <span className="flex items-center gap-1.5 text-amber-300">
                <span className="h-0.5 w-4 bg-amber-400 inline-block" />
                Decision Threshold (50%)
              </span>
              <span
                data-testid="temporal-disagreement-summary"
                className="rounded border border-purple-500/30 bg-purple-500/10 px-2 py-0.5 text-purple-300"
              >
                Disagreement Windows: {temporalData.summary_statistics.disagreement_window_count} /{" "}
                {temporalData.summary_statistics.sonics_gamma_5s.total_window_count}
              </span>
            </div>

            {selectedRange && (
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="text-slate-400">
                  Selected Span:{" "}
                  <strong className="text-cyan-300">
                    [{selectedRange[0].toFixed(2)}s – {selectedRange[1].toFixed(2)}s]
                  </strong>
                </span>
                <button
                  type="button"
                  data-testid="play-selected-region-btn"
                  onClick={() => onSeekAndPlayRange(selectedRange[0], selectedRange[1], true)}
                  className="inline-flex items-center gap-1 rounded bg-cyan-500 px-2.5 py-1 text-[11px] font-semibold text-slate-950 hover:bg-cyan-400 transition"
                >
                  <Play className="h-3 w-3" />
                  Audition Segment
                </button>
              </div>
            )}
          </div>

          {/* Interactive Synchronized Timeline Canvas */}
          <div className="space-y-2">
            <div className="relative h-64 w-full overflow-hidden rounded-lg border border-slate-800 bg-[#07090e] cursor-pointer">
              <canvas
                ref={canvasRef}
                data-testid="temporal-timeline-canvas"
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoveredStepIdx(null)}
                onClick={handleCanvasClick}
                className="h-full w-full block"
              />
              <div className="pointer-events-none absolute top-2 left-2.5 rounded bg-slate-950/85 px-2 py-0.5 font-mono text-[10px] text-slate-300 border border-slate-800">
                Aligned Waveform + Sliding-Window Probability Trajectory (Click any window to seek & audition)
              </div>
              <div
                data-testid="temporal-playhead-readout"
                className="pointer-events-none absolute top-2 right-2.5 rounded bg-slate-950/85 px-2 py-0.5 font-mono text-[10px] text-cyan-300 border border-slate-800"
              >
                Playhead: {currentTime.toFixed(2)}s / {durationSec.toFixed(2)}s
              </div>
            </div>

            {/* Horizontal Pan Slider when Zoom > 1x */}
            {zoomLevel > 1 && (
              <div className="flex items-center gap-3 px-2">
                <span className="font-mono text-[11px] text-slate-400">
                  Pan Timeline ({visibleTimeSpan.start.toFixed(1)}s – {visibleTimeSpan.end.toFixed(1)}s):
                </span>
                <input
                  type="range"
                  data-testid="temporal-pan-slider"
                  min={0}
                  max={1}
                  step={0.01}
                  value={panRatio}
                  onChange={(e) => setPanRatio(parseFloat(e.target.value))}
                  className="flex-1 accent-cyan-500"
                />
              </div>
            )}
          </div>

          {/* Active / Hovered Window Inspector Card */}
          {inspectedStep && (
            <div
              data-testid="temporal-window-inspector"
              className="rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-2.5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <span className="rounded bg-cyan-500/15 border border-cyan-500/30 px-2 py-0.5 font-mono text-xs font-semibold text-cyan-300">
                    Window #{inspectedStep.window_index + 1} [{inspectedStep.start_sec.toFixed(2)}s –{" "}
                    {inspectedStep.end_sec.toFixed(2)}s]
                  </span>
                  {renderRegionBadge(inspectedStep.region_label)}
                  {inspectedStep.is_incomplete_final_window && (
                    <span className="rounded bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 font-mono text-[10px] text-amber-300">
                      Incomplete Final Window (Zero-Pad: {inspectedStep.zero_pad_sec.toFixed(2)}s)
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() =>
                    onSeekAndPlayRange(inspectedStep.start_sec, inspectedStep.end_sec, true)
                  }
                  className="inline-flex items-center gap-1 rounded border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1 font-mono text-xs text-cyan-300 hover:bg-cyan-500/20"
                >
                  <Play className="h-3 w-3" />
                  Play Window [{inspectedStep.start_sec.toFixed(1)}s–{inspectedStep.end_sec.toFixed(1)}s]
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 font-mono text-xs">
                <div className="rounded border border-slate-800/90 bg-slate-900/50 p-2">
                  <span className="text-[10px] text-slate-400 block">SONICS-γ (5s) Score</span>
                  <span className="text-cyan-300 font-semibold">
                    {inspectedStep.sonics_gamma_probability !== null
                      ? `${(inspectedStep.sonics_gamma_probability * 100).toFixed(2)}%`
                      : "N/A"}{" "}
                    {inspectedStep.sonics_gamma_logit !== null && (
                      <span className="text-slate-400 text-[10px]">
                        (logit {inspectedStep.sonics_gamma_logit.toFixed(2)})
                      </span>
                    )}
                  </span>
                </div>

                <div className="rounded border border-slate-800/90 bg-slate-900/50 p-2">
                  <span className="text-[10px] text-slate-400 block">
                    Fakeprint Segment (Exploratory)
                  </span>
                  <span className="text-purple-300 font-semibold">
                    {inspectedStep.fakeprint_exploratory_probability !== null
                      ? `${(inspectedStep.fakeprint_exploratory_probability * 100).toFixed(2)}%`
                      : "Disabled"}
                  </span>
                </div>

                <div className="rounded border border-slate-800/90 bg-slate-900/50 p-2">
                  <span className="text-[10px] text-slate-400 block">Segment RMS / Crest</span>
                  <span className="text-slate-200">
                    {inspectedStep.rms_dbfs.toFixed(1)} dBFS / {inspectedStep.crest_factor_db.toFixed(1)} dB
                  </span>
                </div>

                <div className="rounded border border-slate-800/90 bg-slate-900/50 p-2">
                  <span className="text-[10px] text-slate-400 block">Unpadded Duration</span>
                  <span className="text-slate-200">
                    {inspectedStep.unpadded_duration_sec.toFixed(2)}s
                  </span>
                </div>

                <div className="rounded border border-slate-800/90 bg-slate-900/50 p-2">
                  <span className="text-[10px] text-slate-400 block">Detector Disagreement</span>
                  <span
                    className={
                      inspectedStep.disagreement ? "text-purple-300 font-semibold" : "text-slate-400"
                    }
                  >
                    {inspectedStep.disagreement
                      ? `Yes (Δ=${((inspectedStep.disagreement_spread ?? 0) * 100).toFixed(1)}%)`
                      : "None"}
                  </span>
                </div>
              </div>

              <div className="text-xs text-slate-300 font-mono">
                <strong className="text-slate-400">Window Rationale:</strong> {inspectedStep.rationale}
              </div>

              {inspectedStep.warnings.length > 0 && (
                <div className="rounded border border-amber-500/30 bg-amber-950/20 p-2 text-[11px] text-amber-200 font-mono space-y-0.5">
                  {inspectedStep.warnings.map((w, i) => (
                    <div key={i}>▲ {w}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Contiguous Neutral Region Summaries & Section Transitions */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2 rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
                  Contiguous Forensic Region Summaries (Click Any Region to Audition)
                </h4>
                <span className="font-mono text-[11px] text-slate-400">
                  {temporalData.region_summaries.length} region(s)
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse font-mono text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                      <th className="py-2 px-2.5">Time Span</th>
                      <th className="py-2 px-2.5">Neutral Forensic Classification</th>
                      <th className="py-2 px-2.5">SONICS-γ Mean / Max</th>
                      <th className="py-2 px-2.5">Fakeprint (Expl.)</th>
                      <th className="py-2 px-2.5">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {temporalData.region_summaries.map((reg) => (
                      <tr
                        key={reg.region_id}
                        data-testid={`temporal-region-row-${reg.region_index}`}
                        onClick={() => handleRegionSelectAndPlay(reg, false)}
                        className="hover:bg-slate-900/60 cursor-pointer transition"
                      >
                        <td className="py-2 px-2.5 text-slate-200 font-semibold">
                          {reg.start_sec.toFixed(2)}s – {reg.end_sec.toFixed(2)}s{" "}
                          <span className="text-slate-500 text-[10px]">
                            ({reg.window_count}w)
                          </span>
                        </td>
                        <td className="py-2 px-2.5">{renderRegionBadge(reg.region_label)}</td>
                        <td className="py-2 px-2.5 text-cyan-300">
                          {reg.mean_sonics_probability !== null
                            ? `${(reg.mean_sonics_probability * 100).toFixed(1)}% / ${(
                                (reg.max_sonics_probability ?? 0) * 100
                              ).toFixed(1)}%`
                            : "N/A"}
                        </td>
                        <td className="py-2 px-2.5 text-purple-300">
                          {reg.mean_fakeprint_exploratory_probability !== null
                            ? `${(reg.mean_fakeprint_exploratory_probability * 100).toFixed(1)}%`
                            : "N/A"}
                        </td>
                        <td className="py-2 px-2.5">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRegionSelectAndPlay(reg, true);
                            }}
                            className="inline-flex items-center gap-1 rounded bg-cyan-500/15 border border-cyan-500/40 px-2 py-0.5 text-[11px] text-cyan-300 hover:bg-cyan-500/25"
                          >
                            <Play className="h-3 w-3" />
                            Play
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Hybrid Recording & Section Transition Analysis */}
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-4 space-y-3 flex flex-col justify-between">
              <div className="space-y-2.5">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                  <GitBranch className="h-3.5 w-3.5" />
                  Section Transitions & Hybrid Analysis
                </h4>

                <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                  <div className="rounded border border-slate-800 bg-slate-900/50 p-2">
                    <span className="text-[10px] text-slate-400 block">Inter-Window σ</span>
                    <span className="text-slate-100 font-semibold">
                      {temporalData.hybrid_forensic_analysis.temporal_heterogeneity_metrics.inter_window_std.toFixed(
                        4
                      )}
                    </span>
                  </div>
                  <div className="rounded border border-slate-800 bg-slate-900/50 p-2">
                    <span className="text-[10px] text-slate-400 block">Transitions</span>
                    <span className="text-amber-300 font-semibold">
                      {temporalData.section_transitions.length} detected
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {temporalData.hybrid_forensic_analysis.forensic_interpretation}
                </p>

                {temporalData.section_transitions.length > 0 && (
                  <div className="max-h-32 overflow-y-auto rounded border border-slate-800 bg-slate-900/40 p-2 font-mono text-[11px] text-slate-300 space-y-1">
                    {temporalData.section_transitions.map((tr) => (
                      <div key={tr.transition_index} className="flex items-start gap-1.5">
                        <span className="text-rose-400 font-bold">•</span>
                        <span>{tr.description}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Separate Long-Context SONICS Alpha 120s Box */}
              <div className="rounded border border-slate-800 bg-slate-900/60 p-2.5 text-[11px] font-mono space-y-1">
                <div className="flex items-center justify-between text-slate-200 font-semibold">
                  <span>Long-Context Model: SONICS SpecTTTra-α (120s)</span>
                  <span
                    className={
                      temporalData.long_context_predictions.abstained
                        ? "text-rose-300"
                        : "text-cyan-300"
                    }
                  >
                    {temporalData.long_context_predictions.abstained
                      ? "ABSTAINED (<115s or Disabled)"
                      : `${(
                          (temporalData.long_context_predictions.aggregate_probability ?? 0) * 100
                        ).toFixed(1)}%`}
                  </span>
                </div>
                <p className="text-slate-400 text-[10px] leading-snug">
                  {temporalData.long_context_predictions.representation_notice ||
                    temporalData.long_context_predictions.status_notice}
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
