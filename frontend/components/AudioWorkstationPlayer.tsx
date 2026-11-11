"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Download,
  Activity,
  Repeat,
} from "lucide-react";
import { WaveformData } from "@/types/spectra";

export interface SeekRangeRequest {
  startSec: number;
  endSec?: number | null;
  autoPlay?: boolean;
  nonce: number;
}

interface AudioWorkstationPlayerProps {
  title: string;
  subtitle?: string;
  audioUrl: string;
  secondaryAudioUrl?: string | null;
  secondaryLabel?: string;
  waveform?: WaveformData | null;
  downloadUrl?: string;
  externalSeekRequest?: SeekRangeRequest | null;
  onTimeChange?: (currentTime: number, isPlaying: boolean) => void;
}

export default function AudioWorkstationPlayer({
  title,
  subtitle,
  audioUrl,
  secondaryAudioUrl,
  secondaryLabel = "Processed Variant",
  waveform,
  downloadUrl,
  externalSeekRequest,
  onTimeChange,
}: AudioWorkstationPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(waveform?.duration_seconds || 0);
  const [muted, setMuted] = useState(false);
  const [loop, setLoop] = useState(false);
  const [abMode, setAbMode] = useState<"A" | "B">("A");
  const [activeRangeEnd, setActiveRangeEnd] = useState<number | null>(null);

  const activeUrl = abMode === "B" && secondaryAudioUrl ? secondaryAudioUrl : audioUrl;

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setAbMode("A");
  }, [audioUrl]);

  useEffect(() => {
    if (waveform?.duration_seconds) {
      setDuration(waveform.duration_seconds);
    }
  }, [waveform]);

  useEffect(() => {
    if (!externalSeekRequest || !audioRef.current) return;
    const el = audioRef.current;
    const targetStart = Math.max(0, externalSeekRequest.startSec);
    el.currentTime = targetStart;
    setCurrentTime(targetStart);
    setActiveRangeEnd(
      externalSeekRequest.endSec !== undefined && externalSeekRequest.endSec !== null
        ? externalSeekRequest.endSec
        : null
    );
    if (externalSeekRequest.autoPlay) {
      el.play()
        .then(() => {
          setIsPlaying(true);
          onTimeChange?.(targetStart, true);
        })
        .catch(() => {
          setIsPlaying(false);
          onTimeChange?.(targetStart, false);
        });
    } else {
      onTimeChange?.(targetStart, isPlaying);
    }
  }, [externalSeekRequest]);

  const togglePlay = async () => {
    const el = audioRef.current;
    if (!el) return;
    setActiveRangeEnd(null);
    if (isPlaying) {
      el.pause();
      setIsPlaying(false);
      onTimeChange?.(el.currentTime, false);
    } else {
      try {
        await el.play();
        setIsPlaying(true);
        onTimeChange?.(el.currentTime, true);
      } catch {
        setIsPlaying(false);
        onTimeChange?.(el.currentTime, false);
      }
    }
  };

  const handleRestart = () => {
    const el = audioRef.current;
    if (!el) return;
    setActiveRangeEnd(null);
    el.currentTime = 0;
    setCurrentTime(0);
    onTimeChange?.(0, isPlaying);
  };

  const switchAb = async (mode: "A" | "B") => {
    if (mode === abMode) return;
    const el = audioRef.current;
    const prevTime = el ? el.currentTime : 0;
    const wasPlaying = isPlaying;
    setAbMode(mode);
    setTimeout(async () => {
      if (audioRef.current) {
        audioRef.current.currentTime = prevTime;
        if (wasPlaying) {
          try {
            await audioRef.current.play();
            setIsPlaying(true);
          } catch {
            setIsPlaying(false);
          }
        }
      }
    }, 40);
  };

  // Draw precision waveform envelope onto HTML5 Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Background grid lines
    ctx.strokeStyle = "rgba(30, 41, 59, 0.55)";
    ctx.lineWidth = 1;
    for (let yFrac of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, h * yFrac);
      ctx.lineTo(w, h * yFrac);
      ctx.stroke();
    }

    if (waveform && waveform.peaks_max.length > 0) {
      const n = waveform.peaks_max.length;
      const progressRatio = duration > 0 ? Math.min(1, currentTime / duration) : 0;
      const playheadX = progressRatio * w;

      for (let i = 0; i < n; i++) {
        const x = (i / n) * w;
        const pMax = Math.min(1, Math.max(-1, waveform.peaks_max[i]));
        const pMin = Math.min(1, Math.max(-1, waveform.peaks_min[i]));
        const yTop = midY - pMax * (midY - 6);
        const yBot = midY - pMin * (midY - 6);

        const isPlayed = x <= playheadX;
        if (abMode === "B") {
          ctx.strokeStyle = isPlayed ? "#f59e0b" : "rgba(245, 158, 11, 0.35)";
        } else {
          ctx.strokeStyle = isPlayed ? "#06b6d4" : "rgba(6, 182, 212, 0.35)";
        }
        ctx.lineWidth = Math.max(1.2, w / n - 0.5);
        ctx.beginPath();
        ctx.moveTo(x, yTop);
        ctx.lineTo(x, Math.max(yTop + 1.5, yBot));
        ctx.stroke();
      }

      // Playhead line
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.stroke();
    } else {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.fillText("Loading PCM waveform envelope...", 14, midY + 4);
    }
  }, [waveform, currentTime, duration, abMode]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const el = audioRef.current;
    if (!canvas || !el || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetTime = ratio * duration;
    setActiveRangeEnd(null);
    el.currentTime = targetTime;
    setCurrentTime(targetTime);
    onTimeChange?.(targetTime, isPlaying);
  };

  const formatTime = (sec: number) => {
    if (!Number.isFinite(sec) || sec < 0) return "00:00.00";
    const mins = Math.floor(sec / 60);
    const rem = sec % 60;
    return `${String(mins).padStart(2, "0")}:${rem.toFixed(2).padStart(5, "0")}`;
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4 shadow-lg">
      <audio
        ref={audioRef}
        src={activeUrl}
        loop={loop}
        muted={muted}
        onTimeUpdate={(e) => {
          const t = e.currentTarget.currentTime;
          if (activeRangeEnd !== null && t >= activeRangeEnd) {
            e.currentTarget.pause();
            setIsPlaying(false);
            setActiveRangeEnd(null);
            setCurrentTime(activeRangeEnd);
            onTimeChange?.(activeRangeEnd, false);
            return;
          }
          setCurrentTime(t);
          onTimeChange?.(t, !e.currentTarget.paused);
        }}
        onLoadedMetadata={(e) => {
          if (e.currentTarget.duration && Number.isFinite(e.currentTarget.duration)) {
            setDuration(e.currentTarget.duration);
          }
        }}
        onEnded={() => {
          setIsPlaying(false);
          setActiveRangeEnd(null);
          onTimeChange?.(currentTime, false);
        }}
      />

      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Activity className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
              <span
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${
                  abMode === "B"
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                    : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                }`}
              >
                {abMode === "B" ? `B: ${secondaryLabel}` : "A: Original Source"}
              </span>
            </div>
            {subtitle && <p className="text-xs text-slate-400 font-mono">{subtitle}</p>}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {secondaryAudioUrl && (
            <div className="flex items-center rounded-lg border border-slate-700 bg-slate-900/90 p-0.5">
              <button
                type="button"
                onClick={() => switchAb("A")}
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-medium transition ${
                  abMode === "A"
                    ? "bg-cyan-500 text-slate-950 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                A: Original
              </button>
              <button
                type="button"
                onClick={() => switchAb("B")}
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-medium transition ${
                  abMode === "B"
                    ? "bg-amber-500 text-slate-950 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                B: Processed
              </button>
            </div>
          )}

          {downloadUrl && (
            <a
              href={downloadUrl}
              download
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/70 px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:border-slate-600 transition"
            >
              <Download className="h-3.5 w-3.5 text-cyan-400" />
              WAV/Audio
            </a>
          )}
        </div>
      </div>

      {/* Waveform Scrubbing Canvas */}
      <div className="relative h-24 w-full overflow-hidden rounded-lg border border-slate-800/90 bg-[#07090e] cursor-pointer">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="h-full w-full block"
        />
        <div className="pointer-events-none absolute top-1.5 right-2 rounded bg-slate-950/80 px-1.5 py-0.5 font-mono text-[10px] text-slate-400 border border-slate-800">
          {waveform ? `${waveform.sample_rate} Hz • ${waveform.channels}ch PCM` : "PCM Stream"}
        </div>
      </div>

      {/* Transport Controls Bar */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePlay}
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500 text-slate-950 hover:bg-cyan-400 transition shadow"
            title={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </button>

          <button
            type="button"
            onClick={handleRestart}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-800/60 text-slate-300 hover:bg-slate-700 hover:text-white transition"
            title="Restart from 00:00"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={() => setLoop(!loop)}
            className={`flex h-9 w-9 items-center justify-center rounded-lg border transition ${
              loop
                ? "border-cyan-500/60 bg-cyan-500/20 text-cyan-300"
                : "border-slate-700 bg-slate-800/60 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle Loop"
          >
            <Repeat className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={() => setMuted(!muted)}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-800/60 text-slate-300 hover:bg-slate-700 transition"
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX className="h-4 w-4 text-rose-400" /> : <Volume2 className="h-4 w-4" />}
          </button>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="rounded-md border border-slate-800 bg-slate-950 px-2.5 py-1 text-cyan-300">
            {formatTime(currentTime)}{" "}
            <span className="text-slate-600">/</span>{" "}
            <span className="text-slate-400">{formatTime(duration)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
