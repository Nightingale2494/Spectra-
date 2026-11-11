"use client";

import React, { useEffect, useRef, useState } from "react";
import { SpectrogramData } from "@/types/spectra";

interface SpectrogramCanvasProps {
  spectrogram: SpectrogramData | null;
}

function dbToColor(db: number): [number, number, number] {
  // Map [-90 dB, 0 dB] to a high-contrast scientific inferno-cyan colormap
  const norm = Math.max(0, Math.min(1, (db + 85) / 85));
  if (norm < 0.25) {
    const t = norm / 0.25;
    return [Math.round(7 + t * 25), Math.round(9 + t * 18), Math.round(20 + t * 65)];
  } else if (norm < 0.55) {
    const t = (norm - 0.25) / 0.3;
    return [Math.round(32 + t * 130), Math.round(27 + t * 30), Math.round(85 + t * 65)];
  } else if (norm < 0.82) {
    const t = (norm - 0.55) / 0.27;
    return [Math.round(162 + t * 83), Math.round(57 + t * 115), Math.round(150 - t * 110)];
  } else {
    const t = (norm - 0.82) / 0.18;
    return [Math.round(245 + t * 10), Math.round(172 + t * 80), Math.round(40 + t * 195)];
  }
}

export default function SpectrogramCanvas({ spectrogram }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    timeSec: number;
    freqHz: number;
    magDb: number;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spectrogram || !spectrogram.magnitude_db.length) return;

    const matrix = spectrogram.magnitude_db; // [freq_bins][time_frames]
    const nFreq = matrix.length;
    const nTime = matrix[0]?.length || 0;
    if (nFreq === 0 || nTime === 0) return;

    canvas.width = nTime;
    canvas.height = nFreq;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imgData = ctx.createImageData(nTime, nFreq);
    const data = imgData.data;

    for (let f = 0; f < nFreq; f++) {
      // Draw lowest frequency at bottom of canvas
      const y = nFreq - 1 - f;
      const row = matrix[f];
      for (let t = 0; t < nTime; t++) {
        const db = row[t];
        const [r, g, b] = dbToColor(db);
        const idx = (y * nTime + t) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [spectrogram]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !spectrogram || !spectrogram.magnitude_db.length) return;
    const rect = canvas.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const nFreq = spectrogram.frequencies.length;
    const nTime = spectrogram.times.length;
    const tIdx = Math.min(nTime - 1, Math.floor(relX * nTime));
    const fIdx = Math.min(nFreq - 1, Math.floor((1 - relY) * nFreq));

    setHoverInfo({
      timeSec: spectrogram.times[tIdx] ?? 0,
      freqHz: spectrogram.frequencies[fIdx] ?? 0,
      magDb: spectrogram.magnitude_db[fIdx]?.[tIdx] ?? -90,
    });
  };

  if (!spectrogram) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-slate-800 bg-[#0d111a] font-mono text-xs text-slate-400">
        Computing 2D STFT log-power spectrogram...
      </div>
    );
  }

  const maxFreq =
    spectrogram.frequencies[spectrogram.frequencies.length - 1] ||
    spectrogram.sample_rate / 2;

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold text-slate-100">
            2D STFT Log-Power Spectrogram
          </h4>
          <p className="text-xs text-slate-400 font-mono">
            SR: {spectrogram.sample_rate} Hz • FFT: {spectrogram.n_fft} • Hop:{" "}
            {spectrogram.hop_length} • Band: 0 – {Math.round(maxFreq)} Hz
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          {hoverInfo ? (
            <div className="rounded border border-cyan-500/30 bg-cyan-950/30 px-2.5 py-1 text-cyan-200">
              t={hoverInfo.timeSec.toFixed(2)}s | f={Math.round(hoverInfo.freqHz)} Hz |{" "}
              {hoverInfo.magDb.toFixed(1)} dB
            </div>
          ) : (
            <span className="text-slate-400">Hover heatmap to inspect (t, f, dB)</span>
          )}
        </div>
      </div>

      <div className="relative flex gap-2">
        {/* Frequency axis labels */}
        <div className="flex flex-col justify-between py-1 font-mono text-[10px] text-slate-400 w-14 text-right pr-1 select-none">
          <span>{Math.round(maxFreq)} Hz</span>
          <span>{Math.round(maxFreq * 0.75)} Hz</span>
          <span>{Math.round(maxFreq * 0.5)} Hz</span>
          <span>{Math.round(maxFreq * 0.25)} Hz</span>
          <span>0 Hz</span>
        </div>

        <div className="relative flex-1 h-60 overflow-hidden rounded-lg border border-slate-800 bg-[#07090e]">
          <canvas
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverInfo(null)}
            className="h-full w-full block cursor-crosshair"
            style={{ imageRendering: "pixelated" }}
          />
        </div>
      </div>

      {/* Colorbar Legend */}
      <div className="mt-2.5 flex items-center justify-end gap-2 font-mono text-[10px] text-slate-400">
        <span>-85 dB</span>
        <div
          className="h-2 w-36 rounded"
          style={{
            background:
              "linear-gradient(to right, #070914, #201b55, #a23996, #f5ac28, #fff8dc)",
          }}
        />
        <span>0 dB</span>
      </div>
    </div>
  );
}
