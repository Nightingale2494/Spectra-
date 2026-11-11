"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from "recharts";
import { ComparisonResult, SpectrogramData } from "@/types/spectra";

export function FakeprintInspectorChart({
  spectrogram,
}: {
  spectrogram: SpectrogramData | null;
}) {
  const data = useMemo(() => {
    if (!spectrogram || !spectrogram.fakeprint_freqs.length) return [];
    return spectrogram.fakeprint_freqs.map((f, idx) => ({
      freq: Math.round(f),
      spectrum_db: Number((spectrogram.fakeprint_spectrum_db[idx] ?? 0).toFixed(2)),
      hull_db: Number((spectrogram.fakeprint_hull_db[idx] ?? 0).toFixed(2)),
      residual_norm: Number((spectrogram.fakeprint_residual_norm[idx] ?? 0).toFixed(4)),
    }));
  }, [spectrogram]);

  if (!data.length) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-slate-800 bg-[#0d111a] font-mono text-xs text-slate-400">
        Loading 1–8 kHz neural-vocoder fakeprint curve...
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
        <div className="mb-3">
          <h4 className="text-sm font-semibold text-slate-100">
            1–8 kHz Spectrum vs Lower Hull Envelope
          </h4>
          <p className="text-xs text-slate-400 font-mono">
            8192-point STFT @ 16 kHz mono • Lower hull subtraction isolates transposed-convolution peaks
          </p>
        </div>
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="freq"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                unit=" Hz"
              />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit=" dB" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0d111a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line
                type="monotone"
                dataKey="spectrum_db"
                name="Mean Log-Power (dB)"
                stroke="#06b6d4"
                strokeWidth={1.6}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="hull_db"
                name="Lower Spectral Hull (dB)"
                stroke="#f59e0b"
                strokeWidth={1.6}
                strokeDasharray="4 2"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
        <div className="mb-3">
          <h4 className="text-sm font-semibold text-slate-100">
            Normalized Spectral Residual (&ldquo;Fakeprint&rdquo; Vector)
          </h4>
          <p className="text-xs text-slate-400 font-mono">
            Input to ONNX Logistic Regression Detector (3,585 bins in 1,000–8,000 Hz)
          </p>
        </div>
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="freq"
                stroke="#64748b"
                tick={{ fontSize: 11 }}
                unit=" Hz"
              />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0d111a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Area
                type="monotone"
                dataKey="residual_norm"
                name="Normalized Fakeprint Residual"
                stroke="#f43f5e"
                fill="rgba(244, 63, 94, 0.22)"
                strokeWidth={1.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export function ComparisonSpectraChart({
  comparison,
}: {
  comparison: ComparisonResult;
}) {
  const specData = useMemo(() => {
    const freqs = comparison.spectrum_comparison?.frequencies_hz || [];
    const orig = comparison.spectrum_comparison?.original_db || [];
    const proc = comparison.spectrum_comparison?.processed_db || [];
    return freqs.map((f, i) => ({
      freq: Math.round(f),
      original_db: Number((orig[i] ?? 0).toFixed(2)),
      processed_db: Number((proc[i] ?? 0).toFixed(2)),
    }));
  }, [comparison]);

  const resData = useMemo(() => {
    const freqs = comparison.residual_comparison?.frequencies_hz || [];
    const orig = comparison.residual_comparison?.original_residual || [];
    const proc = comparison.residual_comparison?.processed_residual || [];
    return freqs.map((f, i) => ({
      freq: Math.round(f),
      original_res: Number((orig[i] ?? 0).toFixed(4)),
      processed_res: Number((proc[i] ?? 0).toFixed(4)),
    }));
  }, [comparison]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
        <div className="mb-3">
          <h4 className="text-sm font-semibold text-slate-100">
            Original vs Processed Mean Log-Power Spectrum
          </h4>
          <p className="text-xs text-slate-400 font-mono">
            Full-band spectral envelope overlay (dB)
          </p>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={specData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="freq" stroke="#64748b" tick={{ fontSize: 11 }} unit=" Hz" />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit=" dB" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0d111a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line
                type="monotone"
                dataKey="original_db"
                name="Original Spectrum (dB)"
                stroke="#06b6d4"
                strokeWidth={1.8}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="processed_db"
                name="Processed Spectrum (dB)"
                stroke="#f59e0b"
                strokeWidth={1.8}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-[#0d111a] p-4">
        <div className="mb-3">
          <h4 className="text-sm font-semibold text-slate-100">
            Original vs Processed 1–8 kHz Fakeprint Residual
          </h4>
          <p className="text-xs text-slate-400 font-mono">
            Comb Periodicity: {(comparison.residual_comparison?.original_comb_periodicity ?? 0).toFixed(3)}{" "}
            → {(comparison.residual_comparison?.processed_comb_periodicity ?? 0).toFixed(3)}
          </p>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={resData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="freq" stroke="#64748b" tick={{ fontSize: 11 }} unit=" Hz" />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0d111a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line
                type="monotone"
                dataKey="original_res"
                name="Original Residual"
                stroke="#f43f5e"
                strokeWidth={1.6}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="processed_res"
                name="Processed Residual"
                stroke="#10b981"
                strokeWidth={1.6}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
