import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory",
  description:
    "Local-first audio forensics, C2PA Content Credentials verification, neural-vocoder spectral fakeprint extraction, multi-detector AI music analysis, and experimental DSP laboratory.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#07090e] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
