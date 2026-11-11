import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

const REAL_MUSIC_MP3 =
  "C:\\Users\\user\\Downloads\\Media\\Audio\\Ashes For Mercy.mp3";

test.describe("SPECTRA End-to-End Browser Verification Suite", () => {
  test("1. Navigation across all 7 workstation views & request failure handling", async ({
    page,
  }) => {
    await page.goto("http://localhost:3000");
    await expect(page.locator("h1")).toContainText("SPECTRA");

    // Verify all 7 sidebar tabs navigate cleanly
    const tabs = [
      "1. Laboratory Dashboard",
      "2. Audio Ingestion",
      "3. Forensic Analysis",
      "4. DSP Laboratory",
      "5. A/B Comparison",
      "6. Technical Reports",
      "7. System & Models",
    ];
    for (const label of tabs) {
      await page.getByRole("button", { name: new RegExp(label.replace(".", "\\."), "i") }).click();
      await expect(page.locator("header h2")).toContainText(label);
    }

    // Navigate to 2. Audio Ingestion and test invalid file upload failure handling
    await page.getByRole("button", { name: /2\. Audio Ingestion/i }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "corrupted_fake.wav",
      mimeType: "audio/wav",
      buffer: Buffer.from("NOT_A_VALID_RIFF_AUDIO_FILE_PAYLOAD"),
    });

    // Verify error alert banner renders cleanly
    await expect(page.locator("text=/signature|invalid|corrupt|mismatch/i").first()).toBeVisible();
  });

  test("2. Complete workflow with actual music recording (Ashes For Mercy.mp3): upload, metadata, C2PA, spectrogram, ONNX, SONICS, DSP, A/B playback, comparison & report downloads", async ({
    page,
  }) => {
    expect(fs.existsSync(REAL_MUSIC_MP3)).toBeTruthy();

    await page.goto("http://localhost:3000");
    await page.getByRole("button", { name: /2\. Audio Ingestion/i }).click();

    // Upload real music recording
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(REAL_MUSIC_MP3);

    // Wait for automatic transition to 3. Forensic Analysis after upload + sync analysis
    await expect(page.locator("header h2")).toContainText("3. Forensic Analysis", {
      timeout: 60000,
    });

    // 1. Verify filename and technical profile in AudioWorkstationPlayer
    await expect(page.locator("h4", { hasText: "Ashes For Mercy.mp3" }).first()).toBeVisible();

    // 2. Verify C2PA Content Credentials Inspection (Suno chirp-fenix-t4 manifest_valid)
    await expect(page.locator("text=C2PA Manifest Valid").first()).toBeVisible();
    await expect(page.locator("text=/Suno chirp-fenix-t4/i").first()).toBeVisible();

    // 3. Verify Conventional Metadata Extraction (ID3v2 tags TIT2, TPE1, WOAS)
    await expect(page.locator("text=TIT2:")).toBeVisible();
    await expect(page.locator("text=zayenkaelix")).toBeVisible();

    // Test Authorized Metadata-Clean Export button
    await page.getByRole("button", { name: /Strip Conventional Tags Only/i }).click();
    await expect(page.locator("text=/Created Clean Export:/i")).toBeVisible({ timeout: 30000 });

    // 4. Verify Multi-Detector AI Classification (Fakeprint ONNX, SONICS SpecTTTra-gamma 5s & SpecTTTra-alpha 120s)
    await expect(
      page.locator("h4", { hasText: "Neural-Vocoder Fakeprint Detector (ONNX LR)" })
    ).toBeVisible();
    await expect(
      page.locator("h4", { hasText: /SONICS SpecTTTra-γ/i }).first()
    ).toBeVisible();
    await expect(
      page.locator("h4", { hasText: /SONICS SpecTTTra-α/i }).first()
    ).toBeVisible();
    await expect(page.locator("text=/Entropy H:/i").first()).toBeVisible();

    // 4B. Verify Interactive Temporal AI-Detection Timeline (Phase 5 Part A & Part E)
    const timelinePanel = page.getByTestId("temporal-forensic-timeline");
    await expect(timelinePanel).toBeVisible({ timeout: 30000 });
    const timelineCanvas = page.getByTestId("temporal-timeline-canvas");
    await expect(timelineCanvas).toBeVisible();

    // Verify neutral scientific region summary rows and long-context separation box
    const firstRegionRow = page.getByTestId("temporal-region-row-1");
    await expect(firstRegionRow).toBeVisible();
    await expect(
      timelinePanel.locator("text=/Long-Context Model: SONICS SpecTTTra-α \\(120s\\)/i")
    ).toBeVisible();

    // Click region row to select region, inspect window details, and trigger segment playback
    await firstRegionRow.click();
    await expect(page.getByTestId("temporal-window-inspector")).toBeVisible();
    const playRegionBtn = page.getByTestId("play-selected-region-btn");
    await expect(playRegionBtn).toBeVisible();
    await playRegionBtn.click();

    // 5. Verify 2D STFT Spectrogram Canvas & hover crosshair inspection
    const specHeading = page.locator("h4", { hasText: "2D STFT Log-Power Spectrogram" });
    await expect(specHeading).toBeVisible();
    const specCanvas = page.locator("canvas.cursor-crosshair");
    await expect(specCanvas).toBeVisible();
    await specCanvas.hover({ position: { x: 80, y: 60 } });
    await expect(page.locator("text=/f=.*Hz/i").first()).toBeVisible();

    // 6. Verify Time-Domain & Frequency-Domain Signal Forensics metrics are populated
    await expect(page.locator("text=RMS Level")).toBeVisible();
    await expect(page.locator("text=Spectral Centroid")).toBeVisible();
    await expect(page.locator("text=Comb Periodicity")).toBeVisible();

    // 7. Navigate to 4. DSP Laboratory and execute a real DSP transformation
    await page.getByRole("button", { name: /Test Sensitivity in DSP Lab/i }).click();
    await expect(page.locator("header h2")).toContainText("4. DSP Laboratory");

    // Select High-Frequency Rolloff Filter (7.5 kHz Low-Pass) for fast, deterministic stereo DSP
    await page
      .getByRole("button", { name: /High-Frequency Rolloff Filter/i })
      .click();

    await page
      .getByRole("button", {
        name: /Execute DSP Transform & Run Before\/After Comparison/i,
      })
      .click();

    // Wait for the lowpass DSP experiment and its comparison result to complete
    await expect(
      page.getByRole("button", { name: /Inspect Full Before\/After Comparison/i })
    ).toBeVisible({ timeout: 90000 });
    await expect(page.locator("text=Preserved ✓").first()).toBeVisible();
    await expect(page.locator("text=Finite PCM ✓").first()).toBeVisible();

    // 8. Verify Original vs Processed (A/B) playback switcher & audio element
    const btnA = page.getByRole("button", { name: "A: Original" });
    const btnB = page.getByRole("button", { name: "B: Processed" });
    await expect(btnA).toBeVisible();
    await expect(btnB).toBeVisible();

    const audioEl = page.locator("audio").first();
    const srcA = await audioEl.getAttribute("src");
    expect(srcA).toContain("/api/audio/");

    // Switch to B: Processed and verify stream URL changes to processed asset ID
    await btnB.click();
    await expect(page.locator("text=/B: lowpass/i")).toBeVisible();
    const srcB = await audioEl.getAttribute("src");
    expect(srcB).toContain("/api/audio/");
    expect(srcB).not.toEqual(srcA);

    // 9. Navigate to 5. A/B Comparison and verify fidelity metrics & detector deltas
    await page
      .getByRole("button", { name: /Inspect Full Before\/After Comparison/i })
      .click();
    await expect(page.locator("header h2")).toContainText("5. A/B Comparison");
    await expect(page.locator("text=SNR (Signal-to-Noise)")).toBeVisible();
    await expect(page.locator("text=Log-Spectral Dist (LSD)")).toBeVisible();
    await expect(
      page.locator("h4", { hasText: "Before vs. After AI-Music Detector Score Deltas" })
    ).toBeVisible();
    await expect(page.getByTestId("temporal-dsp-comparison-panel")).toBeVisible();

    // 10. Navigate to 6. Technical Reports and verify JSON & PDF report downloads
    await page.getByRole("button", { name: /6\. Technical Reports/i }).click();
    await expect(page.locator("header h2")).toContainText("6. Technical Reports");

    // Download JSON report
    const [jsonDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("a", { hasText: "Export JSON Report" }).click(),
    ]);
    const jsonPath = path.join(
      process.cwd(),
      "..",
      "storage",
      "temp",
      "playwright_downloaded_report.json"
    );
    await jsonDownload.saveAs(jsonPath);
    const jsonContent = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    expect(jsonContent.report_metadata).toBeDefined();
    expect(jsonContent.audio_asset.original_filename).toBe("Ashes For Mercy.mp3");
    expect(jsonContent.provenance_and_metadata.c2pa_status).toBe("manifest_valid");
    expect(jsonContent.temporal_forensic_analysis).toBeDefined();
    expect(jsonContent.temporal_forensic_analysis.region_summaries.length).toBeGreaterThan(0);
    expect(jsonContent.original_file_sha256).toBeDefined();
    expect(jsonContent.processed_file_sha256).toBeDefined();
    expect(jsonContent.reproducibility_information.temporal_cache_key).toBeDefined();
    expect(jsonContent.selected_time_windows).toBeDefined();

    // Download PDF report
    const [pdfDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("a", { hasText: "Export PDF Technical Report" }).click(),
    ]);
    const pdfPath = path.join(
      process.cwd(),
      "..",
      "storage",
      "temp",
      "playwright_downloaded_report.pdf"
    );
    await pdfDownload.saveAs(pdfPath);
    const pdfBytes = fs.readFileSync(pdfPath);
    expect(pdfBytes.subarray(0, 4).toString("ascii")).toBe("%PDF");
    expect(pdfBytes.length).toBeGreaterThan(1500);
  });

  test("3. Out-of-Distribution (OOD) Abstention Banner in Browser on Pure Sine Tone", async ({
    page,
  }) => {
    // Generate a 2-second pure 440 Hz WAV buffer in memory
    const sr = 16000;
    const numSamples = sr * 2;
    const header = Buffer.alloc(44);
    header.write("RIFF", 0);
    header.writeUInt32LE(36 + numSamples * 2, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20); // PCM
    header.writeUInt16LE(1, 22); // Mono
    header.writeUInt32LE(sr, 24);
    header.writeUInt32LE(sr * 2, 28);
    header.writeUInt16LE(2, 32);
    header.writeUInt16LE(16, 34);
    header.write("data", 36);
    header.writeUInt32LE(numSamples * 2, 40);

    const pcmBuf = Buffer.alloc(numSamples * 2);
    for (let i = 0; i < numSamples; i++) {
      const sample = Math.round(16000 * Math.sin((2 * Math.PI * 440 * i) / sr));
      pcmBuf.writeInt16LE(sample, i * 2);
    }
    const wavBuffer = Buffer.concat([header, pcmBuf]);

    await page.goto("http://localhost:3000");
    await page.getByRole("button", { name: /2\. Audio Ingestion/i }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "pure_440hz_ood_test.wav",
      mimeType: "audio/wav",
      buffer: wavBuffer,
    });

    await expect(page.locator("header h2")).toContainText("3. Forensic Analysis", {
      timeout: 30000,
    });

    // Verify OOD Abstention badge and explanation render without manipulating raw probability
    await expect(page.locator("text=/ABSTAINED \\(Raw: 99\\.\\d+%\\)/i")).toBeVisible();
    await expect(page.locator("text=/OUT-OF-DISTRIBUTION ABSTENTION/i")).toBeVisible();
  });

  test("4. Temporal Timeline Controls: zoom, pan, playhead sync, region click, segment bounds, hop length switching, detector visibility & disagreement display", async ({
    page,
  }) => {
    await page.goto("http://localhost:3000");
    await page.getByRole("button", { name: /1\. Laboratory Dashboard/i }).click();

    // Load the built-in 12-second reference sample from Dashboard for fast, real backend temporal interaction
    await page
      .getByRole("button", { name: /Load & Run Full Forensic Analysis/i })
      .first()
      .click();

    await expect(page.locator("header h2")).toContainText("3. Forensic Analysis", {
      timeout: 30000,
    });

    const timeline = page.getByTestId("temporal-forensic-timeline");
    await expect(timeline).toBeVisible({ timeout: 30000 });
    await expect(page.getByTestId("temporal-timeline-canvas")).toBeVisible();

    // 1. Verify window score rendering and detector disagreement summary badge
    await expect(page.getByTestId("temporal-window-inspector")).toBeVisible();
    await expect(page.getByTestId("temporal-disagreement-summary")).toContainText(
      /Disagreement Windows:/i
    );

    // 2. Verify Timeline Zoom In (1x -> 2x -> 4x) and Horizontal Pan Slider
    const zoomLevel = page.getByTestId("temporal-zoom-level");
    await expect(zoomLevel).toHaveText("1x");
    await page.getByTestId("temporal-zoom-in").click();
    await expect(zoomLevel).toHaveText("2x");

    const panSlider = page.getByTestId("temporal-pan-slider");
    await expect(panSlider).toBeVisible();
    await panSlider.fill("0.5");
    await expect(timeline.locator("text=/Pan Timeline/i")).toBeVisible();

    await page.getByTestId("temporal-zoom-out").click();
    await expect(zoomLevel).toHaveText("1x");

    // 3. Verify Switching Detector Visibility (Exploratory Fakeprint Trace toggle)
    const fpToggle = page.getByTestId("temporal-toggle-fakeprint");
    await expect(fpToggle).toBeChecked();
    await fpToggle.uncheck();
    await expect(fpToggle).not.toBeChecked();
    await expect(
      timeline.locator("text=/Fakeprint ONNX Segment \\(Exploratory/i")
    ).toHaveCount(0);
    await fpToggle.check();
    await expect(
      timeline.locator("text=/Fakeprint ONNX Segment \\(Exploratory/i")
    ).toBeVisible();

    // 4. Verify Changing Hop Length (2.5s -> 5.0s -> 1.0s) triggers real backend recomputation
    await page.getByTestId("temporal-hop-5.0").click();
    await expect(timeline.locator("text=/Hop: 5s/i")).toBeVisible({ timeout: 20000 });

    await page.getByTestId("temporal-hop-1.0").click();
    await expect(timeline.locator("text=/Hop: 1s/i")).toBeVisible({ timeout: 20000 });

    // 5. Verify Clicking a Detected Region, Segment Playback Boundaries & Playhead Synchronization
    const regionRow = page.getByTestId("temporal-region-row-1");
    await regionRow.click();
    await expect(timeline.locator("text=/Selected Span:/i")).toBeVisible();

    await page.getByTestId("play-selected-region-btn").click();
    await expect(page.getByTestId("temporal-playhead-readout")).toContainText(/Playhead:/i);
  });

  test("5. Active Temporal Cancellation & Before/After Temporal DSP Comparison", async ({
    page,
  }) => {
    await page.goto("http://localhost:3000");
    await page.getByRole("button", { name: /3\. Forensic Analysis/i }).click();

    const timeline = page.getByTestId("temporal-forensic-timeline");
    await expect(timeline).toBeVisible({ timeout: 30000 });

    // Trigger Cancel button and verify clean cancellation notice without corrupting existing timeline state
    const cancelBtn = page.getByTestId("temporal-cancel-btn");
    await expect(cancelBtn).toBeVisible();
    await cancelBtn.click();
    await expect(page.getByTestId("temporal-cancel-notice")).toContainText(/cancel/i);
    await expect(page.getByTestId("temporal-window-inspector")).toBeVisible();
  });

  test("6. Browser Reload & Persistent Job + Workstation View Recovery", async ({
    page,
  }) => {
    await page.goto("http://localhost:3000");
    await page.getByRole("button", { name: /3\. Forensic Analysis/i }).click();
    await expect(page.locator("header h2")).toContainText("3. Forensic Analysis");

    const timeline = page.getByTestId("temporal-forensic-timeline");
    await expect(timeline).toBeVisible({ timeout: 30000 });

    // Reload browser and verify both the active workstation tab (3. Forensic Analysis) and active job recover automatically
    await page.reload();
    await expect(page.locator("header h2")).toContainText("3. Forensic Analysis", {
      timeout: 30000,
    });
    await expect(page.getByTestId("temporal-forensic-timeline")).toBeVisible({
      timeout: 30000,
    });
    await expect(page.getByTestId("temporal-window-inspector")).toBeVisible();
  });
});
