const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "CNN-LSTM Evaluation";
pres.title = "CNN-LSTM Traffic Sign Recognition - Robustness Evaluation";

const PRIMARY = "1E2761";
const SECONDARY = "CADCFC";
const ACCENT = "2196F3";
const BG_DARK = "0F1729";
const BG_LIGHT = "F8FAFC";
const TEXT_DARK = "1E293B";
const TEXT_MUTED = "64748B";
const GREEN = "4CAF50";
const ORANGE = "FF9800";
const RED = "F44336";

const figDir = path.resolve("eval_outputs/figures");

// ── Slide 1: Title ──────────────────────────────────
let s1 = pres.addSlide();
s1.background = { color: BG_DARK };
s1.addText("CNN-LSTM Robustness Evaluation", {
    x: 0.8, y: 1.2, w: 8.4, h: 1.2,
    fontSize: 36, fontFace: "Georgia", color: "FFFFFF", bold: true
});
s1.addText("Traffic Sign Recognition Under Adversarial Corruption", {
    x: 0.8, y: 2.5, w: 8.4, h: 0.8,
    fontSize: 18, fontFace: "Calibri", color: SECONDARY
});
s1.addText("GTSRB Dataset  |  ResNet-18 + LSTM  |  0% / 30% / 50% Corruption", {
    x: 0.8, y: 3.8, w: 8.4, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: TEXT_MUTED
});
s1.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 3.4, w: 2.5, h: 0.04, fill: { color: ACCENT }
});

// ── Slide 2: Key Metrics Summary ────────────────────
let s2 = pres.addSlide();
s2.background = { color: BG_LIGHT };
s2.addText("Key Metrics Summary", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});

const metrics = [
    { label: "Clean Accuracy", value: "93.3%", color: GREEN },
    { label: "30% Corruption", value: "92.2%", color: ORANGE },
    { label: "50% Corruption", value: "90.5%", color: RED },
];
metrics.forEach((m, i) => {
    const xPos = 0.6 + i * 3.1;
    s2.addShape(pres.shapes.RECTANGLE, {
        x: xPos, y: 1.3, w: 2.8, h: 1.8,
        fill: { color: "FFFFFF" },
        shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.1 }
    });
    s2.addText(m.value, {
        x: xPos, y: 1.5, w: 2.8, h: 0.9,
        fontSize: 40, fontFace: "Georgia", color: m.color, bold: true, align: "center"
    });
    s2.addText(m.label, {
        x: xPos, y: 2.35, w: 2.8, h: 0.5,
        fontSize: 13, fontFace: "Calibri", color: TEXT_MUTED, align: "center"
    });
});

// Bottom stats row
const bottomStats = [
    { label: "ECE (Clean)", value: "0.0427" },
    { label: "ECE (50%)", value: "0.0651" },
    { label: "Safety Rate (Clean)", value: "94.2%" },
    { label: "Safety Rate (50%)", value: "92.1%" },
];
bottomStats.forEach((st, i) => {
    const xPos = 0.6 + i * 2.3;
    s2.addText(st.value, {
        x: xPos, y: 3.5, w: 2.0, h: 0.6,
        fontSize: 20, fontFace: "Georgia", color: TEXT_DARK, bold: true, align: "center"
    });
    s2.addText(st.label, {
        x: xPos, y: 4.05, w: 2.0, h: 0.4,
        fontSize: 10, fontFace: "Calibri", color: TEXT_MUTED, align: "center"
    });
});

// ── Slide 3: Robustness Curves ──────────────────────
let s3 = pres.addSlide();
s3.background = { color: BG_LIGHT };
s3.addText("Robustness Analysis", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});
s3.addImage({
    path: path.join(figDir, "robustness_curves.png"),
    x: 0.3, y: 1.1, w: 9.4, h: 4.3
});

// ── Slide 4: Attention Heatmaps ─────────────────────
let s4 = pres.addSlide();
s4.background = { color: BG_LIGHT };
s4.addText("Attention Heatmaps: Clean vs Attacked", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});
s4.addImage({
    path: path.join(figDir, "attention_heatmaps.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.4
});

// ── Slide 5: Confidence & Calibration ───────────────
let s5 = pres.addSlide();
s5.background = { color: BG_LIGHT };
s5.addText("Confidence & Calibration", {
    x: 0.6, y: 0.2, w: 8.8, h: 0.6,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});
s5.addImage({
    path: path.join(figDir, "confidence_distributions.png"),
    x: 0.2, y: 0.85, w: 9.6, h: 2.2
});
s5.addImage({
    path: path.join(figDir, "calibration_diagrams.png"),
    x: 0.2, y: 3.1, w: 9.6, h: 2.2
});

// ── Slide 6: Confusion Matrices ─────────────────────
let s6 = pres.addSlide();
s6.background = { color: BG_LIGHT };
s6.addText("Confusion Matrices", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});
s6.addImage({
    path: path.join(figDir, "confusion_heatmaps.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.3
});

// ── Slide 7: Per-Class Breakdown ────────────────────
let s7 = pres.addSlide();
s7.background = { color: BG_LIGHT };
s7.addText("Per-Class Robustness", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
});
s7.addImage({
    path: path.join(figDir, "per_class_robustness.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.3
});

// ── Slide 8: Conclusions ────────────────────────────
let s8 = pres.addSlide();
s8.background = { color: BG_DARK };
s8.addText("Conclusions", {
    x: 0.8, y: 0.4, w: 8.4, h: 0.8,
    fontSize: 32, fontFace: "Georgia", color: "FFFFFF", bold: true
});
s8.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 1.2, w: 2.0, h: 0.04, fill: { color: ACCENT }
});

const conclusions = [
    { text: "Temporal LSTM reasoning provides robustness gains over single-frame models", options: { bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
    { text: "Clean accuracy: 93.3% — model learns GTSRB features effectively", options: { bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
    { text: "30% corruption: accuracy drops by 1.1pp — moderate resilience", options: { bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
    { text: "50% corruption: accuracy drops by 2.8pp — significant but not catastrophic", options: { bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
    { text: "Calibration degrades under attack — model becomes overconfident on wrong predictions", options: { bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
    { text: "Action safety rate remains strong — the model's low-confidence failures are recoverable", options: { bullet: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" } },
];
s8.addText(conclusions, {
    x: 0.8, y: 1.6, w: 8.4, h: 3.5
});

pres.writeFile({ fileName: "eval_outputs/CNN_LSTM_Evaluation.pptx" }).then(() => {
    console.log("Presentation saved: CNN_LSTM_Evaluation.pptx");
});
