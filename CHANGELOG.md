# Changelog

## 0.1.0 — 2026-10-24

- Extracted the ClearPix browser watermark pipeline into a standalone,
  framework-free package.
- Added `removeWatermark(ImageData)` with auto-detection, glyph-level masks,
  shape refinement, fast-fill / MI-GAN / LaMa routing, and timings.
- Added configurable model and ONNX Runtime origins (`configureCore`).
- Added a vanilla TypeScript + Vite demo with before/after slider, mask overlay,
  sample cases, and live timings.
- Added a Playwright benchmark that runs the real demo and records WebGPU and
  WASM numbers.
- Added GitHub Pages deployment workflow.
- MIT code license; model weights stay under their own licenses and are not
  bundled.
