# ClearPix integration notes

ClearPix already owns the UI: upload, mask history, presets, before/after, batch,
video, and feedback. This package is the headless core behind the image path, so
integration should be an adapter, not a rewrite.

## Image path

```ts
import { configureCore, removeWatermark, toImageData } from 'browser-watermark-removal';

configureCore({
  modelBase: import.meta.env.VITE_STATIC_ORIGIN, // ClearPix model origin
  ortWasmPaths: 'https://registry.npmmirror.com/onnxruntime-web/1.30.0/files/dist/',
  cacheName: 'clearpix-models',
});

const input = await toImageData(file);
const result = await removeWatermark(input, {
  quality: 'balanced',
  onStage: (stage) => setStatus(stage),
  onProgress: (p) => setProgress(p),
});

if (result.removed) {
  setPreview(result.image);
  setMask(result.mask);
  setBoxes(result.boxes);
  logTimings(result.timings);
}
```

The package returns the mask and boxes, so the existing brush editor can still
let the user correct a bad automatic result. Manual paths can pass `mask` or
`boxes` and skip detection.

## What stays in ClearPix

- The workbench, result feedback events, history/undo, and export.
- Video frame loop and ffmpeg assembly.
- The manual selection UI and its shape-refinement commands.
- Product copy, pricing, and the ClearPix brand.

## What moves here

- `detect.ts` / `shape-refine.ts` / `fast-fill.ts` / the image part of
  `inpainter.ts`.
- Model loading, runtime-origin racing, IndexedDB cache, and SHA-256 checks.
- The auto-detect → glyph mask → route → inpaint pipeline.

## Upgrade path

ClearPix should depend on a version range (`^0.1`) and let package releases
improve detection/mask/inpaint quality without changing the workbench. If a
release changes model file names, bump the package minor version and update
`modelBase` hosting in the same deploy.

## Guardrails

- Do not bundle model weights in the Pages package; keep R2 + SHA-256 dedupe.
- Keep `useYolo` off unless the AGPL-3.0 implications are accepted.
- Keep the exact Impact meta and the split Clarity events in the app layer.
- Do not move Russian copy or SEO route generation into this package.
