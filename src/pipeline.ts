/**
 * High-level watermark-removal pipeline.
 *
 * This is the "no UI" path: detect → build a glyph/shape mask → pick a
 * backend → inpaint → return pixels plus timings. Everything runs in the
 * browser; no image ever leaves the device.
 */
import {
  backgroundComplexity,
  buildGlyphBitmap,
  buildGlyphMaskFullRes,
  detectWatermarks,
  dilate,
  type Box,
  type BlobMask,
} from './detect';
import { fastFillEligible, fastFillRegion } from './fast-fill';
import {
  findMaskBbox,
  getInpaintBackend,
  initInpainter,
  initLama,
  inpaint,
  maskStats,
  type MaskStats,
} from './inpainter';
import type { ModelProgress } from './model-loader';
import { refineShapeMask } from './shape-refine';

export type RemoveQuality = 'fast' | 'balanced' | 'hq';
export type RemoveEngine = 'auto' | 'migan' | 'lama' | 'fast-fill';

export interface RemoveWatermarkOptions {
  /** Speed/quality preset. Default: `balanced`. */
  quality?: RemoveQuality;
  /** Force a backend. `auto` follows `quality`. */
  engine?: RemoveEngine;
  /** Manual mask. When present, detection is skipped. */
  mask?: ImageData;
  /** Manual boxes. When present, detection is skipped. */
  boxes?: Box[];
  /** Restrict auto-detection to a region (original-image coordinates). */
  region?: Box;
  useText?: boolean;
  useLight?: boolean;
  useDark?: boolean;
  useYolo?: boolean;
  autoGate?: boolean;
  autoArbitrate?: boolean;
  maxSide?: number;
  textDetCap?: number;
  /** MI-GAN passes. Default: 2 for balanced, 1 for fast/hq. */
  passes?: 1 | 2;
  /** Allow the classical fast-fill path when the mask is small/simple. */
  fastFill?: boolean;
  onProgress?: (p: ModelProgress) => void;
  onStage?: (stage: string) => void;
  onPass?: (passIndex: number, totalPasses: number) => void;
}

export interface RemoveWatermarkTimings {
  detectMs: number;
  maskMs: number;
  initMs: number;
  inpaintMs: number;
  totalMs: number;
}

export interface RemoveWatermarkResult {
  /** Result pixels (same dimensions as the input). */
  image: ImageData;
  /** Binary mask used for inpainting (white = removed, transparent = kept). */
  mask: ImageData;
  boxes: Box[];
  removed: boolean;
  engine: 'migan' | 'lama' | 'fast-fill' | 'none';
  backend: string;
  stats: MaskStats;
  timings: RemoveWatermarkTimings;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function fillBox(
  out: Uint8Array,
  W: number,
  H: number,
  box: { x: number; y: number; w: number; h: number },
  pad = 2,
): void {
  const x0 = Math.max(0, Math.floor(box.x - pad));
  const y0 = Math.max(0, Math.floor(box.y - pad));
  const x1 = Math.min(W, Math.ceil(box.x + box.w + pad));
  const y1 = Math.min(H, Math.ceil(box.y + box.h + pad));
  for (let y = y0; y < y1; y++) {
    const row = y * W;
    for (let x = x0; x < x1; x++) out[row + x] = 1;
  }
}

/** Stamp a low-resolution binary mask into the full-resolution mask. */
function stampMask(
  out: Uint8Array,
  W: number,
  H: number,
  src: { data: Uint8Array; w: number; h: number },
): void {
  if (src.w <= 0 || src.h <= 0) return;
  const fx = W / src.w;
  const fy = H / src.h;
  for (let y = 0; y < src.h; y++) {
    const y0 = Math.max(0, Math.floor(y * fy));
    const y1 = Math.min(H, Math.ceil((y + 1) * fy));
    for (let x = 0; x < src.w; x++) {
      if (!src.data[y * src.w + x]) continue;
      const x0 = Math.max(0, Math.floor(x * fx));
      const x1 = Math.min(W, Math.ceil((x + 1) * fx));
      for (let yy = y0; yy < y1; yy++) {
        const row = yy * W;
        for (let xx = x0; xx < x1; xx++) out[row + xx] = 1;
      }
    }
  }
}

function countInBox(
  mask: Uint8Array,
  W: number,
  H: number,
  box: { x: number; y: number; w: number; h: number },
): number {
  const x0 = Math.max(0, Math.floor(box.x));
  const y0 = Math.max(0, Math.floor(box.y));
  const x1 = Math.min(W, Math.ceil(box.x + box.w));
  const y1 = Math.min(H, Math.ceil(box.y + box.h));
  let n = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * W;
    for (let x = x0; x < x1; x++) n += mask[row + x];
  }
  return n;
}

function toMaskImageData(bits: Uint8Array, W: number, H: number): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < bits.length; i++) {
    if (!bits[i]) continue;
    const j = i * 4;
    data[j] = 255;
    data[j + 1] = 255;
    data[j + 2] = 255;
    data[j + 3] = 255;
  }
  return new ImageData(data, W, H);
}

function buildMask(
  image: ImageData,
  boxes: Box[],
  candidates?: { combined?: BlobMask },
): ImageData {
  const W = image.width;
  const H = image.height;
  const out = new Uint8Array(W * H);
  const textBoxes = boxes.filter((b) => b.source === 'text');

  // Text boxes get a glyph-level mask when the detector's probability map is
  // available. If a box ends up nearly empty (faint text), fall back to its
  // rectangle so the watermark is not silently left behind.
  if (textBoxes.length > 0) {
    const glyph =
      buildGlyphMaskFullRes(textBoxes, image) ?? buildGlyphBitmap(textBoxes);
    if (glyph) stampMask(out, W, H, glyph);
    for (const b of textBoxes) {
      if (countInBox(out, W, H, b) < Math.max(8, b.w * b.h * 0.02)) fillBox(out, W, H, b, 2);
    }
  }

  for (const b of boxes) {
    if (b.source === 'text') continue;
    const refined = candidates?.combined
      ? refineShapeMask(image, b, candidates.combined)
      : null;
    if (refined) stampMask(out, W, H, refined);
    else fillBox(out, W, H, b, 2);
  }

  return toMaskImageData(dilate(out, W, H, 2), W, H);
}

/**
 * Remove watermarks from an ImageData object.
 *
 * The function is intentionally stateful only through the model/session caches:
 * call it repeatedly on frames or images and the model is reused.
 */
export async function removeWatermark(
  image: ImageData,
  options: RemoveWatermarkOptions = {},
): Promise<RemoveWatermarkResult> {
  const t0 = now();
  const quality = options.quality ?? 'balanced';
  const W = image.width;
  const H = image.height;
  if (W <= 0 || H <= 0) throw new Error('Image has no pixels');

  let boxes = options.boxes ?? [];
  let mask: ImageData;
  let detectMs = 0;

  if (options.mask) {
    if (options.mask.width !== W || options.mask.height !== H) {
      throw new Error('Mask and image dimensions must match');
    }
    mask = options.mask;
  } else if (boxes.length === 0) {
    options.onStage?.('Detecting watermarks…');
    const tDetect = now();
    const candidates: { combined?: BlobMask } = {};
    boxes = await detectWatermarks(image, {
      region: options.region,
      useText: options.useText,
      useLight: options.useLight,
      useDark: options.useDark,
      useYolo: options.useYolo,
      autoGate: options.autoGate ?? false,
      autoArbitrate: options.autoArbitrate ?? false,
      maxSide: options.maxSide,
      textDetCap: options.textDetCap,
      candidatesOut: candidates,
      onModelProgress: options.onProgress,
    });
    detectMs = now() - tDetect;
    const tMask = now();
    mask = buildMask(image, boxes, candidates);
    const maskMs = now() - tMask;
    if (maskStats(mask).comps === 0) {
      return {
        image,
        mask,
        boxes,
        removed: false,
        engine: 'none',
        backend: '',
        stats: maskStats(mask),
        timings: { detectMs, maskMs, initMs: 0, inpaintMs: 0, totalMs: now() - t0 },
      };
    }
    return runInpaint(image, mask, boxes, quality, options, detectMs, maskMs, t0);
  } else {
    options.onStage?.('Building mask…');
    const tMask = now();
    mask = buildMask(image, boxes);
    const maskMs = now() - tMask;
    return runInpaint(image, mask, boxes, quality, options, 0, maskMs, t0);
  }

  const tMask = now();
  const maskMs = now() - tMask;
  return runInpaint(image, mask, boxes, quality, options, detectMs, maskMs, t0);
}

async function runInpaint(
  image: ImageData,
  mask: ImageData,
  boxes: Box[],
  quality: RemoveQuality,
  options: RemoveWatermarkOptions,
  detectMs: number,
  maskMs: number,
  t0: number,
): Promise<RemoveWatermarkResult> {
  const stats = maskStats(mask);
  if (stats.comps === 0) {
    return {
      image,
      mask,
      boxes,
      removed: false,
      engine: 'none',
      backend: '',
      stats,
      timings: { detectMs, maskMs, initMs: 0, inpaintMs: 0, totalMs: now() - t0 },
    };
  }

  const bbox = findMaskBbox(mask);
  const ringStd =
    bbox != null
      ? backgroundComplexity(image, { ...bbox, source: 'light' }).std
      : 0;
  const fastOk =
    options.fastFill !== false &&
    bbox != null &&
    fastFillEligible(image, bbox, ringStd, stats.maskFrac).ok;

  let engine: 'migan' | 'lama' | 'fast-fill';
  if (options.engine === 'migan' || options.engine === 'lama') {
    engine = options.engine;
  } else if (options.engine === 'fast-fill') {
    engine = fastOk ? 'fast-fill' : 'migan';
  } else if (quality === 'hq') {
    engine = 'lama';
  } else if (quality === 'fast' && fastOk) {
    engine = 'fast-fill';
  } else {
    engine = 'migan';
  }

  if (engine === 'fast-fill') {
    options.onStage?.('Fast fill…');
    const t = now();
    const filled = fastFillRegion(image, mask, bbox ?? undefined);
    const inpaintMs = now() - t;
    return {
      image: filled?.data ?? image,
      mask,
      boxes,
      removed: !!filled,
      engine: 'fast-fill',
      backend: 'classical',
      stats,
      timings: {
        detectMs,
        maskMs,
        initMs: 0,
        inpaintMs,
        totalMs: now() - t0,
      },
    };
  }

  options.onStage?.(engine === 'lama' ? 'Loading LaMa HQ model…' : 'Loading MI-GAN model…');
  const tInit = now();
  if (engine === 'lama') await initLama(options.onProgress);
  else await initInpainter(options.onProgress);
  const initMs = now() - tInit;

  options.onStage?.(engine === 'lama' ? 'Inpainting (HQ)…' : 'Inpainting…');
  const tInpaint = now();
  const result = await inpaint(image, mask, {
    engine,
    passes: options.passes ?? (quality === 'fast' ? 1 : 2),
    onPass: options.onPass,
  });
  const inpaintMs = now() - tInpaint;

  return {
    image: result,
    mask,
    boxes,
    removed: true,
    engine,
    backend: getInpaintBackend(),
    stats,
    timings: {
      detectMs,
      maskMs,
      initMs,
      inpaintMs,
      totalMs: now() - t0,
    },
  };
}
