/**
 * Automatic watermark detection — 100% in-browser.
 *
 * Three complementary detectors:
 *
 * 1. Text detector: PP-OCRv5 mobile DB text-detection model (ONNX, ~4.7 MB)
 *    via onnxruntime-web. Script-agnostic — boxes "where text is".
 *
 * 2. Light-overlay heuristic: semi-transparent white watermarks. Candidates
 *    are bright/low-saturation pixels brighter than their local background;
 *    blobs are validated by internal edge density so flat bright regions
 *    (sky, walls) are rejected — this is what keeps precision high.
 *
 * 3. Dark-badge heuristic: solid dark logo badges (the "black rounded-rect"
 *    style). Candidates are dark/low-saturation pixels darker than their
 *    local background; blobs must be solid (high fill ratio) so textured
 *    dark regions are rejected.
 *
 * All return boxes in original-image coordinates, painted into the same mask
 * canvas the brush editor uses.
 */

import type * as ort from 'onnxruntime-web';
import { createSession, getOrt } from './ort-setup';
import { loadModel, type ModelProgress, type ModelSpec } from './model-loader';

export const TEXT_DET_MODEL: ModelSpec = {
  key: 'ppocr_v5_mobile_det_v1',
  urls: [
    'models/ppocr_v5_mobile_det.onnx',
    'https://huggingface.co/snowfluke/ppu-paddle-ocr-models/resolve/main/detection/PP-OCRv5_mobile_det_infer.onnx',
    'https://hf-mirror.com/snowfluke/ppu-paddle-ocr-models/resolve/main/detection/PP-OCRv5_mobile_det_infer.onnx',
  ],
  sha256: 'd7fe3ea74652890722c0f4d02458b7261d9f5ae6c92904d05707c9eb155c7924',
  expectedSize: 4748769,
};

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  source: 'yolo' | 'text' | 'light' | 'dark';
  /** S1 solid/stroke 语义（.review/2026-10-02 §7）：solid=实心徽章体
   *  （矩形蒙版正确）、stroke=笔画/文字（字形蒙版正确）。 */
  wmType?: 'solid' | 'stroke';
  /** 镜像补跑来源（边界碎片召回）。语义护栏不信任它杀互补源框。 */
  sec?: boolean;
}

export const YOLO_MODEL: ModelSpec = {
  key: 'yolo11_watermark_det_v1',
  urls: [
    'models/yolo11_watermark_det.onnx',
    'https://huggingface.co/ayan4m1/Watermark-Detection-YOLO11-ONNX/resolve/main/onnx/model.onnx',
    'https://hf-mirror.com/ayan4m1/Watermark-Detection-YOLO11-ONNX/resolve/main/onnx/model.onnx',
  ],
  sha256: 'f3638870eedb2ac4ea202d25da35ac2ae520ba33d572479d1d9cfc671aaa253e',
  expectedSize: 11068970,
};

const READBACK = { willReadFrequently: true } as const;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function downscaleImageData(imageData: ImageData, maxSide: number): {
  data: ImageData;
  scale: number;
} {
  const { width: W, height: H } = imageData;
  const scale = Math.min(1, maxSide / Math.max(W, H));
  if (scale === 1) return { data: imageData, scale };
  const w = Math.round(W * scale);
  const h = Math.round(H * scale);
  const src = new OffscreenCanvas(W, H);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const dst = new OffscreenCanvas(w, h);
  const ctx = dst.getContext('2d', READBACK)!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h), scale };
}

/** 局部对比笔画蒙版：box-blur 局部背景估计 + 高通二值化——半透明文字
 *  与其背景的色差/亮度离群（概率图死区时 R1 之外的另一路精确笔画信号）。
 *  区域坐标系、返回 1 = 笔画像素。 */
export function localContrastStrokeMask(
  full: ImageData,
  region: { x: number; y: number; w: number; h: number },
  thr = 26,
): { data: Uint8Array; w: number; h: number } {
  const W = full.width;
  const H = full.height;
  const x0 = Math.max(0, region.x);
  const y0 = Math.max(0, region.y);
  const x1 = Math.min(W, region.x + region.w);
  const y1 = Math.min(H, region.y + region.h);
  const rw = x1 - x0;
  const rh = y1 - y0;
  if (rw <= 0 || rh <= 0) return { data: new Uint8Array(0), w: 0, h: 0 };
  const R = 12;
  // 可分离 box blur（局部背景估计：慢变背景通过，笔画被抹平）
  const blur = new Float32Array(rw * rh * 3);
  const px = new Float32Array(rw * rh * 3);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const j = ((y0 + y) * W + (x0 + x)) * 4;
      px[(y * rw + x) * 3] = full.data[j];
      px[(y * rw + x) * 3 + 1] = full.data[j + 1];
      px[(y * rw + x) * 3 + 2] = full.data[j + 2];
    }
  }
  const tmp = new Float32Array(rw * rh * 3);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      for (let c = 0; c < 3; c++) {
        let s2 = 0;
        let n = 0;
        for (let dx = -R; dx <= R; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= rw) continue;
          s2 += px[(y * rw + xx) * 3 + c];
          n++;
        }
        tmp[(y * rw + x) * 3 + c] = s2 / n;
      }
    }
  }
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      for (let c = 0; c < 3; c++) {
        let s2 = 0;
        let n = 0;
        for (let dy = -R; dy <= R; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= rh) continue;
          s2 += tmp[(yy * rw + x) * 3 + c];
          n++;
        }
        blur[(y * rw + x) * 3 + c] = s2 / n;
      }
    }
  }
  const mask = new Uint8Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const i = (y * rw + x) * 3;
      const dev = Math.max(
        Math.abs(px[i] - blur[i]),
        Math.abs(px[i + 1] - blur[i + 1]),
        Math.abs(px[i + 2] - blur[i + 2]),
      );
      if (dev > thr) mask[y * rw + x] = 1;
    }
  }
  return { data: mask, w: rw, h: rh };
}

/** 字形蒙版膨胀（暴露给 applyBoxes 低覆盖/边缘吞并路径）。 */
export function dilate(bin: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const out = new Uint8Array(bin.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!bin[y * w + x]) continue;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          out[ny * w + nx] = 1;
        }
      }
    }
  }
  return out;
}

function erode(bin: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const out = new Uint8Array(bin.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let ok = 1;
      for (let dy = -r; dy <= r && ok; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) {
          ok = 0;
          break;
        }
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w || !bin[ny * w + nx]) {
            ok = 0;
            break;
          }
        }
      }
      if (ok) out[y * w + x] = 1;
    }
  }
  return out;
}

function close(bin: Uint8Array, w: number, h: number, r: number): Uint8Array {
  return erode(dilate(bin, w, h, r), w, h, r);
}

interface Blob {
  x: number;
  y: number;
  w: number;
  h: number;
  area: number;
  id: number; // 连通域标签值（labelsOut 中该 blob 像素的取值）
}

function connectedComponents(
  bin: Uint8Array,
  w: number,
  h: number,
  minArea: number,
  labelsOut?: Int32Array,
): Blob[] {
  const label = new Int32Array(w * h).fill(-1);
  const blobs: Blob[] = [];
  const stack: number[] = [];
  let cur = 0;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || label[i] >= 0) continue;
    let minX = w,
      minY = h,
      maxX = -1,
      maxY = -1,
      area = 0;
    stack.length = 0;
    stack.push(i);
    label[i] = cur;
    while (stack.length) {
      const p = stack.pop()!;
      const x = p % w;
      const y = (p / w) | 0;
      area++;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (x > 0 && bin[p - 1] && label[p - 1] < 0) {
        label[p - 1] = cur;
        stack.push(p - 1);
      }
      if (x < w - 1 && bin[p + 1] && label[p + 1] < 0) {
        label[p + 1] = cur;
        stack.push(p + 1);
      }
      if (y > 0 && bin[p - w] && label[p - w] < 0) {
        label[p - w] = cur;
        stack.push(p - w);
      }
      if (y < h - 1 && bin[p + w] && label[p + w] < 0) {
        label[p + w] = cur;
        stack.push(p + w);
      }
    }
    if (area >= minArea && maxX >= 0) {
      blobs.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area, id: cur });
    }
    cur++;
  }
  if (labelsOut) labelsOut.set(label);
  return blobs;
}

function expandBox(b: Box, ratio: number, W: number, H: number): Box {
  const cx = b.x + b.w / 2;
  const cy = b.y + b.h / 2;
  const wFinal = Math.max(b.w * ratio, b.w + 10);
  const h = Math.max(b.h * ratio, b.h + 10);
  const x = Math.max(0, Math.min(W - 1, cx - wFinal / 2));
  const y = Math.max(0, Math.min(H - 1, cy - h / 2));
  return {
    x: Math.round(x),
    y: Math.round(y),
    w: Math.round(Math.min(W - x, wFinal)),
    h: Math.round(Math.min(H - y, h)),
    source: b.source,
    wmType: b.wmType,
    sec: b.sec,
  };
}

const SOURCE_PRIORITY: Record<Box['source'], number> = { yolo: 0, text: 1, light: 2, dark: 3 };

function dedupeBoxes(boxes: Box[]): Box[] {
  const prio = (b: Box) => SOURCE_PRIORITY[b.source];
  // Pass 1: 包含去重（原逻辑）
  let out = boxes.filter((b, i) => {
    for (let j = 0; j < boxes.length; j++) {
      if (i === j) continue;
      const o = boxes[j];
      if (
        b.x >= o.x &&
        b.y >= o.y &&
        b.x + b.w <= o.x + o.w &&
        b.y + b.h <= o.y + o.h
      ) {
        // Contained box loses to a higher-priority source, then to the larger box.
        if (prio(o) < prio(b)) return false;
        if (prio(o) === prio(b) && (b.w * b.h < o.w * o.h || j < i)) return false;
      }
    }
    return true;
  });
  // Pass 2: S1 语义护栏去重（2026-10-02，替代三轮失败的跨源去重）：
  // light 框为 **stroke** 型（白字笔画）且与 text 框重叠（IoU≥0.3 或
  // light 中心在 text 框内）→ 丢弃——OCR 字形路径（glyph+膨胀）覆盖同一
  // 白字，light 白字笔画是冗余从属。solid 型 light 框（白底实体徽章）
  // 与 dark/yolo 框一律保留（互补部件，徽章体必须保留——无语义护栏的
  // 三轮去重把 logo recall 打到 27.6~29% 的教训）。
  const textBoxes = out.filter((b) => b.source === 'text' && !b.sec);
  out = out.filter((b) => {
    // S1 冗余判据（coverage-based）：light+stroke 框与 text 框重叠，且其
    // 区域被 **OCR 概率图实际覆盖 ≥50%**（prob>0.3 采样）→ OCR 字形路径
    // 覆盖同一笔画，light 框冗余可丢。仅凭重叠不足立冗余（logo-br 教训：
    // light 框承载独立召回时，重叠判定会误杀）。
    if (b.source !== 'light' || b.wmType !== 'stroke') return true;
    if (!lastTextProb) return true;
    for (const t of textBoxes) {
      const iouOk = iou(b, t) >= 0.3;
      const cx = b.x + b.w / 2;
      const cy = b.y + b.h / 2;
      const inBox = cx >= t.x && cx < t.x + t.w && cy >= t.y && cy < t.y + t.h;
      if (!iouOk && !inBox) continue;
      const lp = lastTextProb;
      let sampled = 0;
      let covered = 0;
      const step = 2;
      for (let y = b.y + 1; y < b.y + b.h; y += step) {
        for (let x = b.x + 1; x < b.x + b.w; x += step) {
          const px = Math.min(lp.w - 1, Math.max(0, Math.round(x / (lp.inv * lp.mapX))));
          const py = Math.min(lp.h - 1, Math.max(0, Math.round(y / (lp.inv * lp.mapY))));
          sampled++;
          if (lp.prob[py * lp.w + px] > 0.3) covered++;
        }
      }
      if (sampled >= 8 && covered / sampled >= 0.5) return false;
    }
    return true;
  });
  return out;
}

/** Luminance map of a (downscaled) ImageData. */
function luminanceOf(data: ImageData): Float32Array {
  const d = data.data;
  const n = data.width * data.height;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    lum[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return lum;
}

/** Box-blurred copy of a luminance map. */
function boxBlur(lum: Float32Array, w: number, h: number, r: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0,
        n = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          sum += lum[ny * w + nx];
          n++;
        }
      }
      out[y * w + x] = sum / n;
    }
  }
  return out;
}

/** Mean |gradient| inside a blob — watermark text/logos are edge-rich. */
function meanGradientInside(
  lum: Float32Array,
  w: number,
  h: number,
  blob: Blob,
  mask: Uint8Array,
): number {
  let sum = 0,
    n = 0;
  for (let y = blob.y; y < blob.y + blob.h; y++) {
    for (let x = blob.x; x < blob.x + blob.w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const gx = x > 0 && x < w - 1 ? Math.abs(lum[i + 1] - lum[i - 1]) : 0;
      const gy = y > 0 && y < h - 1 ? Math.abs(lum[i + w] - lum[i - w]) : 0;
      sum += gx + gy;
      n++;
    }
  }
  return n ? sum / n : 0;
}

// ---------------------------------------------------------------------------
// 1. PP-OCR text detection
// ---------------------------------------------------------------------------

let detSession: ort.InferenceSession | null = null;

/** 最近一次文字检测的概率图（检测输入坐标系），供字形级蒙版使用。 */
interface TextProbCache {
  prob: Float32Array;
  w: number;
  h: number;
  mapX: number; // 检测输入宽 / 概率图宽
  mapY: number;
  inv: number; // 原图 / 检测输入（maxSide 缩放的还原系数）
  // region 精修的坐标契约（评审问题 B）：概率图只覆盖裁剪区，缓存记裁剪
  // 偏移与全图尺寸，buildGlyphBitmap 先把框减偏移再换算、并把蒙版拼回全图
  // 外幅（mask-history 的 bitmap op 按全幅拉伸绘制）。非 region 检测时
  // origin=0、full=0，行为与旧契约逐位一致。
  originX: number;
  originY: number;
  fullW: number; // >0 表示 region 模式（全图尺寸）
  fullH: number;
  image: Uint8ClampedArray; // 检测输入 RGBA（引用，不拷贝）
  iw: number;
  ih: number;
}
let lastTextProb: TextProbCache | null = null;

/** 诊断用只读快照（概率图坐标/数值，不导出可变引用）。 */
export function getLastTextProb(): TextProbCache | null {
  return lastTextProb;
}

function setTextProbInv(inv: number): void {
  if (lastTextProb) lastTextProb.inv = inv;
}

function setTextProbTransform(
  inv: number,
  originX: number,
  originY: number,
  fullW: number,
  fullH: number,
): void {
  if (!lastTextProb) return;
  lastTextProb.inv = inv;
  lastTextProb.originX = originX;
  lastTextProb.originY = originY;
  lastTextProb.fullW = fullW;
  lastTextProb.fullH = fullH;
}

/**
 * 用缓存的文字检测概率图，为 source='text' 的框生成字形级蒙版
 * （概率图分辨率，1 字节/像素，远小于全分辨率蒙版——内存红线考虑）。
 *
 * DB 检测头训练目标是 Vatti 收缩核，概率图高区天然比真实笔画小一圈，
 * 纯阈值蒙版欠覆盖（bench recall 崩到 ~50%、残边把 PSNR 打穿）。改为
 * 种子+生长：prob>0.5 作种子，框内与环带中值色差显著的像素作候选，
 * 只保留与种子连通的候选（背景杂色无种子不入选），最后膨胀 2px。
 * 非 text 来源的框（yolo/启发式）不在此处理，调用方维持矩形；
 * 无缓存概率图或没有 text 框时返回 null，调用方回退矩形。
 */
/**
 * P1 全分辨率字形生长（超清选区方案，2026-10-02）：buildGlyphBitmap 的
 * 全分辨率版。种子+生长全程在输出网格（≤2048 cap 的原图分辨率）上跑——
 * 候选判据（环带中值色差>30）在原生像素上采样，边界精度不再受概率图
 * 分辨率（原图 1/4-1/16）限制；A5 引导吸附同样在全分辨率执行。
 *
 * 坐标契约：lastTextProb 的概率网格覆盖 (originX, originY) 偏移的检测
 * 输入，full 像素 → 概率采样 = ((x-origin)/(inv·mapX), (y-origin)/(inv·mapY))。
 * 输出为**全幅**位图（cap 2048，MaskHistory op 内存红线），bitmap op 拉伸
 * 语义与 buildGlyphBitmap 一致。
 *
 * 调用方契约：必须先跑过 detectTextBoxes（lastTextProb 就位）；full 为
 * 原图 ImageData（非 text 框忽略）。null = 无缓存/无 text 框，调用方回退
 * buildGlyphBitmap 或矩形。
 */
export function buildGlyphMaskFullRes(
  boxes: Box[],
  full: ImageData,
): { data: Uint8Array; w: number; h: number } | null {
  const cache = lastTextProb;
  if (!cache) return null;
  const textBoxes = boxes.filter((b) => b.source === 'text');
  if (textBoxes.length === 0) return null;
  const W = full.width;
  const H = full.height;
  const s = Math.min(1, 2048 / Math.max(W, H)); // 输出网格缩放（cap 2048）
  const gw = Math.max(1, Math.round(W * s));
  const gh = Math.max(1, Math.round(H * s));
  const fd = full.data;

  const out = new Uint8Array(gw * gh);
  const fullR = (gx: number, gy: number) => {
    const fx = (gx + 0.5) / s;
    const fy = (gy + 0.5) / s;
    const x = Math.min(W - 1, Math.max(0, Math.round(fx)));
    const y = Math.min(H - 1, Math.max(0, Math.round(fy)));
    const j = (y * W + x) * 4;
    return [fd[j], fd[j + 1], fd[j + 2]] as const;
  };

  for (const b of textBoxes) {
    const padX = Math.max(4, Math.round(b.w * 0.1));
    const padY = Math.max(4, Math.round(b.h * 0.1));
    const gx0 = toGrid(b.x - padX, s, gw);
    const gy0 = toGrid(b.y - padY, s, gh);
    const gx1 = toGrid(b.x + b.w + padX, s, gw);
    const gy1 = toGrid(b.y + b.h + padY, s, gh);
    if (gx1 - gx0 < 2 || gy1 - gy0 < 2) continue;

    // 环带（框外扩）中值色——全分辨率采样
    const ring: number[][] = [];
    const ringX0 = toGrid(b.x - padX * 2, s, gw);
    const ringY0 = toGrid(b.y - padY * 2, s, gh);
    const ringX1 = toGrid(b.x + b.w + padX * 2, s, gw);
    const ringY1 = toGrid(b.y + b.h + padY * 2, s, gh);
    for (let gy = ringY0; gy <= ringY1; gy++) {
      for (let gx = ringX0; gx <= ringX1; gx++) {
        if (gx >= gx0 && gx < gx1 && gy >= gy0 && gy < gy1) continue;
        ring.push([...fullR(gx, gy)]);
      }
    }
    if (ring.length < 6) continue;
    const med = [0, 1, 2].map((ch) =>
      ring.map((p) => p[ch]).sort((a, b2) => a - b2)[ring.length >> 1],
    );

    // 候选（色差>30）+ 种子（OCR 概率>0.5），输出网格、框+pad 范围内
    const cand = new Uint8Array(gw * gh);
    const seed = new Uint8Array(gw * gh);
    let seedCount = 0;
    const bx0 = toGrid(b.x, s, gw);
    const by0 = toGrid(b.y, s, gh);
    for (let gy = gy0; gy < gy1; gy++) {
      for (let gx = gx0; gx < gx1; gx++) {
        const i = gy * gw + gx;
        const [r, g, b2] = fullR(gx, gy);
        const dist = Math.abs(r - med[0]) + Math.abs(g - med[1]) + Math.abs(b2 - med[2]);
        if (dist > 30) cand[i] = 1;
        // 全图坐标 → 概率网格（region 偏移 + inv/mapX/mapY 契约）
        const fx = (gx + 0.5) / s;
        const fy = (gy + 0.5) / s;
        const px = Math.min(cache.w - 1, Math.max(0, Math.round((fx - cache.originX) / (cache.inv * cache.mapX))));
        const py = Math.min(cache.h - 1, Math.max(0, Math.round((fy - cache.originY) / (cache.inv * cache.mapY))));
        // 种子阈值 0.3 有意低于检测框二值化阈值 0.4（淡字概率峰值 0.3-0.5，
        // 0.5 对淡字永不点火）；精度由色差候选门控保护
        if (cache.prob[py * cache.w + px] > 0.3) {
          seed[i] = 1;
          seedCount++;
        }
      }
    }
    if (seedCount === 0) continue;

    // 洪水生长（种子起、仅经候选）
    const seen = new Uint8Array(gw * gh);
    const stack: number[] = [];
    let floodN = 0;
    for (let i = 0; i < gw * gh; i++) {
      if (seed[i] && cand[i]) {
        seen[i] = 1;
        stack.push(i);
        floodN++;
      }
    }
    while (stack.length) {
      const i = stack.pop()!;
      out[i] = 1;
      const x = i % gw;
      const nb = [i - 1, i + 1, i - gw, i + gw];
      for (let k = 0; k < 4; k++) {
        const j = nb[k];
        if (k === 0 && x === 0) continue;
        if (k === 1 && x === gw - 1) continue;
        if (j < 0 || j >= gw * gh || seen[j] || !cand[j]) continue;
        seen[j] = 1;
        stack.push(j);
        floodN++;
      }
    }

    // 覆盖率双闸门（003 批次实证）：
    //  - 洪泛泄漏：cov > 60% 且 flood/seed > 8——种子稀疏却洪水满贯，说明
    //    色差候选门在大框（多为误检/过大框）围住的杂色区里失效（环带中值
    //    被图像边缘截断污染；人像图 372×349 大框 seeds=7954 却洪泛出
    //    136178 px=17×种子，实心蒙版把皮肤/丝袜交给模型重绘毁图）。
    //    退回种子核（prob>0.3 ∩ 色差>30），稀疏但精确。注意 cov 单独不可
    //    判：case-6 云上大字带 cov=0.71 但 flood/seed=3.0（种子密集的真
    //    文字），只看 cov 会把蒙版砍成字核、消除后留白块（实证回退）。
    //  - 欠覆盖 <8% 且小框（长边 ≤256px）：淡水印保底整体填充；大框欠
    //    覆盖保留稀疏字形——大框实心填充的破坏半径不可接受（003 评审
    //    判词："检测不出都比检测出错好"，漏检由手动涂抹兜底）。
    let glyphN = 0;
    for (let gy = by0; gy < by0 + Math.round(b.h * s) && gy < gh; gy++) {
      for (let gx = bx0; gx < bx0 + Math.round(b.w * s) && gx < gw; gx++) glyphN += out[gy * gw + gx];
    }
    const boxGridArea = (gx1 - gx0) * (gy1 - gy0);
    const cov = glyphN / Math.max(1, boxGridArea);
    if (cov > 0.6 && floodN / Math.max(1, seedCount) > 8) {
      for (let gy = gy0; gy < gy1; gy++) {
        for (let gx = gx0; gx < gx1; gx++) out[gy * gw + gx] = seed[gy * gw + gx];
      }
    } else if (cov < 0.08 && Math.max(b.w, b.h) <= 256) {
      for (let gy = gy0; gy < gy1; gy++) {
        for (let gx = gx0; gx < gx1; gx++) out[gy * gw + gx] = 1;
      }
    }
  }

  // A5 引导吸附 + 1px 膨胀：全分辨率引导（亮度 [0,1]）
  const srcCv = new OffscreenCanvas(W, H);
  srcCv.getContext('2d', READBACK)!.putImageData(full, 0, 0);
  const outCv = new OffscreenCanvas(gw, gh);
  const octx = outCv.getContext('2d', READBACK)!;
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = 'high';
  octx.drawImage(srcCv, 0, 0, gw, gh);
  const od = octx.getImageData(0, 0, gw, gh).data;
  const guide = new Float32Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) {
    guide[i] = (0.299 * od[i * 4] + 0.587 * od[i * 4 + 1] + 0.114 * od[i * 4 + 2]) / 255;
  }
  const refined = guidedRefineGlyph(out, gw, gh, guide);
  // 1px 8 邻域膨胀（抗锯齿边缘余量；全分辨率 1px ≈ 概率网格 2 轮的精度）
  const dil = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x;
      if (!refined[i]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= gh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < gw) dil[ny * gw + nx] = 1;
        }
      }
    }
  }
  // 概率引导生长：笔画延伸出检出框的部分（端部碎片）概率虽低于种子阈值
  // （0.5/0.3）但仍显著高于背景（~0）——条件膨胀沿笔画方向生长蒙版，
  // 背景概率 ~0 不生长。10 轮 × 1px 自限：笔画尽头概率归零即停止。
  const GROW_ITERS = 48;
  const GROW_PROB = 0.12;
  const pd = cache.prob;
  const pw = cache.w;
  const ph = cache.h;
  const pInv = cache.inv || 1;
  const pAt = (gx: number, gy: number) => {
    const fx = (gx + 0.5) / s;
    const fy = (gy + 0.5) / s;
    const px = Math.min(pw - 1, Math.max(0, Math.round(fx / (pInv * cache.mapX))));
    const py = Math.min(ph - 1, Math.max(0, Math.round(fy / (pInv * cache.mapY))));
    return pd[py * pw + px] ?? 0;
  };
  let growSrc = dil;
  for (let it = 0; it < GROW_ITERS; it++) {
    const next = new Uint8Array(gw * gh);
    let added = 0;
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const i = y * gw + x;
        if (growSrc[i]) {
          next[i] = 1;
          continue;
        }
        const nb =
          (x > 0 && growSrc[i - 1]) ||
          (x < gw - 1 && growSrc[i + 1]) ||
          (y > 0 && growSrc[i - gw]) ||
          (y < gh - 1 && growSrc[i + gw]);
        if (nb && pAt(x, y) > GROW_PROB) {
          next[i] = 1;
          added++;
        }
      }
    }
    if (added === 0) break;
    growSrc = next;
  }
  return { data: growSrc, w: gw, h: gh };
}

function toGrid(v: number, s: number, max: number): number {
  return Math.min(max - 1, Math.max(0, Math.round(v * s)));
}

export function buildGlyphBitmap(
  boxes: Box[],
): { data: Uint8Array; w: number; h: number } | null {
  const cache = lastTextProb;
  if (!cache) return null;
  const textBoxes = boxes.filter((b) => b.source === 'text');
  if (textBoxes.length === 0) return null;
  const { prob, w, h, mapX, mapY, inv, originX, originY, fullW, fullH, image, iw, ih } = cache;
  const out = new Uint8Array(w * h);

  const imgAt = (px: number, py: number) => {
    const x = Math.min(iw - 1, Math.max(0, Math.round(px * mapX)));
    const y = Math.min(ih - 1, Math.max(0, Math.round(py * mapY)));
    const i = (y * iw + x) * 4;
    return [image[i], image[i + 1], image[i + 2]] as const;
  };

  for (const b of textBoxes) {
    const x0 = Math.max(0, Math.floor((b.x - originX) / inv / mapX));
    const y0 = Math.max(0, Math.floor((b.y - originY) / inv / mapY));
    const x1 = Math.min(w, Math.ceil((b.x - originX + b.w) / inv / mapX));
    const y1 = Math.min(h, Math.ceil((b.y - originY + b.h) / inv / mapY));
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;

    // 环带（框外扩 3px）中值色作为"此处背景"估计
    const ring: number[][] = [];
    const rx0 = Math.max(0, x0 - 3), ry0 = Math.max(0, y0 - 3);
    const rx1 = Math.min(w - 1, x1 + 3), ry1 = Math.min(h - 1, y1 + 3);
    for (let y = ry0; y <= ry1; y++) {
      for (let x = rx0; x <= rx1; x++) {
        if (x >= x0 && x < x1 && y >= y0 && y < y1) continue;
        ring.push([...imgAt(x, y)]);
      }
    }
    if (ring.length < 6) continue;
    const med = [0, 1, 2].map((ch) => ring.map((p) => p[ch]).sort((a, b2) => a - b2)[ring.length >> 1]);

    // 候选 = 与环带中值色差显著；种子 = prob>0.5
    const cand = new Uint8Array(w * h);
    let seedCount = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * w + x;
        const [r, g, b2] = imgAt(x, y);
        const dist = Math.abs(r - med[0]) + Math.abs(g - med[1]) + Math.abs(b2 - med[2]);
        if (dist > 30) cand[i] = 1;
        if (prob[i] > 0.5) seedCount++;
      }
    }
    if (seedCount === 0) continue;

    // 从种子洪水生长（只经候选像素）
    const seen = new Uint8Array(w * h);
    const stack: number[] = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * w + x;
        if (prob[i] > 0.5 && cand[i] && !seen[i]) {
          seen[i] = 1;
          stack.push(i);
        }
      }
    }
    while (stack.length) {
      const i = stack.pop()!;
      out[i] = 1;
      const x = i % w;
      const neighbors = [i - 1, i + 1, i - w, i + w];
      for (let k = 0; k < 4; k++) {
        const j = neighbors[k];
        if (k === 0 && x === 0) continue;
        if (k === 1 && x === w - 1) continue;
        if (j < 0 || j >= w * h || seen[j] || !cand[j]) continue;
        seen[j] = 1;
        stack.push(j);
      }
    }

    // 淡水印候选色差不足 → 生长覆盖率过低，字形路径失效——该框回退矩形
    // （tiled-light 实测：覆盖率 <8% 时 recall 会从 54% 掉到 50% 以下）。
    let glyphN = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) glyphN += out[y * w + x];
    }
    if (glyphN < (x1 - x0) * (y1 - y0) * 0.08) {
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) out[y * w + x] = 1;
      }
    }
  }

  // A5 蒙版引导精修（消除质量优化方案 P2，2026-10-01）：字形蒙版天生在概率图
  // 分辨率（约原图 1/4–1/16），双线性放大前边界位置误差 4–16px——"毛边"的上游
  // 根源。用检测输入图（重采样到概率图网格）作引导做一次 guided filter，把
  // 生长结果的 0.5 等值线吸附到图像的真实字形边缘（可见水印的边），而不是停在
  // 洪水生长的偶然边界。只动边界位置，不动种子+生长语义（禁区表）；之后的
  // 2 轮四邻膨胀照旧兜底抗锯齿边缘覆盖。
  let hasGlyph = false;
  for (let i = 0; i < w * h; i++) {
    if (out[i]) {
      hasGlyph = true;
      break;
    }
  }
  if (hasGlyph) {
    const guide = resampleGuideLum(image, iw, ih, w, h, mapX, mapY);
    out.set(guidedRefineGlyph(out, w, h, guide));
  }

  // region 模式：概率图只覆盖裁剪区，先把生长结果按裁剪偏移拼回全图外幅
  // （bitmap op 按全幅拉伸绘制，局部位图会被错误拉满全图）；非 region 保持原样。
  let grid = out;
  let gw = w;
  let gh = h;
  if (fullW > 0 && fullH > 0) {
    gw = Math.max(1, Math.round(fullW / (mapX * inv)));
    gh = Math.max(1, Math.round(fullH / (mapY * inv)));
    const offX = Math.round(originX / (mapX * inv));
    const offY = Math.round(originY / (mapY * inv));
    const full = new Uint8Array(gw * gh);
    for (let y = 0; y < h; y++) {
      const fy = y + offY;
      if (fy < 0 || fy >= gh) continue;
      for (let x = 0; x < w; x++) {
        if (!out[y * w + x]) continue;
        const fx = x + offX;
        if (fx >= 0 && fx < gw) full[fy * gw + fx] = 1;
      }
    }
    grid = full;
  }

  // 2 轮四邻膨胀（概率图分辨率，约等于原图 4-8px）
  let cur = grid;
  for (let k = 0; k < 2; k++) {
    const next = cur.slice();
    for (let y = 1; y < gh - 1; y++) {
      for (let x = 1; x < gw - 1; x++) {
        const i = y * gw + x;
        if (!cur[i]) continue;
        next[i - 1] = next[i + 1] = next[i - gw] = next[i + gw] = 1;
      }
    }
    cur = next;
  }
  return { data: cur, w: gw, h: gh };
}

/** A5 引导参数：半径/eps 都在概率图像素与 [0,1] 引导值尺度上。eps 越小贴边
 *  越紧（边缘处 q 越跟随引导图），越大越接近原二值结果的局部平滑。 */
const GLYPH_GUIDE_R = 4;
const GLYPH_GUIDE_EPS = 2e-3;

/**
 * 检测输入图 → 概率图网格的灰度引导图（[0,1]）。每格取四角 2×2 子采样均值
 * ——点采样会引入半格抖动，让引导边缘阶梯化；直接 JS 采样避免为重采样
 * 再建大 canvas（6000px 输入的 putImageData 是 ~96MB 临时缓冲，内存红线）。
 */
function resampleGuideLum(
  image: Uint8ClampedArray,
  iw: number,
  ih: number,
  w: number,
  h: number,
  mapX: number,
  mapY: number,
): Float32Array {
  const lumAt = (px: number, py: number) => {
    const x = Math.min(iw - 1, Math.max(0, Math.round(px)));
    const y = Math.min(ih - 1, Math.max(0, Math.round(py)));
    const i = (y * iw + x) * 4;
    return 0.299 * image[i] + 0.587 * image[i + 1] + 0.114 * image[i + 2];
  };
  const g = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy0 = y * mapY;
    const sy1 = Math.min(ih - 1, (y + 1) * mapY - 1);
    for (let x = 0; x < w; x++) {
      const sx0 = x * mapX;
      const sx1 = Math.min(iw - 1, (x + 1) * mapX - 1);
      g[y * w + x] =
        (lumAt(sx0, sy0) + lumAt(sx1, sy0) + lumAt(sx0, sy1) + lumAt(sx1, sy1)) / 4 / 255;
    }
  }
  return g;
}

/**
 * A5 核心：灰度引导滤波（He et al. 2010）把二值生长结果 p 的边界吸附到引导图
 * I 的边缘。q = ā·I + b̄：引导边缘处局部协方差大 → a 大 → q 跟随 I（0.5 等值线
 * 落在可见字形边缘上）；平坦区 a≈0 → q≈局部均值（回退到原结果的平滑版，边界
 * 不乱跑）。输出仍为 0/1（阈值 0.5）——bitmap op 的二值契约与下游 A4 软边不受
 * 影响，本函数只负责"边界落到正确的地方"。
 */
function guidedRefineGlyph(
  bin: Uint8Array,
  w: number,
  h: number,
  guide: Float32Array,
): Uint8Array {
  const n = w * h;
  const W1 = w + 1;
  const mk = () => new Float64Array(W1 * (h + 1));
  const sI = mk();
  const sP = mk();
  const sII = mk();
  const sIP = mk();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const Iv = guide[y * w + x];
      const Pv = bin[y * w + x];
      const o = (y + 1) * W1 + (x + 1);
      sI[o] = sI[o - 1] + sI[o - W1] - sI[o - W1 - 1] + Iv;
      sP[o] = sP[o - 1] + sP[o - W1] - sP[o - W1 - 1] + Pv;
      sII[o] = sII[o - 1] + sII[o - W1] - sII[o - W1 - 1] + Iv * Iv;
      sIP[o] = sIP[o - 1] + sIP[o - W1] - sIP[o - W1 - 1] + Iv * Pv;
    }
  }
  const box = (pre: Float64Array, x0: number, y0: number, x1: number, y1: number) =>
    pre[(y1 + 1) * W1 + (x1 + 1)] -
    pre[y0 * W1 + (x1 + 1)] -
    pre[(y1 + 1) * W1 + x0] +
    pre[y0 * W1 + x0];

  // 逐像素局部线性回归出 a/b，再对 a/b 做同窗盒滤波（标准引导滤波的两段结构）
  const aArr = new Float32Array(n);
  const bArr = new Float32Array(n);
  const r = GLYPH_GUIDE_R;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mI = box(sI, x0, y0, x1, y1) / cnt;
      const mP = box(sP, x0, y0, x1, y1) / cnt;
      const vI = box(sII, x0, y0, x1, y1) / cnt - mI * mI;
      const cIP = box(sIP, x0, y0, x1, y1) / cnt - mI * mP;
      const a = cIP / (vI + GLYPH_GUIDE_EPS);
      aArr[y * w + x] = a;
      bArr[y * w + x] = mP - a * mI;
    }
  }
  const sA = mk();
  const sB = mk();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y + 1) * W1 + (x + 1);
      sA[o] = sA[o - 1] + sA[o - W1] - sA[o - W1 - 1] + aArr[y * w + x];
      sB[o] = sB[o - 1] + sB[o - W1] - sB[o - W1 - 1] + bArr[y * w + x];
    }
  }
  const out = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      const q =
        (box(sA, x0, y0, x1, y1) / cnt) * guide[y * w + x] + box(sB, x0, y0, x1, y1) / cnt;
      out[y * w + x] = q > 0.5 ? 1 : 0;
    }
  }
  return out;
}

// 并发去重：auto-detect 与引擎预热同时调 initTextDetector 时共享同一次
// 创建（同步的 detSession 守卫挡不住异步竞态——双 create 在 ORT wasm 全局
// 初始化上相撞）。
let textDetectorInflight: Promise<void> | null = null;

export function initTextDetector(
  onProgress?: (p: ModelProgress) => void,
): Promise<void> {
  if (detSession) return Promise.resolve();
  textDetectorInflight ??= (async () => {
    const bytes = await loadModel(TEXT_DET_MODEL, onProgress);
    const s = await createSession(bytes, (status) =>
      onProgress?.({ done: 1, total: 1, stage: status }),
    );
    detSession = s.session;
  })();
  return textDetectorInflight.finally(() => {
    textDetectorInflight = null;
  });
}

export async function detectTextBoxes(imageData: ImageData, cap = 960): Promise<Box[]> {
  const primary = await detectTextBoxesOnce(imageData, cap, true);

  // 边界镜像补跑（2026-10-06 用户实测平铺水印图）：det 是卷积网络，对贴边
  // 物体预测弱——被图像边界裁切的文字碎片（只露半个字母）概率图低于阈值
  // 漏检，表现为"边缘白色文字残留"。镜像 pad 48px 让贴边碎片变成图内对象
  // 再检一遍：映射回原图坐标、钳到图内、与主结果去重合并。镜像不会制造
  // 假阳性（原图边界无文字时镜像区也无文字）；概率图缓存仍以主跑为准
  // （字形生长的坐标链不变）。
  const PAD = 48;
  const padded = mirrorPadImageData(imageData, PAD);
  // 补跑放宽：贴边碎片概率信号弱（部分字母 + 边界效应），阈值降到 0.22、
  // 最小连通域面积减半——只影响补跑；垃圾框由下方准入规则（触边 + ≤220）
  // 过滤，主跑精度不变
  const secondary = await detectTextBoxesRelaxed(padded, cap, 0.22, 40000);
  const { width: W, height: H } = imageData;
  const mapped: Box[] = [];
  for (const b of secondary) {
    const rx = Math.round(b.x - PAD);
    const ry = Math.round(b.y - PAD);
    const rw = Math.round(b.w);
    const rh = Math.round(b.h);
    // 准入：原始映射框必须触碰图像边界（补跑只为贴边碎片服务），且扩展后
    // 长边 ≤220——镜像补跑会把边缘镜像内容并成大团块（tiled-light 顶部
    // 281×197 巨框吞掉 light 框的教训），大框交给主跑负责
    const touchesBorder =
      rx <= 2 || ry <= 2 || rx + rw >= W - 2 || ry + rh >= H - 2;
    if (!touchesBorder) continue;
    const x0 = Math.max(0, rx);
    const y0 = Math.max(0, ry);
    const x1 = Math.min(W, rx + rw);
    const y1 = Math.min(H, ry + rh);
    if (x1 - x0 < 3 || y1 - y0 < 3) continue;
    const expanded = expandBox(
      { x: x0, y: y0, w: x1 - x0, h: y1 - y0, source: 'text' },
      1.7,
      W,
      H,
    );
    if (Math.max(expanded.w, expanded.h) > 220) continue;
    // sec 标记：镜像补跑来源。语义护栏（light/dark 冗余判定）不信任它——
    // 补跑框边界松，误杀互补源框的代价大于收益
    mapped.push({ ...expanded, sec: true });
  }
  // 去重：与主结果 IoU ≥0.25 或交集/较小框 ≥0.6 视为同一水印
  const merged = [...primary];
  for (const b of mapped) {
    const dup = merged.some((m) => {
      const ix = Math.max(0, Math.min(m.x + m.w, b.x + b.w) - Math.max(m.x, b.x));
      const iy = Math.max(0, Math.min(m.y + m.h, b.y + b.h) - Math.max(m.y, b.y));
      if (ix <= 0 || iy <= 0) return false;
      const inter = ix * iy;
      const uni = m.w * m.h + b.w * b.h - inter;
      return inter / uni >= 0.25 || inter / Math.min(m.w * m.h, b.w * b.h) >= 0.6;
    });
    if (!dup) merged.push(b);
  }
  return merged;
}

/** 镜像 pad：四边 + 四角反射延拓（9-patch），边缘文字获得连续上下文。 */
function mirrorPadImageData(imageData: ImageData, pad: number): ImageData {  const W = imageData.width;
  const H = imageData.height;
  const src = new OffscreenCanvas(W, H);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const out = new OffscreenCanvas(W + pad * 2, H + pad * 2);
  const ctx = out.getContext('2d', READBACK)!;
  ctx.imageSmoothingEnabled = false;
  // 中间
  ctx.drawImage(src, 0, 0, W, H, pad, pad, W, H);
  // 四边（镜像）
  ctx.save(); ctx.translate(pad * 2 + W, pad); ctx.scale(-1, 1);
  ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H); ctx.restore(); // 右
  ctx.save(); ctx.translate(pad, pad * 2 + H); ctx.scale(1, -1);
  ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H); ctx.restore(); // 下
  ctx.save(); ctx.translate(pad, pad); ctx.scale(-1, 1);
  ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H); ctx.restore(); // 左
  ctx.save(); ctx.translate(pad, pad); ctx.scale(1, -1);
  ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H); ctx.restore(); // 上
  // 四角（双向镜像）
  const corner = (sx: number, sy: number, dx: number, dy: number) => {
    ctx.save();
    ctx.translate(dx, dy);
    ctx.scale(sx, sy);
    ctx.drawImage(src, 0, 0, pad, pad, 0, 0, pad, pad);
    ctx.restore();
  };
  corner(1, 1, pad, pad);                    // 左上（源 0,0 镜到角内）
  corner(-1, 1, pad * 2 + W, pad);           // 右上
  corner(1, -1, pad, pad * 2 + H);           // 左下
  corner(-1, -1, pad * 2 + W, pad * 2 + H);  // 右下
  return ctx.getImageData(0, 0, W + pad * 2, H + pad * 2);
}

/** 放宽阈值的补跑封装（贴边碎片召回专用，配合 sec 准入规则使用）。 */
function detectTextBoxesRelaxed(
  imageData: ImageData,
  cap: number,
  probThreshold: number,
  minBlobDivisor: number,
): Promise<Box[]> {
  // minBoxScore=0：补跑框是贴边真碎片，概率天然偏弱（0.22 门槛的由来），
  // 框级均分门控只挂主跑——误检由触边 + ≤220 准入规则兜
  return detectTextBoxesOnce(imageData, cap, false, probThreshold, minBlobDivisor, 0);
}

// 框级质量门控（2026-10-14 多余框治理）：二值化阈值只保证"有响应"——纹理
// 误检成片贴着阈值。主跑对每张框算"前景像素（merged==1）的平均概率"作框级
// 分数：真文字笔画普遍 0.7+，误检纹理 0.3~0.5；均值 < 0.5 或映射面积
// < 12×12 的框不上画布。minBoxScore=0 关闭（镜像补跑的贴边碎片概率天然弱，
// 走自己的准入规则）。
// 批二（2026-10-19 QA 定稿）：框级均分 0.5 → 0.6——二值化阈值只保证"有响
// 应"，0.3~0.5 的误检纹理被 0.5 放行。QA 集 125 个真框最低分 0.739，
// 0.55/0.6 两档逐框出框完全一致（均零误杀）；取 0.6：对未见图的
// 0.55~0.6 纹理段多一层拦截，且距真框下限仍有余量。
const BOX_MEAN_SCORE_MIN = 0.6;
const BOX_MIN_AREA_PX2 = 144;
// 批一：上限闸门（detectWatermarks 合并去重后、返回前的统一过滤层，全图坐标）。
// 哲学：漏检 > 误检（检测不出用户可手动框选兜底）——上限闸砍"罩住主体/喷框"，
// 下限门已有（BOX_MEAN_SCORE_MIN / BOX_MIN_AREA_PX2），两侧合起来才是完整门控。
/** ① 框面积 > 整图 20% 丢弃（QA 定稿 2026-10-19：真框面积最大 19.5%，
 *  0.12 实测误杀 2 个真框；>0.2 的 merged 怪物本集为零，上限仍保留防罩主体）。 */
const BOX_MAX_AREA_RATIO = 0.2;
/** ② 填充率下限（前景像素/框面积）：稀疏巨框 = 合并噪声特征而非真水印。
 *  text 框用连通域前景占比（detectTextBoxesOnce 记账），启发式框用其 mask
 *  填充率（b.area/bboxArea）。阈值取 0.05–0.1 区间中值，QA 数据定稿。 */
const BOX_MIN_FILL = 0.07;
/** ③ 框数 top-N 砍尾——2026-10-19 QA 实测后停用（1e9）："真水印极少 >3 个"
 *  前提被证伪（case-1/2 人工采纳 37/22 个真框，top-5 砍掉 55 个真框）；
 *  喷框由批四 YOLO 佐证 + 批二分数门承接。排序分口径保留：text 用框级均分
 *  （__score），yolo 用置信（__score），启发式无置信用填充率。 */
const BOX_TOP_N = 1e9;
/** 批四：主跑出框 > 3（疑似喷框）才懒加载 YOLO 做第二意见——常规场景不白付
 *  11MB。启发式框与任一 YOLO 框 IoU < 0.1 丢弃；text/yolo 框不动；YOLO 框
 *  不入结果；零命中/异常静默回退不过滤。 */
const ARB_TRIGGER_COUNT = 3;
const ARB_MIN_IOU = 0.1;
/** 批四兜底：YOLO 佐证触发但 YOLO 零检出（arbEmpty，模型盲区而非异常）时，
 *  启发式框按填充率排序只留前 8——QA 集 8 张盲区图全为喷框（人工拒收代理
 *  140/140），真框清一色 text 不受影响；异常路径（arbFail）不兜底（瞬态
 *  失败不该惩罚，且下一张会重试）。 */
const ARB_EMPTY_HEUR_TOP_N = 8;

// ---------------------------------------------------------------------------
// 自动检测保守闸门（2026-10-20 用户决策）
// ---------------------------------------------------------------------------
// 产品判断：端上消除只在**背景简单**时可靠——复杂纹理上的大空洞会幻觉出
// 结构（.review/2026-10-02、批次 002/003 实证：MI-GAN/LaMa 在结构场景的
// 大蒙版均会补出可信但错误的物体）。因此自动检测放弃"尽量多召回"，只保留
// **明显文字/水印 + 局部背景简单**的框；复杂背景交给用户手动框选（精修
// 路径不受本闸门影响）。哲学从"漏检 > 误检"进一步收紧为"自动模式零误检
// 优先，漏检由手动兜底"。
//
// 判据一（明显文字）：text 框必须是**文本行形状**——长条（宽高比 ≥1.5）
// 或竖排（≤0.4）。小框（短边 <40px）豁免：平铺小字的单字框天然接近方形，
// 且字号小、局部填充无风险。近方形的大框是 OCR 在**人脸/身体/物体纹理**
// 上的典型误检形状（QA 集亲测：婴儿脸 111×85、电影剧照 66×54/133×107、
// 食物图 46×100、cosplay 人像 277×344 等）。
//
// 判据二（背景简单）：框外环带（pad=35% 短边，8~64px）的亮度标准差与
// 梯度均值。只统计**环带内**像素（框内文字笔画不参与），梯度也只取
// 环带↔环带邻接对，避免水印自身的边缘把背景判复杂。QA 集标定：
//   真水印（纯墙/水面/天空）std 7~30、grad 3~10；
//   误检（人脸/手臂/食物/衣物/人群）std 32~89、grad 4~27。
// 阈值取 std≤35 且 grad≤14。QA 集 24 图：text 框 125→76（拦截 49：
// 形状 10 / 背景 39），人脸/电影剧照/食物/购物/手部等误检全部拦下；
// Luma AI、LIONSGATE、手部纹身等**复杂背景上的真水印**会被拦下——
// 这是刻意取舍（手动框选兜底）。平铺水印由下方 findTiledGroups 豁免。
const AUTO_GATE_MAX_RING_STD = 35;
const AUTO_GATE_MAX_RING_GRAD = 14;
const AUTO_GATE_SHAPE_MIN_SIDE = 40;
const AUTO_GATE_SHAPE_MIN_AR = 0.4;
const AUTO_GATE_SHAPE_MAX_AR = 1.5;
/** 平铺水印聚类阈值：尺寸相近（±30% 宽高）的 text 框 ≥4 个 = 重复叠加的
 *  平铺水印。重复出现本身是"这是水印而非画面内容"的强证据——平铺水印每处
 *  字号小、单处填充局部，端上消除在复杂背景上也可靠（QA：case-1/case-2
 *  山体/雪地平铺消除全净），因此平铺组豁免保守闸门的两条判据。 */
const TILED_GROUP_MIN = 4;
const TILED_GROUP_SIZE_TOLERANCE = 0.3;

/** 尺寸相近的 text 框簇（≥ TILED_GROUP_MIN）= 平铺水印实例。 */
function findTiledGroups(boxes: Box[]): Set<Box> {
  const text = boxes.filter((b) => b.source === 'text');
  const tiled = new Set<Box>();
  const taken = new Set<Box>();
  for (const seed of text) {
    if (taken.has(seed)) continue;
    const group = text.filter(
      (b) =>
        !taken.has(b) &&
        Math.abs(b.w - seed.w) <= seed.w * TILED_GROUP_SIZE_TOLERANCE &&
        Math.abs(b.h - seed.h) <= seed.h * TILED_GROUP_SIZE_TOLERANCE,
    );
    if (group.length >= TILED_GROUP_MIN) {
      for (const b of group) {
        taken.add(b);
        tiled.add(b);
      }
    }
  }
  return tiled;
}

interface RingComplexity {
  std: number;
  grad: number;
}

/** 框外环带的亮度复杂度（全图坐标；只统计环带像素）。
 *  导出给视频路径的经典快速填充判据复用（fast-fill.ts）——两处必须同一
 *  口径：自动检测只报简单背景、快速填充也只在简单背景用。 */
export function backgroundComplexity(imageData: ImageData, b: Box): RingComplexity {
  const W = imageData.width;
  const H = imageData.height;
  const bx0 = Math.max(0, Math.floor(b.x));
  const by0 = Math.max(0, Math.floor(b.y));
  const bx1 = Math.min(W, Math.ceil(b.x + b.w));
  const by1 = Math.min(H, Math.ceil(b.y + b.h));
  if (bx1 <= bx0 || by1 <= by0) return { std: 0, grad: 0 };
  const pad = Math.min(64, Math.max(8, Math.round(0.35 * Math.min(b.w, b.h))));
  const rx0 = Math.max(0, bx0 - pad);
  const ry0 = Math.max(0, by0 - pad);
  const rx1 = Math.min(W, bx1 + pad);
  const ry1 = Math.min(H, by1 + pad);
  const bw = rx1 - rx0;
  const bh = ry1 - ry0;
  if (bw <= 1 || bh <= 1) return { std: 0, grad: 0 };

  // 环带 bbox 的亮度缓存（框内也缓存，但统计/梯度都不使用框内像素）
  const lum = new Float32Array(bw * bh);
  const d = imageData.data;
  for (let y = 0; y < bh; y++) {
    const sy = (ry0 + y) * W;
    for (let x = 0; x < bw; x++) {
      const j = (sy + rx0 + x) * 4;
      lum[y * bw + x] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
    }
  }

  let sum = 0;
  let sumSq = 0;
  let gradSum = 0;
  let n = 0;
  for (let y = 0; y < bh; y++) {
    const gy = ry0 + y;
    const inBoxY = gy >= by0 && gy < by1;
    for (let x = 0; x < bw; x++) {
      const gx = rx0 + x;
      if (inBoxY && gx >= bx0 && gx < bx1) continue;
      const v = lum[y * bw + x];
      sum += v;
      sumSq += v * v;
      n++;
      // 梯度只统计环带↔环带邻接（框内/图外邻接不计），避免文字笔画边缘
      // 把背景复杂度抬高。
      const rightIn = x + 1 < bw && !(inBoxY && rx0 + x + 1 >= bx0 && rx0 + x + 1 < bx1);
      if (rightIn) gradSum += Math.abs(lum[y * bw + x + 1] - v);
      const downY = ry0 + y + 1;
      const downIn =
        y + 1 < bh && !(downY >= by0 && downY < by1 && gx >= bx0 && gx < bx1);
      if (downIn) gradSum += Math.abs(lum[(y + 1) * bw + x] - v);
    }
  }
  if (n === 0) return { std: 0, grad: 0 };
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  return { std, grad: gradSum / Math.max(1, n * 2) };
}

/** 自动模式保守闸门的拒绝原因（null = 保留）。tiled=平铺水印实例（豁免）。 */
function autoGateRejectReason(
  imageData: ImageData,
  b: Box,
  tiled = false,
): 'shape' | 'background' | null {
  if (b.source !== 'text' && b.source !== 'yolo') return null;
  if (tiled) return null;
  if (b.source === 'text') {
    const ar = b.h > 0 ? b.w / b.h : 1;
    if (
      Math.min(b.w, b.h) >= AUTO_GATE_SHAPE_MIN_SIDE &&
      ar > AUTO_GATE_SHAPE_MIN_AR &&
      ar < AUTO_GATE_SHAPE_MAX_AR
    ) {
      return 'shape';
    }
  }
  const cpx = backgroundComplexity(imageData, b);
  if (cpx.std > AUTO_GATE_MAX_RING_STD || cpx.grad > AUTO_GATE_MAX_RING_GRAD) {
    return 'background';
  }
  return null;
}

/** 闸门命中统计（bench-detect.mjs 读取；detectWatermarks 入口清零）。
 *  arbRan=YOLO 佐证实际生效次数（加载成功且 arb 非空才计数）——区分
 *  "仲裁跑了但零丢弃"与"YOLO 压根没加载"。autoGate=保守闸门丢弃总数
 *  （autoGateShape=形状拦截 / autoGateCpx=背景复杂度拦截）。 */
export const detectGateStats = {
  areaCap: 0,
  fillFloor: 0,
  topN: 0,
  yoloArb: 0,
  arbRan: 0,
  arbEmpty: 0,
  arbFail: 0,
  autoGate: 0,
  autoGateShape: 0,
  autoGateCpx: 0,
};
/** 框级内部质量分（不进公开 Box 类型）：text=均分 / yolo=置信 / 启发式=填充率。 */
type ScoredBox = Box & { __score?: number; __fill?: number };

async function detectTextBoxesOnce(
  imageData: ImageData,
  cap: number,
  cacheProb: boolean,
  probThreshold = 0.4,
  minBlobDivisor = 20000,
  minBoxScore = BOX_MEAN_SCORE_MIN,
): Promise<Box[]> {
  if (!detSession) throw new Error('Text detector not initialized');
  const { width: W, height: H } = imageData;

  // P2（超清选区方案）：cap 可配——大图 HQ 路径提到 1920，概率图网格精细
  // 2×，字形生长/A5 全链路精度同步提升；代价是 det 耗时 ~4×（仅大图触发）。
  const scale = Math.min(1, cap / Math.max(W, H));
  const inW = Math.max(32, Math.round((W * scale) / 32) * 32);
  const inH = Math.max(32, Math.round((H * scale) / 32) * 32);

  const src = new OffscreenCanvas(W, H);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const resized = new OffscreenCanvas(inW, inH);
  const rctx = resized.getContext('2d', READBACK)!;
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = 'high';
  rctx.drawImage(src, 0, 0, inW, inH);
  const img = rctx.getImageData(0, 0, inW, inH);

  const ort = await getOrt();
  const size = inW * inH;
  const input = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    input[i] = (img.data[i * 4] / 255 - 0.5) / 0.5;
    input[i + size] = (img.data[i * 4 + 1] / 255 - 0.5) / 0.5;
    input[i + size * 2] = (img.data[i * 4 + 2] / 255 - 0.5) / 0.5;
  }

  const feeds: Record<string, ort.Tensor> = {};
  feeds[detSession.inputNames[0]] = new ort.Tensor('float32', input, [1, 3, inH, inW]);
  const output = await detSession.run(feeds);
  const outTensor = output[detSession.outputNames[0]];
  const outH = outTensor.dims[2] as number;
  const outW = outTensor.dims[3] as number;
  const prob = outTensor.data as Float32Array;

  const bin = new Uint8Array(outW * outH);
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] > probThreshold) bin[i] = 1;
  }
  const merged = close(bin, outW, outH, Math.max(1, Math.round(outW / 320)));
  const blobs = connectedComponents(merged, outW, outH, Math.max(8, (outW * outH) / minBlobDivisor));

  const mapX = W / outW;
  const mapY = H / outH;
  // 缓存概率图供字形级蒙版使用（detectWatermarks 会补上 inv / region 偏移系数）。
  if (cacheProb) {
    lastTextProb = {
      prob,
      w: outW,
      h: outH,
      mapX,
      mapY,
      inv: 1,
      originX: 0,
      originY: 0,
      fullW: 0,
      fullH: 0,
      image: imageData.data,
      iw: W,
      ih: H,
    };
  }
  return blobs
    .filter((b) => b.w >= 3 && b.h >= 3)
    .filter((b) => {
      if (minBoxScore <= 0) return true;
      if (Math.round(b.w * mapX) * Math.round(b.h * mapY) < BOX_MIN_AREA_PX2) return false;
      let sum = 0;
      let n = 0;
      for (let y = b.y; y < b.y + b.h; y++) {
        const row = y * outW;
        for (let x = b.x; x < b.x + b.w; x++) {
          if (merged[row + x]) {
            sum += prob[row + x];
            n++;
          }
        }
      }
      const mean = n > 0 ? sum / n : 0;
      // 批一②③ 记账：框级均分（top-N 排序分）+ 填充率（稀疏巨框闸门）
      (b as { __score?: number }).__score = mean;
      (b as { __fill?: number }).__fill = b.w * b.h > 0 ? n / (b.w * b.h) : 0;
      return n > 0 && mean >= minBoxScore;
    })
    .map((b) => {
      const box: ScoredBox = expandBox(
        {
          x: Math.round(b.x * mapX),
          y: Math.round(b.y * mapY),
          w: Math.round(b.w * mapX),
          h: Math.round(b.h * mapY),
          source: 'text',
        },
        1.7,
        W,
        H,
      );
      box.__score = (b as { __score?: number }).__score;
      box.__fill = (b as { __fill?: number }).__fill;
      return box;
    });
}

// ---------------------------------------------------------------------------
// 2. Light-overlay heuristic (edge-density validated)
// ---------------------------------------------------------------------------

/** 单个连通域的形状蒙版（检测小图分辨率，1 字节/像素）。 */
export interface BlobMask {
  data: Uint8Array;
  w: number;
  h: number;
}

interface HeuristicOpts {
  /**
   * 阈值归一化（评审问题 A）：minArea / maxBoxArea 等面积类阈值是"输入图
   * 面积相对量"，region 裁剪后直接按裁剪图算会让紧贴的目标撞 8% 上限被
   * 静默丢弃。传原图尺寸时阈值仍按"整图跑检测"的口径换算到本图尺度，
   * 只有连通域搜索限制在裁剪图上。
   */
  fullArea?: { w: number; h: number };
  /** 传入空数组时，为每个返回的框按序收集其连通域形状（region 精修画形状用）。 */
  masksOut?: BlobMask[];
  /** R1 形状精修：外送 morphological close 后的候选图（本检测器 640 网格全幅）。 */
  binOut?: { data: Uint8Array; w: number; h: number };
}

/** 面积类阈值换算：fullArea（原图）口径 → 本图小图尺度的像素数。 */
function areaThresholds(
  w: number,
  h: number,
  scale: number,
  fullArea: { w: number; h: number } | undefined,
  divisor: number,
  floor: number,
): { minArea: number; maxBoxArea: number } {
  if (!fullArea) {
    return { minArea: Math.max(floor, ((w * h) / divisor) | 0), maxBoxArea: w * h * 0.08 };
  }
  const fScale = Math.min(1, 640 / Math.max(fullArea.w, fullArea.h));
  const fw = Math.round(fullArea.w * fScale);
  const fh = Math.round(fullArea.h * fScale);
  // 全图小图的 1 像素 ≈ (scale/fScale)² 本图小图像素（裁剪图更小、降采样更少）
  const k = (scale / fScale) ** 2;
  return {
    minArea: Math.max(floor, ((fw * fh) / divisor) | 0) * k,
    maxBoxArea: fw * fh * 0.08 * k,
  };
}

/** 从连通域标签图提取单个 blob 的形状。 */
function blobMaskOf(labels: Int32Array, id: number, w: number, h: number): BlobMask {
  const data = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (labels[i] === id) data[i] = 1;
  return { data, w, h };
}

export function detectLightWatermarkBoxes(imageData: ImageData, opts?: HeuristicOpts): Box[] {
  const { width: W, height: H } = imageData;
  const { data: small, scale } = downscaleImageData(imageData, 640);
  const w = small.width;
  const h = small.height;
  const d = small.data;

  const lum = luminanceOf(small);
  const blur = boxBlur(lum, w, h, Math.max(2, Math.round(w / 160)));

  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = d[i * 4],
      g = d[i * 4 + 1],
      b = d[i * 4 + 2];
    const sat = Math.max(r, g, b) - Math.min(r, g, b);
    if (lum[i] > 160 && sat < 50 && lum[i] - blur[i] > 7) bin[i] = 1;
  }

  const merged = close(bin, w, h, Math.max(2, Math.round(w / 128)));
  if (opts?.binOut) { opts.binOut.data = merged; opts.binOut.w = w; opts.binOut.h = h; }
  const inv = 1 / scale;
  const { minArea, maxBoxArea } = areaThresholds(w, h, scale, opts?.fullArea, 8000, 6);
  const labels = opts?.masksOut ? new Int32Array(w * h) : undefined;
  const blobs = connectedComponents(merged, w, h, minArea, labels);

  const boxes: Box[] = [];
  for (const b of blobs) {
    if (b.w < 4 || b.h < 4) continue;
    const bboxArea = b.w * b.h;
    if (bboxArea > maxBoxArea) continue;
    // Overlays are coherent: glyphs fill their blob after closing.
    if (b.area / bboxArea < 0.2) continue;
    // Reject flat bright regions: a real overlay has texture (glyph edges).
    const edgeDensity = meanGradientInside(lum, w, h, b, merged);
    if (edgeDensity < 6) continue;
    // Overlays are near-neutral: low mean saturation inside the blob.
    let satSum = 0,
      satN = 0;
    for (let y = b.y; y < b.y + b.h; y++) {
      for (let x = b.x; x < b.x + b.w; x++) {
        const i = y * w + x;
        if (!merged[i]) continue;
        const r = d[i * 4],
          g = d[i * 4 + 1],
          bl = d[i * 4 + 2];
        satSum += Math.max(r, g, bl) - Math.min(r, g, bl);
        satN++;
      }
    }
    if (satN && satSum / satN > 45) continue;
    // S1：原始 bin（close 前）上的 solid/stroke 判别，附加到框
    const wmType = wmTypeOfBin(bin, w, h, b.x, b.y, b.w, b.h);
    const lightBox: ScoredBox = expandBox(
      {
        x: Math.round(b.x * inv),
        y: Math.round(b.y * inv),
        w: Math.round(b.w * inv),
        h: Math.round(b.h * inv),
        source: 'light',
        wmType,
      },
      1.25,
      W,
      H,
    );
    // 批一②③ 记账：启发式无置信用填充率（稀疏巨框闸门 + top-N 排序分）
    lightBox.__fill = bboxArea > 0 ? b.area / bboxArea : 0;
    boxes.push(lightBox);
    if (labels && opts?.masksOut) opts.masksOut.push(blobMaskOf(labels, b.id, w, h));
  }
  return boxes;
}

// ---------------------------------------------------------------------------
// 3. Dark-badge heuristic (solid dark logo blocks)
// ---------------------------------------------------------------------------

export function detectDarkBadgeBoxes(imageData: ImageData, opts?: HeuristicOpts): Box[] {
  const { width: W, height: H } = imageData;
  const { data: small, scale } = downscaleImageData(imageData, 640);
  const w = small.width;
  const h = small.height;
  const d = small.data;

  const lum = luminanceOf(small);
  const blur = boxBlur(lum, w, h, Math.max(2, Math.round(w / 160)));

  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = d[i * 4],
      g = d[i * 4 + 1],
      b = d[i * 4 + 2];
    const sat = Math.max(r, g, b) - Math.min(r, g, b);
    if (lum[i] < 100 && sat < 55 && blur[i] - lum[i] > 18) bin[i] = 1;
  }

  const merged = close(bin, w, h, Math.max(2, Math.round(w / 128)));
  if (opts?.binOut) { opts.binOut.data = merged; opts.binOut.w = w; opts.binOut.h = h; }
  const inv = 1 / scale;
  const { minArea, maxBoxArea } = areaThresholds(w, h, scale, opts?.fullArea, 4000, 20);
  const labels = opts?.masksOut ? new Int32Array(w * h) : undefined;
  const blobs = connectedComponents(merged, w, h, minArea, labels);

  const boxes: Box[] = [];
  for (const b of blobs) {
    const bboxArea = b.w * b.h;
    const fill = b.area / bboxArea;
    // Badges are solid rectangles; textured dark regions are not.
    if (fill < 0.5) continue;
    if (bboxArea > maxBoxArea || b.w < 8 || b.h < 8) continue;
    // A badge carries content (icon/text) — require internal structure.
    const edgeDensity = meanGradientInside(lum, w, h, b, merged);
    if (edgeDensity < 5) continue;
    const wmTypeD = wmTypeOfBin(bin, w, h, b.x, b.y, b.w, b.h);
    const darkBox: ScoredBox = expandBox(
      {
        x: Math.round(b.x * inv),
        y: Math.round(b.y * inv),
        w: Math.round(b.w * inv),
        h: Math.round(b.h * inv),
        source: 'dark',
        wmType: wmTypeD,
      },
      1.3,
      W,
      H,
    );
    // 批一②③ 记账：启发式无置信用填充率（稀疏巨框闸门 + top-N 排序分）
    darkBox.__fill = fill;
    boxes.push(darkBox);
    if (labels && opts?.masksOut) opts.masksOut.push(blobMaskOf(labels, b.id, w, h));
  }
  return boxes;
}

// ---------------------------------------------------------------------------
// 4. YOLO watermark detection (purpose-trained, optional deep tier)
// ---------------------------------------------------------------------------

const YOLO_SIZE = 640;
const YOLO_CONF = 0.35;
const YOLO_IOU = 0.5;

let yoloSession: ort.InferenceSession | null = null;

export async function initYoloDetector(
  onProgress?: (p: ModelProgress) => void,
): Promise<void> {
  if (yoloSession) return;
  const bytes = await loadModel(YOLO_MODEL, onProgress);
  const s = await createSession(bytes, (status) =>
    onProgress?.({ done: 1, total: 1, stage: status }),
  );
  yoloSession = s.session;
}

interface Det {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
}

/**
 * S1 solid/stroke 判别（.review/2026-10-02 §7）：在**原始 bin**（形态学
 * close 之前——close 会把文字笔画融成厚块，判别特征会被抹掉）的 bbox 内
 * 计算填充率与 2px 抗腐蚀存活率：solid 实体厚（存活 >50%、填充 >40%），
 * stroke 笔画细（640 网格上 1-3px，两轮腐蚀即消失）。
 */
function wmTypeOfBin(
  bin: Uint8Array,
  w: number,
  h: number,
  bx: number,
  by: number,
  bw: number,
  bh: number,
): 'solid' | 'stroke' {
  const x0 = Math.max(0, bx);
  const y0 = Math.max(0, by);
  const x1 = Math.min(w, bx + bw);
  const y1 = Math.min(h, by + bh);
  if (x1 <= x0 || y1 <= y0) return 'stroke';
  let area = 0;
  let survive = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (!bin[y * w + x]) continue;
      area++;
      let solid = 1;
      for (let dy = -2; dy <= 2 && solid; dy++) {
        for (let dx = -2; dx <= 2 && solid; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w || !bin[ny * w + nx]) solid = 0;
        }
      }
      survive += solid;
    }
  }
  if (area === 0) return 'stroke';
  const fill = area / (bw * bh);
  const solidity = survive / area;
  return solidity > 0.5 && fill > 0.4 ? 'solid' : 'stroke';
}

function iou(a: Pick<Det, 'x' | 'y' | 'w' | 'h'>, b: Pick<Det, 'x' | 'y' | 'w' | 'h'>): number {
  const x = Math.max(a.x, b.x);
  const y = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w);
  const bm = Math.min(a.y + a.h, b.y + b.h);
  if (r <= x || bm <= y) return 0;
  const inter = (r - x) * (bm - y);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function nms(dets: Det[], iouThresh: number): Det[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept: Det[] = [];
  for (const d of sorted) {
    if (kept.every((k) => iou(k, d) < iouThresh)) kept.push(d);
  }
  return kept;
}

export async function detectYoloBoxes(imageData: ImageData): Promise<Box[]> {
  if (!yoloSession) throw new Error('YOLO detector not initialized');
  const { width: W, height: H } = imageData;

  // Letterbox: longest edge → 640, pad to 640×640 with neutral gray.
  const scale = Math.min(1, YOLO_SIZE / Math.max(W, H));
  const rw = Math.round(W * scale);
  const rh = Math.round(H * scale);
  const padX = Math.floor((YOLO_SIZE - rw) / 2);
  const padY = Math.floor((YOLO_SIZE - rh) / 2);

  const src = new OffscreenCanvas(W, H);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const canvas = new OffscreenCanvas(YOLO_SIZE, YOLO_SIZE);
  const ctx = canvas.getContext('2d', READBACK)!;
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, W, H, padX, padY, rw, rh);
  const img = ctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);

  const ort = await getOrt();
  const size = YOLO_SIZE * YOLO_SIZE;
  const input = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    input[i] = img.data[i * 4] / 255;
    input[i + size] = img.data[i * 4 + 1] / 255;
    input[i + size * 2] = img.data[i * 4 + 2] / 255;
  }

  const feeds: Record<string, ort.Tensor> = {};
  feeds[yoloSession.inputNames[0]] = new ort.Tensor('float32', input, [
    1,
    3,
    YOLO_SIZE,
    YOLO_SIZE,
  ]);
  const output = await yoloSession.run(feeds);
  const outTensor = output[yoloSession.outputNames[0]];
  const dims = outTensor.dims; // [1, 5, N] or [1, N, 5]
  const data = outTensor.data as Float32Array;

  const dets: Det[] = [];
  if (dims[1] === 5) {
    const n = dims[2] as number;
    for (let i = 0; i < n; i++) {
      const score = data[4 * n + i];
      if (score < YOLO_CONF) continue;
      const cx = data[i];
      const cy = data[n + i];
      const w = data[2 * n + i];
      const h = data[3 * n + i];
      dets.push({ x: cx - w / 2, y: cy - h / 2, w, h, score });
    }
  } else {
    const n = dims[1] as number;
    for (let i = 0; i < n; i++) {
      const score = data[i * 5 + 4];
      if (score < YOLO_CONF) continue;
      const cx = data[i * 5];
      const cy = data[i * 5 + 1];
      const w = data[i * 5 + 2];
      const h = data[i * 5 + 3];
      dets.push({ x: cx - w / 2, y: cy - h / 2, w, h, score });
    }
  }

  return nms(dets, YOLO_IOU).map((d) => {
    // Reverse letterbox → original coords
    const x = (d.x - padX) / scale;
    const y = (d.y - padY) / scale;
    const w = d.w / scale;
    const h = d.h / scale;
    const box: ScoredBox = expandBox(
      {
        x: Math.max(0, Math.round(x)),
        y: Math.max(0, Math.round(y)),
        w: Math.min(W, Math.round(w)),
        h: Math.min(H, Math.round(h)),
        source: 'yolo',
      },
      1.15,
      W,
      H,
    );
    // 批一③ 记账：YOLO 置信用作 top-N 排序分
    box.__score = d.score;
    return box;
  });
}

// ---------------------------------------------------------------------------
// Combined
// ---------------------------------------------------------------------------

export interface DetectOptions {
  useText?: boolean;
  useLight?: boolean;
  useDark?: boolean;
  /** Purpose-trained YOLO watermark detector (extra ~11 MB download). */
  useYolo?: boolean;
  /**
   * Optional second-opinion YOLO pass when heuristics return more than
   * `ARB_TRIGGER_COUNT` boxes. Off by default in this package: it can trigger
   * an unexpected 11 MB model download. ClearPix enables it for its image page.
   */
  autoArbitrate?: boolean;
  /**
   * 自动检测保守闸门（2026-10-20）：只返回**明显文字/水印 + 局部背景简单**
   * 的框，复杂背景上的检出丢弃（手动框选兜底）。图片页自动检测启用；
   * bench/视频/手动精修默认关闭。判据见 `autoGateRejectReason`。
   */
  autoGate?: boolean;
  /** Cap the detector input's longest side (low-end fast mode, PRD FR-3.5).
   * Boxes are mapped back to original coordinates. Default: no cap. */
  maxSide?: number;
  /**
   * 约束检测的用户选区（原图坐标）。设置后入口先把 imageData 按 region
   * 外扩 15%（下限 8px）裁剪，各检测器在裁剪图上运行，输出框加偏移还原
   * 为原图坐标（与 maxSide/inv 同一映射模式）；启发式的面积类阈值仍按
   * 原图面积计算（评审问题 A），文字概率图缓存写入裁剪偏移（评审问题 B）。
   */
  region?: { x: number; y: number; w: number; h: number };
  /** P2（超清选区方案）：文字检测输入上限（默认 960）。HQ 路径 1920 ——
   *  概率图网格精细 2×，大图字形边界精度同步提升。 */
  textDetCap?: number;
  /**
   * region 精修用：传入空数组，检测完成后与返回值逐框对齐填充连通域形状
   * （全图外幅 1 字节位图，已膨胀 2px）；无形状的来源（yolo/text）为 null。
   */
  shapesOut?: (BlobMask | null)[];
  /**
   * R1 形状精修（.review/2026-10-02-selection-precision-analysis.md）：外送
   * light/dark 启发式的候选图（按位或合并，本检测器 640 网格全幅）——
   * refineShapeMask 的软基底。未启用的检测器无对应条目。
   */
  candidatesOut?: { combined?: BlobMask };
  onModelProgress?: (p: ModelProgress) => void;
}

/** 启发式框内部携带的连通域形状（shapesOut 组装后剥除，不进公开 Box）。 */
type BoxWithMask = Box & { mask?: BlobMask };

export async function detectWatermarks(
  imageData: ImageData,
  opts: DetectOptions = {},
): Promise<Box[]> {
  lastTextProb = null; // 防止检测失败时沿用上一张图的概率图
  detectGateStats.areaCap = 0;
  detectGateStats.fillFloor = 0;
  detectGateStats.topN = 0;
  detectGateStats.yoloArb = 0;
  detectGateStats.arbRan = 0;
  detectGateStats.arbEmpty = 0;
  detectGateStats.arbFail = 0;
  detectGateStats.autoGate = 0;
  detectGateStats.autoGateShape = 0;
  detectGateStats.autoGateCpx = 0;
  let data = imageData;
  let inv = 1;
  let ox = 0;
  let oy = 0;
  if (opts.region) {
    // 用户框外扩 15%（下限 8px）作为搜索区，钳到图内
    const r = opts.region;
    const padX = Math.max(8, Math.round(r.w * 0.15));
    const padY = Math.max(8, Math.round(r.h * 0.15));
    ox = Math.max(0, Math.floor(r.x - padX));
    oy = Math.max(0, Math.floor(r.y - padY));
    const x1 = Math.min(imageData.width, Math.ceil(r.x + r.w + padX));
    const y1 = Math.min(imageData.height, Math.ceil(r.y + r.h + padY));
    const cw = Math.max(1, x1 - ox);
    const ch = Math.max(1, y1 - oy);
    const src = new OffscreenCanvas(imageData.width, imageData.height);
    src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
    const cv = new OffscreenCanvas(cw, ch);
    const cctx = cv.getContext('2d', READBACK)!;
    cctx.drawImage(src, ox, oy, cw, ch, 0, 0, cw, ch);
    data = cctx.getImageData(0, 0, cw, ch);
  }
  if (opts.maxSide && Math.max(data.width, data.height) > opts.maxSide) {
    const r = downscaleImageData(data, opts.maxSide);
    data = r.data;
    inv = 1 / r.scale;
  }
  const fullArea = opts.region
    ? { w: imageData.width, h: imageData.height }
    : undefined;
  let boxes: Box[] = [];

  let yoloBoxes: Box[] = [];
  if (opts.useYolo) {
    await initYoloDetector(opts.onModelProgress);
    yoloBoxes = await detectYoloBoxes(data);
    boxes.push(...yoloBoxes);
  }
  if (opts.useText !== false) {
    await initTextDetector(opts.onModelProgress);
    // 文字检测**恒在全图**上跑（.review/2026-10-02 §9）：det 模型对裁剪
    // 放大的文字存在尺度敏感——紧贴用户框的放大裁剪会把小文字推出分布
    // （实测 text-br-small 区域裁剪检出 0 框 vs 全图 4 框）。概率图缓存
    // 随之是全图坐标，字形生长的映射保持一致；region 模式下检出后按
    // 框中心过滤（框外的文字水印不属于本次选区）。
    const textBoxes = await detectTextBoxes(imageData, opts.textDetCap);
    boxes.push(
      ...textBoxes.filter((b) => {
        if (!opts.region) return true;
        const cx = b.x + b.w / 2;
        const cy = b.y + b.h / 2;
        const r = opts.region;
        return cx >= r.x && cx < r.x + r.w && cy >= r.y && cy < r.y + r.h;
      }),
    );
  }
  const candBins: { data: Uint8Array; w: number; h: number }[] = [];
  if (opts.useLight !== false) {
    const masks: BlobMask[] | undefined = opts.shapesOut ? [] : undefined;
    const binRef: { data: Uint8Array; w: number; h: number } = { data: new Uint8Array(0), w: 0, h: 0 };
    const found = detectLightWatermarkBoxes(data, { fullArea, masksOut: masks, binOut: binRef });
    if (opts.candidatesOut && binRef.w > 0) candBins.push(binRef);
    if (masks) found.forEach((b, i) => ((b as BoxWithMask).mask = masks[i]));
    boxes.push(...found);
  }
  if (opts.useDark !== false) {
    const masks: BlobMask[] | undefined = opts.shapesOut ? [] : undefined;
    const binRef: { data: Uint8Array; w: number; h: number } = { data: new Uint8Array(0), w: 0, h: 0 };
    const found = detectDarkBadgeBoxes(data, { fullArea, masksOut: masks, binOut: binRef });
    if (opts.candidatesOut && binRef.w > 0) candBins.push(binRef);
    if (masks) found.forEach((b, i) => ((b as BoxWithMask).mask = masks[i]));
    boxes.push(...found);
  }
  if (opts.candidatesOut && candBins.length > 0) {
    // 同一输入的两次 640 降采样网格维度一致；按位或合并 light/dark 候选。
    // region 模式：bin 是裁剪区坐标——重排到**全图外幅**（refineShapeMask
    // 以全图坐标采样；refineSelection 无 maxSide → inv=1 直接平移）。
    const w = candBins[0].w;
    const h = candBins[0].h;
    const combined = new Uint8Array(w * h);
    for (const b of candBins) {
      if (b.w !== w || b.h !== h) continue;
      for (let i = 0; i < combined.length; i++) if (b.data[i]) combined[i] = 1;
    }
    const fullBin = new Uint8Array(imageData.width * imageData.height);
    for (let y = 0; y < h; y++) {
      const fy = Math.round(oy + y * inv);
      if (fy < 0 || fy >= imageData.height) continue;
      for (let x = 0; x < w; x++) {
        const fx = Math.round(ox + x * inv);
        if (fx < 0 || fx >= imageData.width) continue;
        if (combined[y * w + x]) fullBin[fy * imageData.width + fx] = 1;
      }
    }
    opts.candidatesOut.combined = { data: fullBin, w: imageData.width, h: imageData.height };
  }

  // YOLO arbitration: when the purpose-trained detector found watermarks,
  // heuristic boxes are only kept where they confirm a YOLO detection.
  // (YOLO is the high-precision source; heuristics are the noisy high-recall
  // source — gating noisy-by-precise is the correct direction. When YOLO
  // finds nothing we fall back to the heuristics as-is.)
  if (yoloBoxes.length > 0) {
    const overlapsYolo = (b: Box) =>
      yoloBoxes.some((y) => {
        const x = Math.max(b.x, y.x);
        const yy = Math.max(b.y, y.y);
        const r = Math.min(b.x + b.w, y.x + y.w);
        const bm = Math.min(b.y + b.h, y.y + y.h);
        if (r <= x || bm <= yy) return false;
        const inter = (r - x) * (bm - yy);
        return inter / (b.w * b.h) > 0.15;
      });
    boxes = boxes.filter(
      (b) => b.source === 'yolo' || b.source === 'text' || overlapsYolo(b),
    );
  }
  const out = dedupeBoxes(boxes);
  // 批四：YOLO 第二意见（2026-10-19 拍板）——主跑出框 >3（疑似喷框）才懒加载
  // YOLO 佐证。启发式框（误检大户）与任一 YOLO 框 IoU<0.1 丢弃；text 框不动
  // （已有字形级概率门控，YOLO 对淡文字召回差，丢它违反漏检哲学）；YOLO 框
  // 本身不入结果。零检出分两种：arbEmpty=模型盲区（实测 8/24 图，此时启发式
  // 按填充率砍到前 8 兜底，见 ARB_EMPTY_HEUR_TOP_N）；arbFail=异常（静默回退
  // 不过滤，漏检 > 误检）。坐标同为 data 空间（dedupe 后、映射前），与上方
  // 既有 useYolo 仲裁（inter>0.15 覆盖率，仅 useYolo 开启时激活）互补不冲突。
  let finalBoxes = out;
  if (opts.autoArbitrate && out.length > ARB_TRIGGER_COUNT) {
    let arb = yoloBoxes;
    let arbFailed = false;
    if (arb.length === 0) {
      try {
        await initYoloDetector(opts.onModelProgress);
        arb = await detectYoloBoxes(data);
      } catch {
        arb = [];
        arbFailed = true;
      }
    }
    if (arb.length > 0) {
      detectGateStats.arbRan++;
      finalBoxes = out.filter((b) => {
        if (b.source !== 'light' && b.source !== 'dark') return true;
        const hit = arb.some((y) => {
          const x0 = Math.max(b.x, y.x);
          const y0 = Math.max(b.y, y.y);
          const x1 = Math.min(b.x + b.w, y.x + y.w);
          const y1 = Math.min(b.y + b.h, y.y + y.h);
          if (x1 <= x0 || y1 <= y0) return false;
          const inter = (x1 - x0) * (y1 - y0);
          const uni = b.w * b.h + y.w * y.h - inter;
          return inter / uni >= ARB_MIN_IOU;
        });
        if (!hit) detectGateStats.yoloArb++;
        return hit;
      });
    } else {
      detectGateStats[arbFailed ? 'arbFail' : 'arbEmpty']++;
      if (!arbFailed) {
        // 盲区兜底：启发式框按填充率留前 N，其余计入 topN 统计（同为
        // "多框砍尾"语义，bench 只读一个数即可对比）。
        const heur = finalBoxes.filter(
          (b) => b.source === 'light' || b.source === 'dark',
        );
        if (heur.length > ARB_EMPTY_HEUR_TOP_N) {
          const keep = new Set(
            [...heur]
              .sort(
                (a, b) =>
                  ((b as ScoredBox).__fill ?? 0) - ((a as ScoredBox).__fill ?? 0),
              )
              .slice(0, ARB_EMPTY_HEUR_TOP_N),
          );
          detectGateStats.topN += heur.length - ARB_EMPTY_HEUR_TOP_N;
          finalBoxes = finalBoxes.filter(
            (b) =>
              (b.source !== 'light' && b.source !== 'dark') || keep.has(b),
          );
        }
      }
    }
  }
  if (ox !== 0 || oy !== 0 || opts.region) {
    // region 模式：文字检测已恒在全图上跑（§9 尺度敏感），概率图缓存就是
    // 全图坐标——transform 保持 detectTextBoxes 的默认（origin 0 / inv 1），
    // 不得写入裁剪偏移（否则字形生长的种子采样整体错位，蒙版空/NaN）。
    setTextProbTransform(1, 0, 0, imageData.width, imageData.height);
  } else {
    setTextProbInv(inv);
  }
  // text 源框例外：文字检测恒在全图上跑（§9），坐标已是全图系——不得
  // 再叠加裁剪偏移/缩放（否则 double-offset，实测框平移出图界）。
  const mapped =
    inv === 1 && ox === 0 && oy === 0
      ? finalBoxes
      : finalBoxes.map((b) =>
          b.source === 'text'
            ? b
            : {
                ...b,
                x: Math.round(b.x * inv) + ox,
                y: Math.round(b.y * inv) + oy,
                w: Math.round(b.w * inv),
                h: Math.round(b.h * inv),
              },
        );
  // 批一：上限闸门（全图坐标，shapesOut 对齐前统一过滤——shapesOut 逐框对齐
  // 返回值，必须在闸门之后组装）。①面积上限 ②填充率下限 ③top-N 砍尾
  // （③已停用：QA 证实多水印图的真框数远超 3，砍尾=砍真框，见常量注释）。
  const gW = imageData.width;
  const gH = imageData.height;
  let gated = mapped.filter((b) => {
    if ((b.w * b.h) / (gW * gH) > BOX_MAX_AREA_RATIO) {
      detectGateStats.areaCap++;
      return false;
    }
    const fill = (b as ScoredBox).__fill;
    if (fill !== undefined && fill < BOX_MIN_FILL) {
      detectGateStats.fillFloor++;
      return false;
    }
    return true;
  });
  // 自动模式保守闸门：在面积/填充闸之后、shapesOut 组装之前执行（shapesOut
  // 逐框对齐返回值；被闸门丢弃的框不能留下形状占位）。
  if (opts.autoGate) {
    const before = gated.length;
    const tiled = findTiledGroups(gated);
    gated = gated.filter((b) => {
      const reason = autoGateRejectReason(imageData, b, tiled.has(b));
      if (reason === 'shape') {
        detectGateStats.autoGateShape++;
        return false;
      }
      if (reason === 'background') {
        detectGateStats.autoGateCpx++;
        return false;
      }
      return true;
    });
    detectGateStats.autoGate += before - gated.length;
  }
  if (gated.length > BOX_TOP_N) {
    detectGateStats.topN += gated.length - BOX_TOP_N;
    const scoreOf = (b: Box) => (b as ScoredBox).__score ?? (b as ScoredBox).__fill ?? 0;
    gated.sort((a, b) => scoreOf(b) - scoreOf(a));
    gated.length = BOX_TOP_N;
  }
  if (opts.shapesOut) {
    // 连通域形状（检测小图空间）拼回全图外幅位图，并膨胀 2px（§3.2.2）。
    // 位图分辨率取检测输入尺度：像素 X ↔ 原图 X*inv，与 bitmap op 的全幅
    // 拉伸绘制约定一致。
    const bw = Math.max(1, Math.round(imageData.width / inv));
    const bh = Math.max(1, Math.round(imageData.height / inv));
    const offX = ox / inv;
    const offY = oy / inv;
    for (const b of gated) {
      const mask = (b as BoxWithMask).mask;
      delete (b as BoxWithMask).mask;
      if (!mask) {
        opts.shapesOut.push(null);
        continue;
      }
      const kx = data.width / mask.w;
      const ky = data.height / mask.h;
      const bmp = new Uint8Array(bw * bh);
      for (let y = 0; y < mask.h; y++) {
        const by = Math.round(y * ky + offY);
        if (by < 0 || by >= bh) continue;
        for (let x = 0; x < mask.w; x++) {
          if (!mask.data[y * mask.w + x]) continue;
          const bx = Math.round(x * kx + offX);
          if (bx >= 0 && bx < bw) bmp[by * bw + bx] = 1;
        }
      }
      opts.shapesOut.push({ data: dilate(bmp, bw, bh, 2), w: bw, h: bh });
    }
  }
  return gated;
}
