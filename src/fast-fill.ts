/**
 * 经典 inpaint 快速填充（push-pull 归一化卷积金字塔，2026-10-21）。
 *
 * 动机：端上消除的内网开销几乎全在神经推理（MI-GAN 单 tile 190-290ms），
 * 而**自动检测只报"背景简单"的框**（保守闸门，见 detect.ts）——简单背景
 * 上的小蒙版，经典扩散填充（Telea/NS 的思路）与 AI 填充肉眼无差，却便宜
 * 两个数量级。复杂背景/手动大蒙版仍走神经路径（见 fastFillEligible）。
 *
 * 算法：归一化卷积（push-pull）金字塔
 *   push：从最细层向上，洞（蒙版内）像素用 4 邻域的加权均值填补；
 *   pull：从粗层向下，洞像素用粗层的值回填；已知像素保留原值。
 * 复杂度 O(区域面积)，250×80 的蒙版区域 < 1ms。
 *
 * 口径与既有管线一致：**不**做羽化/负收益门控（那些在 mask-history 粘贴层
 * 统一做），本函数只负责"把洞填成合理值"。
 */

export interface FastFillResult {
  /** 全幅结果（与输入同尺寸，未改动的像素逐位保留）。 */
  data: ImageData;
  /** 实际参与填充的区域（蒙版 bbox 外扩 2px，钳到图内）。 */
  bbox: { x: number; y: number; w: number; h: number };
}

interface Level {
  w: number;
  h: number;
  /** 颜色 [r,g,b] 累加值（未知像素为 0）。 */
  acc: Float32Array;
  /** 权重（已知像素为 1，未知为 0）。 */
  wgt: Float32Array;
}

/** 蒙版是否被填充（alpha > 16 视为被覆盖，与 mask-history 口径一致）。 */
function maskHit(alpha: number): boolean {
  return alpha > 16;
}

/**
 * 经典填充：把 imageData 中 maskData 覆盖的像素用周围像素扩散填补。
 * 区域 = 蒙版 bbox 外扩 2px（钳到图内）；区域外像素逐位保留。
 */
export function fastFillRegion(
  imageData: ImageData,
  maskData: ImageData,
  bbox?: { x: number; y: number; w: number; h: number },
): FastFillResult | null {
  const W = imageData.width;
  const H = imageData.height;
  const MW = maskData.width;
  const MH = maskData.height;
  const sx = MW / W;
  const sy = MH / H;

  // ① 找蒙版覆盖的像素（蒙版与图不同尺寸时按比例映射，与管线其它处一致）
  let minX = W;
  let minY = H;
  let maxX = -1;
  let maxY = -1;
  let maskN = 0;
  for (let y = 0; y < H; y++) {
    const my = Math.min(MH - 1, Math.max(0, Math.round(y * sy)));
    for (let x = 0; x < W; x++) {
      if (!maskHit(maskData.data[(my * MW + Math.min(MW - 1, Math.max(0, Math.round(x * sx)))) * 4 + 3])) {
        continue;
      }
      maskN++;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0 || maskN === 0) return null;

  const pad = 2;
  const rx0 = Math.max(0, (bbox ? bbox.x : minX) - pad);
  const ry0 = Math.max(0, (bbox ? bbox.y : minY) - pad);
  const rx1 = Math.min(W, (bbox ? bbox.x + bbox.w : maxX + 1) + pad);
  const ry1 = Math.min(H, (bbox ? bbox.y + bbox.h : maxY + 1) + pad);
  const rw = rx1 - rx0;
  const rh = ry1 - ry0;
  if (rw < 2 || rh < 2) return null;

  // ② 金字塔：level0 = 区域原分辨率，向上每次 2× 降采样
  const levels: Level[] = [];
  {
    const acc = new Float32Array(rw * rh * 3);
    const wgt = new Float32Array(rw * rh);
    const d = imageData.data;
    for (let y = 0; y < rh; y++) {
      for (let x = 0; x < rw; x++) {
        const gx = rx0 + x;
        const gy = ry0 + y;
        const j = (gy * W + gx) * 4;
        const mj =
          (Math.min(MH - 1, Math.max(0, Math.round(gy * sy))) * MW +
            Math.min(MW - 1, Math.max(0, Math.round(gx * sx)))) *
          4;
        const known = !maskHit(maskData.data[mj + 3]);
        const i = y * rw + x;
        if (known) {
          acc[i * 3] = d[j];
          acc[i * 3 + 1] = d[j + 1];
          acc[i * 3 + 2] = d[j + 2];
          wgt[i] = 1;
        }
      }
    }
    levels.push({ w: rw, h: rh, acc, wgt });
  }
  while (levels[levels.length - 1].w > 2 && levels[levels.length - 1].h > 2) {
    const prev = levels[levels.length - 1];
    const w = Math.max(1, prev.w >> 1);
    const h = Math.max(1, prev.h >> 1);
    const acc = new Float32Array(w * h * 3);
    const wgt = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = (y * 2) * prev.w + x * 2;
        const i = y * w + x;
        let a0 = 0;
        let a1 = 0;
        let a2 = 0;
        let w0 = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const si = s + dy * prev.w + dx;
            if (si < 0 || si >= prev.w * prev.h) continue;
            a0 += prev.acc[si * 3];
            a1 += prev.acc[si * 3 + 1];
            a2 += prev.acc[si * 3 + 2];
            w0 += prev.wgt[si];
          }
        }
        acc[i * 3] = a0;
        acc[i * 3 + 1] = a1;
        acc[i * 3 + 2] = a2;
        wgt[i] = w0;
      }
    }
    levels.push({ w, h, acc, wgt });
  }

  // ③ push：自粗向细，把粗层的洞用 4 邻域已知值填上（归一化卷积）
  for (let li = levels.length - 1; li > 0; li--) {
    const fine = levels[li];
    const coarse = levels[li - 1];
    const cw = coarse.w;
    const ch = coarse.h;
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = y * cw + x;
        if (coarse.wgt[i] > 0) continue;
        let a0 = 0;
        let a1 = 0;
        let a2 = 0;
        let w0 = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= cw || ny < 0 || ny >= ch) continue;
            const ni = ny * cw + nx;
            if (coarse.wgt[ni] <= 0) continue;
            a0 += coarse.acc[ni * 3];
            a1 += coarse.acc[ni * 3 + 1];
            a2 += coarse.acc[ni * 3 + 2];
            w0 += coarse.wgt[ni];
          }
        }
        if (w0 > 0) {
          coarse.acc[i * 3] = a0 / w0;
          coarse.acc[i * 3 + 1] = a1 / w0;
          coarse.acc[i * 3 + 2] = a2 / w0;
          coarse.wgt[i] = 1;
        }
      }
    }
    void fine;
  }

  // ④ pull：自细向粗，洞像素取粗层的值
  for (let li = 1; li < levels.length; li++) {
    const fine = levels[li];
    const coarse = levels[li - 1];
    const cw = coarse.w;
    const ch = coarse.h;
    for (let y = 0; y < fine.h; y++) {
      for (let x = 0; x < fine.w; x++) {
        const i = y * fine.w + x;
        if (fine.wgt[i] > 0) continue;
        const ci = Math.min(ch - 1, y >> 1) * cw + Math.min(cw - 1, x >> 1);
        if (coarse.wgt[ci] <= 0) continue;
        fine.acc[i * 3] = coarse.acc[ci * 3];
        fine.acc[i * 3 + 1] = coarse.acc[ci * 3 + 1];
        fine.acc[i * 3 + 2] = coarse.acc[ci * 3 + 2];
        fine.wgt[i] = 1;
      }
    }
  }

  // ⑤ 回写：只改蒙版覆盖的像素（区域外逐位保留）
  const out = new ImageData(new Uint8ClampedArray(imageData.data), W, H);
  const od = out.data;
  const fine = levels[0];
  for (let y = 0; y < rh; y++) {
    const gy = ry0 + y;
    const my = Math.min(MH - 1, Math.max(0, Math.round(gy * sy)));
    for (let x = 0; x < rw; x++) {
      const gx = rx0 + x;
      const mx = Math.min(MW - 1, Math.max(0, Math.round(gx * sx)));
      if (!maskHit(maskData.data[(my * MW + mx) * 4 + 3])) continue;
      const i = y * rw + x;
      if (fine.wgt[i] <= 0) continue;
      const j = (gy * W + gx) * 4;
      od[j] = fine.acc[i * 3];
      od[j + 1] = fine.acc[i * 3 + 1];
      od[j + 2] = fine.acc[i * 3 + 2];
    }
  }
  return { data: out, bbox: { x: rx0, y: ry0, w: rw, h: rh } };
}

/**
 * 能否用经典填充替代神经 inpaint：
 *   ① 蒙版面积占比小（≤ FAST_FILL_MAX_AREA_FRAC）——大洞需要语义补全，
 *      扩散只会糊成一片；
 *   ② 局部背景简单（环带亮度标准差 ≤ FAST_FILL_MAX_RING_STD）——纹理背景
 *      扩散会拉出条纹/糊块（与 detect.ts 保守闸门同一判据口径）；
 *   ③ 蒙版不与图像边缘大面积相交（贴边区域可用上下文太少）。
 */
export const FAST_FILL_MAX_AREA_FRAC = 0.15; // 蒙版是**笔画**（细长），不是实心块：
                                        // 4.3% 的字形笔画在简单背景上扩散填充无损
                                        // （e2e 实测 38.5dB，与神经路径同档）
export const FAST_FILL_MAX_RING_STD = 35;

export function fastFillEligible(
  imageData: ImageData,
  bbox: { x: number; y: number; w: number; h: number },
  ringStd: number,
  maskAreaFrac: number,
): { ok: boolean; reason: string } {
  if (maskAreaFrac > FAST_FILL_MAX_AREA_FRAC) {
    return { ok: false, reason: `area ${(maskAreaFrac * 100).toFixed(2)}% > ${FAST_FILL_MAX_AREA_FRAC * 100}%` };
  }
  if (ringStd > FAST_FILL_MAX_RING_STD) {
    return { ok: false, reason: `ring std ${ringStd.toFixed(1)} > ${FAST_FILL_MAX_RING_STD}` };
  }
  // 贴边（角标水印是主场景）：上下文只有 3 侧，允许——金字塔填充用可用邻居
  // 归一化卷积；实测角标位置 PSNR 与神经路径同档。
  const W = imageData.width;
  const H = imageData.height;
  const touches =
    bbox.x <= 1 || bbox.y <= 1 || bbox.x + bbox.w >= W - 1 || bbox.y + bbox.h >= H - 1;
  return { ok: true, reason: touches ? 'ok (edge)' : 'ok' };
}
