/**
 * R1 统一形状精修管线（.review/2026-10-02-selection-precision-analysis.md，
 * 2026-10-02）：把 text 档已验证的"框内形状吸附"泛化为所有来源共用。
 *
 * 输入：原图 + 用户/检出框 + 候选图（启发式二值化 bin，640 网格，全幅）
 * 输出：全幅 0/1 位图（精修网格分辨率），供 mask-history bitmap op 使用。
 *
 * 算法：候选 bin 上采样为软基底（自动解决多部件合并：基底含框内全部
 * 连通域）→ 以裁剪图亮度为引导做 guided filter（边界吸附真实图像边缘，
 * A5 同族方法）→ 0.5 阈值 → 保留与框相交的连通域（去环带杂斑）。
 *
 * 设计要点：
 * - 基底限定在用户框内（精修可向外长入 15% 搜索环，但不会被邻区候选污染）；
 * - 精修网格 cap 1024（内存红线：全幅位图随 op 常驻 MaskHistory）；
 * - 候选支持为零时返回 null（调用方回退矩形——诚实路径，不凭空造形状）。
 */

const REFINE_PAD_RATIO = 0.15;
const REFINE_PAD_MIN = 8;
const REFINE_GRID_CAP = 1024;
const REFINE_RADIUS = 6;
const REFINE_EPS = 2e-3;
/** 色差候选通道阈值（3 通道绝对差之和）：红章体 vs 场景 ≈ 200+，
 *  框内场景像素 ≈ 0-40——宽余量。 */
const REFINE_CHROMA_DIFF = 90;

/** 灰度引导滤波（He et al. 2010；A5 同款数学，泛化为任意软基底）。 */
function guidedFilter(
  guide: Float32Array,
  p: Float32Array,
  w: number,
  h: number,
  r: number,
  eps: number,
): Float32Array {
  const W1 = w + 1;
  const mk = () => new Float64Array(W1 * (h + 1));
  const sI = mk();
  const sP = mk();
  const sII = mk();
  const sIP = mk();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const Iv = guide[y * w + x];
      const Pv = p[y * w + x];
      const o = (y + 1) * W1 + (x + 1);
      sI[o] = sI[o - 1] + sI[o - W1] - sI[o - W1 - 1] + Iv;
      sP[o] = sP[o - 1] + sP[o - W1] - sP[o - W1 - 1] + Pv;
      sII[o] = sII[o - 1] + sII[o - W1] - sII[o - W1 - 1] + Iv * Iv;
      sIP[o] = sIP[o - 1] + sIP[o - W1] - sIP[o - W1 - 1] + Iv * Pv;
    }
  }
  const rect = (pre: Float64Array, x0: number, y0: number, x1: number, y1: number) =>
    pre[(y1 + 1) * W1 + (x1 + 1)] -
    pre[y0 * W1 + (x1 + 1)] -
    pre[(y1 + 1) * W1 + x0] +
    pre[y0 * W1 + x0];
  const aArr = new Float32Array(w * h);
  const bArr = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mI = rect(sI, x0, y0, x1, y1) / cnt;
      const mP = rect(sP, x0, y0, x1, y1) / cnt;
      const vI = rect(sII, x0, y0, x1, y1) / cnt - mI * mI;
      const cIP = rect(sIP, x0, y0, x1, y1) / cnt - mI * mP;
      const a = cIP / (vI + eps);
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
  const q = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      q[y * w + x] =
        (rect(sA, x0, y0, x1, y1) / cnt) * guide[y * w + x] + rect(sB, x0, y0, x1, y1) / cnt;
    }
  }
  return q;
}

const READBACK = { willReadFrequently: true } as const;

/**
 * 形状精修：候选 bin + 框 → 全幅 0/1 位图（精修网格）。
 * 候选在框内无支持时返回 null（调用方回退矩形）。
 */
export function refineShapeMask(
  imageData: ImageData,
  box: { x: number; y: number; w: number; h: number },
  cand: { data: Uint8Array; w: number; h: number },
): { data: Uint8Array; w: number; h: number } | null {
  const W = imageData.width;
  const H = imageData.height;
  const padX = Math.max(REFINE_PAD_MIN, Math.round(box.w * REFINE_PAD_RATIO));
  const padY = Math.max(REFINE_PAD_MIN, Math.round(box.h * REFINE_PAD_RATIO));
  const ox = Math.max(0, Math.floor(box.x - padX));
  const oy = Math.max(0, Math.floor(box.y - padY));
  const x1 = Math.min(W, Math.ceil(box.x + box.w + padX));
  const y1 = Math.min(H, Math.ceil(box.y + box.h + padY));
  const cw = Math.max(1, x1 - ox);
  const ch = Math.max(1, y1 - oy);

  // 精修网格（cap 1024，内存红线）
  const scale = Math.min(1, REFINE_GRID_CAP / Math.max(cw, ch));
  const rw = Math.max(1, Math.round(cw * scale));
  const rh = Math.max(1, Math.round(ch * scale));
  if (rw < 2 || rh < 2) return null;

  // 引导图：裁剪 → 精修网格 → 亮度 [0,1]
  const src = new OffscreenCanvas(imageData.width, imageData.height);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const cropCv = new OffscreenCanvas(rw, rh);
  const cctx = cropCv.getContext('2d', READBACK)!;
  cctx.imageSmoothingEnabled = true;
  cctx.imageSmoothingQuality = 'high';
  cctx.drawImage(src, ox, oy, cw, ch, 0, 0, rw, rh);
  const cd = cctx.getImageData(0, 0, rw, rh).data;
  const guide = new Float32Array(rw * rh);
  for (let i = 0; i < rw * rh; i++) {
    guide[i] = (0.299 * cd[i * 4] + 0.587 * cd[i * 4 + 1] + 0.114 * cd[i * 4 + 2]) / 255;
  }

  // 软基底：候选 bin 双线性采样到精修网格，并限定在用户框内
  const base = new Float32Array(rw * rh);
  let support = 0;
  // 用户框 = 全图坐标（候选 bin 已重排为全图外幅，色差通道用 fx/fy 图像
  // 坐标——region 模式下此前混用裁剪/全图坐标导致双通道全空）
  const bx0 = box.x;
  const by0 = box.y;
  const bx1 = bx0 + box.w;
  const by1 = by0 + box.h;
  for (let y = 0; y < rh; y++) {
    // 精修网格 y → 原图 y → 候选网格 y
    const oyImg = oy + (y + 0.5) / scale;
    for (let x = 0; x < rw; x++) {
      const oxImg = ox + (x + 0.5) / scale;
      if (oxImg < bx0 || oxImg >= bx1 || oyImg < by0 || oyImg >= by1) continue;
      const cx = Math.min(cand.w - 1, Math.max(0, Math.round((oxImg / W) * cand.w)));
      const cy = Math.min(cand.h - 1, Math.max(0, Math.round((oyImg / H) * cand.h)));
      const v = cand.data[cy * cand.w + cx];
      if (v) {
        base[y * rw + x] = 1;
        support++;
      }
    }
  }
  // 色差候选通道（用户实拍红章案例，2026-10-02）：饱和色徽章体不在
  // light/dark 判据内（中亮度高饱和）——但与**框外环带**（pad 区场景色）
  // 的色差显著。参照色**逐行双中值**（该行左 pad / 右 pad 各取中值，像素
  // 按位置线性插值）——城市天际线等垂直渐变场景下单一全局中值会失准
  // （红章体只抓到 41% 的教训），行级参照贴合局部场景色。
  {
    const med3 = (a: number[]) => (a.length ? a.slice().sort((p, q) => p - q)[a.length >> 1] : null);
    const rowRef: ({ r: number; g: number; b: number } | null)[] = [];
    for (let y = 0; y < rh; y++) {
      const l: number[][] = [[], [], []];
      const r2: number[][] = [[], [], []];
      for (let x = 0; x < rw; x += 2) {
        const fx = ox + (x + 0.5) / scale;
        const fy = oy + (y + 0.5) / scale;
        if (fx >= bx0 && fx < bx1 && fy >= by0 && fy < by1) continue;
        const j = (y * rw + x) * 4;
        const tgt = fx < bx0 ? l : r2;
        tgt[0].push(cd[j]);
        tgt[1].push(cd[j + 1]);
        tgt[2].push(cd[j + 2]);
      }
      const mL = med3(l[0]) !== null && med3(l[1]) !== null && med3(l[2]) !== null
        ? { r: med3(l[0])!, g: med3(l[1])!, b: med3(l[2])! } : null;
      const mR = med3(r2[0]) !== null && med3(r2[1]) !== null && med3(r2[2]) !== null
        ? { r: med3(r2[0])!, g: med3(r2[1])!, b: med3(r2[2])! } : null;
      rowRef.push(mL || mR || mR === null && mL ? (mL && mR ? { r: (mL.r + mR.r) / 2, g: (mL.g + mR.g) / 2, b: (mL.b + mR.b) / 2 } : mL ?? mR) : null);
    }
    for (let y = 0; y < rh; y++) {
      const ref = rowRef[y];
      if (!ref) continue;
      for (let x = 0; x < rw; x++) {
        const i = y * rw + x;
        const fx = ox + (x + 0.5) / scale;
        const fy = oy + (y + 0.5) / scale;
        if (fx < bx0 || fx >= bx1 || fy < by0 || fy >= by1) continue;
        const j = (y * rw + x) * 4;
        const dist =
          Math.abs(cd[j] - ref.r) + Math.abs(cd[j + 1] - ref.g) + Math.abs(cd[j + 2] - ref.b);
        if (dist > REFINE_CHROMA_DIFF) base[i] = 1;
      }
    }
  }
  // 候选支持过低：无法造形状（诚实回退矩形）
  console.debug('[refine-dbg] support=', support, 'thresh=', Math.max(24, box.w * box.h * 0.004 * scale * scale), 'chromaBase=', (() => { let n = 0; for (let i = 0; i < base.length; i++) if (base[i]) n++; return n; })());
  if (support < Math.max(24, box.w * box.h * 0.004 * scale * scale)) { console.debug('[refine-dbg] NULL at support'); return null; }

  const q = guidedFilter(guide, base, rw, rh, REFINE_RADIUS, REFINE_EPS);

  // 0.5 阈值 → 连通域 → 保留与用户框相交的域（去搜索环杂斑）
  const bin = new Uint8Array(rw * rh);
  for (let i = 0; i < rw * rh; i++) bin[i] = q[i] > 0.5 ? 1 : 0;
  // 孔洞填充：实心徽章内的亮字/图标会被引导滤波挖成洞（候选图只有暗底），
  // 漏检的主因（logo-tl recall 100→68.6 的教训）。从边界洪水：未触及边界的
  // 背景区 = 洞 → 置 1。文字笔画间的封闭小孔同样被填，但 text 档走 glyph
  // 路径不经此处；logo/watermark 类 badge 语义下填洞正是所需。
  {
    const flood = new Uint8Array(rw * rh);
    const st: number[] = [];
    for (let x = 0; x < rw; x++) {
      if (!bin[x]) { flood[x] = 1; st.push(x); }
      const b = (rh - 1) * rw + x;
      if (!bin[b]) { flood[b] = 1; st.push(b); }
    }
    for (let y = 0; y < rh; y++) {
      const l = y * rw;
      if (!bin[l]) { flood[l] = 1; st.push(l); }
      const r2 = y * rw + rw - 1;
      if (!bin[r2]) { flood[r2] = 1; st.push(r2); }
    }
    while (st.length) {
      const p = st.pop()!;
      const px = p % rw;
      if (px > 0 && !bin[p - 1] && !flood[p - 1]) { flood[p - 1] = 1; st.push(p - 1); }
      if (px < rw - 1 && !bin[p + 1] && !flood[p + 1]) { flood[p + 1] = 1; st.push(p + 1); }
      if (p >= rw && !bin[p - rw] && !flood[p - rw]) { flood[p - rw] = 1; st.push(p - rw); }
      if (p < rw * (rh - 1) && !bin[p + rw] && !flood[p + rw]) { flood[p + rw] = 1; st.push(p + rw); }
    }
    for (let i = 0; i < rw * rh; i++) if (!flood[i]) bin[i] = 1;
  }
  const label = new Int32Array(rw * rh).fill(-1);
  const keep = new Uint8Array(rw * rh);
  const stack: number[] = [];
  let cur = 0;
  for (let i = 0; i < rw * rh; i++) {
    if (!bin[i] || label[i] >= 0) continue;
    const comp: number[] = [];
    let touchesBox = false;
    stack.push(i);
    label[i] = cur;
    while (stack.length) {
      const p = stack.pop()!;
      comp.push(p);
      const px = p % rw;
      const py = (p / rw) | 0;
      if (px + ox >= bx0 && px + ox < bx1 && py + oy >= by0 && py + oy < by1) touchesBox = true;
      if (px > 0 && bin[p - 1] && label[p - 1] < 0) { label[p - 1] = cur; stack.push(p - 1); }
      if (px < rw - 1 && bin[p + 1] && label[p + 1] < 0) { label[p + 1] = cur; stack.push(p + 1); }
      if (py > 0 && bin[p - rw] && label[p - rw] < 0) { label[p - rw] = cur; stack.push(p - rw); }
      if (py < rh - 1 && bin[p + rw] && label[p + rw] < 0) { label[p + rw] = cur; stack.push(p + rw); }
    }
    cur++;
    if (touchesBox) for (const p of comp) keep[p] = 1;
  }

  // 全幅重组（精修网格尺度，bitmap op 按全幅拉伸绘制）
  const fullW = Math.max(1, Math.round(W * scale));
  const fullH = Math.max(1, Math.round(H * scale));
  const full = new Uint8Array(fullW * fullH);
  const offX = Math.round(ox / scale);
  const offY = Math.round(oy / scale);
  for (let y = 0; y < rh; y++) {
    const fy = y + offY;
    if (fy < 0 || fy >= fullH) continue;
    for (let x = 0; x < rw; x++) {
      if (!keep[y * rw + x]) continue;
      const fx = x + offX;
      if (fx >= 0 && fx < fullW) full[fy * fullW + fx] = 1;
    }
  }
  // 面积守门：精修结果收缩超过 40%（vs 框面积）视为精修吃掉了真实水印
  // （候选出洞的场景，如暗徽章坐在暗背景上）——该框回退矩形。合法收缩
  // 上限 = 检出框膨胀系数（≤1.3×→收缩 ≤23%），0.6 有充分余量。
  let kept = 0;
  for (let i = 0; i < full.length; i++) kept += full[i];
  const boxAreaRefine = box.w * box.h * scale * scale;
  console.debug('[refine-dbg] kept=', kept, 'boxAreaRefine=', boxAreaRefine, 'ratio=', (kept / Math.max(1, boxAreaRefine)).toFixed(2));
  if (kept < Math.max(24, boxAreaRefine * 0.6)) { console.debug('[refine-dbg] NULL at area guard'); return null; }
  return { data: full, w: fullW, h: fullH };
}

/** 多个精修位图取并集（多部件水印/多框联合）。 */
export function unionMasks(masks: { data: Uint8Array; w: number; h: number }[]): {
  data: Uint8Array;
  w: number;
  h: number;
} | null {
  if (masks.length === 0) return null;
  const w = masks[0].w;
  const h = masks[0].h;
  const out = new Uint8Array(w * h);
  for (const m of masks) {
    if (m.w !== w || m.h !== h) continue;
    for (let i = 0; i < out.length; i++) if (m.data[i]) out[i] = 1;
  }
  return out.some((v) => v) ? { data: out, w, h } : null;
}
