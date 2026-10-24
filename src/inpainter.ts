/**
 * MI-GAN inpainting via ONNX Runtime Web — 100% in-browser.
 *
 * Model: MI-GAN (Picsart Research, ICCV 2023), ONNX export fixed at 512×512
 * with a single 4-channel float32 input. Pipeline per pass:
 *   1. Find the mask bounding box.
 *   2. Square-crop image + mask around it.
 *   3. Resize the crop to 512×512.
 *   4. Build MI-GAN's 4-channel input (mask polarity: 1 = keep, 0 = inpaint).
 *   5. Run the model; map the [-1,1] output back to RGB.
 *   6. Composite back into the original — only masked pixels change.
 */

import type * as ort from 'onnxruntime-web';
import { createSession, getOrt, isAppleMobile } from './ort-setup';
import { loadModel, type ModelProgress, type ModelSpec } from './model-loader';

/**
 * MI-GAN fp32 基线（29.5 MB）——fp16 版的**兜底 spec**。
 * key 保持 `migan_v1` 不动：既有已缓存用户在 fp16 不可用时直接命中本地缓存，
 * 不会重复下载（与 LaMa 的 lama_fp32_v1 兜底同款结构）。
 */
export const MIGAN_FP32_MODEL: ModelSpec = {
  key: 'migan_v1',
  urls: [
    'models/migan.onnx',
    'https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx',
    'https://hf-mirror.com/lxfater/inpaint-web/resolve/main/migan.onnx',
  ],
  sha256: 'bb7189b2523b8485d9dd6baa2e7e8bccce4493760daa33ffcd432f667945bf62',
  expectedSize: 29541096,
};

/**
 * MI-GAN fp16（14.8 MB，−50% 下载；WebGPU 推理 171→133ms，实测 1.29×）。
 *
 * 为什么 fp16 这次直接能用（方法论见 blog #13 更正篇 + lab/migan-quant）：
 * ① **先审计数值量程**：MI-GAN 的图里**没有 BatchNormalization**（BN 已折进
 *    Conv），307 个 fp32 初始化子的最大 |值| = 256，远低于 fp16 上限 65504
 *    → 没有 LaMa 那种 running_var 溢出被静默夹断的坑；
 * ② 模型级画质：13 个 bench 用例上 fp16 vs fp32 输出 PSNR 62.8~80.1 dB
 *    （中位 76.3），端到端 vs 干净原图 Δ = −0.02…+0.02 dB（**零可见差异**）；
 * ③ 浏览器实测（Apple GPU，ORT Web 1.30，512² 单 tile）：fp32 171ms/29.5MB
 *    → fp16 **133ms/14.8MB**（1.29×，下载减半，会话创建 711ms→58ms）。
 *
 * **前提**：WebGPU 端要能建 fp16 session——ORT 的 WGSL 生成器需要设备支持
 * `shader-f16`（无此特性的适配器会在建 session 时报 "Program Transpose
 * requires f16"）。因此选模型前先查适配器特性，不支持就用 fp32（见
 * `miganPrefersFp16`）。iOS 强制 WASM（WebKit 雷区），WASM 上 fp16 无原生
 * 核 → 按实测结论选型（见 lab/migan-quant/results）。
 */
export const INPAINT_MODEL: ModelSpec = {
  key: 'migan_fp16_v1',
  urls: ['models/migan_fp16.onnx'],
  sha256: '89a09168e647d6ddec5fa76d4afb7b18af62922ff82c821be509dacdc26196bc',
  expectedSize: 14822988,
  fallback: MIGAN_FP32_MODEL,
};

const MODEL_SIZE = 512;
const READBACK = { willReadFrequently: true } as const;

/**
 * LaMa fp32 基线（198 MiB / 208,044,816 B）——量化版的**兜底 spec**。
 * key 保持 `lama_fp32_v1` 不动：既有已缓存用户在下游 fp16 失败时直接命中本地缓存，
 * 不会重复下载。
 */
export const LAMA_FP32_MODEL: ModelSpec = {
  key: 'lama_fp32_v1',
  urls: [
    'models/lama_fp32.onnx',
    'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
    'https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
  ],
  sha256: '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6',
  expectedSize: 208044816,
};

/**
 * LaMa HQ 量化版（fp16，107.8 MB；线上 brotli q11 后用户实际下载 86.3 MiB，−52%）。
 *
 * 为什么 fp16 这次能用了（证据见 lab/lama-quant/FINDINGS.md）：旧结论「掩码门控模型
 * 对精度固有敏感」是错的——真因是 16 个 BatchNorm 的 `running_var` 最大 484,322，
 * 超过 fp16 上限 65,504 被转换器**静默夹断**。用代数重标定
 * （`var/K²`、`γ/K`、`ε/K²`，K=16）修好后：14 例 bench 掩码区 PSNR 29.73 vs fp32
 * 29.74 dB，浏览器真实运行时与 fp32 参考 66.5 dB（亚灰阶，视觉不可见）。
 *
 * **WASM 上这是下载体积优化，不是速度优化**：WASM EP 没有原生 fp16 卷积内核，
 * 算子会退化成 Cast + fp32 计算，单 tile 反而 11.4s → 17.4s（1.5× 慢）。
 * WebGPU 上 fp16 走原生半精度：实测 0.86s/tile（WASM 4.06s，约 4.7×），
 * 与 WASM 输出 PSNR 71.6dB——HQ 现在默认 WebGPU 优先（见 initLama）。
 */
export const LAMA_MODEL: ModelSpec = {
  key: 'lama_fp16_v1',
  urls: ['models/lama_fp16.onnx'],
  sha256: 'ef3973c7704d46ffebf247b4d985ecb9d562c01cfeb1517e1d2df22f3d77475e',
  expectedSize: 107782990,
  fallback: LAMA_FP32_MODEL,
};

let session: ort.InferenceSession | null = null;
let lamaSession: ort.InferenceSession | null = null;
let activeBackend = '';
/** 当前 MI-GAN 会话用的 spec（fp16 或 fp32）——运行期回退重建会话时必须用同一个
 *  （否则会出现"WASM 会话拿 fp16 权重、fp32 回退拿另一份"的错配）。 */
let activeMiganSpec: ModelSpec = MIGAN_FP32_MODEL;

/**
 * 该设备用哪个 MI-GAN spec（**同步**判定，页面预取/隐私面板共用同一条规则）：
 *   - 有 WebGPU（`navigator.gpu` 存在）→ fp16：Apple GPU 实测 171ms → 133ms
 *     （1.29×），下载 29.5MB → 14.8MB，会话创建 711ms → 58ms；
 *   - 无 WebGPU（iOS/Safari、纯 WASM 通路）→ fp32：WASM 上 fp16 无原生核，
 *     实测 326ms → 408ms/tile（慢 25%），而下载只省一次性的 14.8MB——
 *     每 tile 的代价会重复上千次，**fp32 更划算**。
 * 边缘情况：有 WebGPU 但适配器缺 `shader-f16`（老 GPU）→ fp16 session 建不起来
 * （ORT 报 "Program Transpose requires f16"），initInpainter 会自动换 fp32
 * 重试（那台设备会多下 14.8MB，罕见，可接受）。
 */
export function miganSpec(): ModelSpec {
  const hasGpu =
    typeof navigator !== 'undefined' &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;
  // iOS/iPadOS 固定 WASM 通路（WebKit 的 WebGPU 仍是雷区，见 isAppleMobile）——
  // WASM 上 fp16 无原生核（实测慢 25%），所以 Apple 移动端一律 fp32。
  if (hasGpu && !isAppleMobile()) return INPAINT_MODEL;
  return MIGAN_FP32_MODEL;
}

/**
 * 精确选型（异步）：在同步规则之上再查一次适配器是否支持 `shader-f16`
 * （ORT 生成 fp16 WGSL 的前提）。**先查再下载**——否则缺该特性的老 GPU 会
 * 先下 14.8MB fp16、建 session 失败、再下 29.5MB fp32，白费一半流量。
 * 结果缓存（一次会话只查一次）。
 */
let miganSpecPromise: Promise<ModelSpec> | null = null;
export function resolveMiganSpec(): Promise<ModelSpec> {
  miganSpecPromise ??= (async () => {
    const spec = miganSpec();
    if (spec !== INPAINT_MODEL) return spec;
    const gpu = (navigator as unknown as {
      gpu?: { requestAdapter?: () => Promise<{ features?: Set<string> } | null> };
    }).gpu;
    if (!gpu?.requestAdapter) return MIGAN_FP32_MODEL;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter?.features?.has('shader-f16')) {
        console.info('[inpaint] adapter lacks shader-f16 → using fp32 MI-GAN');
        return MIGAN_FP32_MODEL;
      }
    } catch {
      return MIGAN_FP32_MODEL;
    }
    return spec;
  })();
  return miganSpecPromise;
}

export function getInpaintBackend() {
  return activeBackend;
}

/**
 * WebGPU 会话创建成功不代表推理能跑——LaMa 这类大模型在部分 GPU/驱动
 * 组合上会在 session.run 时才失败（显存上限、未实现算子），而创建期的
 * providers 回退覆盖不到运行期。这里兜底：webgpu 推理失败 → 用缓存的
 * 模型字节重建 WASM 会话重试一次，成功后固定走 WASM（不再每遍重试）。
 * loadModel 命中 IndexedDB 缓存，不会重新下载。
 */
async function runWithFallback(
  which: 'migan' | 'lama',
  feeds: Record<string, ort.Tensor>,
): Promise<Awaited<ReturnType<ort.InferenceSession['run']>>> {
  const current = which === 'lama' ? lamaSession : session;
  if (!current) throw new Error(`${which === 'lama' ? 'LaMa' : 'Inpainter'} engine not initialized`);
  try {
    // WebGPU 推理可能死锁（不报错也不返回），看门狗按失败处理走 WASM 回退。
    // WASM 是最终路径、合法耗时可能很长，不加超时。
    if (activeBackend === 'webgpu') {
      // LaMa 单 tile 实测 ~0.9s（M4/Metal），60s 足够覆盖冷启动与低端设备；
      // 300s 的旧预算会让真正的死锁把用户卡住五分钟。
      const budget = which === 'lama' ? 60_000 : 120_000;
      return await Promise.race([
        current.run(feeds),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('webgpu inference watchdog')), budget),
        ),
      ]);
    }
    return await current.run(feeds);
  } catch (e) {
    if (activeBackend !== 'webgpu') throw e;
    console.warn(`[inpaint] ${which} run failed on WebGPU, falling back to WASM:`, e);
    const bytes = await loadModel(which === 'lama' ? LAMA_MODEL : activeMiganSpec);
    const s = await createSession(bytes, undefined, ['wasm']);
    if (which === 'lama') lamaSession = s.session;
    else session = s.session;
    activeBackend = s.backend;
    return (which === 'lama' ? lamaSession : session)!.run(feeds);
  }
}

/** 空闲时跑一次 dummy 前向（错误静默）——把 shader 编译/计划缓存的
 *  1-5s 冷启动从用户首次处理前藏掉（时长优化第 3 步·预热）。 */
const warmPromises: Partial<Record<'migan' | 'lama', Promise<void>>> = {};

function scheduleWarmup(which: 'migan' | 'lama', run: () => Promise<unknown>): void {
  const idle = (cb: () => void) =>
    'requestIdleCallback' in window
      ? window.requestIdleCallback(cb, { timeout: 3000 })
      : setTimeout(cb, 0);
  warmPromises[which] = new Promise<void>((resolve) => {
    idle(() => {
      const t0 = performance.now();
      run()
        .then(() => console.debug('[warmup] inference ready in', Math.round(performance.now() - t0), 'ms'))
        .catch(() => {})
        .finally(resolve);
    });
  });
}

/** 首次真实推理前等 pending 预热收尾：ORT WebGPU EP 不容忍同一 session 并发
 *  run（实测竞态翻车：'Session already started'/'Session mismatch'/张量
 *  detach，2026-09-30 三连跑复现）。promise 已 settled 时 await 零成本。 */
async function awaitWarmup(which: 'migan' | 'lama'): Promise<void> {
  const p = warmPromises[which];
  if (p) {
    delete warmPromises[which];
    await p.catch(() => {});
  }
}

export async function initInpainter(
  onProgress?: (p: ModelProgress) => void,
): Promise<void> {
  if (session) return;
  // fp16 vs fp32 选型（2026-10-21 实测）：
  //   WebGPU（Apple GPU，512² 单 tile）：fp32 171ms/29.5MB → fp16 **133ms/14.8MB**
  //   WASM（4 线程，同机）：                fp32 326ms        → fp16 408ms（慢 25%）
  // → WebGPU 且适配器支持 shader-f16 才用 fp16；其余（WASM 通路 / 老 GPU）用
  //   fp32：WASM 上每次 tile 都要多付 80ms，一次性下载省下的 14.8MB 换不来。
  let spec: ModelSpec = await resolveMiganSpec();
  let bytes = await loadModel(spec, onProgress);
  let s: { session: ort.InferenceSession; backend: string };
  try {
    s = await createSession(bytes, (status) =>
      onProgress?.({ done: 1, total: 1, stage: status }),
    );
  } catch (e) {
    // fp16 session 建不起来（设备缺 shader-f16 / 驱动差异）→ 换 fp32 重试一次。
    if (spec !== INPAINT_MODEL) throw e;
    console.warn('[inpaint] fp16 session failed, falling back to fp32:', e);
    spec = MIGAN_FP32_MODEL;
    bytes = await loadModel(spec, onProgress);
    s = await createSession(bytes, (status) =>
      onProgress?.({ done: 1, total: 1, stage: status }),
    );
  }
  activeMiganSpec = spec;
  session = s.session;
  activeBackend = s.backend;
  const sess = session;
  scheduleWarmup('migan', async () => {
    const ort = await getOrt();
    // 形状与真实输入一致：MI-GAN 4 通道 512²（全零张量）
    const feeds: Record<string, ort.Tensor> = {};
    feeds[sess.inputNames[0]] = new ort.Tensor(
      'float32',
      new Float32Array(4 * MODEL_SIZE * MODEL_SIZE),
      [1, 4, MODEL_SIZE, MODEL_SIZE],
    );
    // 走 runWithFallback：webgpu 死锁有看门狗兜底，预热不会挂住会话
    await runWithFallback('migan', feeds);
  });
}

/** LaMa HQ 引擎初始化（仅用户显式开启 HQ 时调用——108MB 量化版不预取）。
 *
 *  WebGPU 优先：历史上 LaMa 在 WebGPU 上的「FFC/convg2g/Add: Can't perform
 *  binary op on the given tensors」并非算子缺口，而是 `onnxruntime-web` 默认
 *  构建（ort.bundle）里 Slice 负步长（step=-1）的 off-by-one——FFT 分支的
 *  `x[..., 1:-1][..., ::-1]` 少一个元素，64→63，随后与卷积输出 Add 维度不
 *  匹配。改用 `onnxruntime-web/webgpu` 构建后同图主线程/proxy worker 均正确
 *  （实测 WebGPU vs WASM 输出 PSNR 71.6dB，单 tile 4.06s→0.86s，约 4.7×）。
 *  `['webgpu','wasm']` 仍保留运行期看门狗 + WASM 回退：GPU 驱动死锁/显存
 *  不足时重建 WASM 会话继续跑。
 *
 *  iOS/iPadOS 例外：WebKit 的 WebGPU 仍是雷区（推理后内存/CPU 暴涨、驱动
 *  差异），这些设备固定 WASM，不做 WebGPU 尝试。 */
export async function initLama(
  onProgress?: (p: ModelProgress) => void,
): Promise<void> {
  if (lamaSession) return;
  const bytes = await loadModel(LAMA_MODEL, onProgress);
  const s = await createSession(
    bytes,
    (status) => onProgress?.({ done: 1, total: 1, stage: status }),
    isAppleMobile() ? ['wasm'] : undefined,
  );
  lamaSession = s.session;
  activeBackend = s.backend;
  const sess = lamaSession;
  scheduleWarmup('lama', async () => {
    const ort = await getOrt();
    // 形状与真实输入一致：image [1,3,512²] + mask [1,1,512²]（全零）
    const feeds: Record<string, ort.Tensor> = {};
    feeds[sess.inputNames[0]] = new ort.Tensor(
      'float32',
      new Float32Array(3 * MODEL_SIZE * MODEL_SIZE),
      [1, 3, MODEL_SIZE, MODEL_SIZE],
    );
    feeds[sess.inputNames[1]] = new ort.Tensor(
      'float32',
      new Float32Array(MODEL_SIZE * MODEL_SIZE),
      [1, 1, MODEL_SIZE, MODEL_SIZE],
    );
    await runWithFallback('lama', feeds);
  });
}

/**
 * 释放推理会话（wasm 堆数百 MB）。视频逐帧处理完成后、ffmpeg x264 组装
 * 之前调用——组装是整条链路的内存峰值点，ort 会话若还占着堆，低端环境
 * 会把渲染进程直接 OOM 杀掉（页面无声崩溃）。模型字节在 IndexedDB，
 * 下次使用重新 init 即可。
 */
export async function releaseInpainter(): Promise<void> {
  const a = session;
  const b = lamaSession;
  session = null;
  lamaSession = null;
  activeBackend = '';
  try { await a?.release(); } catch { /* noop */ }
  try { await b?.release(); } catch { /* noop */ }
}

export interface InpaintOptions {
  onPass?: (passIndex: number, totalPasses: number) => void;
  /** 2 (default, image quality) or 1 (video frames — halves per-frame time). MI-GAN only. */
  passes?: 1 | 2;
  /** 'migan' 默认（快/小区域）；'lama' HQ（大空洞结构重建，108MB 量化模型）。 */
  engine?: 'migan' | 'lama';
  /** 分段计时：planMs = labelMaskComponents+聚簇+tile 规划；runMs = 模型前向+粘贴+门控。 */
  onTiming?: (t: { planMs: number; runMs: number }) => void;
}

type BBox = { x: number; y: number; w: number; h: number };
type Cluster = { bbox: BBox; comps: number[] };
type LamaPlan = { cl: Cluster; tiles: BBox[] | null };
type MiganPlan = { cl: Cluster; fineTiles: FineTile[] | null };

/**
 * 杠杆 C（时长优化第 1 步）：连通域/聚簇/tile 规划/簇蒙版都是纯蒙版函数，
 * 与帧内容无关——按蒙版对象身份缓存。视频页 detectEvery=0 时整个循环复用
 * 同一个 maskData 对象，WeakMap 天然命中；采样重检替换蒙版（新对象）自动
 * 失效重算。缓存不改变任何修复语义。
 */
interface PlanEntry {
  comps: MaskComp[];
  anyComp: Uint8Array; // 全蒙版 comp 像素查找表（buildClusterMask 软边带防偷邻 tile 用）
  clusters: Cluster[];
  clusterMasks: ImageData[]; // 与 clusters 同序，懒构建
  migan: { twoPass: boolean; plans: MiganPlan[] } | null; // plans 依赖 passes 变体
  lama: LamaPlan[] | null;
}
const planCache = new WeakMap<ImageData, PlanEntry>();

export async function inpaint(
  imageData: ImageData,
  maskData: ImageData,
  opts: InpaintOptions = {},
): Promise<ImageData> {
  const engine = opts.engine ?? 'migan';
  if (engine === 'lama') {
    if (!lamaSession) throw new Error('LaMa engine not initialized');
  } else if (!session) throw new Error('Inpainter not initialized');
  if (imageData.width !== maskData.width || imageData.height !== maskData.height) {
    throw new Error('Image and mask dimensions must match');
  }
  await awaitWarmup(engine);

  // Round B' (2026-09-25 验收根因)：此前用全局 bbox 取一次裁剪，
  // 裁剪上限 min(W,H) —— 蒙版铺得开时（双角标/平铺/分散框）外围区域
  // 从未被修复（插桩抓到 R=0），这是"残留"家族的真正根因。
  // 现在：蒙版连通域 → 邻近分组（bbox 外扩 MERGE_GAP 相交即同组）
  // → 每组独立走 coarse-to-fine 两遍，保证每个区域都被裁进上下文。
  const tPlan0 = performance.now();
  let entry = planCache.get(maskData);
  if (!entry) {
    const comps = labelMaskComponents(maskData);
    const anyComp = new Uint8Array(maskData.width * maskData.height);
    for (const c of comps) for (const p of c.pixels) anyComp[p] = 1;
    entry = {
      comps,
      anyComp,
      clusters: comps.length > 0 ? clusterComponents(comps.map((c) => c.bbox)) : [],
      clusterMasks: [],
      migan: null,
      lama: null,
    };
    planCache.set(maskData, entry);
  }
  const { comps, clusters } = entry;
  if (comps.length === 0) {
    opts.onTiming?.({ planMs: performance.now() - tPlan0, runMs: 0 });
    return imageData;
  }

  // Round 11（2026-09-25 用户实测反馈"看得出被消除过"）：MI-GAN 输入固定
  // 512²，旧管线把任意大的组裁剪整体压到 512 再放大回去——修补区比周围糊，
  // 糊块在清晰背景上就是"消除痕迹"。改为：粗遍保持全局结构（context 3.0），
  // 精修遍对大组按 ≤512 原生 tile 逐块精修（FINE_TILE_CORE + context 1.5
  // 保证 crop ≤512，零压缩），小组维持原路径。tile 从粗遍的一致结果出发，
  // 避免密集场独立幻觉的拼接问题（第 10 轮教训）。
  const FINE_TILE_TRIGGER = 384; // 单连通域长边超过它才分 tile（>1.5× 压缩阈值）
  const FINE_TILE_CORE = 340;    // tile 蒙版 bbox 上限：340×1.5=510 ≤ 512
  const twoPass = (opts.passes ?? 2) > 1;

  // 预先算出每组计划，确定总遍数用于进度。
  // 触发条件：最大单连通域超阈值（大片连通水印的 512 压缩糊），或簇并集
  // 呈长条（文字行/列——软化幻觉在笔画带上同样显眼）；近方形密集小框场
  // （成片误检/小字群）分块会重新引入 per-tile 幻觉拼接（bench
  // text-br-small 19.7→11.9 的教训），维持全局精修遍。
  // LaMa HQ 路径：FFC 全局感受野，单遍即可（无需 coarse-to-fine）。
  // 组 crop（context 1.3）≤512 直接单次；更大则按像素级网格 tile（每格
  // ≤340px → crop ≤442 原生分辨率）逐格修，羽化粘贴，末端同一负收益门控。
  if (engine === 'lama') {
    const LAMA_CONTEXT = 1.3;
    const LAMA_TILE = 340;
    if (!entry.lama) {
      entry.lama = clusters.map((cl) => {
        const maskDim = Math.max(cl.bbox.w, cl.bbox.h);
        if (maskDim * LAMA_CONTEXT <= MODEL_SIZE) return { cl, tiles: null };
        // 网格切分蒙版像素（无需整格覆盖，空格跳过）
        const cells = new Map<string, BBox>();
        for (const ci of cl.comps) {
          const c = comps[ci];
          const gx0 = Math.floor((c.bbox.x - cl.bbox.x) / LAMA_TILE);
          const gy0 = Math.floor((c.bbox.y - cl.bbox.y) / LAMA_TILE);
          const gx1 = Math.floor((c.bbox.x + c.bbox.w - 1 - cl.bbox.x) / LAMA_TILE);
          const gy1 = Math.floor((c.bbox.y + c.bbox.h - 1 - cl.bbox.y) / LAMA_TILE);
          for (let gy = gy0; gy <= gy1; gy++) {
            for (let gx = gx0; gx <= gx1; gx++) {
              const key = `${gx},${gy}`;
              if (!cells.has(key)) {
                cells.set(key, {
                  x: cl.bbox.x + gx * LAMA_TILE,
                  y: cl.bbox.y + gy * LAMA_TILE,
                  w: LAMA_TILE,
                  h: LAMA_TILE,
                });
              }
            }
          }
        }
        return { cl, tiles: [...cells.values()] };
      });
    }
    const lamaPlans = entry.lama;
    const totalLama = lamaPlans.reduce((n, p) => n + (p.tiles ? p.tiles.length : 1), 0);
    const planMs = performance.now() - tPlan0;
    const tRun0 = performance.now();
    let current = imageData;
    let pi = 0;
    for (let k = 0; k < lamaPlans.length; k++) {
      const { cl, tiles } = lamaPlans[k];
      const clusterMask = (entry.clusterMasks[k] ??= buildClusterMask(maskData, cl.comps, comps, entry.anyComp));
      if (tiles === null) {
        opts.onPass?.(pi++, totalLama);
        current = await lamaRunPass(current, clusterMask, cl.bbox, LAMA_CONTEXT);
      } else {
        for (const cell of tiles) {
          // 格内蒙版 = 簇蒙版 ∩ 格子矩形
          const cellMask = clipMaskToRect(clusterMask, cell);
          opts.onPass?.(pi++, totalLama);
          current = await lamaRunPass(current, cellMask, cell, LAMA_CONTEXT);
        }
      }
    }
    const result = applyNegativeGainGuard(imageData, current, maskData);
    opts.onTiming?.({ planMs, runMs: performance.now() - tRun0 });
    return result;
  }

  // plans 依赖 passes 变体（twoPass 决定 fineTiles），缓存按变体存；
  // 同一蒙版换档位（图像页两遍 vs 视频页一遍）时重算覆盖，正确性不变。
  if (!entry.migan || entry.migan.twoPass !== twoPass) {
    entry.migan = {
      twoPass,
      plans: clusters.map((cl) => {
        const maxCompDim = Math.max(...cl.comps.map((ci) => Math.max(comps[ci].bbox.w, comps[ci].bbox.h)));
        // 长条簇（文字行/列，并集长宽比 > 3）即使单连通域不大也分 tile：
        // 整簇一次压到 512 会让填充发软、与原纹理不匹配——这正是文字水印
        // "消除痕迹"的主要来源之一（实测 text-center MAE 24.8，比不处理还差
        // 8 倍）。近方形密集框场维持全局精修（text-br-small 19.7→11.9 的教训）。
        const unionLong = Math.max(cl.bbox.w, cl.bbox.h);
        const unionShort = Math.max(1, Math.min(cl.bbox.w, cl.bbox.h));
        const isBand = unionLong > FINE_TILE_TRIGGER && unionLong / unionShort > 3;
        const fineTiles =
          twoPass && (maxCompDim > FINE_TILE_TRIGGER || isBand)
            ? partitionByCap(cl.comps.map((ci) => comps[ci].bbox), FINE_TILE_CORE).flatMap((t) =>
                splitOversizedTile(t, FINE_TILE_CORE),
              )
            : null;
        return { cl, fineTiles };
      }),
    };
  }
  const plans = entry.migan.plans;
  const totalPasses = plans.reduce(
    (n, p) => n + 1 + (twoPass ? (p.fineTiles ? p.fineTiles.length : 1) : 0),
    0,
  );
  const planMs = performance.now() - tPlan0;
  const tRun0 = performance.now();
  let current = imageData;
  let pi = 0;
  for (let k = 0; k < plans.length; k++) {
    const { cl, fineTiles } = plans[k];
    // 每簇独立蒙版：裁剪会盖住邻簇像素，若用全量蒙版，邻簇区域会被
    // 本簇 inpaint+粘贴一次、再被自己的簇重复一遍——重复重绘会劣化。
    const clusterMask = (entry.clusterMasks[k] ??= buildClusterMask(maskData, cl.comps, comps, entry.anyComp));
    opts.onPass?.(pi++, totalPasses);
    current = await runPass(current, clusterMask, cl.bbox, 3.0);
    if (!twoPass) continue;
    if (fineTiles === null) {
      opts.onPass?.(pi++, totalPasses);
      current = await runPass(current, clusterMask, cl.bbox, 2.0);
    } else {
      for (const t of fineTiles) {
        const tileMask = buildClusterMask(
          maskData,
          t.comps.map((j) => cl.comps[j]),
          comps,
          entry.anyComp,
        );
        opts.onPass?.(pi++, totalPasses);
        if (t.fx || t.fy) {
          // 拆分 cell：蒙版裁到 cell + 内部边界升坡，比例 alpha 粘贴与
          // 前一 cell 的填充线性 cross-fade（消灭接缝硬边界）
          const cellMask = buildCellMask(tileMask, t.bbox, t.fx, t.fy);
          current = await runPass(current, cellMask, t.bbox, 1.5, { proportionalPaste: true });
        } else {
          current = await runPass(current, tileMask, t.bbox, 1.5);
        }
      }
    }
  }
  const result = applyNegativeGainGuard(imageData, current, maskData);
  opts.onTiming?.({ planMs, runMs: performance.now() - tRun0 });
  return result;
}

/**
 * 每簇蒙版。A4 配套（2026-10-01）：不再强制 255——comp 像素保留蒙版原始
 * alpha（模型输入与粘贴的硬阈值都按 >64 二值化，语义不变），并在 comp 外
 * 2px 内收集 alpha≤64 的软边斜坡（mask-history 以 alpha≈140 绘制 + 双线性
 * 放大产生的插值带），让粘贴边界天然抗锯齿。anyComp 是全蒙版 comp 像素
 * 查找表：软边带只收"不属于任何 comp"的像素——细 tile 相邻时防止把邻
 * tile 的 comp 像素偷进本 tile（会被重复重绘）；alpha>64 但落在 <16px 被
 * 丢弃的微型 comp 里的像素维持现状不收（不扩大修复足迹）。
 */
function buildClusterMask(
  full: ImageData,
  compIdx: number[],
  comps: MaskComp[],
  anyComp: Uint8Array,
): ImageData {
  const W = full.width;
  const H = full.height;
  const out = new ImageData(W, H);
  for (const ci of compIdx) {
    for (const p of comps[ci].pixels) {
      out.data[p * 4 + 3] = full.data[p * 4 + 3];
    }
  }
  for (const ci of compIdx) {
    for (const p of comps[ci].pixels) {
      const x = p % W;
      const y = (p / W) | 0;
      const y0 = Math.max(0, y - 2);
      const y1 = Math.min(H - 1, y + 2);
      const x0 = Math.max(0, x - 2);
      const x1 = Math.min(W - 1, x + 2);
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          const q = ny * W + nx;
          if (anyComp[q]) continue;
          const v = full.data[q * 4 + 3];
          if (v > 0 && v <= 64) out.data[q * 4 + 3] = v;
        }
      }
    }
  }
  return out;
}

interface MaskComp {
  pixels: number[];
  bbox: { x: number; y: number; w: number; h: number };
}

/** Mask alpha threshold — unified with buildMIGANInput / pasteCropWithMask.
 *  >128 会把笔刷抗锯齿边缘（alpha 64–128 的 1px 环带）丢出连通域，
 *  簇蒙版随之丢失这层边缘 → 笔刷边缘可能残留 1px 水印，统一为 >64。 */
const MASK_ALPHA_MIN = 64;

/** Connected components of the binarized mask, ≥16px. */
function labelMaskComponents(mask: ImageData): MaskComp[] {
  const W = mask.width;
  const H = mask.height;
  const n = W * H;
  const bin = new Uint8Array(n);
  for (let i = 0; i < n; i++) bin[i] = mask.data[i * 4 + 3] > MASK_ALPHA_MIN ? 1 : 0;
  const label = new Int32Array(n).fill(-1);
  const comps: MaskComp[] = [];
  const stack: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!bin[i] || label[i] >= 0) continue;
    const cur = comps.length;
    const pixels: number[] = [];
    let minX = W, minY = H, maxX = -1, maxY = -1;
    stack.push(i);
    label[i] = cur;
    while (stack.length) {
      const p = stack.pop()!;
      pixels.push(p);
      const x = p % W;
      const y = (p / W) | 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (x > 0 && bin[p - 1] && label[p - 1] < 0) { label[p - 1] = cur; stack.push(p - 1); }
      if (x < W - 1 && bin[p + 1] && label[p + 1] < 0) { label[p + 1] = cur; stack.push(p + 1); }
      if (y > 0 && bin[p - W] && label[p - W] < 0) { label[p - W] = cur; stack.push(p - W); }
      if (y < H - 1 && bin[p + W] && label[p + W] < 0) { label[p + W] = cur; stack.push(p + W); }
    }
    if (pixels.length >= 16 && maxX >= 0) {
      comps.push({
        pixels,
        bbox: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
      });
    }
  }
  return comps;
}

/** 邻近合并间隙：bbox 各向外扩 MERGE_GAP 后相交的区域属同一片水印场。 */
const MERGE_GAP = 80;

/**
 * 超大 tile 网格拆分（2026-10-06 用户实测平铺水印图定位）：
 * partitionByCap 只合并、不拆分——相邻旋转检出框互相重叠时
 * labelMaskComponents 把它们连成单个巨大连通域（长边可达全图宽），
 * 它自己就是一个超过 cap 的 tile，其 crop 窗口（≤min(W,H)）装不下自己，
 * 外围蒙版像素永不重绘——即"只消除了中间的一个矩形框"。
 * 把超大 tile 按网格拆成 ≤cap 的 cell：mask 裁到 cell 范围（fx/fy 标记
 * 内部边界，粘贴时与前一个 cell 的填充交叉渐融），runPass 的 crop 以
 * cell 为中心，各 cell 合计覆盖整个连通域。
 */
interface FineTile {
  bbox: BBox;
  comps: number[];
  /** 内部边界（左侧/上侧邻 cell 存在）：粘贴 alpha 在此羽化交叉渐融 */
  fx?: boolean;
  fy?: boolean;
}

function splitOversizedTile(t: { bbox: BBox; comps: number[] }, cap: number): FineTile[] {
  if (Math.max(t.bbox.w, t.bbox.h) <= cap) return [t];
  const nx = Math.max(1, Math.ceil(t.bbox.w / cap));
  const ny = Math.max(1, Math.ceil(t.bbox.h / cap));
  const cw = t.bbox.w / nx;
  const ch = t.bbox.h / ny;
  const cells: FineTile[] = [];
  for (let gy = 0; gy < ny; gy++) {
    for (let gx = 0; gx < nx; gx++) {
      const cx = Math.round(t.bbox.x + gx * cw);
      const cy = Math.round(t.bbox.y + gy * ch);
      cells.push({
        bbox: {
          x: cx,
          y: cy,
          w: (gx === nx - 1 ? t.bbox.x + t.bbox.w : Math.round(t.bbox.x + (gx + 1) * cw)) - cx,
          h: (gy === ny - 1 ? t.bbox.y + t.bbox.h : Math.round(t.bbox.y + (gy + 1) * ch)) - cy,
        },
        comps: t.comps,
        fx: gx > 0,
        fy: gy > 0,
      });
    }
  }
  return cells;
}

/** cell 交叉渐融带宽（px）：相邻 cell 的填充在此带线性过渡。 */
const FINE_CELL_FEATHER = 40;

/**
 * cell 蒙版：tile 蒙版 ∩ cell bbox（右/下外扩 F+羽化余量），内部边界
 * （fx/fy）按带宽线性升坡。关键：右/下外扩让前一个 cell 以 alpha=1 覆盖
 * 后一 cell 的升坡带——alpha-over 粘贴得到纯线性 cross-fade（无原图掺入，
 * 升坡带之外的区域被后一 cell 核心完整覆盖）。blob 外边界（gx=0/gy=0 侧）
 * 不羽化，消除强度不衰减。
 *
 * 硬像素重标定到 255 标尺（2026-10-16，qa/results/002 case-6 实证）：
 * cell 走 proportionalPaste（alpha = m/255），而蒙版原生 alpha≈140
 * （MASK_PAINT）——不重标定时 cell 核心只贴 55%，底层 45% 渗出。cell 链
 * 上底层通常是前一 cell/粗遍的填充所以不可见，但粗遍 crop 被图像边界钳制
 * 盖不到的地带（case-6 文字带右端 x≥945，crop 右缘 136+809=945）底层是
 * 原图——残留 45% 原始锐度的笔画 + crop 边缘矩形接缝。重标定后核心
 * alpha=255 → 满强度粘贴；升坡带 255·f 线性，模型输入 >64 二值化口径不变
 * （f>0.25 入洞，此前 f>0.46——升坡带是邻 cell 已填充区，无害）。
 */
function buildCellMask(tileMask: ImageData, bbox: BBox, fx?: boolean, fy?: boolean): ImageData {
  const W = tileMask.width;
  const H = tileMask.height;
  const out = new ImageData(W, H);
  const F = FINE_CELL_FEATHER;
  const pad = F + PASTE_FEATHER_PX + 2;
  const x0 = Math.max(0, bbox.x);
  const y0 = Math.max(0, bbox.y);
  const x1 = Math.min(W, bbox.x + bbox.w + pad);
  const y1 = Math.min(H, bbox.y + bbox.h + pad);
  for (let y = y0; y < y1; y++) {
    const ry = fy ? Math.min(1, (y - bbox.y + 1) / F) : 1;
    for (let x = x0; x < x1; x++) {
      const a = tileMask.data[(y * W + x) * 4 + 3];
      if (!a) continue;
      const rx = fx ? Math.min(1, (x - bbox.x + 1) / F) : 1;
      const f = rx * ry;
      const hard = a > MASK_ALPHA_MIN;
      if (f >= 1) out.data[(y * W + x) * 4 + 3] = hard ? 255 : a;
      else if (f > 0) out.data[(y * W + x) * 4 + 3] = hard ? (255 * f) | 0 : (a * f) | 0;
    }
  }
  return out;
}

function inflatedIntersect(a: BBox, b: BBox, gap: number): boolean {
  return (
    a.x - gap < b.x + b.w &&
    a.x + a.w + gap > b.x &&
    a.y - gap < b.y + b.h &&
    a.y + a.h + gap > b.y
  );
}

/** 细修 tile 划分：并集 bbox 长边 ≤ cap 才合并（Round 11 用）。 */
function partitionByCap(boxes: BBox[], cap: number): { bbox: BBox; comps: number[] }[] {
  const tiles = boxes.map((b, i) => ({ bbox: { ...b }, comps: [i] }));
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < tiles.length; i++) {
      for (let j = i + 1; j < tiles.length; j++) {
        const a = tiles[i].bbox;
        const b = tiles[j].bbox;
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        const u = {
          x,
          y,
          w: Math.max(a.x + a.w, b.x + b.w) - x,
          h: Math.max(a.y + a.h, b.y + b.h) - y,
        };
        if (Math.max(u.w, u.h) <= cap) {
          tiles[i] = { bbox: u, comps: [...tiles[i].comps, ...tiles[j].comps] };
          tiles.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return tiles;
}

/**
 * 邻近分组（不设尺寸上限）：密集框场合（如成片误检）合成一组走一遍
 * 传统 coarse-to-fine —— 单次一致的幻觉远好于多次独立幻觉的拼接；
 * 真正分离的水印（双角标/平铺实例）才拆组，保证每组都能被裁剪覆盖。
 */
function clusterComponents(boxes: BBox[]): { bbox: BBox; comps: number[] }[] {
  const clusters = boxes.map((b, i) => ({ bbox: { ...b }, comps: [i] }));
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        if (inflatedIntersect(clusters[i].bbox, clusters[j].bbox, MERGE_GAP)) {
          const a = clusters[i].bbox;
          const b = clusters[j].bbox;
          const x = Math.min(a.x, b.x);
          const y = Math.min(a.y, b.y);
          clusters[i].bbox = {
            x,
            y,
            w: Math.max(a.x + a.w, b.x + b.w) - x,
            h: Math.max(a.y + a.h, b.y + b.h) - y,
          };
          clusters[i].comps.push(...clusters[j].comps);
          clusters.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return clusters;
}

/**
 * Round C (2026-09-25 验收)：负收益自检门控。
 * 背景：tiled-faint / tiled-grid 两例淡水印修复后区域 PSNR 46→25 dB——
 * MI-GAN 的幻觉重绘对"几乎看不见的水印"是净伤害。
 * 机制：逐连通蒙版区域，以原始图环带（ring，外扩 10px）均值作为
 * "这片区域本该长什么样"的参照；若修复后区域对环带的偏离反而超过
 * 修复前 ×1.15（+1/255 滞回），说明重绘是负收益，回退该区域为原图。
 * 强水印的 devBefore 远大于 devAfter，数学上不可能触发，因此天然安全；
 * 纹理背景两边同步抬高、比值 ≈1，也不会误回退。
 */
// 校准（2026-09-25 bench 全量插桩）：误触发样本 ratio=1.34（text-br-small
// 细字面框，纹理占蒙版大头），真触发样本 ratio=1.55–1.6（平铺淡水印），
// 阈值取 1.35 + eps 2 拉开两侧间距；R<2 说明该区域未被实际重绘，跳过。
const GUARD_RATIO = 1.35;
const GUARD_EPS = 2.0; // /255
const GUARD_RING = 10;

function applyNegativeGainGuard(orig: ImageData, result: ImageData, mask: ImageData): ImageData {
  const W = orig.width;
  const H = orig.height;
  const comps = labelMaskComponents(mask).map((c) => c.pixels);
  if (comps.length === 0) return result;
  const n = W * H;
  const bin = new Uint8Array(n);
  for (let i = 0; i < n; i++) bin[i] = mask.data[i * 4 + 3] > MASK_ALPHA_MIN ? 1 : 0;

  const out = new ImageData(new Uint8ClampedArray(result.data), W, H);
  const od = orig.data;
  const rd = result.data;
  let reverted = 0;

  for (const pixels of comps) {
    let minX = W, minY = H, maxX = -1, maxY = -1;
    for (const p of pixels) {
      const x = p % W;
      const y = (p / W) | 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    // 环带均值（原始图）：外扩 GUARD_RING 的框内、非本区域像素
    const rx0 = Math.max(0, minX - GUARD_RING);
    const ry0 = Math.max(0, minY - GUARD_RING);
    const rx1 = Math.min(W - 1, maxX + GUARD_RING);
    const ry1 = Math.min(H - 1, maxY + GUARD_RING);
    let rSum = 0, gSum = 0, bSum = 0, ringN = 0;
    for (let y = ry0; y <= ry1; y += 2) {
      for (let x = rx0; x <= rx1; x += 2) {
        const i = y * W + x;
        if (bin[i]) continue;
        rSum += od[i * 4]; gSum += od[i * 4 + 1]; bSum += od[i * 4 + 2];
        ringN++;
      }
    }
    if (ringN < 8) continue;
    const rM = rSum / ringN, gM = gSum / ringN, bM = bSum / ringN;

    let devBefore = 0, devAfter = 0, repaintMag = 0;
    for (const p of pixels) {
      const j = p * 4;
      devBefore +=
        (Math.abs(od[j] - rM) + Math.abs(od[j + 1] - gM) + Math.abs(od[j + 2] - bM)) / 3;
      devAfter +=
        (Math.abs(rd[j] - rM) + Math.abs(rd[j + 1] - gM) + Math.abs(rd[j + 2] - bM)) / 3;
      repaintMag +=
        (Math.abs(rd[j] - od[j]) + Math.abs(rd[j + 1] - od[j + 1]) + Math.abs(rd[j + 2] - od[j + 2])) / 3;
    }
    devBefore /= pixels.length;
    devAfter /= pixels.length;
    repaintMag /= pixels.length;

    const rule1 = devAfter > devBefore * GUARD_RATIO + GUARD_EPS;
    const rule2 = repaintMag > 2 * devBefore + 6;
    console.debug(
      `[guard] n=${pixels.length} T=${devBefore.toFixed(1)} devAfter=${devAfter.toFixed(1)} R=${repaintMag.toFixed(1)} ratio=${(devAfter / Math.max(devBefore, 0.01)).toFixed(2)}`,
    );
    if (repaintMag < 2) continue; // 未被重绘的区域无回退意义
    if (devBefore >= 15) continue; // 强水印/重纹理区不回退（字形边缘 vs 纹理不可分）
    if (rule1 || rule2) {
      for (const p of pixels) {
        const j = p * 4;
        out.data[j] = od[j];
        out.data[j + 1] = od[j + 1];
        out.data[j + 2] = od[j + 2];
      }
      reverted++;
    }
  }
  if (reverted > 0) {
    console.debug(`[inpaint] negative-gain guard: reverted ${reverted}/${comps.length} regions`);
  }
  return out;
}

interface RunPassOpts {
  /** 粘贴 alpha 全程比例（m/255）：cell 交叉渐融用，避免 >64 硬跳变 */
  proportionalPaste?: boolean;
}

async function runPass(
  imageData: ImageData,
  maskData: ImageData,
  bbox: { x: number; y: number; w: number; h: number },
  context: number,
  passOpts?: RunPassOpts,
): Promise<ImageData> {
  const { width: W, height: H } = imageData;

  const maskDim = Math.max(bbox.w, bbox.h);
  const needed = Math.round(maskDim * context);
  const cropSize = Math.min(Math.max(MODEL_SIZE, needed), W, H);

  const centerX = bbox.x + bbox.w / 2;
  const centerY = bbox.y + bbox.h / 2;
  let cropX = Math.round(centerX - cropSize / 2);
  let cropY = Math.round(centerY - cropSize / 2);
  cropX = Math.max(0, Math.min(W - cropSize, cropX));
  cropY = Math.max(0, Math.min(H - cropSize, cropY));

  const cropImage = cropImageData(imageData, cropX, cropY, cropSize, cropSize);
  const cropMask = cropImageData(maskData, cropX, cropY, cropSize, cropSize);
  const inferImage = resizeImageData(cropImage, MODEL_SIZE, MODEL_SIZE);
  const inferMask = resizeImageData(cropMask, MODEL_SIZE, MODEL_SIZE);

  const ort = await getOrt();
  const inputName = session!.inputNames[0];
  const feeds: Record<string, ort.Tensor> = {};
  feeds[inputName] = new ort.Tensor(
    'float32',
    buildMIGANInput(inferImage, inferMask),
    [1, 4, MODEL_SIZE, MODEL_SIZE],
  );

  const output = await runWithFallback('migan', feeds);
  const outTensor = output[session!.outputNames[0]];
  const outW = (outTensor.dims[3] as number) || MODEL_SIZE;
  const outH = (outTensor.dims[2] as number) || MODEL_SIZE;

  const inferResult = new ImageData(
    miganOutputToRGBA(outTensor.data as Float32Array, outW, outH),
    outW,
    outH,
  );
  const cropResult =
    outW === cropSize && outH === cropSize
      ? inferResult
      : resizeImageData(inferResult, cropSize, cropSize);

  return pasteCropWithMask(imageData, cropResult, cropMask, cropX, cropY, {
    proportional: passOpts?.proportionalPaste,
    featherPx: PASTE_FEATHER_PX,
  });
}

/**
 * 蒙版 ∩ 矩形（LaMa 网格 tile 用）：矩形外像素清零；矩形内保留原始 alpha
 * （A4：含簇蒙版收集的 ≤64 软边斜坡——模型输入按 >64 二值化，语义不变）。
 */
function clipMaskToRect(
  mask: ImageData,
  rect: { x: number; y: number; w: number; h: number },
): ImageData {
  const out = new ImageData(mask.width, mask.height);
  const x1 = Math.min(mask.width, rect.x + rect.w);
  const y1 = Math.min(mask.height, rect.y + rect.h);
  for (let y = Math.max(0, rect.y); y < y1; y++) {
    for (let x = Math.max(0, rect.x); x < x1; x++) {
      const i = (y * mask.width + x) * 4;
      const v = mask.data[i + 3];
      if (v > 0) out.data[i + 3] = v;
    }
  }
  return out;
}

/**
 * LaMa 单遍：与 runPass 同构（裁剪/缩放/羽化粘贴），但张量约定不同——
 * 输入 image [1,3,S,S] 0-1 + mask [1,1,S,S] 1=待修，输出 CHW float 0-255。
 */
async function lamaRunPass(
  imageData: ImageData,
  maskData: ImageData,
  bbox: { x: number; y: number; w: number; h: number },
  context: number,
): Promise<ImageData> {
  const { width: W, height: H } = imageData;
  const maskDim = Math.max(bbox.w, bbox.h);
  const needed = Math.round(maskDim * context);
  const cropSize = Math.min(Math.max(MODEL_SIZE, needed), W, H);

  const centerX = bbox.x + bbox.w / 2;
  const centerY = bbox.y + bbox.h / 2;
  let cropX = Math.round(centerX - cropSize / 2);
  let cropY = Math.round(centerY - cropSize / 2);
  cropX = Math.max(0, Math.min(W - cropSize, cropX));
  cropY = Math.max(0, Math.min(H - cropSize, cropY));

  const cropImage = cropImageData(imageData, cropX, cropY, cropSize, cropSize);
  const cropMask = cropImageData(maskData, cropX, cropY, cropSize, cropSize);
  const inferImage = resizeImageData(cropImage, MODEL_SIZE, MODEL_SIZE);
  const inferMask = resizeImageData(cropMask, MODEL_SIZE, MODEL_SIZE);

  const S = MODEL_SIZE;
  const size2 = S * S;
  const imgT = new Float32Array(3 * size2);
  const mskT = new Float32Array(size2);
  for (let i = 0; i < size2; i++) {
    imgT[i] = inferImage.data[i * 4] / 255;
    imgT[i + size2] = inferImage.data[i * 4 + 1] / 255;
    imgT[i + 2 * size2] = inferImage.data[i * 4 + 2] / 255;
    mskT[i] = inferMask.data[i * 4 + 3] > MASK_ALPHA_MIN ? 1 : 0;
  }

  const ort = await getOrt();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[lamaSession!.inputNames[0]] = new ort.Tensor('float32', imgT, [1, 3, S, S]);
  feeds[lamaSession!.inputNames[1]] = new ort.Tensor('float32', mskT, [1, 1, S, S]);
  const output = await runWithFallback('lama', feeds);
  const od = output[lamaSession!.outputNames[0]].data as Float32Array;

  const out = new Uint8ClampedArray(size2 * 4);
  for (let i = 0; i < size2; i++) {
    out[i * 4] = Math.max(0, Math.min(255, od[i]));
    out[i * 4 + 1] = Math.max(0, Math.min(255, od[i + size2]));
    out[i * 4 + 2] = Math.max(0, Math.min(255, od[i + 2 * size2]));
    out[i * 4 + 3] = 255;
  }
  const inferResult = new ImageData(out, S, S);
  const cropResult = resizeImageData(inferResult, cropSize, cropSize);
  return pasteCropWithMask(imageData, cropResult, cropMask, cropX, cropY, {
    featherPx: PASTE_FEATHER_PX,
  });
}

/**
 * MI-GAN's exact expected 4-channel input (matches Picsart's preprocess):
 *   plane 0: mask_norm - 0.5   (1 = keep, 0 = inpaint)
 *   planes 1-3: RGB normalized to [-1,1], zeroed inside the hole
 */
export interface MaskStats {
  maskFrac: number;
  maxCompDim: number;
  maxCompAspect: number;
  comps: number;
}

/** 蒙版统计（引擎自动选档用，inpaint 前调用）：最大连通域长边/长宽比
 *  判定"大空洞型"蒙版——LaMa 的 FFC 全局感受野对它显著更强（bench
 *  text-center +11.6dB），碎片场/长条带则 MI-GAN 同画质快 10×。 */
export function maskStats(mask: ImageData): MaskStats {
  const comps = labelMaskComponents(mask);
  let maskPx = 0;
  let maxDim = 0;
  let maxAspect = 1;
  for (const c of comps) {
    maskPx += c.pixels.length;
    const dim = Math.max(c.bbox.w, c.bbox.h);
    const short = Math.max(1, Math.min(c.bbox.w, c.bbox.h));
    const aspect = dim / short;
    if (dim > maxDim) {
      maxDim = dim;
      maxAspect = aspect;
    }
  }
  const n = mask.width * mask.height;
  return {
    maskFrac: n ? +(maskPx / n).toFixed(4) : 0,
    maxCompDim: maxDim,
    maxCompAspect: +maxAspect.toFixed(2),
    comps: comps.length,
  };
}

export function buildMIGANInput(imageData: ImageData, maskData: ImageData): Float32Array {
  const { width, height, data: img } = imageData;
  const maskBytes = maskData.data;
  const size = width * height;
  const out = new Float32Array(4 * size);
  for (let i = 0; i < size; i++) {
    const isInpaint = maskBytes[i * 4 + 3] > 64;
    const mask = isInpaint ? 0.0 : 1.0;
    const r = (img[i * 4] / 255) * 2 - 1;
    const g = (img[i * 4 + 1] / 255) * 2 - 1;
    const b = (img[i * 4 + 2] / 255) * 2 - 1;
    out[i] = mask - 0.5;
    out[i + size] = r * mask;
    out[i + size * 2] = g * mask;
    out[i + size * 3] = b * mask;
  }
  return out;
}

export function miganOutputToRGBA(
  chw: Float32Array,
  w: number,
  h: number,
) {
  const size = w * h;
  const out = new Uint8ClampedArray(size * 4);
  for (let i = 0; i < size; i++) {
    out[i * 4] = Math.max(0, Math.min(1, chw[i] * 0.5 + 0.5)) * 255;
    out[i * 4 + 1] = Math.max(0, Math.min(1, chw[i + size] * 0.5 + 0.5)) * 255;
    out[i * 4 + 2] = Math.max(0, Math.min(1, chw[i + size * 2] * 0.5 + 0.5)) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function findMaskBbox(
  maskData: ImageData,
): { x: number; y: number; w: number; h: number } | null {
  const { width, height, data } = maskData;
  let minX = width,
    minY = height,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4 + 3] > 16) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

/* ------------------------------------------------------------------ */
/* 杠杆 G/J（时长优化第 2 步）：运动自适应跳帧 + EMA 时序稳定的共用helper */
/* 2026-10-01 锚定重构：跳帧判据/复用锚定最近一次真推理帧（anchorRaw/Out/  */
/* Small），误差被阈值直接锁死；prev 组只服务 EMA 的相邻帧语义。           */
/* ------------------------------------------------------------------ */

export interface RegionGray {
  data: Float32Array;
  w: number;
  h: number;
}

/** 蒙版 bbox 区域降采样灰度（跳帧运动判据用），宽 ≤ targetW。 */
export function makeRegionGray(img: ImageData, bbox: BBox, targetW = 64): RegionGray {
  const scale = Math.min(1, targetW / Math.max(bbox.w, bbox.h));
  const w = Math.max(1, Math.round(bbox.w * scale));
  const h = Math.max(1, Math.round(bbox.h * scale));
  const src = new OffscreenCanvas(img.width, img.height);
  src.getContext('2d', READBACK)!.putImageData(img, 0, 0);
  const dst = new OffscreenCanvas(w, h);
  const ctx = dst.getContext('2d', READBACK)!;
  ctx.drawImage(src, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gray[i] = d[i * 4] * 0.299 + d[i * 4 + 1] * 0.587 + d[i * 4 + 2] * 0.114;
  }
  return { data: gray, w, h };
}

/** 两个同尺寸区域灰度的 MAE（0–255 尺度）。 */
export function regionMAE(a: RegionGray, b: RegionGray): number {
  if (a.w !== b.w || a.h !== b.h) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.data.length; i++) sum += Math.abs(a.data[i] - b.data[i]);
  return sum / a.data.length;
}

/**
 * 平移对齐 MAE 搜索（2026-10-17 运动补偿跳帧）：±maxShift 整数平移穷举，
 * 取重叠区 MAE 最小的 (dx,dy)。口径：curr[x+dx, y+dy] ≈ anchor[x, y]——
 * 即内容从 anchor → current 帧平移了 (+dx,+dy)；当前蒙版位置的内容在
 * anchor 帧位于蒙版 −(dx,dy)（compositePrevRegion 的 shift 参数同一口径）。
 * blockP90 = 对齐后重叠区切 8×8 网格逐块 MAE 的 P90 分位——全区域均值会被
 * 静态多数稀释（水印后走过的人/物），分位块差分才能检出局部运动。
 * 性能：65×65 网格 × 289 候选 ≈ 1.2M 次减法/帧，常数极小（跳帧帧本来就要省
 * 一次推理，~136ms）。
 */
export function bestShiftMAE(
  curr: RegionGray,
  anchor: RegionGray,
  maxShift = 8,
): { dx: number; dy: number; mae: number; blockP90: number } {
  const W = Math.min(curr.w, anchor.w);
  const H = Math.min(curr.h, anchor.h);
  let bestDx = 0;
  let bestDy = 0;
  let bestMae = Infinity;
  for (let dy = -maxShift; dy <= maxShift; dy++) {
    for (let dx = -maxShift; dx <= maxShift; dx++) {
      const x0c = Math.max(0, dx);
      const y0c = Math.max(0, dy);
      const x0a = Math.max(0, -dx);
      const y0a = Math.max(0, -dy);
      const w = W - Math.abs(dx);
      const h = H - Math.abs(dy);
      let sum = 0;
      for (let y = 0; y < h; y++) {
        const rc = (y0c + y) * curr.w + x0c;
        const ra = (y0a + y) * anchor.w + x0a;
        for (let x = 0; x < w; x++) sum += Math.abs(curr.data[rc + x] - anchor.data[ra + x]);
      }
      const mae = sum / (w * h);
      if (mae < bestMae) {
        bestMae = mae;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }
  // 对齐后重叠区 8×8 网格逐块 MAE 的 P90
  const x0c = Math.max(0, bestDx);
  const y0c = Math.max(0, bestDy);
  const x0a = Math.max(0, -bestDx);
  const y0a = Math.max(0, -bestDy);
  const w = W - Math.abs(bestDx);
  const h = H - Math.abs(bestDy);
  const NB = 8;
  const blocks: number[] = [];
  for (let by = 0; by < NB; by++) {
    for (let bx = 0; bx < NB; bx++) {
      const xs = Math.floor((bx * w) / NB);
      const xe = Math.floor(((bx + 1) * w) / NB);
      const ys = Math.floor((by * h) / NB);
      const ye = Math.floor(((by + 1) * h) / NB);
      let s = 0;
      let n = 0;
      for (let y = ys; y < ye; y++) {
        const rc = (y0c + y) * curr.w + x0c;
        const ra = (y0a + y) * anchor.w + x0a;
        for (let x = xs; x < xe; x++) {
          s += Math.abs(curr.data[rc + x] - anchor.data[ra + x]);
          n++;
        }
      }
      if (n > 0) blocks.push(s / n);
    }
  }
  blocks.sort((a, b) => a - b);
  const p90 = blocks.length > 0 ? blocks[Math.min(blocks.length - 1, Math.floor(blocks.length * 0.9))] : Infinity;
  return { dx: bestDx, dy: bestDy, mae: bestMae, blockP90: p90 };
}

/**
 * 区域亮度漂移估计：bbox 内非蒙版像素的 (currRaw − prevRaw) 逐通道均值。
 * 跳帧/EMA 复用历史帧前用它补偿全局漂移（渐变背景、曝光渐变），
 * 否则链式复用会把蒙版区冻结成可见矩形（e2e PSNR 38.7→16.7 的教训）。
 */
export function estimateRegionDelta(
  currRaw: ImageData,
  prevRaw: ImageData,
  maskData: ImageData,
  bbox: BBox,
): [number, number, number] {
  const W = currRaw.width;
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = bbox.y; y < bbox.y + bbox.h; y += 2) {
    for (let x = bbox.x; x < bbox.x + bbox.w; x += 2) {
      const i = (y * W + x) * 4;
      if (maskData.data[i + 3] > MASK_ALPHA_MIN) continue;
      r += currRaw.data[i] - prevRaw.data[i];
      g += currRaw.data[i + 1] - prevRaw.data[i + 1];
      b += currRaw.data[i + 2] - prevRaw.data[i + 2];
      n++;
    }
  }
  return n > 0 ? [r / n, g / n, b / n] : [0, 0, 0];
}

/**
 * 跳帧帧合成：把 prevOut 的蒙版羽化区域贴到当前帧。
 * 与 runPass 的粘贴完全同款语义（pasteCropWithMask：蒙版内 100% 替换 +
 * 羽化带落在蒙版外侧干净像素上），不硬贴矩形。delta 为漂移补偿
 * （estimateRegionDelta），静止场景 ≈0 不影响行为。
 * shift（2026-10-17 运动补偿跳帧）：内容从 anchor → current 平移了 (dx,dy)
 * （bestShiftMAE 口径）时，当前蒙版位置的内容在 prevOut 中位于 bbox−(dx,dy)，
 * 从该处取源块保证补丁对齐。源越界时把 shift 收缩到刚好贴边（保方向），
 * 不会出现部分粘贴；delta 均值补偿照常应用。
 */
export function compositePrevRegion(
  curr: ImageData,
  prevOut: ImageData,
  maskData: ImageData,
  bbox: BBox,
  delta: readonly [number, number, number] = [0, 0, 0],
  shift: { dx: number; dy: number } = { dx: 0, dy: 0 },
): ImageData {
  const { width: W, height: H } = prevOut;
  // 源原点 = bbox − shift；收缩到帧内 [0, W−bbox.w] × [0, H−bbox.h]（保方向）
  const sx = Math.min(Math.max(bbox.x - shift.dx, 0), Math.max(0, W - bbox.w));
  const sy = Math.min(Math.max(bbox.y - shift.dy, 0), Math.max(0, H - bbox.h));
  const cropPrev = cropImageData(prevOut, sx, sy, bbox.w, bbox.h);
  if (delta[0] || delta[1] || delta[2]) {
    const d = cropPrev.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] += delta[0];
      d[i + 1] += delta[1];
      d[i + 2] += delta[2];
    }
  }
  const cropMask = cropImageData(maskData, bbox.x, bbox.y, bbox.w, bbox.h);
  return pasteCropWithMask(curr, cropPrev, cropMask, bbox.x, bbox.y);
}

/** 蒙版 alpha>64 区域膨胀 radius px 的二值图（全尺寸 Uint8Array 0/1，逐蒙版算一次即可）。 */
export function dilateMaskAlpha(mask: ImageData, radius: number): Uint8Array {
  const { width: W, height: H, data } = mask;
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (data[(y * W + x) * 4 + 3] <= MASK_ALPHA_MIN) continue;
      const x1 = Math.min(W - 1, x + radius);
      const y1 = Math.min(H - 1, y + radius);
      for (let dy = Math.max(0, y - radius); dy <= y1; dy++) {
        for (let dx = Math.max(0, x - radius); dx <= x1; dx++) {
          out[dy * W + dx] = 1;
        }
      }
    }
  }
  return out;
}

/**
 * EMA 时序稳定（蒙版膨胀区域内）：out = α·curr + (1-α)·prev。
 * α 由该像素运动强度（当前帧原图 vs 上一帧原图）平滑映射，不硬切换：
 * 静止 → α≈0.4（偏历史，抑制逐帧幻觉闪烁）；运动大 → α→1.0（纯当前帧，不拖影）。
 * 跳帧帧的 curr 是合成帧，EMA 照常跑——它本来就是时序滤波。
 */
export function applyTemporalEMA(
  curr: ImageData,
  prev: ImageData,
  currRaw: ImageData,
  prevRaw: ImageData,
  dilated: Uint8Array,
  delta: readonly [number, number, number] = [0, 0, 0],
): ImageData {
  const { width: W, height: H } = curr;
  const out = new ImageData(new Uint8ClampedArray(curr.data), W, H);
  const cd = curr.data, pd = prev.data, cr = currRaw.data, pr = prevRaw.data, od = out.data;
  for (let i = 0; i < W * H; i++) {
    if (!dilated[i]) continue;
    const j = i * 4;
    const m =
      (Math.abs(cr[j] - pr[j]) + Math.abs(cr[j + 1] - pr[j + 1]) + Math.abs(cr[j + 2] - pr[j + 2])) /
      (3 * 255);
    // 运动 0→8% 映射 α 0.4→1.0，smoothstep 平滑过渡
    const t = Math.min(1, m / 0.08);
    const s = t * t * (3 - 2 * t);
    const a = 0.4 + 0.6 * s;
    // 历史项先加漂移补偿再混合：渐变背景下 EMA 不再累积滞后
    od[j] = cd[j] * a + (pd[j] + delta[0]) * (1 - a);
    od[j + 1] = cd[j + 1] * a + (pd[j + 1] + delta[1]) * (1 - a);
    od[j + 2] = cd[j + 2] * a + (pd[j + 2] + delta[2]) * (1 - a);
  }
  return out;
}

/**
 * Paste the inpainted crop back. A4（消除质量优化方案 P1，2026-10-01）：
 * alpha = max(蒙版自带灰阶, 硬阈值)。蒙版内（alpha>64）永远 100% 用修复像素
 * ——纯 blur 曾让细笔画中心 alpha 只有 ~0.5、半量水印渗回（"消除后仍有重
 * 痕迹"的根因，2026-09-27 修掉），硬阈值保底不动。边界不再"二值化→再造
 * 模糊"，直接用蒙版自带的过渡灰阶（mask-history 以 alpha≈140 绘制 + 双线性
 * 放大，天然抗锯齿；簇蒙版按 comp 原始 alpha 传递，见 buildClusterMask）。
 * 旧实现在蒙版外侧再造 6–22px blur 羽化带，把模型对整幅裁剪图的重建误差
 * 混进干净像素（P0 体检 ringDelta 5.63 的来源）——删除后洞外不再被触碰；
 * 代价是边界台阶暴露，交给 A2（色彩对齐）/A3（梯度域）接手。
 */
/** 粘贴内向羽化（px）：填充与原图交界处 alpha 线性降坡，软化蒙版边界的
 *  矩形痕迹（检出框是硬边矩形，A4 软坡只对手绘笔刷存在）。模型输入蒙版
 *  不羽化（重生成面积不变），只羽化粘贴 alpha——框内水印仍被完整重生成，
 *  交界处与原图线性过渡，不产生水印鬼影（框外扩余量 > 羽化带宽）。
 *  紧凑度门控：羽化只对"大块"蒙版（面积/周长 ≥12，约等效半径 ≥24px 的
 *  团块）生效——字形/笔画蒙版紧贴笔画，羽化会把水印边缘掺回原图
 *  （bench text-br-small 19.7→11.4dB 的教训），必须排除。 */
const PASTE_FEATHER_PX = 8;
const PASTE_FEATHER_MIN_COMPACTNESS = 12;

function pasteCropWithMask(
  original: ImageData,
  inpaintedCrop: ImageData,
  cropMask: ImageData,
  offsetX: number,
  offsetY: number,
  opts?: { proportional?: boolean; featherPx?: number },
): ImageData {
  const W = original.width,
    H = original.height;
  const cw = inpaintedCrop.width,
    ch = inpaintedCrop.height;

  let ramp: Float32Array | null = null;
  if (opts?.featherPx && opts.featherPx > 0) {
    // 紧凑度门控（按连通域）：面积/边界长（4 邻接）≥12 的团块才羽化。
    // 字形笔画比值 2-5；密集小字场的字形像素互连成大团块——全局判定会
    // 误羽化（bench text-br-small 19.7→11.4dB 的教训），必须逐连通域
    const compact = compactHardMask(cropMask, cw, ch);
    let any = false;
    for (let i = 0; i < compact.length; i++)
      if (compact[i]) {
        any = true;
        break;
      }
    if (any) ramp = inwardFeatherRamp(compact, cw, ch, opts.featherPx);
  }

  const out = new Uint8ClampedArray(original.data);
  for (let y = 0; y < ch; y++) {
    const oy = offsetY + y;
    if (oy < 0 || oy >= H) continue;
    for (let x = 0; x < cw; x++) {
      const ox = offsetX + x;
      if (ox < 0 || ox >= W) continue;
      const i = (y * cw + x) * 4;
      const m = cropMask.data[i + 3];
      if (!m) continue;
      let a = opts?.proportional ? m / 255 : m > 64 ? 1 : m / 255;
      // 羽化只压硬蒙版像素；≤64 的自带软坡（笔刷/簇蒙版收集）保持原语义
      if (ramp && m > 64) a = Math.min(a, ramp[y * cw + x]);
      if (a <= 0) continue;
      const dst = (oy * W + ox) * 4;
      out[dst] = inpaintedCrop.data[i] * a + out[dst] * (1 - a);
      out[dst + 1] = inpaintedCrop.data[i + 1] * a + out[dst + 1] * (1 - a);
      out[dst + 2] = inpaintedCrop.data[i + 2] * a + out[dst + 2] * (1 - a);
      out[dst + 3] = 255;
    }
  }
  return new ImageData(out, W, H);
}

/** 硬蒙版（alpha>64）按连通域的紧凑度过滤：返回仅含"团块"连通域
 *  （面积/边界长 ≥ PASTE_FEATHER_MIN_COMPACTNESS）的二值掩码。
 *  追加实心率门控（003 人像图竖排文字实证）：合并字形柱（字母经膨胀/生长
 *  桥接成带字腔与间隙的柱块，面积/边界可达 16.7）会骗过紧凑度门控——
 *  羽化把柱内笔画按 alpha<1 渗回原图（"字形外框残留"根因）。矩形选区
 *  （roundRect 填充）实心率 ≥97%，字形柱 ≤90%（本案 83.8%），以 0.92 分治。 */
function compactHardMask(cropMask: ImageData, cw: number, ch: number): Uint8Array {
  const n = cw * ch;
  const hard = new Uint8Array(n);
  for (let i = 0; i < n; i++) hard[i] = cropMask.data[i * 4 + 3] > 64 ? 1 : 0;
  const comp = new Int32Array(n).fill(-1);
  const out = new Uint8Array(n);
  const stack: number[] = [];
  const isHard = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < cw && y < ch && hard[y * cw + x] === 1;
  for (let s = 0; s < n; s++) {
    if (!hard[s] || comp[s] >= 0) continue;
    const id = s;
    let area = 0;
    let boundary = 0;
    let minX = cw, minY = ch, maxX = -1, maxY = -1;
    const pixels: number[] = [];
    stack.push(s);
    comp[s] = id;
    while (stack.length) {
      const p = stack.pop()!;
      pixels.push(p);
      area++;
      const x = p % cw;
      const y = (p / cw) | 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (!isHard(x - 1, y) || !isHard(x + 1, y) || !isHard(x, y - 1) || !isHard(x, y + 1)) {
        boundary++;
      }
      if (x > 0 && hard[p - 1] && comp[p - 1] < 0) {
        comp[p - 1] = id;
        stack.push(p - 1);
      }
      if (x < cw - 1 && hard[p + 1] && comp[p + 1] < 0) {
        comp[p + 1] = id;
        stack.push(p + 1);
      }
      if (y > 0 && hard[p - cw] && comp[p - cw] < 0) {
        comp[p - cw] = id;
        stack.push(p - cw);
      }
      if (y < ch - 1 && hard[p + cw] && comp[p + cw] < 0) {
        comp[p + cw] = id;
        stack.push(p + cw);
      }
    }
    const bboxArea = maxX >= 0 ? (maxX - minX + 1) * (maxY - minY + 1) : 0;
    const fillRatio = area / Math.max(1, bboxArea);
    if (area / Math.max(1, boundary) >= PASTE_FEATHER_MIN_COMPACTNESS && fillRatio >= 0.92) {
      for (const p of pixels) out[p] = 1;
    }
  }
  return out;
}

/** 3×3 腐蚀 survival 轮数 → 内向线性坡（0=蒙版边界，1=深入 radius 以上）。
 *  crop 边界外视为"存活"（蒙版延续到 crop 外不是真实填充边界，不得褪色——
 *  否则大簇全局精修在 crop 截断处会留 8px 原图残带）。 */
function inwardFeatherRamp(hard: Uint8Array, w: number, h: number, radius: number): Float32Array {
  const cur = Uint8Array.from(hard);
  const ramp = new Float32Array(w * h);
  const tmp = new Uint8Array(w * h);
  for (let round = 1; round <= radius; round++) {
    // 可分离 3×3 min（越界补 1=存活）：水平一趟 + 垂直一趟
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const v = cur[row + x];
        const lf = x > 0 ? cur[row + x - 1] : 1;
        const rt = x < w - 1 ? cur[row + x + 1] : 1;
        tmp[row + x] = v && lf && rt ? 1 : 0;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const v = tmp[i];
        const up = y > 0 ? tmp[i - w] : 1;
        const dn = y < h - 1 ? tmp[i + w] : 1;
        const nv = v && up && dn ? 1 : 0;
        cur[i] = nv;
        if (nv) ramp[i] = round / radius;
      }
    }
  }
  return ramp;
}

/**
 * 纯像素拷贝裁剪（调用方已保证裁剪框完全在图内，见 runPass 的 clamp）。
 * 旧实现每趟都重建全尺寸源 canvas + putImageData 整图，按簇×遍线性放大
 * CPU 开销（1200×800 图 N 簇 × 2 遍 = N×2 次全图拷贝）——行级 memcpy
 * 只拷贝目标区域，复杂度 O(裁剪面积)。
 */
function cropImageData(
  src: ImageData,
  x: number,
  y: number,
  w: number,
  h: number,
): ImageData {
  const out = new ImageData(w, h);
  const rowBytes = w * 4;
  for (let r = 0; r < h; r++) {
    const srcOff = ((y + r) * src.width + x) * 4;
    out.data.set(src.data.subarray(srcOff, srcOff + rowBytes), r * rowBytes);
  }
  return out;
}

function resizeImageData(imageData: ImageData, w: number, h: number): ImageData {
  const src = new OffscreenCanvas(imageData.width, imageData.height);
  src.getContext('2d', READBACK)!.putImageData(imageData, 0, 0);
  const dst = new OffscreenCanvas(w, h);
  const ctx = dst.getContext('2d', READBACK)!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}
