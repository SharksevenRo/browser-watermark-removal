import {
  configureCore,
  removeWatermark,
  toBlob,
  toImageData,
  type RemoveQuality,
  type RemoveWatermarkResult,
} from '@core';

const params = new URLSearchParams(location.search);
configureCore({
  // Demo-only model host; the package itself defaults to ./models.
  modelBase: 'https://static.clearpix.org',
  ortWasmPaths: 'https://registry.npmmirror.com/onnxruntime-web/1.30.0/files/dist/',
  cacheName: 'browser-watermark-removal-demo',
  // `?ep=wasm` is useful for headless benchmarks and for GPUs with broken
  // WebGPU drivers. The default chain stays WebGPU → WASM.
  executionProviders: params.get('ep') === 'wasm' ? ['wasm'] : undefined,
});

const CASES = [
  {
    id: 'text-br-large',
    label: 'Large white text',
    note: 'bottom-right · opaque',
  },
  {
    id: 'text-tr-faint',
    label: 'Faint white text',
    note: 'top-right · low contrast',
  },
  {
    id: 'halo-flat-text',
    label: 'Halo text on flat sky',
    note: 'center · soft halo',
  },
  {
    id: 'tiled-med',
    label: 'Repeating tile',
    note: 'diagonal · medium',
  },
] as const;

type CaseId = (typeof CASES)[number]['id'];

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el;
}

const canvas = $('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D;

const sample = $('sample') as HTMLSelectElement;
const quality = $('quality') as HTMLSelectElement;
const useYolo = $('useYolo') as HTMLInputElement;
const showMask = $('showMask') as HTMLInputElement;
const runBtn = $('run') as HTMLButtonElement;
const downloadBtn = $('download') as HTMLButtonElement;
const fileInput = $('file') as HTMLInputElement;
const drop = $('drop');
const statusEl = $('status');
const progressBar = $('progressBar');
const emptyEl = $('empty');
const compare = $('compare');
const compareLabels = $('compareLabels');
const slider = $('slider') as HTMLInputElement;
const stageTitle = $('stageTitle');
const stageMeta = $('stageMeta');
const benchBody = $('benchBody');
const caseGrid = $('caseGrid');

let currentImage: ImageData | null = null;
let currentResult: RemoveWatermarkResult | null = null;
let currentName = '';
let busy = false;

function setStatus(text: string, error = false): void {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', error);
}

function setProgress(frac: number): void {
  progressBar.style.width = `${Math.max(0, Math.min(100, frac * 100)).toFixed(1)}%`;
}

function setBusy(value: boolean): void {
  busy = value;
  runBtn.disabled = value;
  downloadBtn.disabled = value || !currentResult?.removed;
  sample.disabled = value;
  quality.disabled = value;
  useYolo.disabled = value;
}

function tempCanvas(image: ImageData): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = image.width;
  c.height = image.height;
  const cctx = c.getContext('2d');
  if (!cctx) throw new Error('Canvas 2D unavailable');
  cctx.putImageData(image, 0, 0);
  return c;
}

function render(): void {
  if (!currentImage) return;
  const W = currentImage.width;
  const H = currentImage.height;
  canvas.width = W;
  canvas.height = H;

  if (currentResult) {
    ctx.putImageData(currentResult.image, 0, 0);
    const pos = Number(slider.value) / 100;
    if (pos > 0) {
      const original = tempCanvas(currentImage);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W * pos, H);
      ctx.clip();
      ctx.drawImage(original, 0, 0);
      ctx.restore();
    }
    if (showMask.checked) {
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = W;
      maskCanvas.height = H;
      const mctx = maskCanvas.getContext('2d');
      if (mctx) {
        mctx.putImageData(currentResult.mask, 0, 0);
        mctx.globalCompositeOperation = 'source-in';
        mctx.fillStyle = 'rgba(239, 68, 68, 0.55)';
        mctx.fillRect(0, 0, W, H);
        ctx.drawImage(maskCanvas, 0, 0);
      }
    }
    compare.hidden = false;
    compareLabels.hidden = false;
  } else {
    ctx.putImageData(currentImage, 0, 0);
    compare.hidden = true;
    compareLabels.hidden = true;
  }
  emptyEl.classList.add('hidden');
}

function updateTimings(result: RemoveWatermarkResult | null): void {
  const values = result
    ? [
        result.timings.detectMs,
        result.timings.maskMs,
        result.timings.initMs,
        result.timings.inpaintMs,
        result.timings.totalMs,
      ]
    : [null, null, null, null, null];
  const cells = document.querySelectorAll<HTMLElement>('#timings strong');
  values.forEach((v, i) => {
    if (cells[i]) cells[i].textContent = v == null ? '—' : `${Math.round(v)} ms`;
  });
}

function resetResult(): void {
  currentResult = null;
  slider.value = '50';
  updateTimings(null);
  downloadBtn.disabled = true;
  render();
}

async function loadCase(id: CaseId): Promise<void> {
  const c = CASES.find((x) => x.id === id) ?? CASES[0];
  setStatus(`Loading ${c.label.toLowerCase()}…`);
  setProgress(0.15);
  try {
    const image = await toImageData(`./cases/${c.id}/input.png`);
    currentImage = image;
    currentName = c.label;
    currentResult = null;
    stageTitle.textContent = c.id;
    stageMeta.textContent = `${image.width} × ${image.height}`;
    resetResult();
    setStatus('Ready. Click “Remove watermark”.');
    setProgress(0);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : 'Could not load the sample', true);
    setProgress(0);
  }
}

async function loadFile(file: File): Promise<void> {
  setStatus(`Loading ${file.name}…`);
  setProgress(0.15);
  try {
    const image = await toImageData(file);
    currentImage = image;
    currentName = file.name;
    currentResult = null;
    stageTitle.textContent = file.name;
    stageMeta.textContent = `${image.width} × ${image.height}`;
    resetResult();
    setStatus('Ready. Click “Remove watermark”.');
    setProgress(0);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : 'Could not read the image', true);
    setProgress(0);
  }
}

async function run(): Promise<void> {
  if (busy || !currentImage) return;
  setBusy(true);
  currentResult = null;
  resetResult();
  setStatus('Detecting watermarks…');
  setProgress(0.04);
  const q = quality.value as RemoveQuality;

  try {
    const result = await removeWatermark(currentImage, {
      quality: q,
      useYolo: useYolo.checked,
      onStage: (stage) => setStatus(stage),
      onProgress: (p) => {
        if (p.total > 0) setProgress(Math.min(0.94, p.done / p.total));
        else setStatus(p.stage);
      },
      onPass: (i, total) => {
        setStatus(`Inpainting pass ${i + 1}/${total}…`);
        setProgress(Math.min(0.98, 0.45 + (i / Math.max(1, total)) * 0.5));
      },
    });
    currentResult = result;
    updateTimings(result);
    render();
    setProgress(1);
    if (!result.removed) {
      setStatus('No watermark detected. Try another sample or the HQ preset.', true);
    } else {
      const backend = result.backend ? ` · ${result.backend}` : '';
      setStatus(
        `Done in ${Math.round(result.timings.totalMs)} ms · ${result.engine}${backend} · ${result.boxes.length} box(es)`,
      );
    }
    downloadBtn.disabled = !result.removed;
  } catch (e) {
    console.error(e);
    setStatus(e instanceof Error ? e.message : 'Removal failed', true);
    setProgress(0);
  } finally {
    setBusy(false);
    setProgress(currentResult?.removed ? 1 : 0);
  }
}

async function download(): Promise<void> {
  if (!currentResult) return;
  const blob = await toBlob(currentResult.image);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${currentName.replace(/\.[^.]+$/, '') || 'watermark-removed'}-clean.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderCaseGrid(): void {
  caseGrid.innerHTML = '';
  for (const c of CASES) {
    const article = document.createElement('article');
    article.className = 'case-card';
    article.innerHTML = `
      <header><h3>${c.id}</h3><span>${c.note}</span></header>
      <div class="case-pair">
        <figure><img loading="lazy" src="./cases/${c.id}/input.png" alt="${c.label} input" /><figcaption>Input</figcaption></figure>
        <figure><img loading="lazy" src="./cases/${c.id}/after.png" alt="${c.label} after removal" /><figcaption>Removed</figcaption></figure>
      </div>`;
    const after = article.querySelectorAll('img')[1] as HTMLImageElement;
    after.onerror = () => {
      after.onerror = null;
      after.src = `./cases/${c.id}/original.png`;
    };
    caseGrid.appendChild(article);
  }
}

interface BenchCase {
  id: string;
  label: string;
  engine: string;
  backend: string;
  detectMs: number;
  maskMs: number;
  initMs: number;
  inpaintMs: number;
  totalMs: number;
  boxes: number;
  maskFrac: number;
  psnrIn: number;
  psnrOut: number;
  psnrGain: number;
  psnrRegionIn: number;
  psnrRegionOut: number;
  psnrRegionGain: number;
  afterDataUrl: string;
}

interface BenchOutput {
  quality: string;
  generatedAt: string;
  userAgent: string;
  cases: BenchCase[];
}

function psnr(a: ImageData, b: ImageData, mask?: ImageData): number {
  if (a.width !== b.width || a.height !== b.height) return 0;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < a.width * a.height; i++) {
    if (mask && mask.data[i * 4 + 3] <= 16) continue;
    const j = i * 4;
    for (let c = 0; c < 3; c++) {
      const d = a.data[j + c] - b.data[j + c];
      sum += d * d;
      n++;
    }
  }
  if (n === 0) return 99;
  const mse = sum / n;
  return mse === 0 ? 99 : 10 * Math.log10((255 * 255) / mse);
}

function imageToDataUrl(image: ImageData): string {
  return tempCanvas(image).toDataURL('image/png');
}

async function runBenchmark(q: RemoveQuality = 'balanced'): Promise<BenchOutput> {
  const results: BenchCase[] = [];
  let warmed = false;
  for (const c of CASES) {
    const input = await toImageData(`./cases/${c.id}/input.png`);
    const original = await toImageData(`./cases/${c.id}/original.png`);
    if (!warmed) {
      // The first call pays for model download + session creation. Warm up on
      // the first case and then measure every case with a live session.
      await removeWatermark(input, { quality: q });
      warmed = true;
    }
    const result = await removeWatermark(input, { quality: q });
    const psnrIn = psnr(input, original);
    const psnrOut = psnr(result.image, original);
    const psnrRegionIn = psnr(input, original, result.mask);
    const psnrRegionOut = psnr(result.image, original, result.mask);
    results.push({
      id: c.id,
      label: c.label,
      engine: result.engine,
      backend: result.backend,
      detectMs: Math.round(result.timings.detectMs),
      maskMs: Math.round(result.timings.maskMs),
      initMs: Math.round(result.timings.initMs),
      inpaintMs: Math.round(result.timings.inpaintMs),
      totalMs: Math.round(result.timings.totalMs),
      boxes: result.boxes.length,
      maskFrac: result.stats.maskFrac,
      psnrIn: +psnrIn.toFixed(2),
      psnrOut: +psnrOut.toFixed(2),
      psnrGain: +(psnrOut - psnrIn).toFixed(2),
      psnrRegionIn: +psnrRegionIn.toFixed(2),
      psnrRegionOut: +psnrRegionOut.toFixed(2),
      psnrRegionGain: +(psnrRegionOut - psnrRegionIn).toFixed(2),
      afterDataUrl: imageToDataUrl(result.image),
    });
  }
  return {
    quality: q,
    generatedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    cases: results,
  };
}

async function loadBenchmark(): Promise<void> {
  try {
    const res = await fetch('./benchmark.json', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as BenchOutput;
    benchBody.innerHTML = '';
    for (const c of data.cases) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${c.id}</td>
        <td>${c.engine}${c.backend ? ` · ${c.backend}` : ''}</td>
        <td>${c.detectMs} ms</td>
        <td>${c.maskMs} ms</td>
        <td>${c.inpaintMs} ms</td>
        <td>${c.psnrRegionIn.toFixed(2)} dB</td>
        <td>${c.psnrRegionOut.toFixed(2)} dB</td>
        <td class="gain">${c.psnrRegionGain >= 0 ? '+' : ''}${c.psnrRegionGain.toFixed(2)} dB</td>`;
      benchBody.appendChild(tr);
    }
  } catch {
    /* no benchmark yet */
  }
}

declare global {
  interface Window {
    __bwr?: {
      ready: boolean;
      runBenchmark: (quality?: RemoveQuality) => Promise<BenchOutput>;
    };
  }
}

runBtn.addEventListener('click', () => void run());
downloadBtn.addEventListener('click', () => void download());
sample.addEventListener('change', () => void loadCase(sample.value as CaseId));
slider.addEventListener('input', render);
showMask.addEventListener('change', render);
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void loadFile(file);
});
drop.addEventListener('dragover', (e) => {
  e.preventDefault();
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (file) void loadFile(file);
});
useYolo.addEventListener('change', () => {
  if (useYolo.checked) setStatus('YOLO adds an 11 MB detector download on first use.');
  else setStatus('Ready.');
});

renderCaseGrid();
void loadBenchmark();
void loadCase('text-br-large');

window.__bwr = {
  ready: true,
  runBenchmark,
};
