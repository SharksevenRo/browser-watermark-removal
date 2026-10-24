/**
 * onnxruntime-web is heavy (~1 MB parsed) and only needed once a model runs,
 * so it is loaded with a dynamic import and stays out of the initial bundle.
 *
 * The runtime also needs its own wasm/asyncify binaries. On first use we race
 * a few origins with a 256 KB range probe and cache the winner for 24 hours.
 * A failed warm-up invalidates the hint, so a dead CDN cannot stick forever.
 */
import type * as ortNS from 'onnxruntime-web/webgpu';
import { getCoreConfig } from './config';

export type Ort = typeof import('onnxruntime-web/webgpu');
export type InferenceSession = ortNS.InferenceSession;

let ortPromise: Promise<Ort> | null = null;
let configured = false;

export function getOrt(): Promise<Ort> {
  return (ortPromise ??= import('onnxruntime-web/webgpu'));
}

/** iOS / iPadOS detection. iPadOS 13+ reports `MacIntel`, so touch points are
 * the only reliable desktop-macOS discriminator (desktop Macs have 0). */
export function isAppleMobile(): boolean {
  if (typeof navigator === 'undefined') return false;
  if (/iPhone|iPad|iPod/.test(navigator.userAgent)) return true;
  return navigator.platform === 'MacIntel' && (navigator.maxTouchPoints ?? 0) > 1;
}

const ORT_VERSION = '1.30.0';
const DEFAULT_ORIGINS = [
  `https://registry.npmmirror.com/onnxruntime-web/${ORT_VERSION}/files/dist/`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
];

/** Probe the actual file ORT downloads, not a tiny JS shim: on some mobile
 * carriers the .mjs is cached but the 25.5 MB asyncify.wasm is not reachable. */
const PROBE_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
const PROBE_BYTES = 256 * 1024;
const PROBE_BUDGET_MS = 2_500;
const HINT_TTL_MS = 24 * 3600_000;

function ortCandidates(): string[] {
  const cfg = getCoreConfig();
  const out = [...(cfg.ortOrigins || []), ...DEFAULT_ORIGINS];
  if (cfg.modelBase && /^https?:/i.test(cfg.modelBase)) {
    out.push(`${cfg.modelBase.replace(/\/+$/, '')}/ort/${ORT_VERSION}/`);
  }
  return [...new Set(out.map((o) => (o.endsWith('/') ? o : `${o}/`)))];
}

interface Probe {
  origin: string;
  ctrl: AbortController;
  done: Promise<{ origin: string; bps: number }>;
}

function startProbe(origin: string): Probe {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PROBE_BUDGET_MS);
  const done = (async () => {
    try {
      const t0 = performance.now();
      const res = await fetch(`${origin}${PROBE_FILE}`, {
        headers: { Range: `bytes=0-${PROBE_BYTES - 1}` },
        signal: ctrl.signal,
        cache: 'no-store',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let got = 0;
      if (res.body) {
        const reader = res.body.getReader();
        for (;;) {
          const { value, done: finished } = await reader.read();
          if (finished) break;
          got += value.byteLength;
          if (got >= PROBE_BYTES) break;
        }
        void reader.cancel().catch(() => {});
      } else {
        got = (await res.arrayBuffer()).byteLength;
      }
      if (got < PROBE_BYTES / 2) throw new Error(`short read ${got}B`);
      return { origin, bps: (got / Math.max(1, performance.now() - t0)) * 1000 };
    } finally {
      clearTimeout(timer);
    }
  })();
  return { origin, ctrl, done };
}

async function raceOrigin(
  candidates: string[],
): Promise<{ origin: string; bps: number } | null> {
  const probes = candidates.map(startProbe);
  try {
    return await Promise.any(probes.map((p) => p.done));
  } catch {
    return null;
  } finally {
    for (const p of probes) p.ctrl.abort();
  }
}

function hintKey(): string {
  return `bwr:ort-origin:v2:${ORT_VERSION}`;
}

function readHint(candidates: string[]): string | null {
  try {
    const raw = localStorage.getItem(hintKey());
    if (!raw) return null;
    const { origin, t } = JSON.parse(raw) as { origin?: unknown; t?: unknown };
    if (typeof origin !== 'string' || typeof t !== 'number') return null;
    if (Date.now() - t > HINT_TTL_MS) return null;
    return candidates.includes(origin) ? origin : null;
  } catch {
    return null;
  }
}

function writeHint(origin: string): void {
  try {
    localStorage.setItem(hintKey(), JSON.stringify({ origin, t: Date.now() }));
  } catch {
    /* private mode / storage disabled */
  }
}

function clearHint(origin: string): void {
  try {
    if (readHint(ortCandidates()) === origin) localStorage.removeItem(hintKey());
  } catch {
    /* ignore */
  }
}

let originPromise: Promise<string> | null = null;

/** Idempotent prefetch hook; call it while the page is idle. */
export function prewarmOrtOrigin(): Promise<string> {
  return resolveOrtOrigin().catch(() => DEFAULT_ORIGINS[1]);
}

async function resolveOrtOrigin(): Promise<string> {
  const cfg = getCoreConfig();
  if (cfg.ortWasmPaths) return cfg.ortWasmPaths;
  if (originPromise) return originPromise;
  originPromise = (async () => {
    const candidates = ortCandidates();
    const cached = readHint(candidates);
    if (cached) return cached;
    const winner = await raceOrigin(candidates);
    if (!winner) {
      const fallback = candidates[0] ?? DEFAULT_ORIGINS[1];
      console.warn('[ort] origin race failed, falling back to', fallback);
      return fallback;
    }
    writeHint(winner.origin);
    console.info(
      `[ort] runtime origin: ${winner.origin} (${(winner.bps / 1024 / 1024).toFixed(1)} MB/s)`,
    );
    return winner.origin;
  })();
  return originPromise;
}

function warmupRuntime(origin: string): void {
  fetch(`${origin}${PROBE_FILE}`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
    })
    .catch((e: unknown) => {
      clearHint(origin);
      console.warn('[ort] runtime warm-up failed on', origin, e);
    });
}

async function configureOrt(ort: Ort): Promise<void> {
  if (configured) return;
  configured = true;
  const cfg = getCoreConfig();
  const wasmPaths = await resolveOrtOrigin();
  ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.simd = true;
  if (typeof navigator !== 'undefined' && 'gpu' in navigator && /^https?:/.test(wasmPaths)) {
    warmupRuntime(wasmPaths);
  }
  ort.env.wasm.numThreads =
    cfg.numThreads > 0
      ? cfg.numThreads
      : globalThis.crossOriginIsolated
        ? Math.min(navigator.hardwareConcurrency || 4, 4)
        : 1;
  ort.env.wasm.proxy = true;
  ort.env.logLevel = cfg.logLevel;
}

const EP_CREATE_TIMEOUT_MS = 45_000;

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(
      () => reject(new Error(`session creation timed out after ${ms / 1000}s`)),
      ms,
    );
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

const INIT_WASM_CONFLICT_BUDGET_MS = 120_000;
const INIT_WASM_BACKOFF_START_MS = 3_000;
const INIT_WASM_BACKOFF_MAX_MS = 30_000;

function isInitWasmConflict(e: unknown): boolean {
  return e instanceof Error && /initWasm|multiple calls/i.test(e.message);
}

/** ORT's wasm runtime initializes once per page. Serialize session creation so
 * concurrent model loads cannot hit "multiple calls to initWasm()". */
let createQueue: Promise<unknown> = Promise.resolve();

export function createSession(
  modelBytes: Uint8Array,
  onStatus?: (status: string) => void,
  providers?: readonly string[],
): Promise<{ session: InferenceSession; backend: string }> {
  const chain = providers ?? getCoreConfig().executionProviders;
  const run = createQueue.then(() => createSessionInner(modelBytes, onStatus, chain));
  createQueue = run.catch(() => undefined);
  return run;
}

async function createSessionInner(
  modelBytes: Uint8Array,
  onStatus?: (status: string) => void,
  providers: readonly string[] = ['webgpu', 'wasm'],
): Promise<{ session: InferenceSession; backend: string }> {
  const ort = await getOrt();
  await configureOrt(ort);
  let lastErr: unknown = null;
  for (const ep of providers) {
    let conflictWaitedMs = 0;
    let backoffMs = INIT_WASM_BACKOFF_START_MS;
    for (let attempt = 1; ; attempt++) {
      try {
        onStatus?.('Preparing AI model…');
        const session = await withTimeout(
          ort.InferenceSession.create(modelBytes, {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
          }),
          EP_CREATE_TIMEOUT_MS,
        );
        return { session, backend: ep };
      } catch (e) {
        if (isInitWasmConflict(e) && conflictWaitedMs < INIT_WASM_CONFLICT_BUDGET_MS) {
          console.warn(
            `[ort] wasm init busy (orphaned create), retry ${attempt} in ${backoffMs / 1000}s`,
          );
          await new Promise((r) => setTimeout(r, backoffMs));
          conflictWaitedMs += backoffMs;
          backoffMs = Math.min(backoffMs * 2, INIT_WASM_BACKOFF_MAX_MS);
          continue;
        }
        console.warn(`[ort] ${ep} unavailable, trying next backend:`, e);
        lastErr = e;
        break;
      }
    }
  }
  const detail = lastErr instanceof Error ? lastErr.message || String(lastErr) : String(lastErr ?? '');
  throw new Error(`No usable ONNX backend in this browser. ${detail}`);
}
