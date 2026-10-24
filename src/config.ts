/**
 * Runtime configuration for the browser watermark-removal core.
 *
 * The library is model-agnostic about hosting: set `modelBase` to the origin
 * that serves the ONNX files, or leave it empty to resolve `models/*` relative
 * to the current page. Model URLs may also be absolute; they are never
 * rewritten.
 */

export interface CoreConfig {
  /**
   * Base URL for relative model paths such as `models/migan_fp16.onnx`.
   * Examples: `https://static.clearpix.org`, `/assets`, or `''` (page-relative).
   */
  modelBase?: string;
  /**
   * Explicit ONNX Runtime Web wasm directory. When set, the runtime-origin
   * race is skipped and this path is used directly. Must end with `/`.
   */
  ortWasmPaths?: string;
  /** Extra ONNX Runtime Web wasm origins to race (must end with `/`). */
  ortOrigins?: string[];
  /** Execution provider chain. Default: `['webgpu', 'wasm']`. */
  executionProviders?: string[];
  /** WASM thread count. Default: `crossOriginIsolated ? min(hardwareConcurrency, 4) : 1`. */
  numThreads?: number;
  /** ONNX Runtime log level. Default: `warning`. */
  logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal';
  /** Optional IndexedDB database name. Default: `browser-watermark-removal`. */
  cacheName?: string;
}

const DEFAULTS: Required<CoreConfig> = {
  modelBase: '',
  ortWasmPaths: '',
  ortOrigins: [],
  executionProviders: ['webgpu', 'wasm'],
  numThreads: 0,
  logLevel: 'warning',
  cacheName: 'browser-watermark-removal',
};

let current: Required<CoreConfig> = { ...DEFAULTS };

export function configureCore(next: CoreConfig): void {
  current = {
    modelBase: next.modelBase ?? current.modelBase,
    ortWasmPaths: next.ortWasmPaths ?? current.ortWasmPaths,
    ortOrigins: next.ortOrigins ? [...next.ortOrigins] : current.ortOrigins,
    executionProviders: next.executionProviders
      ? [...next.executionProviders]
      : current.executionProviders,
    numThreads: next.numThreads ?? current.numThreads,
    logLevel: next.logLevel ?? current.logLevel,
    cacheName: next.cacheName ?? current.cacheName,
  };
}

export function getCoreConfig(): Required<CoreConfig> {
  return current;
}

function trimSlash(s: string): string {
  return s.replace(/\/+$/, '');
}

/**
 * Resolve a model spec URL. Absolute URLs and root-relative URLs are returned
 * unchanged; `models/*` paths are joined to `modelBase`.
 */
export function resolveModelUrl(url: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('/')) return url;
  const base = trimSlash(current.modelBase);
  if (!base) return url;
  return `${base}/${url.replace(/^\/+/, '')}`;
}
