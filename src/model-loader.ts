/**
 * IndexedDB-backed model cache with multi-mirror fallback and SHA-256
 * integrity verification. Models are downloaded once and reused forever.
 */

import { getCoreConfig, resolveModelUrl } from './config';

export interface ModelProgress {
  done: number;
  total: number;
  stage: string;
  cached?: boolean;
}

export interface ModelSpec {
  key: string;
  urls: string[];
  sha256: string;
  expectedSize: number;
  /** 全部镜像失败时的备选 spec(不同权重格式/不同来源,独立 key 与缓存)。 */
  fallback?: ModelSpec;
}

// ---------------------------------------------------------------------------
// Model status bus (drives the Privacy Shield UI; prefetch coordinates here)
// ---------------------------------------------------------------------------

export type ModelPhase = 'downloading' | 'ready' | 'error';

export interface ModelStatus {
  key: string;
  phase: ModelPhase;
  done: number;
  total: number;
  cached?: boolean;
}

type StatusListener = (s: ModelStatus) => void;
const statusListeners = new Set<StatusListener>();
const statusByKey = new Map<string, ModelStatus>();

export function onModelStatus(cb: StatusListener): () => void {
  statusListeners.add(cb);
  return () => {
    statusListeners.delete(cb);
  };
}

export function getModelStatus(key: string): ModelStatus | undefined {
  return statusByKey.get(key);
}

function emitStatus(s: ModelStatus): void {
  statusByKey.set(s.key, s);
  for (const l of statusListeners) {
    try {
      l(s);
    } catch {
      // listener errors must never break model loading
    }
  }
}

const DB_VERSION = 1;
const STORE = 'models';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(getCoreConfig().cacheName, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

const IDB_CHUNK = 32 * 1024 * 1024; // 大模型分块写：整块结构化克隆 198MB(fp32) 会把渲染进程压到 OOM
const IDB_CHUNK_THRESHOLD = 64 * 1024 * 1024;

interface IdbChunkMeta {
  __chunked: true;
  size: number;
  parts: number;
}

async function idbGet(key: string): Promise<ArrayBuffer | null> {
  const db = await openDB();
  const get = (k: string) =>
    new Promise<unknown>((resolve, reject) => {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(k);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
  const val = await get(key);
  if (!val) return null;
  if (val instanceof ArrayBuffer) return val;
  const meta = val as IdbChunkMeta;
  if (!meta.__chunked) return null;
  const parts: ArrayBuffer[] = [];
  for (let i = 0; i < meta.parts; i++) {
    const part = await get(`${key}__part${i}`);
    if (!(part instanceof ArrayBuffer)) return null; // 缺块视为未缓存（触发自愈重下）
    parts.push(part);
  }
  const out = new Uint8Array(meta.size);
  let off = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part), off);
    off += part.byteLength;
  }
  return out.buffer;
}

async function idbSet(key: string, value: ArrayBuffer): Promise<void> {
  const db = await openDB();
  const put = (k: string, v: unknown) =>
    new Promise<void>((resolve, reject) => {
      const req = db.transaction(STORE, 'readwrite').objectStore(STORE).put(v, k);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  if (value.byteLength <= IDB_CHUNK_THRESHOLD) {
    await put(key, value);
    return;
  }
  const parts = Math.ceil(value.byteLength / IDB_CHUNK);
  for (let i = 0; i < parts; i++) {
    await put(
      `${key}__part${i}`,
      value.slice(i * IDB_CHUNK, Math.min(value.byteLength, (i + 1) * IDB_CHUNK)),
    );
  }
  await put(key, { __chunked: true, size: value.byteLength, parts } satisfies IdbChunkMeta);
}

async function idbDelete(key: string): Promise<void> {
  const db = await openDB();
  const del = (k: string) =>
    new Promise<void>((resolve, reject) => {
      const req = db.transaction(STORE, 'readwrite').objectStore(STORE).delete(k);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  const val = await new Promise<unknown>((resolve) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => resolve(null);
  });
  await del(key);
  if (val && !(val instanceof ArrayBuffer) && (val as IdbChunkMeta).__chunked) {
    for (let i = 0; i < (val as IdbChunkMeta).parts; i++) await del(`${key}__part${i}`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  // 直接喂视图（digest 支持 BufferSource），避免大模型再复制一份——
  // 198MB(fp32) 模型叠加 IDB 写与会话创建时的内存峰值曾把标签页压到 OOM。
  const h = await crypto.subtle.digest('SHA-256', bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(h))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

const STALL_MS = 15_000;

async function fetchWithProgress(
  url: string,
  onProgress?: (p: ModelProgress) => void,
  /** 解码体积提示（spec.expectedSize）：br 预压对象（setup-r2.mjs）的
   *  content-length 是压缩体积，而流式读出的是解码字节——口径不一致会把
   *  进度顶穿 100%。有提示时进度按解码字节/原始体积计量。 */
  decodedSizeHint = 0,
): Promise<Uint8Array> {
  // 停滞看门狗：连接建立后 15s 无任何字节到达即中止，交给下一个镜像——
  // 否则边缘节点挂起的连接会让下载永远停在某个 MB 数上（表现为"卡死"）。
  // 移动端运营商网络常见"连接挂着但零流量"，15s 足够区分慢速与假死。
  const ctrl = new AbortController();
  let stall = setTimeout(() => ctrl.abort(), STALL_MS);
  const res = await fetch(url, { signal: ctrl.signal });
  try {
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const total = res.headers.get('content-encoding')
      ? decodedSizeHint
      : parseInt(res.headers.get('content-length') || '0', 10);
    if (!res.body) {
      const buf = await res.arrayBuffer();
      return new Uint8Array(buf);
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let done = 0;
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      chunks.push(value);
      done += value.byteLength;
      clearTimeout(stall);
      stall = setTimeout(() => ctrl.abort(), STALL_MS);
      onProgress?.({ done, total, stage: 'Downloading AI model' });
    }
    const out = new Uint8Array(done);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.byteLength;
    }
    return out;
  } catch (e) {
    if (ctrl.signal.aborted) throw new Error(`download stalled (no data for ${STALL_MS / 1000}s): ${url}`);
    throw e;
  } finally {
    clearTimeout(stall);
  }
}

/**
 * Load model bytes: device cache first, then each mirror URL in order.
 * The first mirror that delivers bytes matching the pinned SHA-256 wins.
 *
 * 同 key 并发去重：prefetch（空闲预热）与真实运行（如用户立刻点击处理）
 * 会同时走到这里，没有去重时同一份模型被并行下载两份、重复写 IndexedDB
 * ——移动端带宽与内存都被白白翻倍，进度条还会各报各的。共享同一次加载，
 * 进度回调以监听器集合转发给所有加入方。
 */
const inflightLoads = new Map<
  string,
  { promise: Promise<Uint8Array>; listeners: Set<(p: ModelProgress) => void> }
>();

/** 引擎自动选档用：该模型是否已完整缓存在 IndexedDB（≥1MB 视为有效，
 *  与 loadModelInner 的自愈阈值一致）。不触发下载。 */
export async function hasCachedModel(spec: ModelSpec): Promise<boolean> {
  try {
    const cached = await idbGet(spec.key);
    return !!cached && cached.byteLength > 1_000_000;
  } catch {
    return false;
  }
}

export function loadModel(
  spec: ModelSpec,
  onProgress?: (p: ModelProgress) => void,
): Promise<Uint8Array> {
  const existing = inflightLoads.get(spec.key);
  if (existing) {
    if (onProgress) existing.listeners.add(onProgress);
    return existing.promise;
  }
  const listeners = new Set<(p: ModelProgress) => void>();
  if (onProgress) listeners.add(onProgress);
  const relay = (p: ModelProgress) => {
    for (const l of listeners) {
      try {
        l(p);
      } catch {
        // listener errors must never break model loading
      }
    }
  };
  const promise = loadModelInner(spec, relay)
    .finally(() => inflightLoads.delete(spec.key));
  inflightLoads.set(spec.key, { promise, listeners });
  return promise;
}

async function loadModelInner(
  spec: ModelSpec,
  onProgress?: (p: ModelProgress) => void,
): Promise<Uint8Array> {
  try {
    const cached = await idbGet(spec.key);
    if (cached && cached.byteLength > 1_000_000) {
      // 自愈：命中也校验一次 SHA-256——下载中断/浏览器崩溃写出的半截
      // 模型若直接信任，会永久卡死该用户（IDB 永不失效）。67MB 约 100ms。
      const bytes = new Uint8Array(cached);
      if (spec.sha256 && globalThis.crypto?.subtle) {
        const hash = await sha256Hex(bytes);
        if (hash !== spec.sha256) {
          console.warn('[model-loader] cached model failed integrity check, refetching:', spec.key);
          try {
            await idbDelete(spec.key);
          } catch {
            /* best-effort */
          }
        } else {
          onProgress?.({
            done: bytes.byteLength,
            total: bytes.byteLength,
            stage: 'Loaded from your device',
            cached: true,
          });
          emitStatus({
            key: spec.key,
            phase: 'ready',
            done: bytes.byteLength,
            total: bytes.byteLength,
            cached: true,
          });
          return bytes;
        }
      } else {
        // crypto.subtle 不可用（非安全上下文）时无法校验，沿用旧行为
        onProgress?.({
          done: bytes.byteLength,
          total: bytes.byteLength,
          stage: 'Loaded from your device',
          cached: true,
        });
        emitStatus({
          key: spec.key,
          phase: 'ready',
          done: bytes.byteLength,
          total: bytes.byteLength,
          cached: true,
        });
        return bytes;
      }
    }
  } catch {
    // IndexedDB unavailable — fall through to network
  }

  let lastErr: unknown = null;
  for (const rawUrl of spec.urls) {
    const url = resolveModelUrl(rawUrl);
    try {
      onProgress?.({ done: 0, total: spec.expectedSize, stage: 'Connecting…' });
      const bytes = await fetchWithProgress(
        url,
        (p) => {
          emitStatus({
            key: spec.key,
            phase: 'downloading',
            done: p.done,
            total: p.total || spec.expectedSize,
          });
          onProgress?.(p);
        },
        spec.expectedSize,
      );
      if (spec.sha256 && globalThis.crypto?.subtle) {
        onProgress?.({ done: bytes.byteLength, total: bytes.byteLength, stage: 'Verifying model…' });
        const hash = await sha256Hex(bytes);
        if (hash !== spec.sha256) {
          throw new Error(`Integrity check failed for ${url}`);
        }
      }
      try {
        await idbSet(spec.key, bytes.buffer as ArrayBuffer);
      } catch {
        // caching is best-effort
      }
      emitStatus({ key: spec.key, phase: 'ready', done: bytes.byteLength, total: bytes.byteLength });
      return bytes;
    } catch (e) {
      console.warn('[model-loader] mirror failed:', url, e);
      lastErr = e;
    }
  }
  if (spec.fallback) {
    console.warn(
      '[model-loader] all mirrors failed for',
      spec.key,
      '— trying fallback spec',
      spec.fallback.key,
    );
    return loadModel(spec.fallback, onProgress);
  }
  emitStatus({ key: spec.key, phase: 'error', done: 0, total: spec.expectedSize });
  throw new Error(
    `Could not download the AI model from any mirror. ${lastErr instanceof Error ? lastErr.message : ''}`,
  );
}
