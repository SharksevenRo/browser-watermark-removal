#!/usr/bin/env node
/**
 * Run the demo in Chromium and write real timings + PSNR to benchmarks/.
 *
 * Usage:
 *   npm run bench            # balanced (MI-GAN)
 *   npm run bench -- fast    # classical fast-fill when eligible
 */
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const gpu = args.includes('--gpu');
const quality = args.find((a) => !a.startsWith('--')) || 'balanced';
const port = 4173;
const base = `http://127.0.0.1:${port}/`;

function run(cmd, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, args, { cwd: root, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) =>
      code === 0 ? resolvePromise() : reject(new Error(`${cmd} exited ${code}`)),
    );
  });
}

async function waitForServer(url, timeoutMs = 60_000) {
  const started = Date.now();
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() - started > timeoutMs) throw new Error(`Server did not start: ${url}`);
    await delay(300);
  }
}

await run('npm', ['run', 'build:demo']);

const server = spawn(
  'npx',
  ['vite', 'preview', '--config', 'vite.demo.config.ts', '--port', String(port), '--strictPort'],
  { cwd: root, stdio: 'ignore' },
);
server.on('error', (e) => console.error('[bench] preview error', e));

let browser;
try {
  await waitForServer(base);
  browser = await chromium.launch({
    headless: !gpu,
    args: gpu
      ? ['--enable-unsafe-webgpu', '--use-angle=metal']
      : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(0);
  page.setDefaultNavigationTimeout(0);
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.error('[browser]', msg.text());
  });
  // Headless Chromium may expose a SwiftShader WebGPU adapter that is far
  // slower than a real GPU. Force WASM for reproducible CI numbers; --gpu
  // runs a headed browser with the real WebGPU adapter.
  await page.goto(gpu ? base : `${base}?ep=wasm`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__bwr?.ready === true, null, { timeout: 60_000 });

  console.log(`[bench] running ${quality}…`);
  const output = await page.evaluate(
    (q) => window.__bwr.runBenchmark(q),
    quality,
  );
  output.generatedAt = new Date().toISOString();

  for (const c of output.cases) {
    const b64 = c.afterDataUrl.replace(/^data:image\/png;base64,/, '');
    const dir = resolve(root, 'demo', 'public', 'cases', c.id);
    await mkdir(dir, { recursive: true });
    await writeFile(resolve(dir, 'after.png'), Buffer.from(b64, 'base64'));
    delete c.afterDataUrl;
  }

  await mkdir(resolve(root, 'benchmarks'), { recursive: true });
  await writeFile(
    resolve(root, 'benchmarks', gpu ? 'results.json' : 'results-wasm.json'),
    `${JSON.stringify(output, null, 2)}\n`,
  );

  // The demo always shows the real-GPU run; the WASM run is kept separately.
  if (gpu) {
    await mkdir(resolve(root, 'demo', 'public'), { recursive: true });
    await writeFile(
      resolve(root, 'demo', 'public', 'benchmark.json'),
      `${JSON.stringify(output, null, 2)}\n`,
    );
  }

  const rows = output.cases.map((c) => ({
    case: c.id,
    engine: c.engine,
    backend: c.backend,
    detect: `${c.detectMs}ms`,
    mask: `${c.maskMs}ms`,
    inpaint: `${c.inpaintMs}ms`,
    total: `${c.totalMs}ms`,
    regionIn: c.psnrRegionIn,
    regionOut: c.psnrRegionOut,
    gain: c.psnrRegionGain,
  }));
  console.table(rows);
  console.log(`[bench] wrote benchmarks/${gpu ? 'results.json' : 'results-wasm.json'}${gpu ? ' and demo/public/benchmark.json' : ''}`);
} finally {
  await browser?.close().catch(() => {});
  server.kill('SIGTERM');
}
