# Benchmarks

`results.json` — Apple M4, macOS 26.6, Chrome 156, WebGPU, MI-GAN, 800×500.
`results-wasm.json` — same machine, headless Chromium, forced WASM.

Each run:

1. builds the demo;
2. starts `vite preview`;
3. loads the demo in Playwright;
4. runs one warm-up inference (model download + session creation);
5. runs every case again and records detect / mask / init / inpaint / total;
6. computes PSNR against the clean reference over the returned mask.

PSNR is `10 * log10(255² / MSE)` over pixels with mask alpha > 16. A value of
`99` means an exact match.

Reproduce:

```bash
npm run bench       # headless WASM
npm run bench:gpu   # headed WebGPU
```

The WebGPU run needs a real GPU and a desktop session. CI uses the WASM run.
