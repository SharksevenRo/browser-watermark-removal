import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = fileURLToPath(new URL('./demo', import.meta.url));
const src = fileURLToPath(new URL('./src', import.meta.url));

export default defineConfig({
  root,
  base: './',
  resolve: {
    alias: {
      '@core': `${src}/index.ts`,
    },
  },
  build: {
    outDir: fileURLToPath(new URL('./demo/dist', import.meta.url)),
    emptyOutDir: true,
    target: 'es2022',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
});
