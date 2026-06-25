import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Errors that are expected on every browser refresh, uvicorn reload, or brief
// backend restart. Vite logs them as errors by default; suppress them so only
// real (unexpected) proxy errors stand out.
const EXPECTED_CODES = new Set(['EPIPE', 'ECONNRESET', 'ECONNREFUSED']);

function isSuppressed(err: Error): boolean {
  // Regular ErrnoException (EPIPE, ECONNRESET, …)
  const code = (err as NodeJS.ErrnoException).code;
  if (code && EXPECTED_CODES.has(code)) return true;
  // AggregateError — ECONNREFUSED during backend restart; code lives on inner errors
  if (err instanceof AggregateError) {
    return (err.errors as Error[]).every(
      e => EXPECTED_CODES.has((e as NodeJS.ErrnoException).code ?? '')
    );
  }
  // Fallback: match by message substring
  return EXPECTED_CODES.has('ECONNREFUSED') && err.message.includes('ECONNREFUSED');
}

const suppressProxy = (proxy: any) => {
  proxy.on('error', (err: Error) => { if (!isSuppressed(err)) console.error('[proxy]', err.message); });
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: suppressProxy,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        configure: suppressProxy,
      },
    },
  },
})
