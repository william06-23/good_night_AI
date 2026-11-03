import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // Bat opens the browser once after the port is up; avoid a second tab / 5174 hop.
    open: false,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Long-running story generation + SSE must not be buffered/timed out.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
})

