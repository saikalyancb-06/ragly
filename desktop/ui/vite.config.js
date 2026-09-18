import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built files land in ui/dist and are served by the Python backend at http://127.0.0.1:8765
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } },
})
