import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': process.env.DOCMIND_API_PROXY || 'http://127.0.0.1:8000' },
    strictPort: true,
  },
});
