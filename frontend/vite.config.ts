import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs keep the compiled bundle self-contained when Render
  // publishes the Vite output directory directly.
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
