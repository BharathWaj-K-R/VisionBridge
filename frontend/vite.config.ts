import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs make the compiled bundle usable both from the
  // correct static publish root (frontend/dist) and from /dist/index.html
  // when an existing Render service still publishes frontend/.
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
