import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',           // Explicit output folder
    emptyOutDir: true,        // Clear old files
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name].js',  // Force assets folder
        assetFileNames: 'assets/[name].[ext]' // CSS/images in assets
      }
    }
  },
  server: {
    proxy: {
      '/chat': 'http://localhost:8000'
    }
  }
});