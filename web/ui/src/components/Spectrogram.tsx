import React, { useRef, useEffect } from 'react';

interface SpectrogramProps {
  data: number[][] | null;
}

const Spectrogram: React.FC<SpectrogramProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Reused render buffers: streaming frames arrive often, so we keep one offscreen canvas +
  // ImageData and only reallocate when the spectrogram dimensions (W/K) actually change.
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const imgDataRef = useRef<ImageData | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const K = data.length;
    const W = data[0].length;

    let offscreen = offscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement('canvas');
      offscreenRef.current = offscreen;
    }
    if (offscreen.width !== W || offscreen.height !== K) {
      offscreen.width = W;
      offscreen.height = K;
      imgDataRef.current = null;  // dimensions changed -> old ImageData is stale
    }
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    if (!imgDataRef.current) imgDataRef.current = offCtx.createImageData(W, K);
    const imgData = imgDataRef.current;
    for (let y = 0; y < K; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        const val = data[y][x];
        const norm = Math.min(1.0, Math.max(0, val));
        
        let r, g, b;
        if (norm < 0.5) {
          r = Math.floor(255 * (norm / 0.5));
          g = 255;
          b = 0;
        } else {
          r = 255;
          g = Math.floor(255 * (1 - (norm - 0.5) / 0.5));
          b = 0;
        }
        
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    // Scaling to canvas size
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, W, K, 0, 0, canvas.width, canvas.height);
  }, [data]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full bg-slate-900 rounded-lg"
    />
  );
};

export default Spectrogram;
