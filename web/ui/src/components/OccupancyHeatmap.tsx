import { useRef, useEffect } from 'react';

interface Props {
  grid?: number[];        // G*G flat array of probabilities in [0,1]
  g?: number;             // grid dimension (default 16)
  track?: { x: number; y: number; measured: boolean };
}

export function OccupancyHeatmap({ grid, g = 16, track }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!grid || !ref.current) return;
    const canvas = ref.current;
    const ctx = canvas.getContext('2d')!;
    const cell = canvas.width / g;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    grid.forEach((v, idx) => {
      const r = Math.floor(idx / g);
      const c = idx % g;
      const intensity = Math.max(0, Math.min(1, v));
      // blue->cyan->green->yellow->red heatmap
      const h = (1 - intensity) * 240;
      ctx.fillStyle = `hsla(${h.toFixed(0)},80%,50%,${(0.2 + 0.8 * intensity).toFixed(2)})`;
      ctx.fillRect(c * cell, r * cell, cell, cell);
    });
    // grid lines
    ctx.strokeStyle = 'rgba(30,41,59,0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= g; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(canvas.width, i * cell); ctx.stroke();
    }
    // Kalman track marker
    if (track) {
      const cx = (track.x / (g * 0.25) + g / 2) * cell;
      const cy = (g - track.y / 0.25) * cell;
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = track.measured ? 'rgba(16,185,129,0.9)' : 'rgba(251,191,36,0.7)';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [grid, g, track]);

  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        Occupancy heatmap
        {track && (
          <span className="ml-2 text-[9px] text-slate-600">
            x={track.x.toFixed(1)}m y={track.y.toFixed(1)}m
            {!track.measured && " (coasting)"}
          </span>
        )}
      </h3>
      <canvas
        ref={ref}
        width={256}
        height={256}
        className="rounded border border-slate-800 w-full"
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
}
