interface Props {
  weights?: number[];   // static (model's first-conv L2 norms)
  live?: number[];      // dynamic (ablation importance for this decision)
}

export function AntennaWeights({ weights, live }: Props) {
  if (!weights || weights.length === 0) {
    return <div className="text-slate-600 text-xs italic">no CNN model loaded</div>;
  }
  const max = Math.max(...weights, ...(live ?? []), 0.001);
  return (
    <div className="space-y-1.5">
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        Antenna weights
      </h3>
      {weights.map((w, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-16 text-[10px] text-slate-400">
            {i >= 100 ? `Pi(5G)` : `ESP ${i + 1}`}
          </span>
          <div className="flex-1 h-3 bg-slate-950 rounded overflow-hidden border border-slate-800 relative">
            <div
              className="h-full bg-blue-500/70 absolute top-0 left-0 transition-all"
              style={{ width: `${(w / max) * 100}%` }}
            />
            {live && live[i] !== undefined && (
              <div
                className="h-1 bg-amber-400 absolute bottom-0 left-0 transition-all"
                style={{ width: `${(live[i] / max) * 100}%` }}
              />
            )}
          </div>
          <span className="w-9 text-[10px] text-right text-slate-400 font-mono">
            {(w * 100).toFixed(0)}%
          </span>
        </div>
      ))}
      {live && (
        <p className="text-[9px] text-slate-600 mt-1">
          blue = model (static) · amber = this decision (ablation)
        </p>
      )}
    </div>
  );
}
