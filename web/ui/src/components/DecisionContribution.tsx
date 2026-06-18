interface Props {
  contribution?: Record<string, number>;
}

export function DecisionContribution({ contribution: c }: Props) {
  if (!c) return null;
  const bands = Object.keys(c).filter(k => k !== 'fused' && k !== 'weights');
  const fused = c['fused'] as number | undefined;
  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        This decision
      </h3>
      <div className="space-y-1">
        {bands.map(b => (
          <div key={b} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-400 w-20">{b} GHz</span>
            <div className="flex-1 h-2 bg-slate-950 rounded overflow-hidden border border-slate-800">
              <div
                className="h-full bg-sky-500/70"
                style={{ width: `${(c[b] ?? 0) * 100}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-slate-300 w-8 text-right">
              {((c[b] ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        ))}
        {fused !== undefined && (
          <div className="flex items-center gap-2 border-t border-slate-800 pt-1 mt-1">
            <span className="text-[10px] text-slate-300 font-bold w-20">fused</span>
            <div className="flex-1 h-2 bg-slate-950 rounded overflow-hidden border border-slate-700">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${fused * 100}%` }}
              />
            </div>
            <span className="text-[10px] font-mono font-bold text-emerald-400 w-8 text-right">
              {(fused * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
