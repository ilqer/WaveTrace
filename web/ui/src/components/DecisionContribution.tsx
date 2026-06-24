interface Props {
  contribution?: Record<string, number>;
}

export function DecisionContribution({ contribution: c }: Props) {
  if (!c) return null;
  // Per-class decision confidence (the full softmax for the latest window). 'fused'/'weights' are
  // reserved keys from the band-fusion path — filter them out here.
  const classes = Object.keys(c).filter(k => k !== 'fused' && k !== 'weights');
  if (classes.length === 0) return null;
  const winner = classes.reduce((a, b) => ((c[b] ?? 0) > (c[a] ?? 0) ? b : a), classes[0]);
  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        This decision (per class)
      </h3>
      <div className="space-y-1">
        {classes.map(cls => {
          const isWin = cls === winner;
          return (
            <div key={cls} className="flex items-center gap-2">
              <span className={`text-[10px] w-20 truncate ${isWin ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>{cls}</span>
              <div className="flex-1 h-2 bg-slate-950 rounded overflow-hidden border border-slate-800">
                <div
                  className={`h-full ${isWin ? 'bg-emerald-500' : 'bg-sky-500/60'}`}
                  style={{ width: `${(c[cls] ?? 0) * 100}%` }}
                />
              </div>
              <span className={`text-[10px] font-mono w-8 text-right ${isWin ? 'text-emerald-400 font-bold' : 'text-slate-300'}`}>
                {((c[cls] ?? 0) * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
