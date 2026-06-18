interface Props {
  probs: number[];
  threshold?: number;
}

export function SegmentVoteTrace({ probs, threshold = 0.5 }: Props) {
  if (probs.length === 0) return null;
  const max = Math.max(...probs, 1e-6);
  const threshPct = Math.min(100, (threshold / max) * 100);

  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
        Segment vote trace
      </h3>
      <div className="relative flex items-end gap-px h-10 bg-slate-950 rounded border border-slate-800 p-0.5 overflow-hidden">
        {/* threshold line — absolute inside relative container */}
        <div
          className="absolute inset-x-0 border-t border-dashed border-amber-500/70 pointer-events-none z-10"
          style={{ bottom: `${threshPct}%` }}
        />
        {probs.map((p, i) => (
          <div
            key={i}
            className={`flex-1 min-w-0 ${p >= threshold ? 'bg-rose-500' : 'bg-slate-600'}`}
            style={{ height: `${(p / max) * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
}
