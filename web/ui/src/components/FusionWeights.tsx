interface Props {
  bands?: string[];
  weights?: number[];
}

const BAND_COLORS = [
  "bg-emerald-600",
  "bg-fuchsia-600",
  "bg-sky-600",
  "bg-amber-600",
];

export function FusionWeights({ bands, weights }: Props) {
  if (!bands || !weights || bands.length === 0) return null;
  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        Band fusion (learned)
      </h3>
      <div className="flex h-6 rounded overflow-hidden border border-slate-800">
        {bands.map((b, i) => (
          <div
            key={b}
            className={`${BAND_COLORS[i % BAND_COLORS.length]} flex items-center justify-center`}
            style={{ width: `${(weights[i] ?? 0) * 100}%`, minWidth: "2px" }}
            title={`${b}: ${((weights[i] ?? 0) * 100).toFixed(0)}%`}
          >
            <span className="text-[9px] text-white font-bold truncate px-1">
              {b} {((weights[i] ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
