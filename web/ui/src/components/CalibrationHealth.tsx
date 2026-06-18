import { clsx } from 'clsx';

interface SepData {
  auc: number;
  weapon_hist: number[];
  none_hist: number[];
  separable: boolean;
  edges?: number[];
}

interface DriftData {
  mean_ratio: number;
  max_ratio: number;
  recalibrate: boolean;
}

interface Props {
  separation?: SepData;
  drift?: DriftData;
}

export function CalibrationHealth({ separation: sep, drift }: Props) {
  return (
    <div className="space-y-3 text-[10px]">
      {sep && (
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-slate-400">σ²[p] separability (AUC):</span>
            <span className={clsx("font-bold", sep.separable ? "text-emerald-400" : "text-rose-400")}>
              {sep.auc}
            </span>
            <span className={clsx("text-[9px]", sep.separable ? "text-emerald-500" : "text-rose-500")}>
              {sep.separable ? "✓ trainable" : "✗ fix geometry/hardware first"}
            </span>
          </div>
          <div className="flex items-end gap-px h-10 bg-slate-950 rounded border border-slate-800 p-0.5">
            {sep.weapon_hist.map((h, i) => {
              const nh = sep.none_hist[i] ?? 0;
              const total = Math.max(h, nh, 1);
              return (
                <div key={i} className="flex-1 flex flex-col-reverse gap-px">
                  <div className="bg-rose-500/70" style={{ height: `${(h / total) * 40}px` }} title={`weapon: ${h}`} />
                  <div className="bg-sky-500/50" style={{ height: `${(nh / total) * 40}px` }} title={`none: ${nh}`} />
                </div>
              );
            })}
          </div>
          <p className="text-[9px] text-slate-600 mt-0.5">red = weapon · blue = none (higher overlap = harder task)</p>
        </div>
      )}
      {drift && (
        <div className={clsx(
          "flex items-center gap-2 rounded px-2 py-1",
          drift.recalibrate ? "bg-amber-900/20 border border-amber-700/30" : "bg-slate-900"
        )}>
          <span className="text-slate-400">Room drift:</span>
          <span className={clsx("font-mono font-bold",
            drift.recalibrate ? "text-amber-400" : "text-slate-300")}>
            {drift.mean_ratio?.toFixed(3)}×
          </span>
          {drift.recalibrate && (
            <span className="text-amber-400 font-bold">→ recalibrate recommended</span>
          )}
        </div>
      )}
    </div>
  );
}
