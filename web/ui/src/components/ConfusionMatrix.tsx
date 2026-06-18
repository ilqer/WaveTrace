import { clsx } from 'clsx';

interface Props {
  matrix?: number[][];
  tpr?: number;
  fp_rate?: number;
  labels?: string[];
}

export function ConfusionMatrix({ matrix: m, tpr, fp_rate: fp, labels = ['No weapon', 'Weapon'] }: Props) {
  if (!m) return null;
  const gate = tpr !== undefined && fp !== undefined && tpr >= 0.9 && fp <= 0.1;
  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        LOGO confusion matrix
      </h3>
      <table className="text-[10px] border-collapse mb-2">
        <thead>
          <tr>
            <th />
            {labels.map(l => (
              <th key={l} className="text-slate-500 font-normal px-2 pb-1">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              <td className="text-slate-500 pr-2 text-right">{labels[i]}</td>
              {row.map((v, j) => (
                <td key={j}
                    className={clsx(
                      "px-3 py-1 text-center font-mono font-bold border border-slate-800",
                      i === j ? "bg-emerald-900/30 text-emerald-300" : "bg-rose-900/20 text-rose-400"
                    )}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {tpr !== undefined && fp !== undefined && (
        <div className={clsx(
          "text-[10px] flex gap-3 px-2 py-1 rounded",
          gate ? "bg-emerald-900/20 text-emerald-400" : "bg-rose-900/20 text-rose-400"
        )}>
          <span>TPR {(tpr * 100).toFixed(1)}%</span>
          <span>FP {(fp * 100).toFixed(1)}%</span>
          <span className="font-bold">{gate ? "✓ gate passed" : "✗ below gate"}</span>
        </div>
      )}
    </div>
  );
}
