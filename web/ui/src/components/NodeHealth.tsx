import { clsx } from 'clsx';
import type { NodeHealth, SyncInfo } from '../hooks/useTelemetry';

interface Props {
  nodes: NodeHealth[];
  sync: SyncInfo;
}

export function NodeHealthTable({ nodes, sync }: Props) {
  if (nodes.length === 0) {
    return <div className="text-slate-600 text-xs italic">no nodes yet</div>;
  }
  return (
    <div>
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 flex items-center gap-2">
        Node health
        <span className={clsx(
          "text-[9px] px-1.5 py-0.5 rounded-full font-bold",
          sync.ok ? "bg-emerald-900/40 text-emerald-400" : "bg-rose-900/40 text-rose-400"
        )}>
          {sync.ok ? `synced` : `de-synced ${sync.spread_s.toFixed(3)}s`}
        </span>
      </h3>
      <table className="w-full text-[10px] border-collapse">
        <thead>
          <tr className="text-slate-600 border-b border-slate-800">
            <th className="text-left py-0.5">node</th>
            <th className="text-left">band</th>
            <th className="text-right">Hz</th>
            <th className="text-right">SNR</th>
            <th className="text-right">CV</th>
            <th className="text-right">gain</th>
          </tr>
        </thead>
        <tbody>
          {nodes.map(n => (
            <tr key={n.node_id}
                className={clsx("border-b border-slate-900",
                  n.hz_ok ? "text-slate-300" : "text-rose-400")}>
              <td className="py-0.5 font-mono">{n.node_id}</td>
              <td>{n.band}</td>
              <td className="text-right font-mono">
                {n.hz}{!n.hz_ok && <span className="ml-0.5">⚠</span>}
              </td>
              <td className="text-right font-mono">{n.snr_db}dB</td>
              <td className="text-right font-mono">{n.cv.toFixed(3)}</td>
              <td className={clsx("text-right font-mono",
                n.gain_drift !== null && Math.abs(n.gain_drift - 1) > 0.2 ? "text-amber-400" : "")}>
                {n.gain_drift !== null ? n.gain_drift.toFixed(2) : "—"}
                {n.gain_drift !== null && Math.abs(n.gain_drift - 1) > 0.2 && " ⚠"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
