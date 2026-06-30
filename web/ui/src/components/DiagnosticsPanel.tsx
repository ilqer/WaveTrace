import { useTelemetry } from '../hooks/useTelemetry';
import { NodeHealthTable } from './NodeHealth';
import { AntennaWeights } from './AntennaWeights';
import { FusionWeights } from './FusionWeights';
import { OccupancyHeatmap } from './OccupancyHeatmap';
import { DecisionContribution } from './DecisionContribution';
import { SegmentVoteTrace } from './SegmentVoteTrace';
import { clsx } from 'clsx';

export function DiagnosticsPanel() {
  const t = useTelemetry();

  if (!t) {
    return (
      <div className="flex items-center justify-center h-full text-slate-600 text-sm italic">
        Waiting for telemetry stream… (start inference first)
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-4 p-4 overflow-y-auto h-full">
      {/* Left column */}
      <div className="space-y-5">
        {/* Alert + drift status badges */}
        <div className="flex gap-2 flex-wrap">
          <span className={clsx(
            "text-[9px] font-bold uppercase tracking-widest px-2 py-1 rounded-full border",
            t.alert_active
              ? "bg-rose-900/40 border-rose-500 text-rose-400 animate-pulse"
              : "bg-slate-900 border-slate-700 text-slate-600"
          )}>
            {t.alert_active ? "⚠ WEAPON ALERT" : "Alert: clear"}
          </span>
          {t.drift_ratio != null && t.drift_ratio > 0 && (
            <span className="text-[9px] font-bold uppercase tracking-widest px-2 py-1 rounded-full border bg-amber-900/30 border-amber-600 text-amber-400">
              Drift {(t.drift_ratio * 100).toFixed(0)}% — recalibrate
            </span>
          )}
        </div>
        <NodeHealthTable nodes={t.nodes} sync={t.sync} links={t.links} />
        <AntennaWeights weights={t.antenna_weights} />
        <FusionWeights bands={t.fusion?.bands} weights={t.fusion?.weights} />
        <DecisionContribution contribution={t.contribution} />
      </div>

      {/* Right column */}
      <div className="space-y-5">
        <OccupancyHeatmap grid={t.heatmap} g={t.grid ?? 16} />
        {(t.voter_trace?.length ?? 0) > 0 && (
          <SegmentVoteTrace probs={t.voter_trace!} threshold={0.5} />
        )}
        {t.nodes.length > 0 && (
          <div className="space-y-1">
            <h3 className="text-[10px] uppercase tracking-widest text-slate-500">Live rates</h3>
            {t.nodes.map(n => (
              <div key={n.node_id} className="flex items-center gap-2 text-[10px]">
                <span className="text-slate-400 w-16">Node {n.node_id}</span>
                <div className="flex-1 h-1.5 bg-slate-950 rounded overflow-hidden border border-slate-800">
                  <div
                    className={`h-full ${n.hz_ok ? 'bg-emerald-500' : 'bg-rose-500'}`}
                    style={{ width: `${Math.min(100, (n.hz / 120) * 100)}%` }}
                  />
                </div>
                <span className="text-slate-500 font-mono w-12 text-right">{n.hz} Hz</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
