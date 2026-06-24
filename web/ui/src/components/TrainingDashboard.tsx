import React, { useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';
import type { TrainingMetrics, TrainingMeta } from '../hooks/useWaveTrace';
import { ConfusionMatrix } from './ConfusionMatrix';

interface TrainingDashboardProps {
  metrics: TrainingMetrics[];
  meta: TrainingMeta | null;
  result: Record<string, any> | null;
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="h-full flex items-center justify-center text-[10px] text-slate-600 italic">
      {label}
    </div>
  );
}

function Stat({ label, value, color = 'text-emerald-400' }: { label: string; value: React.ReactNode; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
      <div className="text-[9px] uppercase tracking-widest text-slate-500">{label}</div>
    </div>
  );
}

// Build an approximate 2×2 confusion matrix from tpr + fp_rate + class counts.
function buildMatrix(logo: any, classCounts: any, axis?: string): number[][] | null {
  const axisData = axis ? logo?.[axis] : (logo?.session ?? logo?.subject);
  if (!axisData) return null;
  const { tpr, fp_rate: fp } = axisData;
  if (tpr == null || fp == null) return null;
  const pos = Number(classCounts?.['1'] ?? 0);
  const neg = Number(classCounts?.['0'] ?? 0);
  if (!pos && !neg) return null;
  const tp = Math.round(tpr * pos);
  const fn = pos - tp;
  const fp_ = Math.round(fp * neg);
  const tn = neg - fp_;
  return [[tn, fp_], [fn, tp]];
}

// Shown when MLP/SVM finishes: no epoch loop, just a final metrics dict.
function ResultCard({ result }: { result: Record<string, any> }) {
  const availableAxes = Object.keys(result.logo ?? {}).filter(
    ax => result.logo[ax]?.tpr != null
  );
  const [selectedAxis, setSelectedAxis] = useState<string>(availableAxes[0] ?? 'session');

  const acc = result.train_accuracy != null ? (result.train_accuracy * 100).toFixed(2) + '%' : '—';
  const logoAcc = result.logo_accuracy != null
    ? (result.logo_accuracy * 100).toFixed(2) + '%'
    : Object.keys(result.logo ?? {}).length > 0
      ? Object.values(result.logo as Record<string, number>)
          .map(v => (v * 100).toFixed(1) + '%')
          .join(', ')
      : '—';
  const fitS = result.fit_seconds != null ? result.fit_seconds.toFixed(2) + 's' : '—';
  const backend = (result.backend ?? '?').toUpperCase();
  const n = result.n_samples ?? result.n ?? '—';
  const k = result.n_features != null ? result.n_features : result.k != null ? result.k : '—';
  const cmMatrix = buildMatrix(result.logo, result.class_counts, selectedAxis);
  const logoAxisData = result.logo?.[selectedAxis];
  const tpr = logoAxisData?.tpr;
  const fp = logoAxisData?.fp_rate;

  return (
    <div className="flex flex-col gap-4 p-5">
      {/* header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Training Complete</span>
        <div className="flex items-center gap-2">
          {result.subtract_ic_baseline && (
            <span className="text-[10px] bg-amber-900/40 text-amber-400 border border-amber-500/30 rounded px-2 py-0.5 font-mono"
                  title="Model was trained with IC background subtraction — serving must match">
              bg-subtract
            </span>
          )}
          <span className="text-[10px] bg-emerald-900/40 text-emerald-400 border border-emerald-500/30 rounded px-2 py-0.5 font-mono">
            {backend}
          </span>
        </div>
      </div>

      {/* key numbers */}
      <div className="grid grid-cols-2 gap-4 bg-slate-950 rounded-xl p-4 border border-slate-800">
        <Stat label="Train accuracy" value={acc} color="text-emerald-400" />
        <Stat label="LOGO accuracy" value={logoAcc} color="text-sky-400" />
        <Stat label="Samples" value={n.toLocaleString?.() ?? n} color="text-slate-200" />
        <Stat label="Fit time" value={fitS} color="text-slate-200" />
      </div>

      {/* detail row */}
      <div className="grid grid-cols-3 gap-3 text-[10px]">
        {[
          ['Stage', result.stage ?? '—'],
          ['Features', k],
          ['K (subcarriers)', result.k ?? '—'],
        ].map(([l, v]) => (
          <div key={l} className="bg-slate-950 rounded-lg p-3 border border-slate-800">
            <div className="text-slate-500 uppercase tracking-widest mb-1">{l}</div>
            <div className="text-slate-200 font-mono font-bold">{v}</div>
          </div>
        ))}
      </div>

      {/* LOGO cross-validation + confusion matrix */}
      {cmMatrix && (
        <div className="bg-slate-950 rounded-xl border border-slate-800 p-3 space-y-2">
          {availableAxes.length > 1 && (
            <div className="flex gap-1">
              {availableAxes.map(ax => (
                <button key={ax} onClick={() => setSelectedAxis(ax)}
                  className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded transition-colors ${
                    selectedAxis === ax
                      ? 'bg-slate-700 text-emerald-400'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}>
                  {ax}
                </button>
              ))}
            </div>
          )}
          <ConfusionMatrix matrix={cmMatrix} tpr={tpr} fp_rate={fp}
            labels={result.stage === 'weapon' ? ['No Weapon', 'Weapon'] : ['Absent', 'Present']} />
        </div>
      )}
      {Object.keys(result.logo ?? {}).length > 0 && (
        <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
          <div className="px-4 py-2 border-b border-slate-900 text-[9px] font-bold uppercase tracking-widest text-slate-500">
            LOGO Cross-Validation
          </div>
          <div className="p-3 font-mono text-[10px] space-y-1">
            {Object.entries(result.logo as Record<string, any>).map(([axis, v]) => (
              <div key={axis}>
                <div className="text-slate-500 uppercase text-[9px] mb-0.5">{axis}</div>
                {typeof v === 'object' && v !== null ? (
                  Object.entries(v as Record<string, number>).map(([k2, v2]) => (
                    <div key={k2} className="flex justify-between text-slate-400 pl-2">
                      <span>{k2}</span>
                      <span className="text-sky-400 font-bold">
                        {typeof v2 === 'number' ? (v2 * 100).toFixed(2) + '%' : String(v2)}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="flex justify-between text-slate-400 pl-2">
                    <span>{axis}</span>
                    <span className="text-sky-400 font-bold">{(Number(v) * 100).toFixed(2)}%</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ metrics, meta, result }) => {
  const distributionData = useMemo(() => {
    if (!meta?.distribution) return [];
    return Object.entries(meta.distribution).map(([cls, count]) => ({
      name: cls === '0' ? 'Absent' : 'Present',
      count: Number(count),
    }));
  }, [meta]);

  if (metrics.length === 0 && !meta && !result) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-2">
        <div className="animate-pulse bg-slate-800 p-4 rounded-full">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <p className="text-sm font-medium tracking-wide">Awaiting training session…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-4 h-full bg-slate-900/30 overflow-y-auto custom-scrollbar">

      {/* Dataset summary + class balance */}
      <div className="grid grid-cols-3 gap-6 shrink-0">
        <div className="col-span-1 bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col justify-center">
          <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-4 tracking-widest">Dataset</h3>
          <div className="space-y-4">
            <div>
              <div className="text-2xl font-bold text-emerald-400">{meta?.n_samples ?? result?.n_samples ?? 0}</div>
              <div className="text-[10px] text-slate-500 uppercase font-medium">Total Samples</div>
            </div>
            <div className="pt-4 border-t border-slate-900">
              <div className="text-xl font-bold text-sky-400">{distributionData.length}</div>
              <div className="text-[10px] text-slate-500 uppercase font-medium">Classes</div>
            </div>
          </div>
        </div>

        <div className="col-span-2 bg-slate-950 p-4 rounded-xl border border-slate-800">
          <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-4 tracking-widest">Class Balance</h3>
          <div className="h-[120px]">
            {distributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip
                    cursor={{ fill: '#1e293b', opacity: 0.4 }}
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {distributionData.map((_e, i) => (
                      <Cell key={i} fill={i % 2 === 0 ? '#4ECDC4' : '#FF6B6B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart label="No class data yet" />
            )}
          </div>
        </div>
      </div>

      {/* Result card (MLP/SVM one-shot) OR epoch curves (CNN) */}
      {result && metrics.length === 0 ? (
        <div className="bg-slate-950 rounded-xl border border-slate-800 shrink-0">
          <ResultCard result={result} />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-6 shrink-0">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Loss</h3>
              <div className="flex gap-4 text-[10px]">
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-rose-500 inline-block" /> Train</span>
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-rose-300 inline-block" /> Val</span>
              </div>
            </div>
            <div className="h-[200px]">
              {metrics.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="epoch" stroke="#475569" fontSize={10} />
                    <YAxis stroke="#475569" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} labelStyle={{ color: '#94a3b8' }} />
                    <Line isAnimationActive={false} type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2} dot={{ r: 2, fill: '#f43f5e' }} />
                    <Line isAnimationActive={false} type="monotone" dataKey="val_loss" stroke="#fda4af" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 2, fill: '#fda4af' }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart label="Waiting for first epoch…" />
              )}
            </div>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Accuracy</h3>
              <div className="flex gap-4 text-[10px]">
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-emerald-500 inline-block" /> Train</span>
                <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-emerald-300 inline-block" /> Val</span>
              </div>
            </div>
            <div className="h-[200px]">
              {metrics.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="epoch" stroke="#475569" fontSize={10} />
                    <YAxis stroke="#475569" fontSize={10} domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} labelStyle={{ color: '#94a3b8' }} />
                    <Line isAnimationActive={false} type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={{ r: 2, fill: '#10b981' }} />
                    <Line isAnimationActive={false} type="monotone" dataKey="val_accuracy" stroke="#6ee7b7" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 2, fill: '#6ee7b7' }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart label="Waiting for first epoch…" />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Per-epoch log (CNN only, irrelevant for MLP) */}
      {metrics.length > 0 && (
        <div className="flex flex-col border border-slate-800 bg-slate-950 rounded-xl shrink-0">
          <div className="px-4 py-2 border-b border-slate-900 bg-slate-900/50 flex justify-between items-center rounded-t-xl">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Epoch Log</h3>
            <span className="text-[9px] text-slate-500 font-mono">Epochs: {metrics.length}</span>
          </div>
          <div className="p-3 font-mono text-[11px]">
            {metrics.map((m, i) => (
              <div key={i} className="flex items-center gap-6 py-1.5 border-b border-slate-900 last:border-0 hover:bg-slate-900/50 transition-colors">
                <span className="text-emerald-500 w-20 shrink-0 font-bold">Epoch {m.epoch}:</span>
                <div className="flex gap-8">
                  <span className="text-slate-400 whitespace-nowrap">
                    Loss: <span className="text-rose-400 font-bold">{m.loss?.toFixed(4) ?? '—'}</span>
                    {' / '}<span className="text-rose-300">{m.val_loss?.toFixed(4) ?? '—'}</span>
                  </span>
                  <span className="text-slate-400 whitespace-nowrap">
                    Acc: <span className="text-sky-400 font-bold">{m.accuracy != null ? (m.accuracy * 100).toFixed(2) + '%' : '—'}</span>
                    {' / '}<span className="text-sky-300">{m.val_accuracy != null ? (m.val_accuracy * 100).toFixed(2) + '%' : '—'}</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingDashboard;
