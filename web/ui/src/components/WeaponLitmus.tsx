import { useState } from 'react';
import { clsx } from 'clsx';
import { Crosshair, ChevronDown, ChevronRight } from 'lucide-react';

// Static σ²[p] go/no-go before ML. per_link=true scores each directed tx→rx link
// separately — surfaces which round-robin directions carry weapon signal (NLOS geometry).
interface HistData { edges: number[]; clear: number[]; weapon: number[]; }

interface LitmusRow {
  label: string;
  ok?: boolean;
  reason?: string;
  auc?: number;
  lower_when_armed?: boolean;
  cohens_d?: number;
  n_clear?: number;
  n_weapon?: number;
  verdict?: string;
  hist?: HistData | null;
}

function aucColor(auc: number): string {
  if (auc < 0.55) return 'text-rose-400';
  if (auc < 0.65) return 'text-amber-400';
  return 'text-emerald-400';
}

// Overlaid density histogram — clear=teal, weapon=red, shared bin grid. Mirrors Yousaf Fig 17.
function SigmaHist({ hist }: { hist: HistData }) {
  const W = 220, H = 52;
  const peak = Math.max(...hist.clear, ...hist.weapon, 1e-12);
  const bins = hist.clear.length;
  const bw = W / bins;
  return (
    <svg width={W} height={H} className="block">
      {hist.clear.map((v, i) => {
        const h = (v / peak) * H;
        return <rect key={`c${i}`} x={i * bw} y={H - h} width={Math.max(bw - 1, 1)} height={h}
                     fill="#2dd4bf" fillOpacity={0.55} />;
      })}
      {hist.weapon.map((v, i) => {
        const h = (v / peak) * H;
        return <rect key={`w${i}`} x={i * bw} y={H - h} width={Math.max(bw - 1, 1)} height={h}
                     fill="#f43f5e" fillOpacity={0.55} />;
      })}
    </svg>
  );
}

export function WeaponLitmus() {
  const [root, setRoot] = useState('data');
  const [perLink, setPerLink] = useState(false);
  const [rows, setRows] = useState<LitmusRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const run = async () => {
    setLoading(true);
    setErr(null);
    setExpanded(new Set());
    try {
      const res = await fetch(
        `/api/weapon/litmus?root=${encodeURIComponent(root)}&per_link=${perLink}`
      );
      const data = await res.json();
      if (data.error) { setErr(data.error); setRows(null); }
      else { setRows(data.rows); }
    } catch (e) {
      setErr(String(e));
    }
    setLoading(false);
  };

  const toggle = (label: string) =>
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(label) ? next.delete(label) : next.add(label);
      return next;
    });

  return (
    <section className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-400">
          <Crosshair size={16} className="text-emerald-500" />
          <h2 className="text-xs font-bold uppercase tracking-widest">Weapon σ²[p] Litmus</h2>
        </div>
        <span className="text-[9px] text-slate-600 uppercase tracking-wider">go / no-go before ML</span>
      </div>

      <div className="flex gap-2 items-center">
        <input
          className="flex-1 bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
          value={root}
          onChange={(e) => setRoot(e.target.value)}
          placeholder="data root (holds weapon_rec/)"
        />
        <label className="flex items-center gap-1.5 text-[10px] text-slate-400 cursor-pointer whitespace-nowrap select-none">
          <input type="checkbox" checked={perLink} onChange={(e) => setPerLink(e.target.checked)} />
          Per-link
        </label>
        <button
          onClick={run}
          disabled={loading}
          className={clsx('px-3 py-1 rounded-md text-xs font-bold transition-all',
            loading ? 'bg-slate-700 text-slate-500' : 'bg-emerald-600 hover:bg-emerald-500 text-white')}
        >
          {loading ? '…' : 'Run'}
        </button>
      </div>

      {err && <p className="text-[11px] text-rose-400 font-mono">{err}</p>}

      {rows && rows.length > 0 && (
        <div className="space-y-1">
          <div className="grid grid-cols-[16px_72px_56px_36px_1fr] gap-2 text-[9px] uppercase tracking-wider text-slate-600 px-1">
            <span />
            <span>{perLink ? 'Link' : 'Node'}</span>
            <span>AUC</span>
            <span>Dir</span>
            <span>Verdict</span>
          </div>
          {rows.map((r) => (
            <div key={r.label}>
              <div
                className="grid grid-cols-[16px_72px_56px_36px_1fr] gap-2 items-center text-[11px] bg-slate-950/60 rounded px-1 py-1 border border-slate-800 cursor-pointer hover:border-slate-700"
                onClick={() => r.hist && toggle(r.label)}
              >
                <span className="text-slate-600">
                  {r.hist
                    ? (expanded.has(r.label) ? <ChevronDown size={10} /> : <ChevronRight size={10} />)
                    : null}
                </span>
                <span className="text-slate-400 font-mono truncate" title={r.label}>{r.label}</span>
                {r.auc != null ? (
                  <>
                    <span className={clsx('font-mono font-bold', aucColor(r.auc))}>{r.auc.toFixed(3)}</span>
                    <span
                      className={clsx('font-mono', r.lower_when_armed ? 'text-slate-400' : 'text-amber-400')}
                      title={r.lower_when_armed
                        ? 'armed σ² lower — matches metal physics'
                        : 'armed σ² HIGHER — anti-physics (geometry/gain issue)'}
                    >
                      {r.lower_when_armed ? 'ok' : 'INV'}
                    </span>
                    <span className="text-slate-400 leading-tight">{r.verdict}</span>
                  </>
                ) : (
                  <span className="col-span-3 text-slate-600 italic">{r.reason}</span>
                )}
              </div>
              {r.hist && expanded.has(r.label) && (
                <div className="ml-[88px] mt-1 mb-1 space-y-1">
                  <div className="flex gap-3 text-[9px] text-slate-500">
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-2 h-2 rounded-sm bg-teal-400 opacity-70" />clear (n={r.n_clear})
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-2 h-2 rounded-sm bg-rose-400 opacity-70" />weapon (n={r.n_weapon})
                    </span>
                    {r.cohens_d != null && (
                      <span className="text-slate-600">d={r.cohens_d.toFixed(2)}</span>
                    )}
                  </div>
                  <SigmaHist hist={r.hist} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {rows && rows.length === 0 && (
        <p className="text-[11px] text-slate-600 italic">
          No recordings found under {root}/weapon_rec/.
        </p>
      )}
    </section>
  );
}
