import { useState } from 'react';
import { clsx } from 'clsx';
import { Crosshair } from 'lucide-react';

// Static σ²[p] go/no-go before ML. per_link=true scores each directed tx→rx link
// separately — surfaces which round-robin directions carry weapon signal (NLOS geometry).
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
}

function aucColor(auc: number): string {
  if (auc < 0.55) return 'text-rose-400';
  if (auc < 0.65) return 'text-amber-400';
  return 'text-emerald-400';
}

export function WeaponLitmus() {
  const [root, setRoot] = useState('data');
  const [perLink, setPerLink] = useState(false);
  const [rows, setRows] = useState<LitmusRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setErr(null);
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
          <div className="grid grid-cols-[72px_56px_36px_1fr] gap-2 text-[9px] uppercase tracking-wider text-slate-600 px-1">
            <span>{perLink ? 'Link' : 'Node'}</span>
            <span>AUC</span>
            <span>Dir</span>
            <span>Verdict</span>
          </div>
          {rows.map((r) => (
            <div
              key={r.label}
              className="grid grid-cols-[72px_56px_36px_1fr] gap-2 items-center text-[11px] bg-slate-950/60 rounded px-1 py-1 border border-slate-800"
            >
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
