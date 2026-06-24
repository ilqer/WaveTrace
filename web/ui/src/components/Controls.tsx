import React, { useState, useEffect } from 'react';
import { Play, Square, Activity, Database, Brain, Target, Camera, type LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';
import type { StartPayload } from '../hooks/useWaveTrace';

interface ControlsProps {
  onStart: (payload: StartPayload) => void;
  onStop: () => void;
  isRunning: boolean;
  onCalibDetected: (k: number) => void;
}

type Action = 'run' | 'calib' | 'collect' | 'train';

const Controls: React.FC<ControlsProps> = ({ onStart, onStop, isRunning, onCalibDetected }) => {
  const [action, setAction] = useState<Action>('run');
  const [calibBadge, setCalibBadge] = useState<string | null>(null);
    const [config, setConfig] = useState({
      antennas: 2,
      subcarriers: 64,
      fs: 100.0,
      udp_port: 9876,
      cam_url: 'http://192.168.1.100/mjpeg',
      // Run
      mode: 'presence',
      calibration: 'data/2g4_ht40/ui/cal',
      model: 'data/2g4_ht40/ui/model/model.joblib',
      gain_lock: true,
      vote: true,
      frame_average: 1,
      use_baseline: false,
      // Calib
      baseline_packets: 300,
      cal_out: 'data/2g4_ht40/ui/cal',
      // Collect
      col_stage: 'presence',
      col_spans: '0:5,10:15,20:25',
      col_window: 128,
      col_hop: 32,
      subtract_ic_baseline: true,
      // Train
      train_backend: 'cnn',
      train_out: 'data/2g4_ht40/ui/model',
      train_data: 'output/dataset_ui',
    });

  // Fetch pinned subcarrier width whenever the calibration path changes.
  useEffect(() => {
    fetch(`/api/calib/info?path=${encodeURIComponent(config.calibration)}`)
      .then(r => r.json())
      .then(d => {
        if (d.K) {
          setCalibBadge(`${d.bw_label} · ${d.K} sc`);
          setConfig(prev => ({ ...prev, subcarriers: d.K }));
          onCalibDetected(d.K);
        } else {
          setCalibBadge(null);
        }
      })
      .catch(() => setCalibBadge(null));
  }, [config.calibration, onCalibDetected]);

    const handleStart = () => {
      onStart({ action, ...config, synthetic: false, duration: 9999.0 } as StartPayload);
    };

    const tabs: { id: Action; label: string; icon: LucideIcon }[] = [
      { id: 'run', label: 'Run', icon: Activity },
      { id: 'calib', label: 'Calib', icon: Target },
      { id: 'collect', label: 'Data', icon: Database },
      { id: 'train', label: 'Train', icon: Brain },
    ];

    return (
      <div className="flex flex-col gap-4 bg-slate-800 p-4 rounded-xl border border-slate-700">
        {/* Hardware config — always shown */}
        <div className="space-y-2 pb-3 border-b border-slate-700">
          <p className="text-[9px] font-bold uppercase tracking-widest text-slate-500">Hardware Config</p>
          <div className="space-y-1">
            <label className="text-[10px] text-slate-400 flex items-center gap-1"><Camera size={10} /> Camera URL</label>
            <input
              type="text"
              className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
              value={config.cam_url}
              onChange={(e) => setConfig({ ...config, cam_url: e.target.value })}
              placeholder="http://..."
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Bandwidth</label>
              <div className={clsx(
                "w-full rounded-md px-2 py-1 text-xs font-mono border",
                calibBadge
                  ? "bg-slate-950 border-emerald-800 text-emerald-400"
                  : "bg-slate-950 border-slate-700 text-slate-600 italic"
              )}>
                {calibBadge ?? "run Calib first"}
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Rate Hz</label>
              <input
                type="number"
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                value={config.fs}
                onChange={(e) => setConfig({ ...config, fs: parseFloat(e.target.value) })}
              />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">UDP Port (nodes push CSI here — match firmware, default 9876)</label>
            <input
              type="number"
              className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
              value={config.udp_port}
              onChange={(e) => setConfig({ ...config, udp_port: parseInt(e.target.value) || 9876 })}
            />
          </div>
        </div>

        {/* Action tabs */}
        <div className="flex gap-1 p-1 bg-slate-900 rounded-lg shrink-0">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setAction(tab.id)}
              className={clsx(
                "flex-1 flex flex-col items-center justify-center gap-1 py-2 px-1 rounded-md transition-all",
                action === tab.id
                  ? "bg-emerald-600 text-white shadow-lg"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800"
              )}
            >
              <tab.icon size={14} />
              <span className="text-[9px] font-bold uppercase tracking-tight leading-none">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab-specific settings */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50 space-y-3 max-h-[280px] overflow-y-auto pr-1 custom-scrollbar">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block border-b border-slate-800 pb-1">
            {tabs.find(t => t.id === action)?.label} Settings
          </label>

          {action === 'run' && (
            <>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Pipeline Mode</label>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-slate-200"
                  value={config.mode}
                  onChange={(e) => setConfig({ ...config, mode: e.target.value })}
                >
                  <option value="presence">Human Presence</option>
                  <option value="weapon">Weapon Detection</option>
                  <option value="count">People Counting</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Model Path (or Root Dir for Mesh)</label>
                <input
                  type="text"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                  value={config.model}
                  onChange={(e) => setConfig({ ...config, model: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Calib Directory</label>
                <input
                  type="text"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                  value={config.calibration}
                  onChange={(e) => setConfig({ ...config, calibration: e.target.value })}
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Temporal Avg</label>
                  <input
                    type="number" min="1"
                    className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                    value={config.frame_average}
                    onChange={(e) => setConfig({ ...config, frame_average: parseInt(e.target.value) || 1 })}
                  />
                </div>
                <div className="space-y-1 flex flex-col justify-end">
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer h-[34px] px-2">
                    <input type="checkbox" checked={config.use_baseline} onChange={(e) => setConfig({ ...config, use_baseline: e.target.checked })} />
                    Sub Baseline
                  </label>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input type="checkbox" checked={config.gain_lock} onChange={(e) => setConfig({ ...config, gain_lock: e.target.checked })} />
                  Gain Lock
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input type="checkbox" checked={config.vote} onChange={(e) => setConfig({ ...config, vote: e.target.checked })} />
                  Voted Mode
                </label>
              </div>
            </>
          )}

          {action === 'calib' && (
            <>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Baseline Packets (Frames)</label>
                <input
                  type="number"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                  value={config.baseline_packets}
                  onChange={(e) => setConfig({ ...config, baseline_packets: parseInt(e.target.value) })}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Output Directory</label>
                <input
                  type="text"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                  value={config.cal_out}
                  onChange={(e) => setConfig({ ...config, cal_out: e.target.value })}
                />
              </div>
            </>
          )}

          {action === 'collect' && (
            <>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Label Stage</label>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200"
                  value={config.col_stage}
                  onChange={(e) => setConfig({ ...config, col_stage: e.target.value })}
                >
                  <option value="presence">Presence</option>
                  <option value="weapon">Weapon</option>
                  <option value="count">Count</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Time Spans (e.g. 0:5,10:15)</label>
                <input
                  type="text"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                  value={config.col_spans}
                  onChange={(e) => setConfig({ ...config, col_spans: e.target.value })}
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Window</label>
                  <input
                    type="number"
                    className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                    value={config.col_window}
                    onChange={(e) => setConfig({ ...config, col_window: parseInt(e.target.value) })}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Hop</label>
                  <input
                    type="number"
                    className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                    value={config.col_hop}
                    onChange={(e) => setConfig({ ...config, col_hop: parseInt(e.target.value) })}
                  />
                </div>
              </div>
              {config.col_stage === 'weapon' && (
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer"
                       title="Subtract the quiet-room baseline from σ²[p] (Item 10/CAUSE 2B). Serving mirrors it from the model.">
                  <input type="checkbox" checked={config.subtract_ic_baseline}
                         onChange={(e) => setConfig({ ...config, subtract_ic_baseline: e.target.checked })} />
                  Subtract room baseline (σ²[p])
                </label>
              )}
            </>
          )}

        {action === 'train' && (
          <>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">ML Backend</label>
              <select
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200"
                value={config.train_backend}
                onChange={(e) => setConfig({ ...config, train_backend: e.target.value })}
              >
                <option value="cnn">CNN (PyTorch)</option>
                <option value="mlp">MLP Classifier</option>
                <option value="svm">SVM (calibrated)</option>
                <option value="variance">Variance Threshold (weapon baseline)</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Dataset Path</label>
              <input
                type="text"
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                value={config.train_data}
                onChange={(e) => setConfig({ ...config, train_data: e.target.value })}
                placeholder="output/dataset_ui or data/weapon_ds/node0"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Output Model Path</label>
              <input
                type="text"
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                value={config.train_out}
                onChange={(e) => setConfig({ ...config, train_out: e.target.value })}
              />
            </div>
          </>
        )}
      </div>

      <div className="flex gap-2 shrink-0 pt-2 mt-auto border-t border-slate-700">
        <button
          onClick={handleStart}
          disabled={isRunning}
          className={clsx(
            "flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold transition-all",
            isRunning
              ? "bg-slate-700 text-slate-500 cursor-not-allowed"
              : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20 active:scale-95"
          )}
        >
          <Play size={18} fill="currentColor" />
          {isRunning ? 'Running…' : 'Start'}
        </button>
        <button
          onClick={onStop}
          disabled={!isRunning}
          className={clsx(
            "flex-none flex items-center justify-center p-3 rounded-xl font-bold transition-all",
            !isRunning
              ? "bg-slate-700 text-slate-500 cursor-not-allowed"
              : "bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-900/20 active:scale-95"
          )}
        >
          <Square size={18} fill="currentColor" />
        </button>
      </div>
    </div>
  );
};

export default Controls;
