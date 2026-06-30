import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, Activity, Database, Brain, Target, Camera, CheckCircle,
         XCircle, FolderOpen, type LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';
import type { StartPayload } from '../hooks/useWaveTrace';

interface ControlsProps {
  onStart: (payload: StartPayload) => void;
  onStop: () => void;
  isRunning: boolean;
  isConnected: boolean;
  onCalibDetected: (k: number) => void;
}

type Action = 'run' | 'calib' | 'collect' | 'train';

// ---------------------------------------------------------------------------
// FilePicker — text input + folder/file icon that opens a native OS dialog
// via the backend /api/paths/browse (osascript on macOS).
// ---------------------------------------------------------------------------
interface FilePickerProps {
  value: string;
  onChange: (v: string) => void;
  type?: 'dir' | 'file';
  prompt?: string;
  ext?: string;          // comma-separated, file mode only e.g. "joblib,pt"
  placeholder?: string;
}

const FilePicker: React.FC<FilePickerProps> = ({
  value, onChange, type = 'dir', prompt = 'Select path', ext = '', placeholder,
}) => {
  const [loading, setLoading] = useState(false);

  const browse = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ type, prompt, ext });
      const res = await fetch(`/api/paths/browse?${params}`);
      const d = await res.json();
      if (d.path) onChange(d.path);
    } catch {}
    finally { setLoading(false); }
  }, [type, prompt, ext, onChange]);

  return (
    <div className="flex gap-0.5">
      <input
        type="text"
        className="flex-1 min-w-0 bg-slate-900 border border-slate-700 rounded-l-md px-2 py-1 text-xs text-slate-200 font-mono focus:outline-none focus:border-emerald-600 transition-colors"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
      />
      <button
        type="button"
        onClick={browse}
        disabled={loading}
        title={type === 'dir' ? 'Browse folder…' : 'Browse file…'}
        className="flex items-center px-2 rounded-r-md border border-l-0 border-slate-700 bg-slate-800 text-slate-400 hover:text-emerald-400 hover:bg-slate-700 hover:border-emerald-700 transition-colors disabled:opacity-40"
      >
        {loading
          ? <span className="w-3 h-3 border border-slate-500 border-t-transparent rounded-full animate-spin" />
          : <FolderOpen size={11} />
        }
      </button>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
const Controls: React.FC<ControlsProps> = ({ onStart, onStop, isRunning, isConnected, onCalibDetected }) => {
  const [action, setAction] = useState<Action>('run');

  const START_LABEL: Record<Action, string> = {
    run: 'Start Inference',
    calib: 'Run Calibration',
    collect: 'Collect Data',
    train: 'Fit Model',
  };
  const [calibBadge, setCalibBadge] = useState<string | null>(null);
  const [camCheckStatus, setCamCheckStatus] = useState<{ok: boolean; msg: string} | null>(null);

  const [config, setConfig] = useState({
    antennas: 2,
    subcarriers: 64,
    fs: 100.0,
    udp_port: 9876,
    cam_url: '/api/camera/stream',
    cam_index: 0,
    // Shared across all tabs
    calibration: 'data/2g4_ht40/ui/cal',
    // Run
    mode: 'presence',
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
    camera_collect: false,
    cam_duration: 30,
    yolo_weights: 'yolov8n-seg.pt',
    // Train
    train_backend: 'cnn',
    train_out: 'data/2g4_ht40/ui/model',
    train_data: 'output/dataset_ui',
    col_per_link: false,
    train_per_link: false,
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

  const handleCamCheck = async () => {
    setCamCheckStatus(null);
    try {
      const res = await fetch(`/api/camera/check?cam_index=${config.cam_index}`);
      const d = await res.json();
      setCamCheckStatus(d.ok
        ? { ok: true, msg: `${d.width}×${d.height}` }
        : { ok: false, msg: d.error ?? 'failed' });
    } catch {
      setCamCheckStatus({ ok: false, msg: 'network error' });
    }
  };

  const handleStart = () => {
    if (action === 'calib' && !window.confirm('This will overwrite calibration files in the output directory. Continue?')) return;
    if (action === 'train' && !window.confirm('This will overwrite model files in the output directory. Continue?')) return;
    const effectiveAction = action === 'collect' && config.camera_collect
      ? 'camera_collect' : action;
    const duration = effectiveAction === 'camera_collect' ? config.cam_duration : 9999.0;
    const perLink = action === 'collect' ? config.col_per_link : config.train_per_link;
    onStart({ action: effectiveAction, ...config, per_link: perLink, synthetic: false, duration } as StartPayload);
  };

  // Ordered to match the workflow: Calib → Data → Fit → Run
  const tabs: { id: Action; label: string; icon: LucideIcon; step: number }[] = [
    { id: 'calib',   label: 'Calib', icon: Target,   step: 1 },
    { id: 'collect', label: 'Data',  icon: Database, step: 2 },
    { id: 'train',   label: 'Fit',   icon: Brain,    step: 3 },
    { id: 'run',     label: 'Run',   icon: Activity, step: 4 },
  ];

  return (
    <div className="flex flex-col gap-4 bg-slate-800 p-4 rounded-xl border border-slate-700">

      {/* ── Hardware / always-visible ── */}
      <div className="space-y-2 pb-3 border-b border-slate-700">
        <p className="text-[9px] font-bold uppercase tracking-widest text-slate-500">Hardware Config</p>
        <p className="text-[9px] font-bold uppercase tracking-widest text-slate-600 pl-1.5 border-l-2 border-slate-600">Camera</p>

        {/* Camera URL */}
        <div className="space-y-1">
          <label className="text-[10px] text-slate-400 flex items-center gap-1"><Camera size={10} /> Camera URL</label>
          <div className="flex gap-1">
            <input
              type="text"
              className="flex-1 bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
              value={config.cam_url}
              onChange={e => setConfig({ ...config, cam_url: e.target.value })}
              placeholder="/api/camera/stream or http://pi:8090/stream.mjpg"
            />
            <button
              onClick={handleCamCheck}
              title="Probe camera (one frame)"
              className="px-2 py-1 rounded-md bg-slate-900 border border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500 transition-colors text-[10px] font-bold"
            >◎</button>
          </div>
          {camCheckStatus && (
            <div className={clsx("flex items-center gap-1 text-[10px] font-mono",
              camCheckStatus.ok ? "text-emerald-400" : "text-rose-400")}>
              {camCheckStatus.ok ? <CheckCircle size={10} /> : <XCircle size={10} />}
              {camCheckStatus.msg}
            </div>
          )}
        </div>

        {/* Cam index */}
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Cam Index (local)</label>
          <input
            type="number" min="0"
            className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
            value={config.cam_index}
            onChange={e => setConfig({ ...config, cam_index: parseInt(e.target.value) || 0 })}
          />
        </div>

        <p className="text-[9px] font-bold uppercase tracking-widest text-slate-600 pl-1.5 border-l-2 border-sky-800 mt-1">RF Capture</p>
        {/* Calibration — SHARED across all tabs */}
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider flex items-center justify-between">
            <span>Calibration Dir</span>
            {calibBadge && (
              <span className="text-emerald-400 font-mono normal-case tracking-normal text-[9px]">{calibBadge}</span>
            )}
          </label>
          <FilePicker
            value={config.calibration}
            onChange={v => setConfig({ ...config, calibration: v })}
            type="dir"
            prompt="Select calibration directory"
            placeholder="data/2g4_ht40/ui/cal"
          />
        </div>

        {/* Bandwidth + Rate */}
        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1">
            <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Bandwidth</label>
            <div className={clsx(
              "w-full px-2 py-1 text-xs font-mono",
              calibBadge ? "text-emerald-400" : "text-slate-600 italic"
            )}>
              {calibBadge ?? "no calib yet"}
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Rate Hz</label>
            <input
              type="number"
              className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
              value={config.fs}
              onChange={e => setConfig({ ...config, fs: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        {/* UDP Port */}
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">UDP Port</label>
          <input
            type="number"
            className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
            value={config.udp_port}
            onChange={e => setConfig({ ...config, udp_port: parseInt(e.target.value) || 9876 })}
          />
        </div>
      </div>

      {/* ── Action tabs ── */}
      <div className="flex gap-1 p-1 bg-slate-900 rounded-lg shrink-0">
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            onClick={() => setAction(tab.id)}
            className={clsx(
              "flex-1 flex flex-col items-center justify-center gap-1 py-2 px-1 rounded-md transition-all relative",
              action === tab.id
                ? "bg-slate-700 text-white shadow-lg ring-1 ring-slate-500"
                : "text-slate-500 hover:text-slate-300 hover:bg-slate-800"
            )}
          >
            {/* step connector */}
            {i < tabs.length - 1 && (
              <span className="absolute -right-0.5 top-1/2 -translate-y-1/2 text-slate-700 text-[8px] z-10">›</span>
            )}
            <span className="absolute top-1 left-1.5 text-[7px] font-bold text-slate-500 leading-none">{tab.step}</span>
            <tab.icon size={14} />
            <span className="text-[9px] font-bold uppercase tracking-tight leading-none">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* ── Tab settings ── */}
      <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50 space-y-3 max-h-[320px] overflow-y-auto pr-1 custom-scrollbar">
        <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block border-b border-slate-800 pb-1">
          {action === 'run' ? 'Inference' : action === 'calib' ? 'Calibration' : action === 'collect' ? 'Data Collection' : 'Fit Model'} Settings
        </label>

        {/* ── RUN ── */}
        {action === 'run' && (
          <>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Pipeline Mode</label>
              <select
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-slate-200"
                value={config.mode}
                onChange={e => setConfig({ ...config, mode: e.target.value })}
              >
                <option value="presence">Human Presence</option>
                <option value="weapon">Weapon Detection</option>
                <option value="count">People Counting</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Model Path / Mesh Root Dir</label>
              <FilePicker
                value={config.model}
                onChange={v => setConfig({ ...config, model: v })}
                type="file"
                prompt="Select model file"
                ext="joblib"
                placeholder="data/.../model.joblib"
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Temporal Avg</label>
                <input
                  type="number" min="1"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                  value={config.frame_average}
                  onChange={e => setConfig({ ...config, frame_average: parseInt(e.target.value) || 1 })}
                />
              </div>
              <div className="space-y-1 flex flex-col justify-end">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer h-[34px] px-2">
                  <input type="checkbox" checked={config.use_baseline} onChange={e => setConfig({ ...config, use_baseline: e.target.checked })} />
                  Sub Baseline
                </label>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input type="checkbox" checked={config.gain_lock} onChange={e => setConfig({ ...config, gain_lock: e.target.checked })} />
                Gain Lock
              </label>
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input type="checkbox" checked={config.vote} onChange={e => setConfig({ ...config, vote: e.target.checked })} />
                Voted Mode
              </label>
            </div>
          </>
        )}

        {/* ── CALIB ── */}
        {action === 'calib' && (
          <>
            <p className="text-[10px] text-slate-500 italic">Calibration dir is set in Hardware Config above.</p>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Output Directory</label>
              <FilePicker
                value={config.cal_out}
                onChange={v => setConfig({ ...config, cal_out: v })}
                type="dir"
                prompt="Select calibration output directory"
                placeholder="data/2g4_ht40/ui/cal"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Baseline Packets (Frames)</label>
              <input
                type="number"
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                value={config.baseline_packets}
                onChange={e => setConfig({ ...config, baseline_packets: parseInt(e.target.value) })}
              />
            </div>
          </>
        )}

        {/* ── COLLECT ── */}
        {action === 'collect' && (
          <>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Label Stage</label>
              <select
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200"
                value={config.col_stage}
                onChange={e => setConfig({ ...config, col_stage: e.target.value })}
              >
                <option value="presence">Presence</option>
                <option value="weapon">Weapon</option>
                <option value="count">Count</option>
              </select>
            </div>
            {!config.camera_collect && (
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Time Spans (e.g. 0:5,10:15)</label>
                <input
                  type="text"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                  value={config.col_spans}
                  onChange={e => setConfig({ ...config, col_spans: e.target.value })}
                />
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Window</label>
                <input type="number"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                  value={config.col_window} onChange={e => setConfig({ ...config, col_window: parseInt(e.target.value) })} />
              </div>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Hop</label>
                <input type="number"
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                  value={config.col_hop} onChange={e => setConfig({ ...config, col_hop: parseInt(e.target.value) })} />
              </div>
            </div>
            {config.col_stage === 'weapon' && (
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer"
                     title="Subtract the quiet-room baseline from σ²[p] (Item 10/CAUSE 2B).">
                <input type="checkbox" checked={config.subtract_ic_baseline}
                       onChange={e => setConfig({ ...config, subtract_ic_baseline: e.target.checked })} />
                Subtract room baseline (σ²[p])
              </label>
            )}
            <div className="pt-1 border-t border-slate-700/50 space-y-2">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer"
                     title="Use webcam + YOLO-seg for live labeling instead of scripted time spans.">
                <input type="checkbox" checked={config.camera_collect}
                       onChange={e => setConfig({ ...config, camera_collect: e.target.checked })} />
                <Camera size={11} /> Camera-supervised (YOLO)
              </label>
              {config.camera_collect && (
                <>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Capture Duration (s)</label>
                    <input type="number" min="5"
                      className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-slate-200"
                      value={config.cam_duration}
                      onChange={e => setConfig({ ...config, cam_duration: parseInt(e.target.value) || 30 })} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">YOLO Weights Path</label>
                    <input type="text"
                      className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
                      value={config.yolo_weights}
                      onChange={e => setConfig({ ...config, yolo_weights: e.target.value })}
                      placeholder="yolov8n-seg.pt" />
                  </div>
                  {config.col_stage === 'weapon' && (
                    <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer"
                           title="Also build per-link (tx→rx) weapon datasets for directional heads">
                      <input type="checkbox" checked={config.col_per_link}
                             onChange={e => setConfig({ ...config, col_per_link: e.target.checked })} />
                      Per-link weapon datasets
                    </label>
                  )}
                </>
              )}
            </div>
          </>
        )}

        {/* ── TRAIN ── */}
        {action === 'train' && (
          <>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">ML Backend</label>
              <select
                className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-200"
                value={config.train_backend}
                onChange={e => setConfig({ ...config, train_backend: e.target.value })}
              >
                <option value="cnn">CNN (PyTorch)</option>
                <option value="mlp">MLP Classifier</option>
                <option value="svm">SVM (calibrated)</option>
                <option value="variance">Variance Threshold (weapon baseline)</option>
              </select>
              {config.train_backend === 'cnn' && (
                <p className="text-[10px] text-amber-400/80 leading-snug">
                  CNN wants hundreds+ of windows. On small or weapon datasets it overfits — start with
                  {' '}<span className="font-mono">Variance Threshold</span>, the honest baseline, and only
                  move to CNN once it clears the σ²[p] litmus.
                </p>
              )}
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Dataset Path</label>
              <FilePicker
                value={config.train_data}
                onChange={v => setConfig({ ...config, train_data: v })}
                type="dir"
                prompt="Select dataset directory"
                placeholder="output/dataset_ui or data/weapon_ds/node0"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-slate-400">Output Model Dir</label>
              <FilePicker
                value={config.train_out}
                onChange={v => setConfig({ ...config, train_out: v })}
                type="dir"
                prompt="Select model output directory"
                placeholder="data/2g4_ht40/ui/model"
              />
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer"
                   title="Train one weapon head per (tx→rx) link direction from node*/link*/ dataset dirs.">
              <input type="checkbox" checked={config.train_per_link}
                     onChange={e => setConfig({ ...config, train_per_link: e.target.checked })} />
              Per-link weapon heads
            </label>
          </>
        )}
      </div>

      {/* ── Start / Stop ── */}
      <div className="flex gap-2 shrink-0 pt-2 mt-auto border-t border-slate-700">
        <button
          onClick={handleStart}
          disabled={isRunning || (action !== 'train' && !isConnected)}
          title={action !== 'train' && !isConnected ? 'Not connected — open Devices to flash firmware, then ensure UDP stream is active' : undefined}
          className={clsx(
            "flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold transition-all",
            isRunning || (action !== 'train' && !isConnected)
              ? "bg-slate-700 text-slate-500 cursor-not-allowed"
              : action === 'calib'   ? "bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-900/20 active:scale-95"
              : action === 'collect' ? "bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-900/20 active:scale-95"
              : action === 'train'   ? "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-900/20 active:scale-95"
              : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20 active:scale-95"
          )}
        >
          <Play size={18} fill="currentColor" />
          {isRunning ? 'Running…' : (action !== 'train' && !isConnected) ? 'Not Connected' : START_LABEL[action]}
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
