import { useWaveTrace } from './hooks/useWaveTrace';
import Controls from './components/Controls';
import Spectrogram from './components/Spectrogram';
import VariancePlot from './components/VariancePlot';
import MockCamera from './components/MockCamera';
import { SpatialView } from './components/SpatialView';
import TrainingDashboard from './components/TrainingDashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { WeaponLitmus } from './components/WeaponLitmus';
import { DevicePanel } from './components/DevicePanel';
import { Activity, Terminal, Wifi, BarChart3, AlertCircle, Box, Map as MapIcon, BrainCircuit, LayoutDashboard, Gauge, AlertTriangle, Upload, Usb, Crosshair } from 'lucide-react';
import { clsx } from 'clsx';
import AnsiImport from 'ansi-to-react';
const Ansi = (AnsiImport as any).default || AnsiImport;
import { useState, useMemo, useRef } from 'react';

function App() {
  const {
    isConnected,
    isRunning,
    verdict,
    driftRatio,
    varianceData,
    antennas,
    nodeIds,
    logs,
    spectrogram,
    spatial,
    heatmapGrid,
    gridSize,
    trainingMetrics,
    trainingMeta,
    trainingResult,
    camUrl,
    start,
    stop,
    addLog,
  } = useWaveTrace();

  const modelUploadRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const bytes = new Uint8Array(await file.arrayBuffer());
    let b64 = '';
    const CHUNK = 8192;
    for (let i = 0; i < bytes.length; i += CHUNK)
      b64 += btoa(String.fromCharCode(...bytes.subarray(i, i + CHUNK)));
    setUploadStatus('Uploading…');
    try {
      const res = await fetch('/api/model/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_b64: b64, dest: `output/model.pkl/${file.name}` }),
      });
      const data = await res.json();
      setUploadStatus(data.error ? `Error: ${data.error}` : `Uploaded → ${data.dest}`);
      addLog(`[MODEL] ${data.error ? 'Upload failed: ' + data.error : 'Uploaded: ' + data.dest}`);
    } catch (err) {
      setUploadStatus(`Upload failed`);
    }
    e.target.value = '';
  };

  const [view3dMode, setView3dMode] = useState<'manifold' | 'heatmap'>('manifold');
  const [activeTab, setActiveTab] = useState<'sensing' | 'training' | 'diagnostics' | 'devices'>('sensing');
  const [calibK, setCalibK] = useState(64);
  const [camEnabled, setCamEnabled] = useState(false);
  const [camAnnotate, setCamAnnotate] = useState(false);
  const [camSessionId, setCamSessionId] = useState(Date.now());

  const verdictInfo = useMemo(() => {
    if (!verdict) return null;
    const isWeaponMode = verdict.mode === "weapon";
    let className = "--";
    let classColor = "text-slate-200";
    if (isWeaponMode) {
      className = verdict.class === 1 ? "Weapon Detected" : "No Weapon";
      classColor = verdict.class === 1 ? "text-rose-500" : "text-emerald-500";
    } else {
      className = verdict.class === 1 ? "Human Present" : "Empty Room";
      classColor = verdict.class === 1 ? "text-sky-500" : "text-slate-400";
    }
    return { className, classColor };
  }, [verdict]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 font-sans">
      {/* Header */}
      <header className="flex justify-between items-center mb-8">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-600 p-2 rounded-lg">
            <Wifi className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">WaveTrace <span className="text-emerald-500">Lab</span></h1>
            <p className="text-slate-500 text-sm font-medium">WiFi-CSI Sensing Research Platform</p>
          </div>
        </div>

        <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button 
            onClick={() => setActiveTab('sensing')}
            className={clsx("flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold transition-all", 
              activeTab === 'sensing' ? "bg-slate-800 text-emerald-400 shadow-lg" : "text-slate-500 hover:text-slate-300")}
          >
            <LayoutDashboard size={14} />
            Live Sensing
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={clsx("flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
              activeTab === 'training' ? "bg-slate-800 text-emerald-400 shadow-lg" : "text-slate-500 hover:text-slate-300")}
          >
            <BrainCircuit size={14} />
            Training Dashboard
          </button>
          <button
            onClick={() => setActiveTab('diagnostics')}
            className={clsx("flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
              activeTab === 'diagnostics' ? "bg-slate-800 text-emerald-400 shadow-lg" : "text-slate-500 hover:text-slate-300")}
          >
            <Gauge size={14} />
            Diagnostics
          </button>
          <button
            onClick={() => setActiveTab('devices')}
            className={clsx("flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
              activeTab === 'devices' ? "bg-slate-800 text-emerald-400 shadow-lg" : "text-slate-500 hover:text-slate-300")}
          >
            <Usb size={14} />
            Devices
          </button>
        </div>

        <div className="flex items-center gap-3">
          {driftRatio != null && driftRatio > 0 && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-amber-600 bg-amber-900/30 text-amber-400 text-xs font-bold">
              <AlertTriangle size={12} />
              Drift {(driftRatio * 100).toFixed(0)}% — recalibrate
            </div>
          )}
          {/* Model upload */}
          <div className="relative">
            <input ref={modelUploadRef} type="file" accept=".joblib,.pkl" className="hidden" onChange={handleModelUpload} />
            <button
              onClick={() => modelUploadRef.current?.click()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-slate-700 bg-slate-900 text-slate-400 text-xs font-bold hover:border-slate-500 hover:text-slate-200 transition-all"
              title={uploadStatus ?? 'Upload model .joblib'}
            >
              <Upload size={12} />
              {uploadStatus ? uploadStatus.slice(0, 20) : 'Upload Model'}
            </button>
          </div>
          <div className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-bold transition-all",
            isConnected
              ? "bg-emerald-900/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
              : "bg-slate-900 border-slate-700 text-slate-500"
          )}>
            <span className={clsx(
              "w-2.5 h-2.5 rounded-full",
              isConnected ? "bg-emerald-500 animate-pulse" : "bg-slate-600"
            )}></span>
            {isConnected ? "LIVE STREAMING" : "DISCONNECTED"}
          </div>
        </div>
      </header>

      <main className="grid grid-cols-12 gap-6 max-w-[1600px] mx-auto">
        {/* Left Column: Controls and Antennas */}
        <div className="col-span-3 space-y-6 flex flex-col">
          <Controls
            onStart={start}
            onStop={stop}
            isRunning={isRunning}
            onCalibDetected={setCalibK}
          />
          
          <section className="bg-slate-900 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2 mb-4 text-slate-400">
              <BarChart3 size={18} />
              <h2 className="text-xs font-bold uppercase tracking-widest">Node Power</h2>
            </div>
            <div className="space-y-4">
              {antennas.length > 0 ? antennas.map((pwr, idx) => {
                // normalize to the strongest node so a dead board reads ~0 and balance is visible
                const maxPwr = Math.max(...antennas, 1e-9);
                const pct = Math.min(100, Math.max(0, (pwr / maxPwr) * 100));
                return (
                  <div key={nodeIds[idx] ?? idx} className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400 font-medium">Node {nodeIds[idx] ?? idx + 1}</span>
                      <span className="font-mono text-emerald-400 font-bold">{pwr.toFixed(3)}</span>
                    </div>
                    <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                      <div 
                        className="bg-emerald-500 h-full transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.4)]" 
                        style={{ width: `${pct}%` }}
                      ></div>
                    </div>
                  </div>
                );
              }) : (
                <p className="text-xs text-slate-600 italic">No data active</p>
              )}
            </div>
          </section>
        </div>

        {/* Middle Column: Visualizations */}
        <div className="col-span-6 space-y-6">
          {activeTab === 'devices' ? (
            <section className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col h-[624px]">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50 shrink-0">
                <Usb size={14} className="text-emerald-500" />
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                  Hardware — Flash · Serial · Pi
                </span>
              </div>
              <div className="flex-1 min-h-0">
                <ErrorBoundary label="Device panel error">
                  <DevicePanel subcarriers={calibK} />
                </ErrorBoundary>
              </div>
            </section>
          ) : activeTab === 'diagnostics' ? (
            <div className="flex flex-col gap-4 h-[624px] overflow-auto custom-scrollbar">
              {/* Weapon litmus is offline — give it its own clearly-labelled section */}
              <section className="bg-slate-900 rounded-xl border border-slate-800 shrink-0">
                <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50">
                  <Crosshair size={14} className="text-emerald-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                    Weapon Litmus — Offline
                  </span>
                </div>
                <div className="p-4">
                  <WeaponLitmus />
                </div>
              </section>
              <section className="bg-slate-900 rounded-xl border border-slate-800 flex flex-col flex-1 min-h-0">
                <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50 shrink-0">
                  <Gauge size={14} className="text-emerald-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                    Live Node Diagnostics
                  </span>
                </div>
                <div className="flex-1 min-h-0 overflow-auto p-4">
                  <DiagnosticsPanel />
                </div>
              </section>
            </div>
          ) : activeTab === 'sensing' ? (
            <>
              <section className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col h-[400px]">
                <div className="px-4 py-2 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                    <div className="flex gap-4">
                      <button 
                        onClick={() => setView3dMode('manifold')}
                        className={clsx("flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors", 
                          view3dMode === 'manifold' ? "text-emerald-400" : "text-slate-500 hover:text-slate-300")}
                      >
                        <Box size={12} />
                        Signal Manifold
                      </button>
                      <button 
                        onClick={() => setView3dMode('heatmap')}
                        className={clsx("flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider transition-colors", 
                          view3dMode === 'heatmap' ? "text-emerald-400" : "text-slate-500 hover:text-slate-300")}
                      >
                        <MapIcon size={12} />
                        Spatial Heatmap
                      </button>
                    </div>
                    <Activity size={12} className="text-emerald-500" />
                  </div>
                  <div className="flex-1 relative">
                    <ErrorBoundary label="3D view error">
                    <SpatialView
                      spectrogramData={spectrogram}
                      heatmapGrid={heatmapGrid}
                      gridSize={gridSize}
                      persons={verdict?.pos ? [{ position: verdict.pos }] : []}
                      mode={view3dMode}
                    />
                    </ErrorBoundary>
                  </div>
              </section>

              <div className="grid grid-cols-2 gap-6 h-[200px]">
                <section className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col">
                  <div className="px-4 py-2 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">2D Spectrogram</span>
                  </div>
                  <div className="flex-1 p-2">
                    <Spectrogram data={spectrogram} />
                  </div>
                </section>
                <section className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col">
                  <div className="px-4 py-2 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">AI Vision Overlay</span>
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-1.5 text-[10px] text-slate-400 font-bold uppercase tracking-wider cursor-pointer hover:text-slate-200 transition-colors">
                        <input type="checkbox" checked={camAnnotate} onChange={(e) => setCamAnnotate(e.target.checked)} className="accent-emerald-500" disabled={!camEnabled} />
                        YOLO Labels
                      </label>
                      <label className="flex items-center gap-1.5 text-[10px] text-slate-400 font-bold uppercase tracking-wider cursor-pointer hover:text-slate-200 transition-colors">
                        <input
                          type="checkbox"
                          checked={camEnabled}
                          onChange={(e) => {
                            const isChecked = e.target.checked;
                            setCamEnabled(isChecked);
                            if (isChecked) {
                              setCamSessionId(Date.now());
                            } else {
                              fetch('/api/camera/stop', { method: 'POST' }).catch(() => {});
                            }
                          }}
                          className="accent-emerald-500"
                        />
                        Camera Power
                      </label>
                      <AlertCircle size={12} className={verdictInfo?.classColor.replace('text', 'text')} />
                    </div>
                  </div>
                  <div className="flex-1 p-2">
                    <MockCamera
                      camUrl={camEnabled ? `${camUrl}${camUrl.includes('?') ? '&' : '?'}annotate=${camAnnotate ? 'true' : 'false'}&_t=${camSessionId}` : ''}
                      label={verdictInfo?.className || ""}
                      isActive={!!verdict && verdict.class === 1}
                    />
                  </div>
                </section>
              </div>
            </>
          ) : (
            <section className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col h-[624px]">
              <div className="px-4 py-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 shrink-0">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                  <BrainCircuit size={14} className="text-emerald-500" />
                  Live Training Performance
                </span>
              </div>
              <div className="flex-1 min-h-0">
                <ErrorBoundary label="Training dashboard error">
                  <TrainingDashboard metrics={trainingMetrics} meta={trainingMeta} result={trainingResult} />
                </ErrorBoundary>
              </div>
            </section>
          )}

          <section className="bg-slate-900 p-4 rounded-xl border border-slate-800">
             <div className="flex items-center justify-between mb-4">
               <div className="flex items-center gap-2 text-slate-400">
                <Activity size={18} />
                <h2 className="text-xs font-bold uppercase tracking-widest">Temporal Variance</h2>
              </div>
              <div className="flex items-center gap-3">
                {verdict && (
                  <>
                    <span className={clsx("text-sm font-bold", verdictInfo?.classColor)}>
                      {verdictInfo?.className}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">
                      Conf: {(verdict.conf * 100).toFixed(1)}% | t={verdict.t.toFixed(2)}s
                    </span>
                  </>
                )}
                {spatial && (
                  <span className="text-[10px] font-mono text-sky-400 border border-sky-800 bg-sky-900/20 rounded px-2 py-0.5">
                    AoA ({spatial.x.toFixed(2)}, {spatial.y.toFixed(2)}m) {(spatial.conf * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
            <div className="bg-slate-950 p-2 rounded-lg border border-slate-800 h-[220px]">
              <VariancePlot data={varianceData} />
            </div>
          </section>
        </div>

        {/* Right Column: Logs */}
        <div className="col-span-3 flex flex-col min-h-[600px]">
           <section className="bg-slate-900 rounded-xl border border-slate-800 flex flex-col flex-1 overflow-hidden h-full">
             <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50">
               <Terminal size={16} className="text-slate-500" />
               <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">System Logs</h2>
             </div>
             <div className="flex-1 p-4 font-mono text-[11px] overflow-y-auto space-y-1 bg-slate-950/50 custom-scrollbar">
                {logs.length > 0 ? logs.map((log) => (
                  <div key={log.id} className={clsx(
                    "border-l-2 pl-2",
                    log.text.includes('ERROR') ? "border-rose-500 text-rose-400 bg-rose-500/5" :
                    log.text.includes('[SYSTEM]') ? "border-emerald-500 text-emerald-400 bg-emerald-500/5" :
                    "border-slate-800 text-slate-500"
                  )}>
                    <Ansi>{log.text}</Ansi>
                  </div>
                )) : (
                  <div className="text-slate-700 italic">No logs yet...</div>
                )}
             </div>
           </section>
        </div>
      </main>
    </div>
  );
}

export default App;
