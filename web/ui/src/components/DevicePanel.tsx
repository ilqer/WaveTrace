import { useState, useRef, useLayoutEffect } from 'react';
import { useDevice } from '../hooks/useDevice';
import { Usb, RefreshCw, Upload, Terminal, Square, Radio, Cpu, Server, Trash2, X } from 'lucide-react';
import { clsx } from 'clsx';
import { Ansi } from '../lib/ansi';

const PI_PRESETS: { label: string; cmd: string }[] = [
  { label: 'Start capture', cmd: 'bash ~/wavetrace/firmware/pi/start_capture.sh' },
  { label: 'Run CSI node', cmd: 'cd ~/wavetrace/firmware/pi && python3 pi5_csi_node.py' },
  { label: 'Setup Nexmon (once)', cmd: 'bash ~/wavetrace/firmware/pi/setup_nexmon.sh' },
];

export function DevicePanel({ subcarriers }: { subcarriers: number }) {
  const {
    lines, ports, monitoring, busy, runningScripts,
    refreshPorts, startMonitor, stopMonitor, flash, runPi, runScript, stopDevice, clearLines, sendInput
  } = useDevice();

  const [port, setPort] = useState('');
  const [role, setRole] = useState<'node' | 'rx' | 'tx'>('node');
  const [nodeId, setNodeId] = useState(1);
  const [piHost, setPiHost] = useState('pi@raspberrypi.local');
  const [piPreset, setPiPreset] = useState(PI_PRESETS[0].cmd);
  const [piCmd, setPiCmd] = useState(PI_PRESETS[0].cmd);
  const [scriptName, setScriptName] = useState('health_monitor.py');
  const [opts, setOpts] = useState<Record<string, string>>({});  // per-option values for the picked script
  const [cleanBuild, setCleanBuild] = useState(false);
  const [logFilter, setLogFilter] = useState('All');
  const [closedTabs, setClosedTabs] = useState<string[]>([]);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const handleScroll = () => {
    if (!logsContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 30);
  };



  // Per-script option schema grounded in each script's real argparse (positionals for the
  // sys.argv scripts, --flags for the rest). Lets the UI render one field per option and assemble
  // the command, instead of a freeform args string. `flag` omitted => positional (order = schema).
  type Opt = { name: string; flag?: string; kind: 'text' | 'number' | 'bool'; def?: string; help?: string };
  const SCRIPT_SCHEMA: Record<string, Opt[]> = {
    'health_monitor.py': [{ name: 'port', kind: 'number', def: '9877', help: 'positional UDP port' }],
    'mesh_verify.py': [
      { name: 'port', kind: 'number', def: '9876', help: 'positional UDP port' },
      { name: 'seconds', kind: 'number', def: '', help: 'positional run time (blank = until stopped)' },
    ],
    'ntp_server.py': [{ name: 'port', kind: 'number', def: '123', help: 'positional UDP port' }],
    'collect_baseline.py': [
      { name: 'node', flag: '--node', kind: 'number', help: 'only this node (blank = all)' },
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'frames', flag: '--frames', kind: 'number', def: '3000' },
      { name: 'min-frames', flag: '--min-frames', kind: 'number', def: '300' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
    ],
    'collect_presence.py': [
      { name: 'node', flag: '--node', kind: 'number', help: 'only this node (blank = all)' },
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'sessions', flag: '--sessions', kind: 'number', def: '3' },
      { name: 'frames', flag: '--frames', kind: 'number', def: '1500' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
    ],
    'collect_weapon.py': [
      { name: 'node', flag: '--node', kind: 'number', help: 'only this node (blank = all)' },
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'sessions', flag: '--sessions', kind: 'number', def: '3' },
      { name: 'frames', flag: '--frames', kind: 'number', def: '1500' },
      { name: 'subject', flag: '--subject', kind: 'text', def: 'p0', help: 'vary across people!' },
      { name: 'carry', flag: '--carry', kind: 'text', def: 'na', help: 'waist/chest/ankle' },
      { name: 'bg-subtract', flag: '--bg-subtract', kind: 'bool', help: 'room-null σ²[p] (Item 10)' },
      { name: 'root', flag: '--root', kind: 'text', def: subcarriers <= 64 ? 'data/2g4_ht20' : subcarriers <= 128 ? 'data/2g4_ht40' : 'data/5g_ht80' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
      { name: 'model', flag: '--model', kind: 'text', help: 'default <root>/model_weapon' },
    ],
    'collect_count.py': [
      { name: 'node', flag: '--node', kind: 'number', help: 'only this node (blank = all)' },
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'sessions', flag: '--sessions', kind: 'number', def: '3' },
      { name: 'frames', flag: '--frames', kind: 'number', def: '1500' },
      { name: 'max-count', flag: '--max-count', kind: 'number', def: '3' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
    ],
    'run_live_mesh.py': [
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
      { name: 'model', flag: '--model', kind: 'text', help: 'default <root>/model' },
    ],
    'run_weapon.py': [
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
      { name: 'model', flag: '--model', kind: 'text', help: 'default <root>/model_weapon' },
    ],
    'run_count.py': [
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
      { name: 'model', flag: '--model', kind: 'text', help: 'default <root>/model_count' },
      { name: 'max-count', flag: '--max-count', kind: 'number', def: '3' },
    ],
    'collect_camera.py': [
      { name: 'stage', flag: '--stage', kind: 'text', def: 'presence', help: 'presence or weapon' },
      { name: 'duration', flag: '--duration', kind: 'number', def: '30', help: 'seconds to capture' },
      { name: 'fps', flag: '--fps', kind: 'number', def: '15', help: 'live label rate' },
      { name: 'grid', flag: '--grid', kind: 'number', def: '16', help: 'heatmap resolution G×G' },
      { name: 'cam-index', flag: '--cam-index', kind: 'number', def: '0' },
      { name: 'weights', flag: '--weights', kind: 'text', help: 'YOLO-seg weights (default yolov8n-seg.pt)' },
      { name: 'conf', flag: '--conf', kind: 'number', def: '0.35', help: 'detector confidence floor' },
      { name: 'subject', flag: '--subject', kind: 'text', def: 'cam', help: 'subject id for LOGO grouping' },
      { name: 'port', flag: '--port', kind: 'number', def: '9876' },
      { name: 'root', flag: '--root', kind: 'text', def: 'data/2g4_ht40' },
      { name: 'cal', flag: '--cal', kind: 'text', help: 'default <root>/cal' },
      { name: 'model', flag: '--model', kind: 'text', help: 'default <root>/model' },
      { name: 'train', flag: '--train', kind: 'bool', help: 'train presence + heatmap after capture' },
    ],
    'weapon_experiments.py': [
      { name: 'root', flag: '--root', kind: 'text', def: 'data/2g4_ht40/ui', help: 'capture-profile root' },
      { name: 'skip-cnn', flag: '--skip-cnn', kind: 'bool', help: 'skip slow CPU CNN experiment' },
    ],
    'weapon_litmus.py': [
      { name: 'root', flag: '--root', kind: 'text', def: 'data', help: 'capture-profile root' },
      { name: 'node', flag: '--node', kind: 'number', help: 'only this node (default: all)' },
      { name: 'per-link', flag: '--per-link', kind: 'bool', help: 'per-link (tx→rx) litmus' },
      { name: 'no-hist', flag: '--no-hist', kind: 'bool', help: 'skip ASCII histograms' },
      { name: 'plot', flag: '--plot', kind: 'bool', help: 'write PNG PDFs (needs matplotlib)' },
    ],
  };

  const LOCAL_SCRIPTS = Object.keys(SCRIPT_SCHEMA);
  const schema = SCRIPT_SCHEMA[scriptName] ?? [];

  // Assemble the CLI args from the filled fields: positionals in order (fill earlier gaps with their
  // default so argparse position is preserved), --flags only when set, bool flags only when checked.
  const buildArgs = (): string => {
    const parts: string[] = [];
    const positionals = schema.filter(o => !o.flag);
    const lastSet = positionals.reduce((acc, o, i) => (opts[o.name]?.trim() ? i : acc), -1);
    positionals.forEach((o, i) => {
      if (i <= lastSet) parts.push((opts[o.name]?.trim() || o.def || '').toString());
    });
    for (const o of schema) {
      if (!o.flag) continue;
      const v = opts[o.name];
      if (o.kind === 'bool') { if (v === 'true') parts.push(o.flag); }
      else if (v != null && v.trim() !== '') parts.push(`${o.flag} ${v.trim()}`);
    }
    return parts.join(' ');
  };
  const previewArgs = buildArgs();

  const selectedPort = port || ports.find(p => p.likely_esp)?.device || ports[0]?.device || '';

  const sources = Array.from(new Set(lines.map(l => l.source)));
  const ttySources = sources.filter(s => s.startsWith('tty:') && !closedTabs.includes(s)).sort();
  const scriptSources = sources.filter(s => s.startsWith('script:') && !closedTabs.includes(s)).sort();
  const tabs = ['All', 'System/Flash', 'Scripts', ...ttySources, ...scriptSources];

  const filteredLines = lines.filter(l => {
    if (logFilter === 'All') return true;
    if (logFilter === 'Scripts') return l.source.startsWith('script:');
    if (logFilter === 'System/Flash') return ['flash', 'pi', 'monitor'].includes(l.source);
    return l.source === logFilter;
  });

  useLayoutEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [filteredLines, autoScroll]);

  useLayoutEffect(() => {
    setAutoScroll(true);
  }, [logFilter]);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-hidden">
      {/* Top: controls */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 shrink-0 max-h-[50%] overflow-y-auto custom-scrollbar pr-1 items-start">
        {/* Serial port picker */}
        <section className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
              <Usb size={12} /> Serial Port
            </span>
            <button onClick={refreshPorts} className="text-slate-500 hover:text-emerald-400" title="Rescan ports">
              <RefreshCw size={13} />
            </button>
          </div>
          <select
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1.5 text-xs text-slate-200 font-mono"
            value={selectedPort}
            onChange={(e) => setPort(e.target.value)}
          >
            {ports.length === 0 && <option value="">No ports — plug in a board & rescan</option>}
            {ports.map((p) => (
              <option key={p.device} value={p.device}>
                {p.likely_esp ? '● ' : '○ '}{p.device}{p.description ? ` — ${p.description}` : ''}
              </option>
            ))}
          </select>
          <div className="flex flex-col gap-2">
            <button
              onClick={() => {
                if (selectedPort) {
                  setClosedTabs(prev => prev.filter(t => t !== `tty:${selectedPort}`));
                  startMonitor(selectedPort);
                }
              }}
              disabled={!selectedPort || monitoring.includes(selectedPort)}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-bold bg-slate-800 hover:bg-slate-700 text-emerald-400 disabled:opacity-40"
            >
              <Radio size={12} /> Monitor Selected
            </button>
            {monitoring.length > 0 && (
              <div className="space-y-1">
                {monitoring.map(p => (
                  <div key={p} className="flex items-center justify-between bg-slate-950 px-2 py-1 rounded border border-emerald-900/30">
                    <span className="text-[10px] text-emerald-500 font-mono truncate mr-2">▶ {p.split('/').pop()}</span>
                    <button onClick={() => { stopMonitor(p); if (logFilter === `tty:${p}`) setLogFilter('All'); }} className="text-rose-500 hover:text-rose-400"><Square size={10} /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Flash */}
        <section className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
            <Cpu size={12} /> Flash Firmware
          </span>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[9px] text-slate-500 uppercase font-bold">Role</label>
              <select
                className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200"
                value={role}
                onChange={(e) => setRole(e.target.value as typeof role)}
              >
                <option value="node">node (mesh)</option>
                <option value="rx">rx (legacy)</option>
                <option value="tx">tx (legacy)</option>
              </select>
            </div>
            <div>
              <label className="text-[9px] text-slate-500 uppercase font-bold">Node ID</label>
              <input
                type="number" min={1}
                disabled={role === 'tx'}
                className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 disabled:opacity-40"
                value={nodeId}
                onChange={(e) => setNodeId(parseInt(e.target.value) || 1)}
              />
            </div>
          </div>
          <button
            onClick={() => selectedPort && flash(role, role === 'tx' ? null : nodeId, selectedPort, cleanBuild)}
            disabled={!selectedPort || busy}
            className="w-full flex items-center justify-center gap-1.5 py-2 rounded-md text-xs font-bold bg-slate-600 hover:bg-slate-500 text-white disabled:opacity-40"
          >
            <Upload size={13} /> {busy ? 'Flashing…' : `Build + Flash ${role}${role !== 'tx' ? ' ' + nodeId : ''}`}
          </button>
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400 mt-2 mb-1 cursor-pointer">
            <input type="checkbox" className="border-slate-700 bg-slate-950 text-emerald-500" checked={cleanBuild} onChange={(e) => setCleanBuild(e.target.checked)} />
            Clean rebuild (apply sdkconfig changes)
          </label>
          <p className="text-[9px] text-slate-600 leading-snug">
            Builds + flashes (no monitor). Needs ESP-IDF at <code className="text-slate-500">~/esp/esp-idf</code>{' '}
            (override with <code className="text-slate-500">IDF_EXPORT</code>).
          </p>
        </section>

        {/* Pi capture */}
        <section className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
            <Server size={12} /> Pi Capture (SSH)
          </span>
          <input
            type="text"
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200 font-mono"
            value={piHost}
            onChange={(e) => setPiHost(e.target.value)}
            placeholder="user@host"
          />
          <select
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200"
            value={piPreset}
            onChange={(e) => { setPiPreset(e.target.value); if (e.target.value) setPiCmd(e.target.value); }}
          >
            <option value="">— pick a preset —</option>
            {PI_PRESETS.map((p) => <option key={p.cmd} value={p.cmd}>{p.label}</option>)}
          </select>
          <input
            type="text"
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-[10px] text-slate-300 font-mono"
            value={piCmd}
            placeholder="or type a custom command…"
            onChange={(e) => { setPiCmd(e.target.value); setPiPreset(''); }}
          />
          <button
            onClick={() => runPi(piHost, piCmd)}
            disabled={!piHost || !piCmd}
            className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-bold bg-sky-700 hover:bg-sky-600 text-white disabled:opacity-40"
          >
            <Terminal size={12} /> Run on Pi
          </button>
        </section>

        {/* Local Scripts */}
        <section className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
            <Terminal size={12} /> Local Scripts
          </span>
          <select
            className="w-full bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-200"
            value={scriptName}
            onChange={(e) => { setScriptName(e.target.value); setOpts({}); }}
          >
            {LOCAL_SCRIPTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>

          {/* One field per real argparse option (no more freeform args string). */}
          <div className="space-y-1.5 max-h-44 overflow-y-auto custom-scrollbar pr-1">
            {schema.map((o) => (
              <div key={o.name} className="flex items-center gap-2">
                <label className="text-[10px] text-slate-400 w-24 shrink-0 truncate font-mono"
                       title={o.help ? `${o.name} — ${o.help}` : o.name}>
                  {o.flag ?? o.name}{!o.flag && <span className="text-slate-600"> (pos)</span>}
                </label>
                {o.kind === 'bool' ? (
                  <input type="checkbox"
                         checked={opts[o.name] === 'true'}
                         onChange={(e) => setOpts({ ...opts, [o.name]: e.target.checked ? 'true' : '' })} />
                ) : (
                  <input
                    type={o.kind === 'number' ? 'number' : 'text'}
                    className="flex-1 min-w-0 bg-slate-950 border border-slate-700 rounded-md px-2 py-1 text-[11px] text-slate-200 font-mono"
                    value={opts[o.name] ?? ''}
                    placeholder={o.def ? `${o.def} (default)` : (o.help ?? '')}
                    onChange={(e) => setOpts({ ...opts, [o.name]: e.target.value })}
                  />
                )}
              </div>
            ))}
            {schema.length === 0 && <p className="text-[10px] text-slate-600 italic">No options.</p>}
          </div>

          <div className="bg-slate-950 p-1.5 rounded border border-slate-800 text-[9px] font-mono text-slate-500 break-all">
            <span className="text-slate-600">$ python </span>{scriptName} {previewArgs || <span className="text-slate-700">(no args)</span>}
          </div>

          <button
            onClick={() => {
              setClosedTabs(prev => prev.filter(t => t !== `script:${scriptName}`));
              runScript(scriptName, previewArgs);
            }}
            disabled={!scriptName}
            className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-bold bg-emerald-700 hover:bg-emerald-600 text-white disabled:opacity-40"
          >
            <Terminal size={12} /> Run Script
          </button>
        </section>

        {/* Active Scripts */}
        {runningScripts.length > 0 && (
          <section className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-2">
            <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400 flex items-center gap-1.5">
              <Terminal size={12} /> Running Scripts
            </span>
            <div className="space-y-1">
              {runningScripts.map(s => (
                <div key={s} className="flex items-center justify-between bg-slate-950 px-2 py-1.5 rounded border border-emerald-900/50">
                  <span className="text-[10px] text-emerald-500 font-mono truncate mr-2">▶ {s}</span>
                  <div className="flex items-center gap-2">
                    <button onClick={() => sendInput(`script:${s}`, '\n')} className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-900/50 hover:bg-emerald-800 text-emerald-300">↵ Continue</button>
                    <button onClick={() => { stopDevice(`script:${s}`); if (logFilter === `script:${s}`) setLogFilter('All'); }} className="text-rose-500 hover:text-rose-400"><Square size={10} /></button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <button
          onClick={() => { stopDevice(); if (['flash', 'pi', 'System/Flash'].includes(logFilter)) setLogFilter('All'); }}
          className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-bold bg-slate-800 hover:bg-rose-700 text-slate-300 border border-slate-700"
        >
          <Square size={12} /> Stop Flash/Pi Process
        </button>
      </div>

      {/* Bottom: device log stream */}
      <div className="flex-1 flex flex-col bg-slate-950/60 rounded-lg border border-slate-800 overflow-hidden min-h-0">
        <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between shrink-0">
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
            <Terminal size={12} /> Device Output
          </span>
          <button onClick={clearLines} className="text-slate-600 hover:text-slate-300" title="Clear">
            <Trash2 size={12} />
          </button>
        </div>
        <div className="flex bg-slate-900 border-b border-slate-800 px-2 pt-2 gap-1 overflow-x-auto custom-scrollbar shrink-0">
          {tabs.map(t => (
            <div key={t} className="group relative flex items-center">
              <button onClick={() => setLogFilter(t)} className={clsx("px-3 py-1 text-[10px] uppercase font-bold rounded-t-md whitespace-nowrap", logFilter === t ? "bg-slate-800 text-slate-200" : "bg-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-800/50", (t.startsWith('tty:') || t.startsWith('script:')) && "pr-6")}>
                {t.replace('tty:', '')}
              </button>
              {(t.startsWith('tty:') || t.startsWith('script:')) && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setClosedTabs(prev => [...prev, t]);
                    if (logFilter === t) setLogFilter('All');
                    if (t.startsWith('tty:')) {
                      const p = t.replace('tty:', '');
                      if (monitoring.includes(p)) stopMonitor(p);
                    }
                  }}
                  className="absolute right-1.5 p-0.5 text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={10} strokeWidth={3} />
                </button>
              )}
            </div>
          ))}
        </div>
        <div 
          ref={logsContainerRef}
          onScroll={handleScroll}
          onWheel={(e) => { if (e.deltaY < 0) setAutoScroll(false); }}
          onTouchMove={() => setAutoScroll(false)}
          className="flex-1 min-h-0 p-2 font-mono text-[10px] overflow-y-auto space-y-0.5 custom-scrollbar"
        >
          {filteredLines.length === 0 ? (
            <div className="text-slate-700 italic p-2">No device output yet…</div>
          ) : filteredLines.map((l) => (
            <div key={l.id} className={clsx(
              'whitespace-pre-wrap break-all border-l-2 pl-2',
              l.level === 'error' ? 'border-rose-500 text-rose-400'
                : l.level === 'system' ? 'border-emerald-600 text-emerald-400'
                : l.source === 'flash' ? 'border-amber-800 text-amber-300/80'
                : l.source === 'pi' ? 'border-sky-800 text-sky-300/80'
                : l.source.startsWith('script:') ? 'border-indigo-800 text-indigo-300/80'
                : 'border-slate-800 text-slate-400',
            )}>
              <span className="text-slate-600">[{l.source}]</span> <Ansi>{l.line}</Ansi>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
