import { useState, useCallback, useRef, useEffect } from 'react';

export interface InferenceResult {
  mode: 'presence' | 'weapon';
  class: number;
  conf: number;
  t: number;
  pos?: [number, number, number];
}

export interface StreamData {
  t: number;
  ic?: number[];
  image?: number[][];
  antennas?: number[];
  node_ids?: number[];
  spatial?: {
    x: number;
    y: number;
    conf: number;
    heatmap: number[];
  };
  heatmap_grid?: number[];
  grid_size?: number;
  error?: string;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  loss_std?: number;       // within-epoch batch-loss spread → curve confidence band
  accuracy: number;
  val_loss?: number;       // optional: the cnn head trains without a held-out val split
  val_accuracy?: number;
}

export interface TrainingMeta {
  n_samples: number;
  distribution: Record<string, number>;
}

export interface StartPayload {
  action: 'run' | 'calib' | 'collect' | 'train' | 'camera_collect';
  synthetic?: boolean;
  duration?: number;
  antennas?: number;
  subcarriers?: number;
  fs?: number;
  udp_port?: number;
  cam_url?: string;
  cam_index?: number;
  mode?: string;
  calibration?: string;
  model?: string;
  gain_lock?: boolean;
  vote?: boolean;
  frame_average?: number;
  use_baseline?: boolean;
  baseline_packets?: number;
  cal_out?: string;
  col_stage?: string;
  col_spans?: string;
  col_window?: number;
  col_hop?: number;
  subtract_ic_baseline?: boolean;
  train_backend?: string;
  train_out?: string;
  train_data?: string;
  per_link?: boolean;
  yolo_weights?: string;
}

const HISTORY_LEN = 300;

export interface LogLine { id: number; text: string; }

const _logSeq = { n: 0 };

export function useWaveTrace() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [verdict, setVerdict] = useState<InferenceResult | null>(null);
  const [driftRatio, setDriftRatio] = useState<number | null>(null);
  const [varianceData, setVarianceData] = useState<{t: number[], v: number[]}>({ t: [], v: [] });
  const [antennas, setAntennas] = useState<number[]>([]);
  const [nodeIds, setNodeIds] = useState<number[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [spectrogram, setSpectrogram] = useState<number[][] | null>(null);
  const [spatial, setSpatial] = useState<StreamData['spatial'] | null>(null);
  const [heatmapGrid, setHeatmapGrid] = useState<number[] | null>(null);
  const [gridSize, setGridSize] = useState<number>(16);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [trainingMeta, setTrainingMeta] = useState<TrainingMeta | null>(null);
  const [trainingResult, setTrainingResult] = useState<Record<string, unknown> | null>(null);
  const [camUrl, setCamUrl] = useState<string>('/api/camera/stream');

  const wsStream = useRef<WebSocket | null>(null);
  const wsInference = useRef<WebSocket | null>(null);
  const wsLogs = useRef<WebSocket | null>(null);
  const wsTraining = useRef<WebSocket | null>(null);

  // Each connect() call bumps this. onclose handlers capture their generation at
  // creation time and bail if the counter has moved on — meaning the socket was
  // intentionally closed by a later connect() or by unmount cleanup.
  // This replaces the old closingRef+setTimeout(0) hack, which was racy because
  // WebSocket onclose fires as a macrotask AFTER setTimeout(0) already reset the flag.
  const generation = useRef(0);

  // Reconnect state — exponential backoff capped at 30 s
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(2000);
  // Independent reconnect for training WS — backend closes it when idle so we
  // can't use the full connect() path (that would disrupt the live stream).
  const trainingReconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const trainingReconnectDelay = useRef(2000);
  // Debounce disconnect so brief WebSocket cycles don't flash the disconnected UI
  const disconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Stable ref so onclose handlers can always call the latest connect()
  const connectRef = useRef<() => void>(() => {});

  const addLog = useCallback((msg: string) => {
    const id = ++_logSeq.n;
    setLogs(prev => [...prev.slice(-100), { id, text: msg }]);
  }, []);

  const connect = useCallback(() => {
    const wsUrl = `ws://${window.location.host}`;
    // Bump generation — any onclose from a prior generation is stale and ignored.
    const gen = ++generation.current;
    const isStale = () => generation.current !== gen;

    let streamOpen = false;
    let inferenceOpen = false;
    let logsOpen = false;

    const scheduleDisconnect = () => {
      if (isStale()) return;
      if (disconnectTimer.current) return;
      disconnectTimer.current = setTimeout(() => {
        disconnectTimer.current = null;
        if (!isStale()) setIsConnected(false);
      }, 6000);
    };

    const checkReady = () => {
      if (isStale()) return;
      if (streamOpen && inferenceOpen && logsOpen) {
        if (disconnectTimer.current) { clearTimeout(disconnectTimer.current); disconnectTimer.current = null; }
        setIsConnected(true);
        reconnectDelay.current = 2000;
      }
    };

    const scheduleReconnect = () => {
      if (isStale()) return;
      if (reconnectTimer.current) return;
      reconnectTimer.current = setTimeout(() => {
        reconnectTimer.current = null;
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
        connectRef.current();
      }, reconnectDelay.current);
    };

    // Close old sockets. Their onclose will fire later but isStale() returns true
    // (generation already bumped above), so scheduleReconnect is never called.
    if (trainingReconnectTimer.current) { clearTimeout(trainingReconnectTimer.current); trainingReconnectTimer.current = null; }
    wsStream.current?.close();
    wsInference.current?.close();
    wsLogs.current?.close();
    wsTraining.current?.close();

    wsStream.current = new WebSocket(`${wsUrl}/ws/stream`);
    wsStream.current.onopen = () => { streamOpen = true; checkReady(); };
    wsStream.current.onclose = () => { if (isStale()) return; streamOpen = false; scheduleDisconnect(); scheduleReconnect(); };
    wsStream.current.onmessage = (event) => {
      const data: StreamData = JSON.parse(event.data);
      if (data.error) { addLog(`ERROR: ${data.error}`); return; }
      if (data.ic) {
        const val = data.ic.reduce((a, b) => a + Math.abs(b), 0) / data.ic.length;
        setVarianceData(prev => {
          const t = [...prev.t, prev.t.length].slice(-HISTORY_LEN);
          const v = [...prev.v, val].slice(-HISTORY_LEN);
          return { t, v };
        });
      }
      if (data.image) setSpectrogram(data.image);
      if (data.antennas) setAntennas(data.antennas);
      if (data.node_ids) setNodeIds(data.node_ids);
      if (data.spatial) setSpatial(data.spatial);
      if (data.heatmap_grid) setHeatmapGrid(data.heatmap_grid);
      if (data.grid_size) setGridSize(data.grid_size);
    };

    wsInference.current = new WebSocket(`${wsUrl}/ws/inference`);
    wsInference.current.onopen = () => { inferenceOpen = true; checkReady(); };
    wsInference.current.onclose = () => { if (isStale()) return; inferenceOpen = false; scheduleReconnect(); };
    wsInference.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.event === 'pipeline_done') { setIsRunning(false); return; }
      if (data.event === 'weapon_alert') return;
      if (data.event === 'clear') return;
      if (data.event === 'recalibrate_advisory') { setDriftRatio(data.drift ?? null); return; }
      setVerdict(data as InferenceResult);
    };

    wsLogs.current = new WebSocket(`${wsUrl}/ws/logs`);
    wsLogs.current.onopen = () => { logsOpen = true; checkReady(); };
    wsLogs.current.onclose = () => { if (isStale()) return; logsOpen = false; scheduleReconnect(); };
    wsLogs.current.onmessage = (event) => {
      addLog(event.data as string);
    };

    // Training WS reconnects independently — backend closes it when idle.
    const openTrainingWs = () => {
      const ws = new WebSocket(`${wsUrl}/ws/training`);
      wsTraining.current = ws;
      ws.onopen = () => { if (!isStale()) trainingReconnectDelay.current = 2000; };
      ws.onclose = () => {
        if (isStale()) return;
        if (wsTraining.current !== ws) return; // superseded by a newer openTrainingWs call
        if (trainingReconnectTimer.current) return;
        trainingReconnectTimer.current = setTimeout(() => {
          trainingReconnectTimer.current = null;
          if (isStale()) return;
          trainingReconnectDelay.current = Math.min(trainingReconnectDelay.current * 2, 30000);
          openTrainingWs();
        }, trainingReconnectDelay.current);
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'train_init') {
          setTrainingMeta({ n_samples: data.n_samples, distribution: data.distribution });
        } else if (data.type === 'epoch') {
          setTrainingMetrics(prev => [...prev, data]);
        } else if (data.type === 'done') {
          setTrainingResult(data.metrics ?? null);
          setIsRunning(false);
        }
      };
    };
    openTrainingWs();
  }, [addLog]);

  // Keep connectRef in sync so reconnect closures call the latest version
  useEffect(() => { connectRef.current = connect; }, [connect]);

  useEffect(() => {
    fetch('/api/pipeline/state')
      .then(r => r.json())
      .then(d => { setIsRunning(d.isRunning ?? false); })
      .catch(e => console.error(e));
    connect();

    // Cleanup: bump generation to invalidate all pending onclose/timer handlers,
    // then close sockets and cancel timers.
    return () => {
      ++generation.current;
      if (reconnectTimer.current) { clearTimeout(reconnectTimer.current); reconnectTimer.current = null; }
      if (disconnectTimer.current) { clearTimeout(disconnectTimer.current); disconnectTimer.current = null; }
      if (trainingReconnectTimer.current) { clearTimeout(trainingReconnectTimer.current); trainingReconnectTimer.current = null; }
      wsStream.current?.close();
      wsInference.current?.close();
      wsLogs.current?.close();
      wsTraining.current?.close();
    };
  }, [connect]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) { clearTimeout(reconnectTimer.current); reconnectTimer.current = null; }
    wsStream.current?.close();
    wsInference.current?.close();
    wsLogs.current?.close();
    wsTraining.current?.close();
    setIsConnected(false);
  }, []);

  const start = useCallback(async (payload: StartPayload) => {
    if (!isConnected) connect();
    if (payload.cam_url) setCamUrl(payload.cam_url);
    if (payload.action === 'train') {
      setTrainingMetrics([]);
      setTrainingMeta(null);
      setTrainingResult(null);
    }
    setDriftRatio(null);
    setIsRunning(true);
    addLog(`[SYSTEM] Triggering action: ${payload.action.toUpperCase()}`);
    try {
      const res = await fetch('/api/action/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        setIsRunning(false);
        addLog(`[ERROR] Start failed: ${(err as any).error ?? res.statusText}`);
      }
    } catch {
      setIsRunning(false);
      addLog(`[ERROR] Could not reach server — is the backend running?`);
    }
  }, [isConnected, connect, addLog]);

  const stop = useCallback(async () => {
    addLog('[SYSTEM] Requesting pipeline stop...');
    await fetch('/api/action/stop', { method: 'POST' });
    setIsRunning(false);
    addLog('[SYSTEM] Pipeline stopped.');
  }, [addLog]);

  return {
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
    disconnect,
    addLog,
  };
}
