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
  val_loss: number;
  accuracy: number;
  val_accuracy: number;
}

export interface TrainingMeta {
  n_samples: number;
  distribution: Record<string, number>;
}

export interface StartPayload {
  action: 'run' | 'calib' | 'collect' | 'train';
  synthetic?: boolean;
  duration?: number;
  antennas?: number;
  subcarriers?: number;
  fs?: number;
  udp_port?: number;
  cam_url?: string;
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
  const [camUrl, setCamUrl] = useState<string>('');

  const wsStream = useRef<WebSocket | null>(null);
  const wsInference = useRef<WebSocket | null>(null);
  const wsLogs = useRef<WebSocket | null>(null);
  const wsTraining = useRef<WebSocket | null>(null);

  // Reconnect state — exponential backoff capped at 30 s
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(2000);
  // Stable ref so onclose handlers can always call the latest connect()
  const connectRef = useRef<() => void>(() => {});

  const addLog = useCallback((msg: string) => {
    const id = ++_logSeq.n;
    setLogs(prev => [...prev.slice(-100), { id, text: msg }]);
  }, []);

  const connect = useCallback(() => {
    // Close any stale sockets before opening new ones
    wsStream.current?.close();
    wsInference.current?.close();
    wsLogs.current?.close();
    wsTraining.current?.close();

    const wsUrl = `ws://${window.location.host}`;
    let streamOpen = false;
    let inferenceOpen = false;
    let logsOpen = false;

    const checkReady = () => {
      if (streamOpen && inferenceOpen && logsOpen) {
        setIsConnected(true);
        reconnectDelay.current = 2000; // reset on successful connect
      }
    };

    const scheduleReconnect = () => {
      if (reconnectTimer.current) return; // already pending
      reconnectTimer.current = setTimeout(() => {
        reconnectTimer.current = null;
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
        connectRef.current();
      }, reconnectDelay.current);
    };

    wsStream.current = new WebSocket(`${wsUrl}/ws/stream`);
    wsStream.current.onopen = () => { streamOpen = true; checkReady(); };
    wsStream.current.onclose = () => { streamOpen = false; setIsConnected(false); scheduleReconnect(); };
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
    wsInference.current.onclose = () => { inferenceOpen = false; scheduleReconnect(); };
    wsInference.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Structured control events from the backend — no log-scraping.
      if (data.event === 'pipeline_done') { setIsRunning(false); return; }
      if (data.event === 'weapon_alert') return;   // consumed by DiagnosticsPanel via telemetry
      if (data.event === 'clear') return;
      if (data.event === 'recalibrate_advisory') { setDriftRatio(data.drift ?? null); return; }
      setVerdict(data as InferenceResult);
    };

    wsLogs.current = new WebSocket(`${wsUrl}/ws/logs`);
    wsLogs.current.onopen = () => { logsOpen = true; checkReady(); };
    wsLogs.current.onclose = () => { logsOpen = false; scheduleReconnect(); };
    wsLogs.current.onmessage = (event) => {
      addLog(event.data as string);
    };

    wsTraining.current = new WebSocket(`${wsUrl}/ws/training`);
    wsTraining.current.onmessage = (event) => {
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
  }, [addLog]);

  // Keep connectRef in sync so reconnect closures call the latest version
  useEffect(() => { connectRef.current = connect; }, [connect]);

  useEffect(() => {
    fetch('/api/pipeline/state')
      .then(r => r.json())
      .then(d => {
        setIsRunning(d.isRunning ?? false);
        if (d.isRunning) connect();
      })
      .catch(e => console.error(e));
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
    await fetch('/api/action/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }, [isConnected, connect, addLog]);

  const stop = useCallback(async () => {
    addLog('[SYSTEM] Requesting pipeline stop...');
    await fetch('/api/action/stop', { method: 'POST' });
    disconnect();
    setIsRunning(false);
    addLog('[SYSTEM] Pipeline stopped.');
  }, [disconnect, addLog]);

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
    addLog,
  };
}
