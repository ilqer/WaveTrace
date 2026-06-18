import { useState, useCallback, useRef } from 'react';

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

// Matches backend log messages that signal a pipeline has finished.
const DONE_PATTERN = /complete|ended|saved|stopped|FATAL|ERROR/i;

export function useWaveTrace() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [verdict, setVerdict] = useState<InferenceResult | null>(null);
  const [alertActive, setAlertActive] = useState(false);
  const [driftRatio, setDriftRatio] = useState<number | null>(null);
  const [varianceData, setVarianceData] = useState<{t: number[], v: number[]}>({ t: [], v: [] });
  const [antennas, setAntennas] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [spectrogram, setSpectrogram] = useState<number[][] | null>(null);
  const [spatial, setSpatial] = useState<StreamData['spatial'] | null>(null);
  const [heatmapGrid, setHeatmapGrid] = useState<number[] | null>(null);
  const [gridSize, setGridSize] = useState<number>(16);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [trainingMeta, setTrainingMeta] = useState<TrainingMeta | null>(null);
  const [trainingResult, setTrainingResult] = useState<Record<string, any> | null>(null);
  const [camUrl, setCamUrl] = useState<string>('');

  const wsStream = useRef<WebSocket | null>(null);
  const wsInference = useRef<WebSocket | null>(null);
  const wsLogs = useRef<WebSocket | null>(null);
  const wsTraining = useRef<WebSocket | null>(null);

  const HISTORY_LEN = 300;

  const addLog = useCallback((msg: string) => {
    setLogs(prev => [...prev.slice(-100), msg]);
  }, []);

  const connect = useCallback(() => {
    const wsUrl = `ws://${window.location.host}`;

    let streamOpen = false;
    let inferenceOpen = false;
    let logsOpen = false;

    const checkReady = () => {
      if (streamOpen && inferenceOpen && logsOpen) setIsConnected(true);
    };

    wsStream.current = new WebSocket(`${wsUrl}/ws/stream`);
    wsStream.current.onopen = () => { streamOpen = true; checkReady(); };
    wsStream.current.onclose = () => { streamOpen = false; setIsConnected(false); };
    wsStream.current.onmessage = (event) => {
      const data: StreamData = JSON.parse(event.data);
      if (data.error) { addLog(`ERROR: ${data.error}`); return; }
      if (data.ic) {
        const val = data.ic.reduce((a, b) => a + Math.abs(b), 0) / data.ic.length;
        setVarianceData(prev => {
          const t = [...prev.t, prev.t.length].slice(-HISTORY_LEN);  // sequential index, not raw timestamp
          const v = [...prev.v, val].slice(-HISTORY_LEN);
          return { t, v };
        });
      }
      if (data.image) setSpectrogram(data.image);
      if (data.antennas) setAntennas(data.antennas);
      if (data.spatial) setSpatial(data.spatial);
      if (data.heatmap_grid) setHeatmapGrid(data.heatmap_grid);
      if (data.grid_size) setGridSize(data.grid_size);
    };

    wsInference.current = new WebSocket(`${wsUrl}/ws/inference`);
    wsInference.current.onopen = () => { inferenceOpen = true; checkReady(); };
    wsInference.current.onclose = () => { inferenceOpen = false; };
    wsInference.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // The inference socket carries both verdicts and control events; only verdicts update verdict.
      if (data.event === 'weapon_alert') { setAlertActive(true); return; }
      if (data.event === 'clear') { setAlertActive(false); return; }
      if (data.event === 'recalibrate_advisory') { setDriftRatio(data.drift ?? null); return; }
      setVerdict(data as InferenceResult);
    };

    wsLogs.current = new WebSocket(`${wsUrl}/ws/logs`);
    wsLogs.current.onopen = () => { logsOpen = true; checkReady(); };
    wsLogs.current.onclose = () => { logsOpen = false; };
    wsLogs.current.onmessage = (event) => {
      const msg: string = event.data;
      addLog(msg);
      // Any completion/error line from the backend resets the running state.
      if (DONE_PATTERN.test(msg)) setIsRunning(false);
    };

    wsTraining.current = new WebSocket(`${wsUrl}/ws/training`);
    wsTraining.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'train_init') {
        setTrainingMeta({ n_samples: data.n_samples, distribution: data.distribution });
      } else if (data.type === 'epoch') {
        // Only epoch messages carry loss/accuracy — push them to the chart.
        setTrainingMetrics(prev => [...prev, data]);
      } else if (data.type === 'done') {
        setTrainingResult(data.metrics ?? null);
        setIsRunning(false);
      }
    };
  }, [addLog]);

  const disconnect = useCallback(() => {
    wsStream.current?.close();
    wsInference.current?.close();
    wsLogs.current?.close();
    wsTraining.current?.close();
    setIsConnected(false);
  }, []);

  const start = useCallback(async (payload: any) => {
    if (!isConnected) connect();
    if (payload.cam_url) setCamUrl(payload.cam_url);
    if (payload.action === 'train') {
      setTrainingMetrics([]);
      setTrainingMeta(null);
      setTrainingResult(null);
    }
    setAlertActive(false);
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
    alertActive,
    driftRatio,
    varianceData,
    antennas,
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
