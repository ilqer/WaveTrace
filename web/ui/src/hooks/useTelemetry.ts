import { useEffect, useRef, useState } from 'react';

export interface NodeHealth {
  node_id: number;
  band: string;
  hz: number;
  hz_ok: boolean;
  frames: number;
  mean_amp: number;
  snr_db: number;
  cv: number;
  gain_drift: number | null;
  subcarriers: number[];
  last_ts: number;
}

export interface SyncInfo {
  spread_s: number;
  ok: boolean;
}

export interface Telemetry {
  nodes: NodeHealth[];
  sync: SyncInfo;
  antenna_weights?: number[];
  fusion?: { bands: string[]; weights: number[] };
  contribution?: Record<string, number>;
  heatmap?: number[];
  grid?: number;
  alert_active?: boolean;
  drift_ratio?: number;
  voter_trace?: number[];
}

export function useTelemetry(
  url = `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost'}/ws/telemetry`
) {
  const [tel, setTel] = useState<Telemetry | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const s = new WebSocket(url);
    ws.current = s;
    s.onmessage = (e) => {
      try { setTel(JSON.parse(e.data)); } catch {}
    };
    s.onerror = () => {};
    return () => { s.close(); };
  }, [url]);

  return tel;
}
