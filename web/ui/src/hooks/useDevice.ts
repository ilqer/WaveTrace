import { useState, useCallback, useEffect, useRef } from 'react';

export interface DeviceLine {
  source: 'monitor' | 'flash' | 'pi' | 'script';
  line: string;
  level: 'info' | 'system' | 'error';
  t: number;
}

export interface SerialPort {
  device: string;
  description: string;
  hwid: string;
  likely_esp: boolean;
}

// Hardware control: serial discovery + monitor, flashing, Pi capture. Streams from /ws/device,
// which is independent of the inference pipeline sockets so it stays live across run/stop.
export function useDevice() {
  const [lines, setLines] = useState<DeviceLine[]>([]);
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [monitoring, setMonitoring] = useState<string[]>([]);
  const [runningScripts, setRunningScripts] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const sock = new WebSocket(`ws://${window.location.host}/ws/device`);
    sock.onmessage = (e) => {
      const d: DeviceLine = JSON.parse(e.data);
      setLines((prev) => {
        let sourceCount = 0;
        let oldestIdx = -1;
        for (let i = 0; i < prev.length; i++) {
          if (prev[i].source === d.source) {
            sourceCount++;
            if (oldestIdx === -1) oldestIdx = i;
          }
        }
        if (sourceCount >= 1000) {
          const next = [...prev];
          next.splice(oldestIdx, 1);
          next.push(d);
          return next;
        }
        return [...prev, d];
      });
      if (d.source === 'flash' && d.line.startsWith('exit code')) setBusy(false);
      if (d.source.startsWith('script:') && d.line.startsWith('exit code')) {
        const script = d.source.split(':')[1];
        setRunningScripts(prev => prev.filter(s => s !== script));
      }
    };
    ws.current = sock;
    return () => sock.close();
  }, []);

  const refreshPorts = useCallback(async () => {
    const r = await fetch('/api/serial/ports');
    const d = await r.json();
    setPorts(d.ports ?? []);
  }, []);

  const refreshState = useCallback(async () => {
    const r = await fetch('/api/device/state');
    const d = await r.json();
    setMonitoring(d.monitoring ?? []);
    setRunningScripts(d.runningScripts ?? []);
    setBusy(d.busy ?? false);
  }, []);

  useEffect(() => { 
    refreshPorts(); 
    refreshState();
  }, [refreshPorts, refreshState]);

  const startMonitor = useCallback(async (port: string, baud = 115200) => {
    await fetch('/api/serial/monitor/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ port, baud }),
    });
    setMonitoring(prev => prev.includes(port) ? prev : [...prev, port]);
  }, []);

  const stopMonitor = useCallback(async (port?: string) => {
    await fetch('/api/serial/monitor/stop', { 
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ port: port || null }),
    });
    if (port) {
      setMonitoring(prev => prev.filter(p => p !== port));
    } else {
      setMonitoring([]);
    }
  }, []);

  const flash = useCallback(async (role: string, node_id: number | null, port: string, clean = false) => {
    setBusy(true);
    if (monitoring.includes(port)) setMonitoring(prev => prev.filter(p => p !== port));
    await fetch('/api/flash', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, node_id, port, clean }),
    });
  }, [monitoring]);

  const runPi = useCallback(async (host: string, command: string) => {
    await fetch('/api/pi/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host, command }),
    });
  }, []);

  const runScript = useCallback(async (script: string, args: string = "") => {
    await fetch('/api/script/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ script, args }),
    });
    setRunningScripts(prev => prev.includes(script) ? prev : [...prev, script]);
  }, []);

  const sendInput = useCallback(async (proc_id: string, input: string) => {
    await fetch('/api/device/input', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proc_id, input }),
    });
  }, []);

  const stopDevice = useCallback(async (proc_id?: string) => {
    await fetch('/api/device/stop', { 
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proc_id: proc_id || null })
    });
    if (proc_id && proc_id.startsWith('script:')) {
      const script = proc_id.split(':')[1];
      setRunningScripts(prev => prev.filter(s => s !== script));
    } else if (!proc_id) {
      setBusy(false);
    }
  }, []);

  const clearLines = useCallback(() => setLines([]), []);
  return {
    lines, ports, monitoring, busy, runningScripts,
    refreshPorts, startMonitor, stopMonitor, flash, runPi, runScript, stopDevice, clearLines, sendInput,
  };
}
