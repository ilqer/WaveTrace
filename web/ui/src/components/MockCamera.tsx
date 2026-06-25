import React, { useState, useCallback, useEffect, useRef } from 'react';
import { VideoOff, RefreshCw, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';

interface CameraFeedProps {
  camUrl: string;
  label: string;
  isActive: boolean;
}

const MockCamera: React.FC<CameraFeedProps> = ({ camUrl, label, isActive }) => {
  const [error, setError]   = useState(false);
  const [key,   setKey]     = useState(0);
  const retryTimer          = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear any pending auto-retry timer when the URL changes
  useEffect(() => {
    setError(false);
    setKey(k => k + 1);
    return () => { if (retryTimer.current) clearTimeout(retryTimer.current); };
  }, [camUrl]);

  const handleError = useCallback(() => {
    setError(true);
    // Auto-retry after 5 s — covers the case where the camera is briefly busy
    // (e.g. another ffmpeg process is releasing it) without requiring user action.
    retryTimer.current = setTimeout(() => {
      setError(false);
      setKey(k => k + 1);
    }, 5000);
  }, []);

  const handleLoad  = useCallback(() => setError(false), []);

  const retry = useCallback(() => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
    setError(false);
    setKey(k => k + 1);
  }, []);

  return (
    <div className="relative w-full h-full bg-slate-950 rounded-lg overflow-hidden">
      {/* Offline overlay */}
      {!camUrl && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 text-slate-600 bg-slate-950">
          <VideoOff size={24} />
          <span className="text-xs">No camera URL set</span>
        </div>
      )}

      {/* The <img> must always exist in the DOM. If we unmount it while it's downloading
          an MJPEG stream, Chrome sometimes leaks the connection. By keeping it mounted
          and setting src to "about:blank", we force the browser to explicitly abort the stream. */}
      <img
        key={key}
        src={camUrl || 'about:blank'}
        className={`w-full h-full object-contain transition-opacity duration-300 ${(!camUrl || error) ? 'opacity-0 absolute inset-0' : 'opacity-100'}`}
        alt="Camera feed"
        onError={camUrl ? handleError : undefined}
        onLoad={camUrl ? handleLoad : undefined}
      />

      {/* Error / retrying overlay */}
      {camUrl && error && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 text-slate-500 bg-slate-950">
          <AlertCircle size={20} className="text-amber-500/70" />
          <span className="text-[10px] text-center px-3 text-slate-600">
            Camera stream unavailable — retrying…
          </span>
          <button
            onClick={retry}
            className="flex items-center gap-1 mt-1 px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-400 hover:text-slate-100 transition-colors"
          >
            <RefreshCw size={9} /> Retry now
          </button>
        </div>
      )}

      {/* Class label badge — top-left, only when inference is active */}
      {camUrl && !error && isActive && label && (
        <div className={clsx(
          'absolute top-1.5 left-1.5 z-20 text-[10px] font-bold px-2 py-0.5 rounded backdrop-blur-sm',
          label.includes('Weapon') ? 'bg-rose-600/80 text-white' : 'bg-sky-600/80 text-white'
        )}>
          {label}
        </div>
      )}

      {/* LIVE badge — bottom-left */}
      {camUrl && !error && (
        <div className="absolute bottom-1.5 left-1.5 z-20 text-[10px] font-mono text-slate-500 bg-slate-900/60 px-1.5 py-0.5 rounded">
          LIVE
        </div>
      )}
    </div>
  );
};

export default MockCamera;
