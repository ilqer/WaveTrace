import React from 'react';
import { VideoOff } from 'lucide-react';

interface CameraFeedProps {
  camUrl: string;
  label: string;
  isActive: boolean;
}

// Renders a real MJPEG camera stream. MJPEG servers respond to a plain <img> src request.
const MockCamera: React.FC<CameraFeedProps> = ({ camUrl, label, isActive }) => {
  if (!camUrl) {
    return (
      <div className="w-full h-full bg-slate-950 rounded-lg flex flex-col items-center justify-center gap-2 text-slate-600">
        <VideoOff size={24} />
        <span className="text-xs">No camera URL set</span>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-slate-950 rounded-lg overflow-hidden">
      <img
        src={camUrl}
        className="w-full h-full object-contain"
        alt="Camera feed"
        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
      />
      {isActive && label && (
        <div className={`absolute top-1.5 left-1.5 text-[10px] font-bold px-2 py-0.5 rounded backdrop-blur-sm ${
          label.includes('Weapon') ? 'bg-rose-600/80 text-white' : 'bg-sky-600/80 text-white'
        }`}>
          {label}
        </div>
      )}
      {!isActive && (
        <div className="absolute bottom-1.5 left-1.5 text-[10px] font-mono text-slate-500 bg-slate-900/60 px-1.5 py-0.5 rounded">
          LIVE
        </div>
      )}
    </div>
  );
};

export default MockCamera;
