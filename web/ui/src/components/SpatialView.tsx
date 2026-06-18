import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { SubcarrierManifold } from './SubcarrierManifold';
import { OccupancyGrid3D } from './OccupancyGrid3D';
import { Suspense } from 'react';

interface SpatialViewProps {
  spectrogramData: number[][] | null;
  heatmapGrid?: number[] | null;
  gridSize?: number;
  persons?: { position: [number, number, number] }[];
  mode: 'manifold' | 'heatmap';
}

export function SpatialView({ spectrogramData, heatmapGrid, gridSize = 16, mode }: SpatialViewProps) {
  return (
    <div className="w-full h-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800 relative">
      <div className="absolute top-2 left-2 z-10">
        <div className="bg-slate-900/80 backdrop-blur px-2 py-1 rounded text-[10px] font-bold text-emerald-400 border border-emerald-500/30">
          3D {mode === 'manifold' ? 'SIGNAL MANIFOLD' : 'SPATIAL HEATMAP'}
        </div>
      </div>

      <Canvas gl={{ antialias: true, powerPreference: 'high-performance' }}>
        <PerspectiveCamera makeDefault position={[0, 6, 12]} fov={40} />
        <OrbitControls enableDamping dampingFactor={0.05} minDistance={2} maxDistance={40} />
        <ambientLight intensity={0.5} />
        <pointLight position={[5, 8, 5]} intensity={1.2} />
        <Suspense fallback={null}>
          {mode === 'manifold' && (
            <SubcarrierManifold data={spectrogramData} />
          )}
          {mode === 'heatmap' && heatmapGrid && heatmapGrid.length > 0 ? (
            <OccupancyGrid3D grid={heatmapGrid} G={gridSize} />
          ) : mode === 'heatmap' ? (
            <mesh>
              <boxGeometry args={[0.4, 0.4, 0.4]} />
              <meshStandardMaterial color="#334155" wireframe />
            </mesh>
          ) : null}
        </Suspense>
      </Canvas>
    </div>
  );
}
