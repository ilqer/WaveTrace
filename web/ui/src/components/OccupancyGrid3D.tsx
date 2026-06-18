import * as THREE from 'three';
import { useMemo } from 'react';

interface Props {
  grid: number[];
  G: number;
  cellSize?: number;
}

const BOX = new THREE.BoxGeometry(1, 1, 1);
const gap = 0.45;

export function OccupancyGrid3D({ grid, G, cellSize = 0.35 }: Props) {
  const cells = useMemo(() => {
    const out = [];
    for (let i = 0; i < G * G; i++) {
      const p = Math.max(0, Math.min(1, grid[i] ?? 0));
      const row = Math.floor(i / G);
      const col = i % G;
      const h = 0.4 + p * 2.2;
      const hue = Math.round((0.6 - p * 0.6) * 360);
      const light = Math.round(28 + p * 14);  // 28% (deep blue) → 42% (deep red)
      out.push({ i, p, row, col, h, color: `hsl(${hue},85%,${light}%)` });
    }
    return out;
  }, [grid, G]);

  const floorSize = G * gap;

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[floorSize, floorSize]} />
        <meshBasicMaterial color="#1e293b" />
      </mesh>
      {cells.map(({ i, row, col, h, color }) => (
        <mesh
          key={i}
          geometry={BOX}
          position={[(col - G / 2 + 0.5) * gap, h / 2, (row - G / 2 + 0.5) * gap]}
          scale={[cellSize, h, cellSize]}
        >
          <meshBasicMaterial color={color} />
        </mesh>
      ))}
    </group>
  );
}
