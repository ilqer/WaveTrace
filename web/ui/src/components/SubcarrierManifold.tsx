import { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';

interface SubcarrierManifoldProps {
  data: number[][] | null;  // data[subcarrier][time_step], newest value last
}

const W = 10;            // world units across subcarrier axis
const D = 8;             // world units across time axis
const H_SCALE = 3.0;     // max height in world units

// CSI spectrogram as a 3D height field. Subcarriers on X, time on Z, amplitude on Y.
// Color: blue (quiet) → cyan → yellow (high amplitude). Standard material, no custom shaders.
export function SubcarrierManifold({ data }: SubcarrierManifoldProps) {
  const meshRef = useRef<THREE.Mesh>(null!);

  const subs  = data?.length       ?? 32;
  const slots = data?.[0]?.length  ?? 128;

  // Rebuild geometry only when dimensions change; pre-allocate color attribute.
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(W, D, subs - 1, slots - 1);
    g.rotateX(-Math.PI / 2);  // lay flat in XZ plane; Y becomes height
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(subs * slots * 3), 3));
    return g;
  }, [subs, slots]);

  // Dispose GPU geometry when it's replaced by a new one.
  useEffect(() => () => { geo.dispose(); }, [geo]);

  // Update vertex heights and colours whenever new data arrives.
  useEffect(() => {
    if (!data) return;
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const col = geo.attributes.color   as THREE.BufferAttribute;
    const c = new THREE.Color();

    for (let ti = 0; ti < slots; ti++) {
      for (let si = 0; si < subs; si++) {
        const idx = ti * subs + si;
        // ti=0 → newest window edge (front), ti=slots-1 → oldest (back)
        const val = Math.max(0, Math.min(1, data[si]?.[data[si].length - 1 - ti] ?? 0));
        pos.setY(idx, val * H_SCALE);
        // blue (0.6) → cyan (0.4) → yellow (0.1)
        c.setHSL(0.6 - val * 0.5, 1.0, 0.25 + val * 0.45);
        col.setXYZ(idx, c.r, c.g, c.b);
      }
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geo.computeVertexNormals();
  }, [data, geo, subs, slots]);

  return (
    <mesh ref={meshRef} geometry={geo} position={[0, -1.5, 0]}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.5}
        metalness={0.1}
      />
    </mesh>
  );
}
