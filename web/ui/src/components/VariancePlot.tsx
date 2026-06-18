import React, { useRef, useEffect } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface VariancePlotProps {
  data: { t: number[], v: number[] };
}

const VariancePlot: React.FC<VariancePlotProps> = ({ data }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const opts: uPlot.Options = {
      width: containerRef.current.clientWidth,
      height: 200,
      scales: { x: { time: false }, y: { min: 0 } },
      series: [
        {},
        {
          stroke: "#10b981",
          width: 2,
          label: "Variance"
        }
      ],
      axes: [
        { stroke: "#94a3b8", grid: { stroke: "#334155" } },
        { stroke: "#94a3b8", grid: { stroke: "#334155" } }
      ]
    };

    plotRef.current = new uPlot(opts, [data.t, data.v], containerRef.current);

    const handleResize = () => {
      if (plotRef.current && containerRef.current) {
        plotRef.current.setSize({
          width: containerRef.current.clientWidth,
          height: 200
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      plotRef.current?.destroy();
    };
  }, []);

  useEffect(() => {
    if (plotRef.current) {
      plotRef.current.setData([data.t, data.v]);
    }
  }, [data]);

  return <div ref={containerRef} className="w-full h-[200px]" />;
};

export default VariancePlot;
