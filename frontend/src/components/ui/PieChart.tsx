import React from 'react';

interface PieChartProps {
  data: Record<string, number>;
}

export const PieChart: React.FC<PieChartProps> = ({ data }) => {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  const total = entries.reduce((sum, [, val]) => sum + Number(val), 0);
  if (total === 0) return null;

  const colors = [
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // purple
    '#06b6d4', // cyan
    '#d946ef', // fuchsia
    '#84cc16', // lime
  ];

  let currentAngle = 0;
  const radius = 50;
  const cx = 50;
  const cy = 50;

  return (
    <div className="flex flex-col items-center gap-5 my-5 p-5 rounded-xl border border-line bg-surface-2/70 shadow-sm">
      <svg viewBox="0 0 100 100" className="w-40 h-40 overflow-visible transform -rotate-90">
        {entries.map(([key, value], i) => {
          const numValue = Number(value);
          if (numValue === 0) return null;
          
          const sliceAngle = (numValue / total) * 360;
          const startAngle = currentAngle;
          currentAngle += sliceAngle;

          const startX = cx + radius * Math.cos((Math.PI * startAngle) / 180);
          const startY = cy + radius * Math.sin((Math.PI * startAngle) / 180);
          const endX = cx + radius * Math.cos((Math.PI * currentAngle) / 180);
          const endY = cy + radius * Math.sin((Math.PI * currentAngle) / 180);

          const largeArcFlag = sliceAngle > 180 ? 1 : 0;

          // If it's a full circle, just draw a circle element or use two arcs
          if (sliceAngle >= 359.9) {
            return (
              <circle
                key={key}
                cx={cx}
                cy={cy}
                r={radius}
                fill={colors[i % colors.length]}
                className="hover:opacity-85 transition-opacity"
              />
            );
          }

          const pathData = `M ${cx} ${cy} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY} Z`;

          return (
            <path
              key={key}
              d={pathData}
              fill={colors[i % colors.length]}
              className="hover:opacity-85 transition-opacity"
            />
          );
        })}
      </svg>
      <div className="flex flex-wrap justify-center gap-3.5 text-[13px] text-ink">
        {entries.map(([key, value], i) => {
          const numValue = Number(value);
          if (numValue === 0) return null;
          return (
            <div key={key} className="flex items-center gap-2 font-medium">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: colors[i % colors.length] }}
                aria-hidden
              />
              <span>
                {key} <span className="text-ink-faint font-normal">({(numValue / total * 100).toFixed(1)}%)</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
