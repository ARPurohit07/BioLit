import "./BarChart.css";

export interface BarChartDatum {
  label: string;
  value: number;
}

interface BarChartProps {
  data: BarChartDatum[];
  unit?: string;
  formatValue?: (value: number) => string;
  height?: number;
  colorVar?: string;
}

const BAR_WIDTH = 64;
const BAR_GAP = 36;
const CHART_PADDING_TOP = 24;
const CHART_PADDING_BOTTOM = 36;

export default function BarChart({
  data,
  unit = "",
  formatValue,
  height = 180,
  colorVar = "--accent",
}: BarChartProps) {
  const max = Math.max(...data.map((d) => d.value), 1);
  const innerHeight = height - CHART_PADDING_TOP - CHART_PADDING_BOTTOM;
  const width = data.length * (BAR_WIDTH + BAR_GAP) + BAR_GAP;
  const fmt = formatValue ?? ((v: number) => v.toFixed(1));

  return (
    <svg
      className="bar-chart"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      role="img"
    >
      {data.map((d, idx) => {
        const barHeight = max > 0 ? (d.value / max) * innerHeight : 0;
        const x = BAR_GAP + idx * (BAR_WIDTH + BAR_GAP);
        const y = CHART_PADDING_TOP + (innerHeight - barHeight);
        return (
          <g key={d.label}>
            <text
              x={x + BAR_WIDTH / 2}
              y={CHART_PADDING_TOP - 8}
              textAnchor="middle"
              className="bar-chart-value"
            >
              {fmt(d.value)}
              {unit}
            </text>
            <rect
              x={x}
              y={y}
              width={BAR_WIDTH}
              height={Math.max(barHeight, 1)}
              rx={3}
              fill={`var(${colorVar})`}
            />
            <text
              x={x + BAR_WIDTH / 2}
              y={height - CHART_PADDING_BOTTOM + 20}
              textAnchor="middle"
              className="bar-chart-label"
            >
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
