import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Dot,
} from 'recharts';

export default function CalorieTrendChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-400 text-sm">
        No calorie data yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: '#9ca3af', fontFamily: 'DM Mono, monospace' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(val) => {
            const d = new Date(val);
            return `${d.getDate()}/${d.getMonth() + 1}`;
          }}
        />
        <YAxis
          tick={{ fontSize: 11, fill: '#9ca3af', fontFamily: 'DM Mono, monospace' }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <Tooltip
          contentStyle={{
            fontSize: '12px',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            fontFamily: 'DM Sans, sans-serif',
          }}
          formatter={(value) => [`${value} kcal`, 'Calories']}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Line
          type="monotone"
          dataKey="calories"
          stroke="#4a7c59"
          strokeWidth={2}
          dot={{ r: 4, fill: '#4a7c59', strokeWidth: 2, stroke: '#fff' }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
