import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const COLORS = ['#4a7c59', '#94b8a0', '#c8ddd0', '#e8f0eb'];

export default function MacroPieChart({ calories, protein, carbohydrates, fat }) {
  const items = [
    { name: 'Calories', value: calories, unit: 'kcal' },
    { name: 'Protein', value: protein, unit: 'g' },
    { name: 'Carbs', value: carbohydrates, unit: 'g' },
    { name: 'Fat', value: fat, unit: 'g' },
  ];

  const data = items
    .filter(item => item.value !== null && item.value !== undefined && item.value > 0)
    .map(item => ({ name: item.name, value: item.value }));

  if (data.length === 0) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-400 text-sm">
        No macro data available
      </div>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            dataKey="value"
            strokeWidth={2}
            stroke="#fff"
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value, name) => [
              `${value}${items.find(i => i.name === name)?.unit || ''}`,
              name,
            ]}
            contentStyle={{ fontSize: '12px', border: '1px solid #e5e7eb', borderRadius: '8px' }}
          />
        </PieChart>
      </ResponsiveContainer>
      {/* Custom legend */}
      <div className="flex flex-wrap justify-center gap-3 mt-1">
        {items.map((item, index) => (
          <div key={item.name} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: COLORS[index % COLORS.length] }}
            />
            <span className="text-xs text-gray-600">{item.name}</span>
            <span className="text-xs font-mono text-gray-800 font-medium">
              {item.value !== null && item.value !== undefined ? `${item.value}${item.unit}` : 'N/A'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
