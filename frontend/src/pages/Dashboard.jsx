import { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import api from '../api/client';
import CalorieTrendChart from '../components/CalorieTrendChart';
import FilterBar from '../components/FilterBar';
import LogTable from '../components/LogTable';

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [logs, setLogs] = useState([]);
  const [filters, setFilters] = useState({});
  const [loading, setLoading] = useState(true);

  const fetchSummary = async () => {
    try {
      const res = await api.get('/logs/summary');
      setSummary(res.data);
    } catch (err) {
      console.error("Failed to fetch summary", err);
    }
  };

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, val]) => {
        if (val) params.append(key, val);
      });
      // For now, load everything (limit 1000) so client-side pagination works for the demo
      params.append('limit', 1000);

      const res = await api.get(`/logs?${params.toString()}`);
      setLogs(res.data.items);
    } catch (err) {
      console.error("Failed to fetch logs", err);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchSummary();
  }, []); // Summary once on mount

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]); // Refetch logs when filters change

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const handleClearFilters = () => {
    setFilters({});
  };

  const handleDelete = async (id) => {
    try {
      await api.delete(`/logs/${id}`);
      // Refresh both summary and logs
      fetchSummary();
      fetchLogs();
    } catch (err) {
      console.error("Failed to delete log", err);
      alert("Failed to delete log entry.");
    }
  };

  if (loading && !summary) {
    return <div className="py-20 text-center text-gray-500">Loading dashboard...</div>;
  }

  // Prep data for meal type bar chart
  const mealTypeData = summary ? [
    { name: 'Breakfast', count: summary.by_meal_type['Breakfast'] || 0 },
    { name: 'Lunch', count: summary.by_meal_type['Lunch'] || 0 },
    { name: 'Snacks', count: summary.by_meal_type['Snacks'] || 0 },
    { name: 'Dinner', count: summary.by_meal_type['Dinner'] || 0 },
  ] : [];

  const COLORS = ['#ea580c', '#16a34a', '#eab308', '#2563eb']; // Tailwind colors for meals

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 px-1">Dashboard</h1>

      {/* Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card shadow-sm bg-brand-50 border-brand-100">
          <div className="text-sm font-medium text-brand-700 uppercase tracking-wide mb-1">Total Meals</div>
          <div className="text-3xl font-mono text-brand-900">{summary?.total_entries || 0}</div>
        </div>
        <div className="card shadow-sm leading-tight">
          <div className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-1">Total Calories</div>
          <div className="text-3xl font-mono text-gray-900">
            {summary?.total_calories || 0} <span className="text-sm text-gray-500 font-sans normal-case">kcal</span>
          </div>
        </div>
        <div className="card shadow-sm">
          <div className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-1">Avg per Meal</div>
          <div className="text-3xl font-mono text-gray-900">
            {summary?.avg_calories_per_meal !== undefined ? Math.round(summary.avg_calories_per_meal) : 0} <span className="text-sm text-gray-500 font-sans normal-case">kcal</span>
          </div>
        </div>
        <div className="card shadow-sm">
          <div className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-1">Top Food</div>
          <div className="text-2xl font-semibold text-gray-900 capitalize truncate mt-1">
            {summary?.top_foods?.[0]?.food || '-'}
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card shadow-sm lg:col-span-2 flex flex-col h-[320px]">
          <h2 className="text-base font-semibold mb-6 text-gray-800">Calorie Trend</h2>
          <div className="flex-1 -ml-4">
            <CalorieTrendChart data={summary?.daily_calories || []} />
          </div>
        </div>
        
        <div className="card shadow-sm flex flex-col h-[320px]">
          <h2 className="text-base font-semibold mb-6 text-gray-800">Meals by Type</h2>
          <div className="flex-1 w-full h-full pb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mealTypeData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <XAxis 
                  dataKey="name" 
                  tick={{ fontSize: 11, fill: '#6b7280' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis 
                  tick={{ fontSize: 11, fill: '#6b7280' }}
                  axisLine={false}
                  tickLine={false}
                />
                <RechartsTooltip 
                  cursor={{fill: '#f3f4f6'}}
                  contentStyle={{ fontSize: '12px', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={40}>
                  {mealTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Logs Table Section */}
      <div className="pt-2">
        <h2 className="text-xl font-bold text-gray-900 px-1 mb-4">Meal Logs</h2>
        <FilterBar 
          filters={filters} 
          onFilterChange={handleFilterChange} 
          onClear={handleClearFilters} 
        />
        <div className={loading ? 'opacity-50' : ''}>
          <LogTable logs={logs} onDelete={handleDelete} />
        </div>
      </div>
    </div>
  );
}
