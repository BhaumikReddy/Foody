import { useState, useEffect } from 'react';

export default function FilterBar({ filters, onFilterChange, onClear }) {
  const [localSearch, setLocalSearch] = useState(filters.food || '');

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      onFilterChange('food', localSearch);
    }, 300);
    return () => clearTimeout(timer);
  }, [localSearch, onFilterChange]);

  const handleClear = () => {
    setLocalSearch('');
    onClear();
  };

  return (
    <div className="bg-white p-4 rounded-xl border border-gray-100 mb-6 flex flex-wrap gap-4 items-end">
      <div className="flex-1 min-w-[200px]">
        <label className="form-label">Search Food</label>
        <input
          type="text"
          className="form-input"
          placeholder="e.g. Apple"
          value={localSearch}
          onChange={(e) => setLocalSearch(e.target.value)}
        />
      </div>

      <div className="w-40">
        <label className="form-label">Meal Type</label>
        <select
          className="form-input"
          value={filters.meal_type || ''}
          onChange={(e) => onFilterChange('meal_type', e.target.value)}
        >
          <option value="">All Meals</option>
          <option value="Breakfast">Breakfast</option>
          <option value="Lunch">Lunch</option>
          <option value="Snacks">Snacks</option>
          <option value="Dinner">Dinner</option>
        </select>
      </div>

      <div className="w-36">
        <label className="form-label">Category</label>
        <select
          className="form-input"
          value={filters.category || ''}
          onChange={(e) => onFilterChange('category', e.target.value)}
        >
          <option value="">All Categories</option>
          <option value="Fruit">Fruit</option>
          <option value="Vegetable">Vegetable</option>
        </select>
      </div>

      <div className="w-40">
        <label className="form-label">From Date</label>
        <input
          type="date"
          className="form-input"
          value={filters.date_from || ''}
          onChange={(e) => onFilterChange('date_from', e.target.value)}
        />
      </div>

      <div className="w-40">
        <label className="form-label">To Date</label>
        <input
          type="date"
          className="form-input"
          value={filters.date_to || ''}
          onChange={(e) => onFilterChange('date_to', e.target.value)}
        />
      </div>

      <button
        type="button"
        className="btn-secondary h-[38px] flex items-center justify-center px-4 whitespace-nowrap"
        onClick={handleClear}
      >
        Clear Filters
      </button>
    </div>
  );
}
