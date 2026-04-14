import { useState, useMemo } from 'react';

export default function LogTable({ logs, onDelete }) {
  const [sortConfig, setSortConfig] = useState({ key: 'timestamp', direction: 'desc' });
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 20;

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const sortedLogs = useMemo(() => {
    let sortableItems = [...logs];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        let aValue = a[sortConfig.key];
        let bValue = b[sortConfig.key];

        // Handle nulls
        if (aValue === null) aValue = '';
        if (bValue === null) bValue = '';

        if (aValue < bValue) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [logs, sortConfig]);

  const currentTableData = useMemo(() => {
    const firstPageIndex = (currentPage - 1) * rowsPerPage;
    const lastPageIndex = firstPageIndex + rowsPerPage;
    return sortedLogs.slice(firstPageIndex, lastPageIndex);
  }, [currentPage, sortedLogs]);

  const totalPages = Math.ceil(sortedLogs.length / rowsPerPage);

  const formatDate = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return '↕';
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  const [deletingId, setDeletingId] = useState(null);

  const handleDeleteClick = (id) => {
    setDeletingId(id);
  };

  const handleConfirmDelete = (id) => {
    onDelete(id);
    setDeletingId(null);
  };

  const handleCancelDelete = () => {
    setDeletingId(null);
  };

  if (logs.length === 0) {
    return (
      <div className="bg-white p-8 rounded-xl border border-gray-100 text-center text-gray-500">
        No logs found. Start by uploading a food image.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-600">
          <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-100">
            <tr>
              <th
                className="px-4 py-3 font-medium cursor-pointer hover:bg-gray-100 whitespace-nowrap"
                onClick={() => handleSort('timestamp')}
              >
                Timestamp {getSortIcon('timestamp')}
              </th>
              <th
                className="px-4 py-3 font-medium cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('food')}
              >
                Food {getSortIcon('food')}
              </th>
              <th className="px-4 py-3 font-medium">Category</th>
              <th className="px-4 py-3 font-medium">Meal</th>
              <th className="px-4 py-3 font-medium text-right whitespace-nowrap">Weight (g)</th>
              <th
                className="px-4 py-3 font-medium cursor-pointer hover:bg-gray-100 text-right whitespace-nowrap"
                onClick={() => handleSort('calories')}
              >
                Cal {getSortIcon('calories')}
              </th>
              <th className="px-4 py-3 font-medium text-right">Pro (g)</th>
              <th className="px-4 py-3 font-medium text-right">Carbs (g)</th>
              <th className="px-4 py-3 font-medium text-right">Fat (g)</th>
              <th className="px-4 py-3 font-medium text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {currentTableData.map((log) => (
              <tr key={log.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                <td className="px-4 py-3 font-mono text-xs">{formatDate(log.timestamp)}</td>
                <td className="px-4 py-3 font-medium text-gray-900 capitalize">{log.food}</td>
                <td className="px-4 py-3">
                  <span className={log.category === 'Fruit' ? 'badge-fruit' : 'badge-vegetable'}>
                    {log.category}
                  </span>
                </td>
                <td className="px-4 py-3">{log.meal_type}</td>
                <td className="px-4 py-3 font-mono text-right">{log.weight_grams}</td>
                <td className="px-4 py-3 font-mono text-right font-medium text-gray-800">
                  {log.calories !== null ? log.calories : '-'}
                </td>
                <td className="px-4 py-3 font-mono text-right text-gray-500">
                  {log.protein !== null ? log.protein : '-'}
                </td>
                <td className="px-4 py-3 font-mono text-right text-gray-500">
                  {log.carbohydrates !== null ? log.carbohydrates : '-'}
                </td>
                <td className="px-4 py-3 font-mono text-right text-gray-500">
                  {log.fat !== null ? log.fat : '-'}
                </td>
                <td className="px-4 py-3 text-center">
                  {deletingId === log.id ? (
                    <div className="flex items-center justify-center gap-2 text-xs">
                      <span className="text-gray-500">Sure?</span>
                      <button
                        onClick={() => handleConfirmDelete(log.id)}
                        className="text-red-600 hover:underline font-medium"
                      >
                        Yes
                      </button>
                      <button
                        onClick={handleCancelDelete}
                        className="text-gray-500 hover:underline"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => handleDeleteClick(log.id)}
                      className="btn-danger w-8 h-8 inline-flex items-center justify-center rounded hover:bg-red-50"
                      title="Delete log"
                    >
                      ×
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-4 py-3 flex items-center justify-between border-t border-gray-100 bg-gray-50">
          <span className="text-sm text-gray-500">
            Showing <span className="font-medium text-gray-900">{(currentPage - 1) * rowsPerPage + 1}</span> to{' '}
            <span className="font-medium text-gray-900">
              {Math.min(currentPage * rowsPerPage, sortedLogs.length)}
            </span>{' '}
            of <span className="font-medium text-gray-900">{sortedLogs.length}</span> entries
          </span>
          <div className="flex gap-2 text-sm">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-white border border-gray-200 rounded text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Prev
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-white border border-gray-200 rounded text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
