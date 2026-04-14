export default function NutritionCard({ label, value, unit }) {
  const display = value !== null && value !== undefined ? value : 'N/A';
  return (
    <div className="flex flex-col gap-0.5 p-3 bg-gray-50 rounded-lg">
      <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
      <span className="font-mono text-gray-900 text-base font-medium">
        {display !== 'N/A' ? (
          <>
            {display}
            <span className="text-xs text-gray-400 ml-1">{unit}</span>
          </>
        ) : (
          <span className="text-gray-400">N/A</span>
        )}
      </span>
    </div>
  );
}
