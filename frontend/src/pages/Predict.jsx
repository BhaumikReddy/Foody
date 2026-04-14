import { useState, useRef } from 'react';
import api from '../api/client';
import NutritionCard from '../components/NutritionCard';
import MacroPieChart from '../components/MacroPieChart';

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mealType, setMealType] = useState('Breakfast');
  const [weight, setWeight] = useState(100);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);
      // Reset result when new image is uploaded
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      const objectUrl = URL.createObjectURL(droppedFile);
      setPreview(objectUrl);
      setResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image first.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('meal_type', mealType);
    formData.append('weight_grams', weight);

    try {
      const response = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred while analyzing the food.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
      {/* Upload Form */}
      <div className="card shadow-sm flex flex-col">
        <h2 className="text-xl mb-6">Analyze Food</h2>
        
        <form onSubmit={handleSubmit} className="flex flex-col gap-5 flex-1">
          {/* Image Upload Zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-colors duration-150 min-h-[240px]
              ${preview ? 'border-brand-300 bg-brand-50/50' : 'border-gray-200 hover:border-brand-400 hover:bg-gray-50'}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-[200px] object-contain rounded-lg shadow-sm" />
            ) : (
              <div className="text-gray-500 flex flex-col items-center">
                <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                </svg>
                <span className="font-medium text-gray-700">Click to upload</span>
                <span className="text-xs mt-1">or drag and drop an image</span>
              </div>
            )}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/jpeg, image/png"
              onChange={handleFileChange}
            />
          </div>

          {/* Form Fields */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="form-label">Meal Type</label>
              <select
                className="form-input"
                value={mealType}
                onChange={(e) => setMealType(e.target.value)}
              >
                <option value="Breakfast">Breakfast</option>
                <option value="Lunch">Lunch</option>
                <option value="Snacks">Snacks</option>
                <option value="Dinner">Dinner</option>
              </select>
            </div>
            <div>
              <label className="form-label">Weight (g)</label>
              <input
                type="number"
                min="1"
                className="form-input"
                value={weight}
                onChange={(e) => setWeight(Number(e.target.value))}
              />
            </div>
          </div>

          {error && <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg">{error}</div>}

          <div className="mt-auto pt-4">
            <button
              type="submit"
              disabled={loading || !file}
              className="btn-primary w-full flex justify-center items-center h-[46px] text-base"
            >
              {loading ? (
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                'Analyse Food'
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Result Panel */}
      <div className="card shadow-sm bg-white">
        {!result ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 min-h-[400px]">
             <svg className="w-16 h-16 mb-4 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
             </svg>
            <p>Upload an image to view nutritional details.</p>
          </div>
        ) : (
          <div className="flex flex-col h-full animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-100">
              <div>
                <h2 className="text-2xl capitalize text-gray-900 mb-1">{result.food}</h2>
                <div className="text-sm text-gray-500 font-mono">{result.weight_grams}g • {result.meal_type}</div>
              </div>
              <span className={result.category === 'Fruit' ? 'badge-fruit text-sm px-3 py-1' : 'badge-vegetable text-sm px-3 py-1'}>
                {result.category}
              </span>
            </div>

            {/* Macros Chart */}
            <div className="mb-6 bg-gray-50 rounded-xl p-4">
               <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 text-center">Macro Profile</h3>
               <MacroPieChart 
                  calories={result.calories}
                  protein={result.protein}
                  carbohydrates={result.carbohydrates}
                  fat={result.fat}
               />
            </div>

            {/* Nutrition Grid */}
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Detailed Nutrition</h3>
            <div className="grid grid-cols-2 gap-3 flex-1 content-start">
              <NutritionCard label="Calories" value={result.calories} unit="kcal" />
              <NutritionCard label="Protein" value={result.protein} unit="g" />
              <NutritionCard label="Carbs" value={result.carbohydrates} unit="g" />
              <NutritionCard label="Fat" value={result.fat} unit="g" />
              <NutritionCard label="Fiber" value={result.fiber} unit="g" />
              <NutritionCard label="Sugars" value={result.sugars} unit="g" />
              <NutritionCard label="Cholesterol" value={result.cholesterol} unit="mg" />
              <NutritionCard label="Sodium" value={result.sodium} unit="mg" />
            </div>
            
            {(result.calories === null) && (
              <div className="mt-4 text-xs text-orange-600 bg-orange-50 p-3 rounded-lg border border-orange-100 italic">
                Note: Could not retrieve reliable nutritional data from the external API for this item.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
