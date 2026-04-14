# Foody — Nutrition Tracker

A modern web application to analyze food images, classify fruits and vegetables
using machine learning, fetch nutritional data from the USDA FoodData Central API,
and visualize consumption logs.

This repository features:
- **Backend:** FastAPI (Python), SQLAlchemy, SQLite, and Keras/TensorFlow.
- **Frontend:** React, Vite, TailwindCSS, and Recharts.

---

## Architecture

1. **User Upload:** Upload an image, input weight, and select meal type.
2. **Analysis:** The FastAPI backend classifies the image using a Keras model (`FV.h5`).
3. **Data Fetching:** Nutritional values are pulled from the USDA FoodData Central
   API and scaled based on the entered weight.
4. **Dashboard:** Built with React and Recharts, it visualizes macros, daily calorie
   trends, and provides a filterable log table of consumption data.

---

## ML Model

The classification model (`FV.h5`) was trained on the
[Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
dataset from Kaggle.

- **Base Model:** MobileNetV2 (transfer learning)
- **Classes:** 36 fruits and vegetables
- **Training Notebook:** The full training pipeline is included in the repository
  as a Jupyter notebook (`.ipynb`) — covering data loading, augmentation,
  fine-tuning, and model export.

---

## Requirements

### Backend
- Python 3.9+
- The `FV.h5` Keras model file must be present in the `backend/` directory.
- USDA FoodData Central API key (free at https://api.nal.usda.gov).

### Frontend
- Node.js v18+
- npm or yarn

---

## Setup & Running

### 1. API Key Setup

Get a free API key from the USDA FoodData Central:
 https://api.nal.usda.gov

Create a `.env` file inside the `backend/` directory:

```env
USDA_API_KEY=your_api_key_here
```


### 2. Run the Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

The API will be running at `http://localhost:8000`  
Interactive docs available at `http://localhost:8000/docs`

### 3. Run the Frontend

Open a new terminal session:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be running at `http://localhost:5173`

> ⚠️ Both terminals must stay open simultaneously for the app to work.
> The frontend communicates with the backend on port 8000.

---

## Key Features

- **Drag-and-Drop Image Prediction:** Seamless upload area for quick inference
  across 36 fruit and vegetable classes.
- **USDA Nutrition Data:** Accurate per-100g nutritional values scaled to the
  user's entered weight.
- **Dynamic Charts:** Calorie trends and macro distributions powered by Recharts.
- **Advanced Filtering Table:** Filter your nutrition history by category,
  date range, food name, and meal type.
- **Reliable Error Handling:** The UI gracefully handles third-party API failures
  without crashing.


---

## Dataset & Model Training

The model was trained using transfer learning on top of MobileNetV2
(pretrained on ImageNet). The Kaggle dataset contains images across 36 classes
of common fruits and vegetables. The `.ipynb` notebook in the root of this
repository documents the full training process including:

- Data loading and augmentation
- MobileNetV2 base with custom classification head
- Training, validation, and export to `.h5` format

The 36 supported classes are:

| Fruits | Vegetables |
|---|---|
| Apple, Banana, Bell Pepper | Beetroot, Cabbage, Capsicum |
| Chilli Pepper, Grapes, Jalepeno | Carrot, Cauliflower, Corn |
| Kiwi, Lemon, Mango | Cucumber, Eggplant, Ginger |
| Orange, Paprika, Pear | Lettuce, Onion, Peas |
| Pineapple, Pomegranate, Watermelon | Potato, Raddish, Soy Beans |
| | Spinach, Sweetcorn, Sweetpotato |
| | Tomato, Turnip |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload image, get prediction + nutrition |
| `GET` | `/api/logs` | Fetch logs with optional filters |
| `GET` | `/api/logs/summary` | Aggregated stats for dashboard |
| `DELETE` | `/api/logs/{id}` | Delete a log entry by ID |

Full interactive API documentation is available at `http://localhost:8000/docs`
when the backend is running.

---

## Environment Variables

| Variable | Location | Description |
|---|---|---|
| `USDA_API_KEY` | `backend/.env` | Your USDA FoodData Central API key |

---

## License

This project is open source and available under the MIT License.
