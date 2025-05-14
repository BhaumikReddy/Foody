# Foody

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)   
## Overview

**Foody** is a web app that detects food items from images using a MobileNetV2-based model and fetches real-time nutritional information using the Nutritionix API. It helps users understand what they’re eating in terms of calories, protein, fats, and carbohydrates.

---

## Features

- 🧠 **AI-Powered Detection**: Detect food images using a trained MobileNetV2 deep learning model.
- 🍽️ **Nutrition Info**: Uses **Nutritionix API** to provide accurate macro breakdowns for recognized food items.
- 📊 **Interactive Charts**: Displays calorie, protein, carbs, and fat breakdown using pie charts.
- 📁 **Export to Excel**: Automatically saves predictions and nutrition info into downloadable Excel files.
- 🖥️ **User-Friendly UI**: Built using **Streamlit** for quick and interactive analysis.

---

## Tech Stack

- **Frontend**: Streamlit
- **Model**: MobileNetV2 (Keras/TensorFlow)
- **Backend**: Python (Pandas, OpenCV, Pillow)
- **API**: [Nutritionix API](https://developer.nutritionix.com/)

---

## Nutritionix API Setup

1. Go to [Nutritionix Developer](https://developer.nutritionix.com/) and create a free account.
2. Create a `.env` file in the root directory and add:
   ```env
   NUTRITIONIX_APP_ID=your_app_id
   NUTRITIONIX_API_KEY=your_api_key



## To Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/foody.git
cd foody

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

```


## Usage:-

- Clone my repository.
- Open CMD in working directory.
- Run following command.

  ```
  pip install -r requirements.txt
  ```
- `Fruits_Vegetable_Classification.py` is the main Python file of Streamlit Web-Application. 
- `Fruit_Veg_Classification_Mobilenet.ipynb` is the Notebook file of the Training
- Dataset that I have used is [Fruit and Vegetable Image Recognition](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition).
- To run app, write following command in CMD. or use any IDE.

  ```
  streamlit run Fruits_Vegetable_Classification.py
  ```
