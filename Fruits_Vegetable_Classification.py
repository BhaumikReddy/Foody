import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
import os
from dotenv import load_dotenv, dotenv_values
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt

model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion',
          21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato',
          28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato',
          34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Nutritionix API function
load_dotenv()

def fetch_nutritionix_data(prediction):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": os.getenv("APP_ID"),
        "x-app-key": os.getenv("API_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "query": prediction
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        food_data = result['foods'][0]
        return {
            'Calories': food_data.get('nf_calories'),
            'Protein': food_data.get('nf_protein'),
            'Carbohydrates': food_data.get('nf_total_carbohydrate'),
            'Fat': food_data.get('nf_total_fat'),
            'Fiber': food_data.get('nf_dietary_fiber'),
            'Sugars': food_data.get('nf_sugars'),
            'Cholesterol': food_data.get('nf_cholesterol'),
            'Sodium': food_data.get('nf_sodium')
        }
    except Exception as e:
        st.error("Failed to fetch data from Nutritionix API")
        print(e)
        return {
            'Calories': "Invalid value",
            'Protein': "Invalid value",
            'Carbohydrates': "Invalid value",
            'Fat': "Invalid value",
            'Fiber': "Invalid value",
            'Sugars': "Invalid value",
            'Cholesterol': "Invalid value",
            'Sodium': "Invalid value"
        }

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = prediction.argmax(axis=-1)[0]
    return labels[predicted_class].capitalize()

def sanitize_value(value):
    if not isinstance(value, str):
        value = str(value)
    match = re.search(r"[-+]?\d*\.\d+|\d+", value)
    return float(match.group()) if match else None

def scale_nutritional_info(nutrition_info, weight):
    scaled_nutrition = {}
    for key, value in nutrition_info.items():
        val = sanitize_value(value)
        scaled_nutrition[key] = round(val * weight / 100, 2) if val is not None else "Invalid value"
    return scaled_nutrition

def run():
    st.title("Snap2Nutrition")
    time_of_day = st.selectbox("When was the food taken?", ["Breakfast", "Lunch",
                                                            "Snacks", "Dinner"])
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    weight = st.number_input("Enter the weight of the food (grams)", min_value=1)

    excel_file = 'prediction_log.xlsx'
    if not os.path.exists(excel_file):
        pd.DataFrame(columns=[
            "Time of Day", "Food", "Category", "Calories", "Protein", "Carbohydrates", "Fat",
            "Fiber", "Sugars", "Cholesterol", "Sodium",
            "Weight (grams)", "Timestamp"
        ]).to_excel(excel_file, index=False)

    if st.button("Start Prediction"):
        if img_file:
            img = Image.open(img_file).resize((250, 250))
            st.image(img, use_column_width=False)

            save_image_path = './upload_images/' + img_file.name
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

            result = processed_img(save_image_path)
            st.success(f"**Predicted: {result}**")
            category = "Vegetable" if result in vegetables else "Fruit"

            nutrition_info = fetch_nutritionix_data(result)
            scaled_nutrition = scale_nutritional_info(nutrition_info, weight)

            for key, value in scaled_nutrition.items():
                st.warning(f"**{key}: {value} (per {weight} grams)**")

            labels_chart = ['Calories', 'Protein', 'Carbohydrates', 'Fat']
            sizes = [
                sanitize_value(scaled_nutrition['Calories']),
                sanitize_value(scaled_nutrition['Protein']),
                sanitize_value(scaled_nutrition['Carbohydrates']),
                sanitize_value(scaled_nutrition['Fat'])
            ]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels_chart, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

            st.pyplot(fig)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_data = {
                "Time of Day": time_of_day,
                "Food": result,
                "Category": category,
                "Calories": scaled_nutrition.get('Calories', 'N/A'),
                "Protein": scaled_nutrition.get('Protein', 'N/A'),
                "Carbohydrates": scaled_nutrition.get('Carbohydrates', 'N/A'),
                "Fat": scaled_nutrition.get('Fat', 'N/A'),
                "Fiber": scaled_nutrition.get('Fiber', 'N/A'),
                "Sugars": scaled_nutrition.get('Sugars', 'N/A'),
                "Cholesterol": scaled_nutrition.get('Cholesterol', 'N/A'),
                "Sodium": scaled_nutrition.get('Sodium', 'N/A'),
                "Weight (grams)": weight,
                "Timestamp": timestamp
            }
            df = pd.read_excel(excel_file, engine="openpyxl")
            df = df.append(new_data, ignore_index=True)
            df.to_excel(excel_file, index=False, engine="openpyxl")
            st.success("Prediction logged successfully.")

    if st.button("View Past Logs"):
        df = pd.read_excel(excel_file, engine="openpyxl")
        if not df.empty:
            for idx, row in df.iterrows():
                if st.button(f"View Log #{idx + 1}"):
                    st.text_area("Prediction Log", f"""
Food: {row['Food']}
Category: {row['Category']}
Time of Day: {row['Time of Day']}
Weight: {row['Weight (grams)']} grams
Calories: {row['Calories']} (per {row['Weight (grams)']} grams)
Protein: {row['Protein']} (per {row['Weight (grams)']} grams)
Carbohydrates: {row['Carbohydrates']} (per {row['Weight (grams)']} grams)
Fat: {row['Fat']} (per {row['Weight (grams)']} grams)
Fiber: {row['Fiber']} (per {row['Weight (grams)']} grams)
Sugars: {row['Sugars']} (per {row['Weight (grams)']} grams)
Cholesterol: {row['Cholesterol']} (per {row['Weight (grams)']} grams)
Sodium: {row['Sodium']} (per {row['Weight (grams)']} grams)
Timestamp: {row['Timestamp']}
""", height=300)
        else:
            st.info("No past logs found.")

run()

