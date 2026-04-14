import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Label map
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = [
    'Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple',
    'Pomegranate', 'Watermelon'
]

vegetables = [
    'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn',
    'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas',
    'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
    'Tomato', 'Turnip', 'Garlic'
]

# Load model once at module level, handling Keras 3 deserialization breaking change
model = load_model('FV.h5', compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})


def predict_image(file_bytes: bytes) -> str:
    """Load image from bytes, resize to (224, 224), normalize, predict."""
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = prediction.argmax(axis=-1)[0]
    return labels[predicted_class].capitalize()
