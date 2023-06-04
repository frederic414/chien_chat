from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
from datetime import datetime

# Charger le modèle pré-entraîné
model = load_model("models/modele_chien_chat_Xception.h5")
class_names = ["Chat", "Chien"]


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image



def predict_image(image):
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    class_label = class_names[class_index]
    return class_label


def generate_unique_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name, file_extension = os.path.splitext(filename)
    unique_filename = f"{file_name}_{timestamp}{file_extension}"
    return unique_filename

