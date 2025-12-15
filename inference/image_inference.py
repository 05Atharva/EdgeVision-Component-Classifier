import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "model/component_classifier_finetuned.h5"
TEST_DIR = "path_to_test_images"
IMG_SIZE = 96

model = tf.keras.models.load_model(MODEL_PATH)

for img_name in os.listdir(TEST_DIR):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
        img_path = os.path.join(TEST_DIR, img_name)

        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]

        label = "Capacitor ⚡" if pred < 0.5 else "Inductor 🌀"
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"{img_name}: {label} (Confidence: {confidence:.4f})")
