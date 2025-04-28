import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('ecg_classifier_final.h5')  # adjust the filename if yours is different

# Path to the folder containing unseen ECG images
unseen_path = 'ecg_prediction'

# Class labels (must match your training labels)
class_names = ['abnormal', 'normal']  # adjust order if needed

# Loop through images in the unseen folder
for img_file in os.listdir(unseen_path):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(unseen_path, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize just like in training

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Show result
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}")
        plt.axis('off')
        plt.show()