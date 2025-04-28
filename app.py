from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid

app = Flask(__name__)
model = load_model('ecg_classifier_final.h5')  # adjust if different

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Give the image a unique name
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess for model
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            result = model.predict(img_array)
            prediction = "Abnormal" if result[0][0] >= 0.5 else "Normal"

            # Path to show on frontend
            img_url = f"/static/uploads/{filename}"

    return render_template("index.html", prediction=prediction, img_url=img_url)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)

