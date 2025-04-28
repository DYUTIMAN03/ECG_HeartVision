# ECG Heartbeat Classification Using VGG16

This project uses a Convolutional Neural Network (CNN) based on the VGG16 architecture to classify ECG heartbeat signals into Normal and Abnormal categories.

## 📚 Project Overview

- **Dataset:** MIT-BIH Arrhythmia Dataset
- **Model:** Fine-tuned VGG16 
- **Goal:** Classify ECG heartbeats from 1D signal images and Fine-tuned on ECG heartbeat images (Normal vs Abnormal).

## 🚀 Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 📈 Project Workflow

1. Preprocessing ECG signals
2. Converting 1D signals to 2D images
3. Building and fine-tuning the VGG16 model
4. Training and Validation
5. Model Evaluation and Metrics
6. Integrate backend with frontend using flask.
7. run app.py

## 🖼️ Sample Results

(sample images or graphs will be added here later!)

## 📦 How to Run

# Clone the repository
git clone https://github.com/DYUTIMAN03/vgg16-medical-imaging.git

# Change directory
cd your-repo-name

# Install requirements
pip install -r requirements.txt

# Run your training script
python train_model.py

## ✨ Features

- Upload ECG images and get instant predictions.
- Visualize training graphs.
- Clean UI using Flask templates.

## 🛠️ Future Improvements

- Improve model accuracy.
- Add multi-class classification.
- Deploy to cloud (AWS, Azure, or Render).


🙌 Acknowledgements:
MIT-BIH Arrhythmia Dataset

Made with ❤️ and TensorFlow/Keras
