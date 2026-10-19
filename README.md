# 🤟 ASL Sign Language Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent American Sign Language (ASL) alphabet recognition system built with **Deep Convolutional Neural Networks (CNN)** and an interactive web interface powered by **Streamlit**.

---

## 🌟 Key Features

- **🎥 Real-Time Webcam Stream with ROI Bounding Box:**
  - Solves background noise and lighting clutter by analyzing only the Region of Interest (green target box: `Place Hand Here`).
  - Seamless start/stop controls without UI freezing.
- **📸 In-Browser Snapshot (Camera Input):**
  - Capture static gestures on any device using Streamlit's native camera component.
- **🖼️ Image Upload Mode:**
  - Upload JPG, JPEG, or PNG images for instant prediction and analysis.
- **🎯 Confidence Scoring & Top-3 Predictions:**
  - Displays predicted letter with percentage confidence.
  - Interactive confidence threshold slider in the sidebar.
  - Visual probability distribution for the top 3 most likely classes.
- **🔍 28x28 Input Preview:**
  - Real-time feedback showing the exact preprocessed image seen by the CNN model.

---

## ℹ️ Supported Characters (24 Signs)

The model supports **24 static letters** of the English alphabet:

| | | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | **B** | **C** | **D** | **E** | **F** | **G** | **H** |
| **I** | **K** | **L** | **M** | **N** | **O** | **P** | **Q** |
| **R** | **S** | **T** | **U** | **V** | **W** | **X** | **Y** |

> ⚠️ **Note:** The letters **J** and **Z** are excluded from the Sign Language MNIST dataset because their gestures require continuous hand motion rather than a single static posture.

---

## 📂 Project Structure

```text
sign_language/
│
├── dataset/                     # Directory containing cropped image samples per class (a-y)
├── Sign_Language.ipynb          # Jupyter notebook for CNN training, evaluation & fine-tuning
├── app.py                       # Streamlit web application
├── label_map.json               # Class index to alphabet letter mapping
├── sign_mnist_finetuned.keras   # Saved fine-tuned Keras model
├── sign_mnist_finetuned.h5      # Fallback HDF5 model weights
├── sign_mnist_cnn.h5            # Base trained CNN model weights
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files (large datasets, CSVs, checkpoints)
└── README.md                    # Project documentation
```

---

## 🧠 Model Architecture & Training

1. **Dataset:** Sign Language MNIST (28x28 Grayscale images of cropped hands).
2. **CNN Architecture:**
   - Multiple Convolutional layers (`Conv2D` + `BatchNormalization` + `ReLU`).
   - Downsampling layers (`MaxPooling2D`).
   - Regularization layers (`Dropout`) to prevent overfitting.
   - Fully connected output layers (`Dense` + `Softmax` for 24 classes).
3. **Data Augmentation & Fine-Tuning:**
   - Augmented with random rotations, shifts, zoom, brightness, and shear using `ImageDataGenerator`.
   - Achieved over **98.5% validation accuracy** during fine-tuning.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Rawda-Hafez/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV (cv2), Pillow
- **Frontend / Deployment:** Streamlit
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib

---

## 👩‍💻 Author

- **Rawda Hafez** - [GitHub Profile](https://github.com/Rawda-Hafez)
