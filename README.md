# 🌱 AgroAI Guardian: AI-Powered Plant Disease Detection and Treatment Recommendation System

AgroAI Guardian is a Deep Learning and Generative AI-based agricultural decision support system designed to detect plant diseases from leaf images and provide intelligent treatment recommendations. The system combines a Convolutional Neural Network (CNN) for disease classification with Google Gemini AI for generating comprehensive disease management reports.

---

## 📌 Project Overview

Plant diseases significantly impact crop yield and food production worldwide. Early and accurate disease diagnosis is essential for effective crop management. AgroAI Guardian helps farmers, researchers, and agricultural practitioners identify plant diseases from leaf images and receive AI-generated recommendations for treatment and prevention.

Unlike traditional plant disease classifiers that only predict disease names, AgroAI Guardian provides:

* Disease Detection
* Confidence Score
* Disease Overview
* Causes of Disease
* Symptoms
* Treatment Recommendations
* Chemical Pesticide Suggestions
* Organic / Natural Cure Options
* Prevention Guidelines
* Severity Assessment

---

## 🚀 Features

### 🌿 Plant Disease Detection

* CNN-based image classification
* Supports 38 plant disease categories
* Real-time disease prediction

### 📊 Confidence Score

* Displays prediction confidence level
* Helps users evaluate prediction reliability

### 🤖 AI-Powered Agricultural Recommendations

After disease prediction, Google Gemini AI generates:

* Disease Overview
* Causes
* Symptoms
* Treatment Steps
* Chemical Pesticides
* Organic / Natural Cure
* Prevention Tips
* Severity Level

### 💻 Interactive Web Application

* Built with Streamlit
* User-friendly interface
* Image upload functionality
* Instant prediction results

---

## 🗂 Dataset

**Dataset Name:** New Plant Diseases Dataset (Augmented)

**Source:** Kaggle

Dataset Link:

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### Dataset Statistics

* Approximately 87,000 RGB images
* 38 disease and healthy classes
* Training, Validation, and Testing sets
* Multiple crop species

### Supported Crops

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

---

## 🧠 Model Architecture

The disease classification model is built using a Sequential Convolutional Neural Network (CNN).

### Components

* Convolution Layers
* ReLU Activation
* Max Pooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

### Workflow

Leaf Image

⬇

Image Preprocessing

⬇

CNN Classification

⬇

Disease Prediction

⬇

Confidence Score

⬇

Gemini AI Analysis

⬇

Treatment Recommendation

⬇

Prevention Guidelines

⬇

Severity Assessment

⬇

Final Report

---

## 🛠 Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* PIL

### Web Framework

* Streamlit

### AI Recommendation Engine

* Google Gemini API

### Data Analysis & Visualization

* NumPy
* Pandas
* Matplotlib
* Seaborn

---

## 📈 Model Performance

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

### Results

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 83%   |
| Macro F1 Score    | 0.83  |
| Weighted F1 Score | 0.83  |

> Note: Performance may vary depending on image quality, disease similarity, and dataset distribution.

---

## 📸 Application Screenshots

### Home Page

Add Screenshot Here

### Disease Prediction

Add Screenshot Here

### AI Recommendation Output

Add Screenshot Here

### Confusion Matrix

Add Screenshot Here

### Accuracy & Loss Graphs

Add Screenshot Here

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AgroAI-Guardian.git
cd AgroAI-Guardian
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

```bash
streamlit run main.py
```

Application will open in your browser:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
AgroAI-Guardian/
│
├── main.py
├── requirements.txt
├── trained_model.h5
├── trained_model.keras
├── training_hist.json
│
├── Train_Plant_Disease.ipynb
├── Test_Plant_Disease.ipynb
│
├── screenshots/
│   ├── homepage.png
│   ├── prediction.png
│   └── recommendation.png
│
└── README.md
```

---

## 🔍 Limitations

* Performance depends on image quality
* Similar diseases may be confused
* AI recommendations depend on generated responses
* Requires internet connection for Gemini AI features

---

## 🔮 Future Improvements

* Mobile Application Development
* Transfer Learning (ResNet50, EfficientNet)
* Vision Transformers (ViTs)
* Real-Time Field Monitoring
* Drone-Based Crop Analysis
* Multilingual Support
* Cloud Deployment

---

## 🎯 Conclusion

AgroAI Guardian demonstrates the integration of Computer Vision and Generative AI for intelligent agricultural decision support. By combining CNN-based disease detection with AI-generated treatment recommendations, the system provides a practical solution for early disease diagnosis and crop health management.

The project highlights the potential of Artificial Intelligence in modern agriculture and contributes toward smarter, more accessible plant disease management systems.

---

## 👨‍💻 Author

Imran Khan

Bachelor of Artificial Intelligence

Computer Vision Project

---

## 📜 License

This project is intended for educational and research purposes.
