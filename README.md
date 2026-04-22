# 🍜 Street Food Image Classification with CNN

A deep learning project that classifies street food images into 9 categories using a custom Convolutional Neural Network (CNN). Built as part of a Kaggle competition.

---

## 📌 Project Overview

This project trains a CNN model from scratch to classify street food images into the following 9 classes:

| # | Class | Emoji |
|---|-------|-------|
| 0 | Falafel | 🧆 |
| 1 | Burger | 🍔 |
| 2 | Pani Puri | 🫓 |
| 3 | Pretzel | 🥨 |
| 4 | Shawarma | 🌯 |
| 5 | Hot Dog | 🌭 |
| 6 | Tacos | 🌮 |
| 7 | Crepes | 🥞 |
| 8 | Pad Thai | 🍜 |

---

## 📁 Repository Structure

```
StreetFoodClassification_CNN_and_TransferLearning/
├── street-food-image-classification-with-cnn.ipynb  # Main notebook
├── app.py                                            # Streamlit web app
├── requirements.txt                                  # Dependencies
├── .gitignore                                        # Git ignore rules
└── README.md                                         # This file
```

> 📦 **Trained models** (`street_food_cnn.h5` / `street_food_cnn.keras`) are stored on Google Drive due to file size (173 MB):
> 👉 [Google Drive - Model Files](https://drive.google.com/drive/folders/1rWbswwCEcDjeKuY6mxqc1kJlxTS4nh2b)

---

## 🧠 Model Architecture

Custom CNN built with TensorFlow/Keras:

- **Input:** 240×240×3
- **4 Convolutional Blocks:** Conv2D → BatchNormalization → MaxPooling → Dropout
- **Filters:** 32 → 64 → 128 → 256
- **Dense Layers:** 256 units + Dropout
- **Output:** 9 units (Softmax)
- **Regularization:** L2 + Dropout + BatchNormalization
- **Callbacks:** EarlyStopping + ReduceLROnPlateau

---

## 🚀 Streamlit Web App

An interactive web app that lets you upload a street food image and get an instant prediction with confidence scores.

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/tugcesi/StreetFoodClassification_CNN_and_TransferLearning.git
cd StreetFoodClassification_CNN_and_TransferLearning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model from Google Drive and place it here:
#    street_food_cnn.h5

# 4. Run the app
streamlit run app.py
```

### App Features
- 📁 Upload JPG / PNG images
- 🔍 Instant prediction with confidence score
- 📊 Probability bar chart for all 9 classes
- ⚡ Model cached for fast repeated predictions

---

## 📊 Dataset

- **Source:** Kaggle Street Food Image Classification Competition
- **Train:** Images grouped into 10 class-named sub-folders
- **Test:** 764 unlabeled images in a flat folder
- **Format:** JPG images + CSV labels

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)

---

## 👩‍💻 Author

**Tugce Basyigit**
- GitHub: [@tugcesi](https://github.com/tugcesi)
