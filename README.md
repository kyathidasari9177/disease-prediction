# 🩺 Disease Prediction System

A machine learning web application that predicts diabetes risk based on clinical data — built with Python, scikit-learn, and Streamlit.

👉 **[Live Demo](https://kyathidasari9177-disease-prediction-app-xmbf4n.streamlit.app/)**

---

## 📌 Project Overview

This project uses the **PIMA Indians Diabetes Dataset** (768 patients, 8 clinical features) to predict whether a patient is at risk of diabetes. It compares 4 ML models, tunes the best one using GridSearchCV, and deploys it as an interactive web app.

---

## ✨ Features

- 🔍 **Real-time Prediction** — Enter patient details and get instant diabetes risk prediction with probability score
- 📋 **Health Indicators** — Shows Normal / Borderline / High status for each input value
- 🩺 **Personalized Health Tips** — Diet, exercise, and medical recommendations based on result
- 📊 **Feature Importance Chart** — Visual explanation of which features affect predictions most
- 🔢 **SHAP Explainability** — Shows WHY the model made a specific prediction
- 📁 **Batch Prediction** — Upload a CSV of multiple patients and get all predictions at once
- 📋 **Prediction History** — Tracks all predictions made during the session
- 📄 **Download Reports** — Export prediction results as CSV

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib |
| Explainability | SHAP |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🔬 ML Pipeline

### 1. Exploratory Data Analysis
- Explored feature distributions, correlations, and class balance
- Identified **374 medically invalid zero-values** in clinical features (Glucose, Insulin, BMI, etc.)
- Found **Glucose** as the strongest predictor (correlation: 0.47)

### 2. Data Preprocessing
- Replaced zero values with **median imputation**
- Capped outliers using **IQR method** (clip instead of drop)
- Applied **StandardScaler** (fit on train only to prevent data leakage)
- Train/test split: **80/20 with stratify**

### 3. Model Training & Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~78% |
| **Random Forest** ⭐ | **~82%** |
| SVM | ~79% |
| KNN | ~75% |

### 4. Hyperparameter Tuning
- Used **GridSearchCV** with 5-fold cross validation
- Tested **180 combinations** of hyperparameters
- Best params: `n_estimators=300, max_depth=None, min_samples_split=5`

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve (AUC ~0.87)

---

## 📁 Project Structure

```
disease-prediction/
├── app.py              ← Streamlit web app
├── model.pkl           ← Trained Random Forest model
├── scaler.pkl          ← StandardScaler
└── requirements.txt    ← Dependencies
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/kyathidasari9177/disease-prediction.git
cd disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository
- **Records:** 768 patients
- **Features:** 8 clinical measurements
- **Target:** Diabetic (1) or Not Diabetic (0)

---

## 👩‍💻 Author

**Kyathi Sujitha Dasari**  
B.Tech Information Technology — Bhoj Reddy Engineering College for Women  
📧 kyathidasari9177@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
