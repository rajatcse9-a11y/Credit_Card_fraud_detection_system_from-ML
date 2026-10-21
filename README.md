# 💳 Credit Card Fraud Detection System

An AI-powered **Credit Card Fraud Detection System** built using Machine Learning, FastAPI, and Streamlit to identify potentially fraudulent financial transactions through a user-friendly web interface.

## 🚀 Project Overview

Credit card fraud is a major challenge in digital payments. This project uses a trained Machine Learning model to classify transactions as **Fraudulent** or **Legitimate**.

The application uses **FastAPI** as the backend API and **Streamlit** as the frontend interface, providing a modular and interactive fraud detection system.

## ✨ Features

- 🤖 Machine Learning-based transaction prediction
- ⚡ FastAPI backend for model inference
- 🎨 Advanced Streamlit dashboard
- 💰 Transaction amount and time inputs
- 📊 Support for transaction features V1–V28
- 🔍 Fraud and legitimate transaction classification
- 📈 Fraud probability display, if supported by the trained model
- 🔗 Frontend-backend communication through REST API

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn / trained ML model
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **API Requests:** Requests
- **Model Serialization:** Joblib or the format used by the trained model

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── model/
│   └── model.pkl
│
├── app.py
├── fastapi.py
├── requirements.txt
└── README.md
```

> Note: Update the structure and model filename according to your actual project files.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rajatcse9-a11y/Credit_Card_fraud_detection_system_from-ML.git
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit fastapi uvicorn requests pandas numpy scikit-learn joblib
```

If you have a `requirements.txt` file, you can instead run:

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

Open the first terminal in your project directory:

```bash
uvicorn fastapi:app --reload
```

The API will generally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit frontend

Open a second terminal in the same project directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 🔄 How It Works

1. The user enters transaction details in the Streamlit dashboard.
2. Streamlit sends the transaction data to the FastAPI `/predict` endpoint.
3. FastAPI validates and prepares the input features.
4. The trained Machine Learning model predicts the transaction class.
5. The API returns the prediction and, if implemented, fraud probability.
6. Streamlit displays the result to the user.

## 📊 Input Features

The application is designed around the following transaction fields:

- `Time`
- `Amount`
- `V1` to `V28`
- `Hour_sin`
- `Hour_cos`

The exact features and their order must match the model's training pipeline. `Hour_sin` and `Hour_cos` should be calculated consistently with the feature engineering used during training.

## 🎯 Expected Output

The application can display one of the following results:

- **Fraud Detected:** The model