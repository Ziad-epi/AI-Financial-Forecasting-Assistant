# 🚀 AI Financial Forecasting & Assistant

An end-to-end AI system combining **time series forecasting** and a **RAG-based LLM assistant**, built with a production-oriented architecture.

---

## 🧠 Overview

This project simulates a real-world AI system used in financial software, integrating:

* 📊 Time series forecasting (Machine Learning)
* 🤖 LLM-powered assistant (RAG pipeline)
* ⚙️ FastAPI backend
* 🎨 Streamlit frontend
* 🐳 Docker deployment

---

## 🎯 Features

### 📊 Forecasting

* Predict financial time series values
* Multiple models (baseline, ML)
* Evaluation with RMSE / MAE

### 💡 Business Decision Layer

* BUY / SELL / HOLD signals
* Risk assessment
* Model explanation

### 🤖 AI Assistant (RAG)

* Ask questions on financial data
* Context-aware responses
* Reduced hallucinations

### 🌐 API

* `/predict` → raw prediction
* `/decision` → business insights
* `/ask` → LLM assistant

### 🎨 Frontend (Streamlit)

* Interactive UI
* Real-time predictions
* AI chat interface

---

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → ML Model / RAG → Response
```

---

## 📁 Project Structure

```
api/        → FastAPI backend
app/        → Streamlit frontend
business/   → business logic
llm/        → RAG pipeline
model/      → trained model
data/       → datasets
notebooks/  → EDA & experiments
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-financial-forecasting.git
cd ai-financial-forecasting
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Locally

### Start API

```bash
uvicorn api.main:app --reload
```

API available at:
👉 http://localhost:8000/docs

---

### Start Streamlit

```bash
streamlit run app/app.py
```

App available at:
👉 http://localhost:8501

---

## 🐳 Docker Deployment

### Run full system

```bash
docker-compose up --build
```

---

## 📊 Example Usage

### Prediction

```json
POST /predict
{
  "features": {
    "lag_1": 0.001,
    "lag_2": 0.0005,
    "lag_3": -0.0002,
    "MA7": 2378,
    "MA30": 2375,
    "volatility": 0.002
  }
}
```

---

### Decision

```json
POST /decision
{
  "features": {...}
}
```

Response:

```json
{
  "prediction": 0.0021,
  "signal": "BUY",
  "risk": "MEDIUM",
  "explanation": "Upward trend detected"
}
```

---

## 🧠 Tech Stack

* Python
* FastAPI
* Streamlit
* scikit-learn
* LangChain
* ChromaDB
* Docker

---

## 🚀 Future Improvements

* Cloud deployment (Azure / AWS)
* CI/CD pipeline
* Model monitoring
* Advanced deep learning models (LSTM)

---

## 💼 About

This project was built to simulate a **production-grade AI system** combining data science, machine learning, and modern LLM applications.

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
