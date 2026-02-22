# 🌍 COVID-19 Global Analytics & Prediction Dashboard

## 📌 Overview
This project is a Streamlit-based interactive dashboard for analyzing global COVID-19 data and predicting total deaths using a trained Machine Learning model.

The application provides:
- Advanced dynamic filters
- KPI monitoring
- Interactive visualizations
- ML-based prediction
- Deployment-ready structure

---

## 🚀 Features

### 🔎 Advanced Filters
- Country multi-select filter
- Date range filter (auto-detection)
- Dynamic numeric range sliders
- Real-time filtered dataset count

### 📊 Dashboard Metrics
- Total Cases
- Total Deaths
- Total Recovered

### 📈 Interactive Visualizations
- Top 10 Countries by Cases
- Top 10 Countries by Deaths
- Dynamic Plotly charts

### 🤖 ML Prediction
- Custom input-based prediction
- Trained model (`best_model.pkl`)
- Real-time prediction output

---
## 📁 Project Structure

```text
covid19-streamlit-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── covid19_global_statistics_2026.csv
│
├── model/
│   └── best_model.pkl
│
└── notebooks/
    └── COVID19.ipynb
```

## ⚙️ Installation (Local Setup)

1. Clone the repository:


git clone <your-repo-link>
cd covid19-streamlit-app


2. Install dependencies:


pip install -r requirements.txt


3. Run the application:


streamlit run app.py


---

## 🌐 Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Select `app.py`
5. Deploy

---

## 🧠 Machine Learning Model

- Model file: `model/best_model.pkl`
- Algorithm: (e.g., XGBoost / Random Forest)
- Target variable: Total Deaths

---

## 📊 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Plotly

---

## 📌 Author
Aanjney Kumawat  
Data Scientist | ML Enthusiast

---

## 📜 License
This project is for educational and demonstration purposes.
