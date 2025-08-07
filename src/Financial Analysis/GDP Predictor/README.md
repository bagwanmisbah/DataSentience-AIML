# 🌍 GDP Prediction App

A simple and interactive web application built with **Streamlit** that predicts a country's **GDP** based on economic indicators using a **CatBoost Regressor**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qdm44qwg7c6sckbuss69kc.streamlit.app/)

---

## 🧠 What it does

- Takes real-time input of economic features
- Loads a trained `.cbm` CatBoost model
- Predicts the GDP (in USD 💵) with a click

---

## 📌 Features

- ✨ Clean UI with number sliders and input fields
- 🔁 Handles negative inputs where appropriate
- 📦 Lightweight and fast CatBoost model
- 🎯 Returns rounded predictions in currency format

---

## 🛠️ Technologies Used

- Python 🐍
- Streamlit
- Pandas
- CatBoost

---

## 🧪 How to Use Locally

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/gdp-predictor.git
   cd gdp-predictor

2. **Install Dependencies:**
   pip install -r requirements.txt

3. **Run the app:**
   streamlit run src/app.py
