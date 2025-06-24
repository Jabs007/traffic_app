# 🚦 Smart Traffic Prediction System

An interactive web application built with **Streamlit** that predicts traffic conditions using machine learning. It supports **traffic condition prediction**, **exploratory data analysis**, and **visual reporting** from smart mobility data.

![App Preview](https://via.placeholder.com/800x300.png?text=Smart+Traffic+Prediction+App)

---

## 🌟 Features

- 🧠 **Predict traffic condition** using a trained ML model
- 📊 **EDA dashboard** to explore traffic data trends
- 🗂️ **SQLite database** logging of predictions
- 📈 **Interactive charts** powered by Plotly
- 🎞️ **Animations** using Lottie for an enhanced UI

---

## 🏗️ Project Structure

```bash
traffic_app/
├── app.py                     # Main Streamlit app
├── traffic_model.pkl          # Trained ML model
├── label_encoders.pkl         # Encoders for categorical data
├── feature_columns.pkl        # Ordered list of feature columns
├── mobility_with_new_features.csv  # Input dataset
├── prediction_history.db      # SQLite DB for logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
