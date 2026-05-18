# 🏥 Insurance Cost Prediction App (ML Project)

## 📌 Project Overview
This is a Machine Learning web application that predicts **health insurance charges** based on user input features such as age, BMI, smoking status, gender, and number of children.

The model is built using **Linear Regression** and deployed using **Streamlit** for an interactive user interface.

This project is presented as a **case study for GLICO Ghana**, demonstrating how machine learning can support insurance risk estimation.

---

## 🚀 Live Demo
https://glicoinsurance.streamlit.app/

---

## 🧠 Problem Statement
Insurance companies need to estimate the cost of coverage for clients based on health and demographic factors. Manual estimation can be inefficient and inconsistent.

This project solves that by building a predictive model that estimates insurance costs automatically.

---

## 📊 Dataset Features

The model uses the following input features:

- **Age** → Age of the insured person  
- **BMI** → Body Mass Index  
- **Children** → Number of dependents  
- **Sex (Male/Female)** → Gender encoded as binary  
- **Smoker (Yes/No)** → Smoking status encoded as binary  

---

## 🤖 Machine Learning Model

- Algorithm: **Linear Regression**
- Library: `scikit-learn`
- Output: Predicted insurance charges (USD)

### Model Performance:
- **R² Score:** 0.7813  
- **MAE:** $4,209.50  
- **RMSE:** $5,827.35  
- **MSE:** 33,957,993.04  

---

## 📉 Why Predictions Can Be Negative
Linear Regression can sometimes produce negative values because it is a linear equation.

To handle this:
- Predictions are capped at **0 minimum**
- This ensures realistic insurance cost outputs

---

## 🖥️ Web App Features

- Interactive sliders for input
- Real-time prediction
- GLICO-themed UI
- Logo integration
- Model performance summary
- Warning system for low/moderate/high risk predictions

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 📊
- Pandas
- NumPy
- Scikit-learn
- Pickle (model serialization)
- PIL (image handling)

---

## 📂 Project Structure
