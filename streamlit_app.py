import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GLICO Insurance Cost Predictor",
    layout="centered"
)

# ---------------- LOAD LOGO ----------------
logo = Image.open("health_logo.JPG")

st.image(logo, width=180)

# ---------------- TITLE ----------------
st.title("GLICO Ghana Insurance Cost Prediction App")

st.markdown("""
This machine learning application predicts estimated health insurance charges
based on client information.

The model was trained using Linear Regression.

By: Gabriel Agana Anongwin  
Doctor of Pharmacy Class of 2026 (KNUST)  
Data Science Student at ALX Ghana (Cohort 10)
""")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open('linear_regression_insurance_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------- USER INPUTS ----------------
st.header("Enter Client Information")

age = st.slider("Age", 18, 100, 30)

bmi = st.slider(
    "BMI (Body Mass Index)",
    15.0,
    50.0,
    25.0
)

children = st.slider(
    "Number of Children",
    0,
    5,
    1
)

gender = st.radio(
    "Gender",
    ["Female", "Male"]
)

smoker = st.radio(
    "Smoker",
    ["No", "Yes"]
)

# Convert categorical variables
sex_male = 1 if gender == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("Predict Insurance Charges"):

    # Create dataframe
    input_data = pd.DataFrame(
        [[age, bmi, children, sex_male, smoker_yes]],
        columns=[
            'age',
            'bmi',
            'children',
            'sex_male',
            'smoker_yes'
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Prevent negative predictions
    prediction = max(prediction, 0)

    # ---------------- RESULTS ----------------
    st.success(
        f"Estimated Insurance Cost: ${prediction:,.2f}"
    )

    # ---------------- WARNINGS ----------------
    if prediction < 5000:
        st.info(
            "Low estimated insurance cost. "
            "This is common for younger non-smokers with healthy BMI values."
        )

    elif prediction < 15000:
        st.warning(
            "Moderate insurance cost predicted."
        )

    else:
        st.error(
            "High insurance cost predicted. "
            "Smoking status, BMI, or age may significantly increase costs."
        )

# ---------------- MODEL PERFORMANCE ----------------
st.markdown("---")

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", "0.7813")
    st.metric("MAE", "$4,209")

with col2:
    st.metric("RMSE", "$5,827")
    st.metric("MSE", "33.96M")

# ---------------- EXPLANATION ----------------
st.subheader("What These Metrics Mean")

st.markdown("""
### R² Score (78.13%)
The model explains approximately **78%** of the variation in insurance costs.
This indicates reasonably strong predictive performance.

### MAE : Mean Absolute Error
On average, predictions differ from actual insurance charges by about **$4,209**.

### RMSE : Root Mean Squared Error
Typical prediction error is around **$5,827**.
This metric penalizes larger mistakes more heavily.

### Why Negative Predictions Can Happen
Linear Regression can sometimes predict negative values because the model uses a mathematical line equation.

Your model has a negative intercept:

```python
Intercept = -11908.61""")
