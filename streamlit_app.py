import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title('Insurance Cost Prediction App')
st.write('Enter the client details to predict their insurance charges.')

# Load the trained model
@st.cache_resource # Cache the model loading for better performance
def load_model():
    with open('linear_regression_insurance_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Input fields for user
age = st.slider('Age', 18, 100, 30)
bmi = st.slider('BMI', 15.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 5, 1)
sex_male = st.radio('Gender', ['Female', 'Male']) == 'Male'
smoker_yes = st.radio('Smoker', ['No', 'Yes']) == 'Yes'

# When the user clicks the 'Predict' button
if st.button('Predict Insurance Charges'):
    # Create a DataFrame from user input, ensuring correct column order and types
    input_data = pd.DataFrame([[age, bmi, children, sex_male, smoker_yes]],
                              columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Insurance Charges: **${prediction:.2f}**")

st.markdown("""
--- 
#### Model Features:
- Age: Age of the primary beneficiary
- BMI: Body mass index
- Children: Number of children covered by health insurance
- Sex_male: Gender of the beneficiary (True for Male, False for Female)
- Smoker_yes: Smokes or not (True for Smoker, False for Non-Smoker)
""")
