#importing Necessary Libraries

import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

model = pkl.load(open('MIPML.pkl', 'rb'))

st.('🩺 Medical Insurance Premium Predictor (Linear Regression)')
st.markdown('This app predicts the **insurance cost** based on your personal and lifestyle information.')

gender = st.selectbox('Choose Gender', ['Male', 'Female'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.slider('Enter BMI', 5.0, 100.0)
children = st.slider('Number of Children', 0, 5)

gender = 0 if gender == 'Female' else 1
smoker = 1 if smoker == 'Yes' else 0

if region == 'Southeast':
    region = 0
elif region == 'Southwest':
    region = 1
elif region == 'Northeast':
     region = 2
else:
    region = 3

input_data = np.array([age, gender, bmi, children, smoker, region], dtype=float).reshape(1, -1)

if st.button('Predict'):
    predicted_prem = model.predict(input_data)
    display_string = f'💰 **Estimated Insurance Premium:** ${round(predicted_prem[0], 2)}'
    st.success(display_string)