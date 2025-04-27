# app.py
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open('mlp_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit interface
def main():
    st.title('Diabetes Prediction App')

    pregnancies = st.number_input('Pregnancies', min_value=0)
    glucose = st.number_input('Glucose')
    blood_pressure = st.number_input('Blood Pressure')
    skin_thickness = st.number_input('Skin Thickness')
    insulin = st.number_input('Insulin')
    bmi = st.number_input('BMI')
    pedigree = st.number_input('Diabetes Pedigree Function')
    age = st.number_input('Age')

    if st.button('Predict'):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
        prediction = predict_diabetes(input_data)

        if prediction == 1:
            st.success('The person has diabetes.')
        else:
            st.success('The person does not have diabetes.')

if __name__ == '__main__':
    main()
