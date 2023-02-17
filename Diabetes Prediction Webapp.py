import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("./Trained_model.sav", 'rb'))


#creating a function for prediction

def diabetes_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1,-1)
    prediction = loaded_model.predict(input_data)
    if prediction[0]==0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user
        #Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    Bloodpressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin levels")
    BMI = st.text_input("Body Mass Index")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of person")
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
