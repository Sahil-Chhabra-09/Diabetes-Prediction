import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("Trained_model.sav", 'rb'))


#creating a function for prediction

def diabetes_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1,-1)
    input_data
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
    
    Pregnancies = st.text_input("Number of Pregnancies (0-17)")
    Glucose = st.text_input("Glucose level (0-199)")
    Bloodpressure = st.text_input("Blood Pressure (0-122)")
    SkinThickness = st.text_input("Skin Thickness (0-99)")
    Insulin = st.text_input("Insulin levels (0-846)")
    BMI = st.text_input("Body Mass Index (0.0-67.1)")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value (0.078-2.42)")
    Age = st.text_input("Age of person (21-81)")
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
