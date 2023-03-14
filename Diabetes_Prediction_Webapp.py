import numpy as np
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

loaded_model = pickle.load(open("Trained_model.sav", 'rb'))


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
    page = st.selectbox('Select what you see',('Homepage', 'Data Analysis', 'Data Set'))
    
    if page == 'HomePage':
    
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
        
        
    elif page == 'Data Analysis':
        st.title("Exploratory Data Analysis")
        df = pd.read_csv(r"./dataset.csv")
        Outcome = df['Outcome']
        df.drop("Outcome", inplace=True, axis = 1)
        columns = st.radio(
            "Data to show:",
            (df.columns),
            horizontal=True
        )
        plt.grid()
        st.line_chart(df[columns])
        st.pyplot(sns.histplot(df[columns]).figure)
        show = st.checkbox("Show Data Description")
        if show:
            st.write("Description of data")
            st.write(df.describe())
        
        
    elif page=='Data Set':
        st.title("Here's the dataset on which ML model is based")
        df = pd.read_csv(r"C:\Users\Sahil Chhabra\Desktop\Streamlit\dataset.csv")
        # print(df.head())
        st.table(df)
    
    
    
    
if __name__ == '__main__':
    main()
