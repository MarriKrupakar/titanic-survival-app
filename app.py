
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('logistic_model.pkl', 'rb'))

st.title("Titanic Survival Prediction")

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 30)
sibsp = st.number_input('Siblings/Spouses Aboard', 0, 8, 0)
parch = st.number_input('Parents/Children Aboard', 0, 6, 0)
fare = st.number_input('Fare', 0.0, 512.0, 32.0)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

sex = 0 if sex == 'male' else 1
embarked_dict = {'S': 2, 'C': 0, 'Q': 1}
embarked = embarked_dict[embarked]

features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

if st.button('Predict'):
    result = model.predict(features)
    st.write("Survived" if result[0] == 1 else "Did not survive")
