import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
image = Image.open('./titanic.jpg')

st.set_page_config(
    page_title="Titanic project",
    page_icon="⛴"
)

st.markdown("# Titanic project")
st.image(image, caption='Titanic project')

st.markdown("## Fill in the passenger information")

clazz = st.selectbox(
    'Class:',
    [1, 2, 3])


sex = st.selectbox(
    'Sex:',
    ['male', 'female'])    


age = st.number_input(
    'Age:',
    min_value=1,
    max_value=80
)

fare = st.number_input(
    'Fare',
    min_value=0.0,
    max_value=512.32,
    step=0.01
)

embarked = st.selectbox(
    'Embarked:',
    ['S', 'C', 'Q', np.nan])

d = {
    'Pclass':[clazz],
    'Sex':[sex],
    'Age':[age],
    'Fare':[fare],
    'Embarked':[embarked]
}   

df = pd.DataFrame(d)
st.dataframe(df)

with open(r"../models/model.pickle", "rb") as input_file:
   model = pickle.load(input_file)

r = model.predict_proba(df)
proba = r[:, 1][0]
if proba > 0.5:
    st.success(f"Survive: {np.round(proba, 2)}")
else:
    st.error(f"Survive: {np.round(proba, 2)}")

