import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd
import os


model_path = os.path.join(os.path.dirname(__file__), "GDP_model.cbm")
model = CatBoostRegressor()
model.load_model(model_path)

st.header("GDP Predictor🌏",divider =True)
st.write("This model predicts a Country's Per Capita GDP based on suitable factors.  \nThe dataset used for this project can be found on kaggle : (https://www.kaggle.com/rutikbhoyar/gdp-prediction-dataset)")

region = st.selectbox("Region",("ASIA (EX. NEAR EAST)","BALTICS","C.W. OF IND. STATES","EASTERN EUROPE","LATIN AMER. & CARIB","NEAR EAST","NORTHERN AFRICA","NORTHERN AMERICA","OCEANIA","SUB-SAHARAN AFRICA","WESTERN EUROPE",))

col1,col2,col3 = st.columns(3)

with col1:
    population = st.number_input("Population",min_value=5000,step = 1000,value = 30000000)
    net_migration = st.number_input("Net migration ratio",min_value=-500.0,value=0.04)
    phones = st.number_input("Phones per 1000",value=233)

with col2:
    area = st.number_input("Area(per sq. mt.)",value = 600000)
    mortality = st.number_input("Infant mortality per 1000 births",value=35)
    birth_rate = st.number_input("Birth Rate per 1000",value=22)

with col3: 
    coastline = st.number_input("Coastline (coast/area ratio)",value = 21)
    literacy = st.number_input("Literacy %",value=77)
    death_rate = st.number_input("Death rate per 1000",value=9)

arable = st.slider("Arable land %",min_value =0,max_value=100,value=14)

pop_density = population/area

input_df = pd.DataFrame([{
    'region': region,
    'population': population,
    'area_(sq._mi.)': area,
    'pop._density_(per_sq._mi.)':pop_density,
    'coastline_(coast/area_ratio)': coastline,
    'net_migration': net_migration,
    'infant_mortality_(per_1000_births)': mortality,
    'literacy_(%)': literacy,
    'phones_(per_1000)': phones,
    'arable_(%)': arable,
    'birthrate': birth_rate,
    'deathrate': death_rate



}])

st.markdown("###")

if st.button("Predict GDP💰"):
    prediction = round(model.predict(input_df)[0],2)
    st.success(f"The predicted gdp is : ${prediction}")








