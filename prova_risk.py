#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit

# Fissa il seme per garantire la riproducibilità
np.random.seed(42)

# Carica i dati (inserisci il percorso corretto per il file CSV)
df = pd.read_csv("Liver_disease_data.csv")

# Pre-elaborazione dei dati
df = df.drop('LiverFunctionTest', axis=1)
df['Gender'] = df['Gender'].map({0: 1, 1: 0})
df = df.rename(columns={'Gender': 'Sex'})

# Dividi i dati in X (features) e y (target)
x = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Suddividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crea e allena il modello di regressione logistica
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Funzione per fare una previsione del rischio
def predict_risk(age, sex, weight, height, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension):
    data = np.zeros(len(x.columns))
    bmi = weight / (height ** 2)
    
    # Assegna i valori delle variabili nel vettore
    data[0] = age
    data[1] = sex
    data[2] = bmi
    data[3] = alcohol_consumption
    data[4] = smoking
    data[5] = genetic_risk
    data[6] = physical_activity
    data[7] = diabetes
    data[8] = hypertension
    
    probabilities = log_reg.predict_proba([data])
    
    # Calcola la probabilità di rischio e restituiscila come stringa
    risk_percentage = probabilities[0, 1] * 100
    risk_percentage_string = f"{risk_percentage:.2f}%"
    
    return risk_percentage_string

# Streamlit App
st.title("Liver Disease Risk Prediction")

st.header("Inserisci i tuoi dati:")

# Widget per raccogliere i dati dell'utente
age = st.slider("Età", 18, 100, 30)
sex = st.selectbox("Sesso", ("Maschio", "Femmina"))
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Altezza (m)", min_value=1.0, max_value=2.5, value=1.75)
alcohol_consumption = st.slider("Consumo di Alcol (unità a settimana)", 0, 100, 10)
st.write("1 unità alcolica corrisponde a un bicchiere di vino")
smoking = st.selectbox("Fumatore", ("No", "Sì"))
genetic_risk = st.selectbox("Rischio Genetico - Alto: parente di primo grado con malattia epatica; Medio: parente di secondo grado con malattia epatica; Basso: nessun parente con malattia epatica", ("Basso", "Medio", "Alto"))
physical_activity = st.slider("Attività Fisica (ore a settimana)", 0, 40, 5)
diabetes = st.selectbox("Diabete", ("No", "Sì"))
hypertension = st.selectbox("Ipertensione", ("No", "Sì"))

# Bottone per calcolare il rischio
if st.button("Calcola Rischio"):
    sex_encoded = 1 if sex == "Maschio" else 0
    smoking_encoded = 1 if smoking == "Sì" else 0
    genetic_risk_encoded = {"Basso": 0, "Medio": 1, "Alto": 2}[genetic_risk]
    diabetes_encoded = 1 if diabetes == "Sì" else 0
    hypertension_encoded = 1 if hypertension == "Sì" else 0

    # Calcola e mostra il rischio
    result = predict_risk(age, sex_encoded, weight, height, alcohol_consumption, smoking_encoded, genetic_risk_encoded, physical_activity, diabetes_encoded, hypertension_encoded)
    st.success(f"Il tuo rischio di malattia epatica è: {result}")

