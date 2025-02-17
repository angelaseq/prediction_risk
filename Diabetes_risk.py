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
df = pd.read_csv("diabetes.csv")

# Dividi i dati in X (features) e y (target)
x = df.drop('Outcome', axis=1)
y = df['Outcome']

# Suddividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crea e allena il modello di regressione logistica
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Funzione per fare una previsione del rischio
def predict_risk(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = np.zeros(len(x.columns))
    
    # Assegna i valori delle variabili nel vettore
    data[0] = Pregnancies
    data[1] = Glucose
    data[2] = BloodPressure
    data[3] = SkinThickness
    data[4] = Insulin
    data[5] = BMI
    data[6] = DiabetesPedigreeFunction
    data[7] = Age
    
    probabilities = log_reg.predict_proba([data])
    
    # Calcola la probabilità di rischio e restituiscila come stringa
    risk_percentage = probabilities[0, 1] * 100
    risk_percentage_string = f"{risk_percentage:.2f}%"
    
    return risk_percentage_string

# Streamlit App
st.title("Diabetes Risk Prediction")

st.header("Inserisci i tuoi dati:")

# Widget per raccogliere i dati dell'utente
Pregnancies = st.slider("Figli", 0, 15, 2)
Glucose = st.slider("Glucosio", 50, 250, 80)
BloodPressure = st.slider("Pressione Sanguigna", 40, 150, 70)
Insulin = st.number_input("Insulina", min_value=0, max_value=800.0, value=70.0)
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Altezza (m)", min_value=1.0, max_value=2.5, value=1.75)
# Bottone per calcolare il BMI
if st.button('Calcola BMI'):
    # Calcola il BMI
    BMI = weight / (height ** 2)
Age = st.slider("Età", 1, 90, 30)

# Bottone per calcolare il rischio
result = predict_risk(Pregnancies, Glucose, BloodPressure, Insulin, BMI, Age)
st.success(f"Il tuo rischio di diabete è: {result}")
