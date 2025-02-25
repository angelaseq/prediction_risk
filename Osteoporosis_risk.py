#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fissa il seme per garantire la riproducibilità
np.random.seed(42)

# Carica i dati (inserisci il percorso corretto per il file CSV)
df = pd.read_csv("Osteoporosis.csv")

# Dividi i dati in X (features) e y (target)
x = df.drop('Osteoporosis', axis=0)
y = df['Osteoporosis']

# Suddividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crea e allena il modello di regressione logistica
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Funzione per fare una previsione del rischio
def predict_risk(Age, Gender, HormonalChanges, FamilyHistory, Ethnicity, BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures):
    data = np.zeros(len(x.columns))
    
    # Assegna i valori delle variabili nel vettore
    data[0] = Age
    data[1] = Gender
    data[2] = HormonalChanges
    data[3] = FamilyHistory
    data[4] = Ethnicity
    data[5] = BodyWeight
    data[6] = CalciumIntake
    data[7] = VitaminDIntake
    data[8] = PhysicalActivity
    data[9] = Smoking
    data[10] = AlcoholConsumption
    data[11] = MedicalConditions
    data[12] = Medications
    data[13] = PriorFractures
    
    probabilities = log_reg.predict_proba([data])
    
    # Calcola la probabilità di rischio e restituiscila come stringa
    risk_percentage = probabilities[0, 1] * 100
    risk_percentage_string = f"{risk_percentage:.2f}%"
    
    return risk_percentage_string

# Streamlit App
st.title("Osteoporosis Risk Prediction")

st.header("Inserisci i tuoi dati:")

# Widget per raccogliere i dati dell'utente
Age = st.slider("Età", 1, 90, 30)
Gender = st.selectbox("Sesso", ["Maschio", "Femmina"])
HormonalChanges = st.selectbox("Cambiamenti Ormonali", ["Postmenopausa", "Normale"])
FamilyHistory = st.selectbox("Storia Familiare di Osteoporosi", ["Sì", "No"])
BodyWeight = st.selectbox("Peso", ["Normale", "Sottopeso", "Sovrappeso"])
CalciumIntake = st.selectbox("Assunzione di Calcio", ["Adeguato", "Basso"])
VitaminDIntake = st.selectbox("Assunzione di Vitamina D", ["Sufficiente", "Insufficiente"])
PhysicalActivity = st.selectbox("Attività Fisica", ["Attivo", "Sedentario"])
Smoking = st.selectbox("Fumo", ["Sì", "No"])
AlcoholConsumption = st.selectbox("Consumo di Alcol", ["No", "Moderato", "Alto"])
MedicalConditions = st.selectbox("Altre patologie", ["No", "Artrite reumatoide", "Ipertiroidismo"])
Medications = st.selectbox("Assunzione di Farmaci", ["Corticosteroidi", "No"])
PriorFractures = st.selectbox("Fratture Precedenti", ["Sì", "No"])



# Bottone per calcolare il rischio
if st.button('Calcola rischio di osteoporosi'):
    result = predict_risk(Age, Gender, HormonalChanges, FamilyHistory, "caucasico", BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures)
    st.success(f"Il tuo rischio di osteoporosi è: {result}")
