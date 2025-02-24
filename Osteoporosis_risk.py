#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fissa il seme per garantire la riproducibilità
np.random.seed(42)

# Carica i dati (inserisci il percorso corretto per il file CSV)
df = pd.read_csv("osteoporosis.csv")

# Codifica le variabili categoriche
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Race/Ethnicity'] = le.fit_transform(df['Race/Ethnicity'])
df['Family History'] = le.fit_transform(df['Family History'])
df['Smoking'] = le.fit_transform(df['Smoking'])
df['Alcohol Consumption'] = le.fit_transform(df['Alcohol Consumption'])

# Dividi i dati in X (features) e y (target)
x = df.drop('Osteoporosis', axis=1)
y = df['Osteoporosis']

# Assicurati che X e y siano numerici
x = x.apply(pd.to_numeric, errors='coerce')  # Converte tutte le colonne in numerico
y = pd.to_numeric(y, errors='coerce')  # Converte la colonna target in numerico

# Suddividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crea e allena il modello di regressione logistica
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Funzione per fare una previsione del rischio
def predict_risk(Age, Gender, HormonalChanges, FamilyHistory, RaceEthnicity, BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures):
    data = np.zeros(len(x.columns))
    
    # Assegna i valori delle variabili nel vettore
    data[0] = Age
    data[1] = Gender
    data[2] = HormonalChanges
    data[3] = FamilyHistory
    data[4] = RaceEthnicity
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
st.title("Previsione del Rischio di Osteoporosi")

st.header("Inserisci i tuoi dati:")

# Widget per raccogliere i dati dell'utente
Age = st.slider("Età", 18, 100, 50)
Gender = st.selectbox("Sesso", ["Maschio", "Femmina"])
HormonalChanges = st.selectbox("Cambiamenti Ormonali", ["Sì", "No"])
FamilyHistory = st.selectbox("Storia Familiare di Osteoporosi", ["Sì", "No"])
RaceEthnicity = st.selectbox("Razza/Etnia", ["Caucasico", "Afroamericano", "Asiatico", "Latino", "Altro"])
BodyWeight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
CalciumIntake = st.number_input("Assunzione di Calcio (mg)", min_value=0.0, max_value=1500.0, value=1000.0)
VitaminDIntake = st.number_input("Assunzione di Vitamina D (UI)", min_value=0.0, max_value=1000.0, value=400.0)
PhysicalActivity = st.selectbox("Attività Fisica (settimanale)", ["Nessuna", "Leggera", "Moderata", "Intensa"])
Smoking = st.selectbox("Fumo", ["Sì", "No"])
AlcoholConsumption = st.selectbox("Consumo di Alcol", ["Sì", "No"])
MedicalConditions = st.selectbox("Condizioni Mediche", ["Sì", "No"])
Medications = st.selectbox("Assunzione di Farmaci", ["Sì", "No"])
PriorFractures = st.selectbox("Fratture Precedenti", ["Sì", "No"])

# Converte i valori categorici in numerici
Gender = 1 if Gender == "Femmina" else 0
HormonalChanges = 1 if HormonalChanges == "Sì" else 0
FamilyHistory = 1 if FamilyHistory == "Sì" else 0
RaceEthnicity = le.transform([RaceEthnicity])[0]
PhysicalActivity = ["Nessuna", "Leggera", "Moderata", "Intensa"].index(PhysicalActivity)
Smoking = 1 if Smoking == "Sì" else 0
AlcoholConsumption = 1 if AlcoholConsumption == "Sì" else 0
MedicalConditions = 1 if MedicalConditions == "Sì" else 0
Medications = 1 if Medications == "Sì" else 0
PriorFractures = 1 if PriorFractures == "Sì" else 0

# Bottone per calcolare il rischio
if st.button('Calcola rischio di osteoporosi'):
    result = predict_risk(Age, Gender, HormonalChanges, FamilyHistory, RaceEthnicity, BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures)
    st.success(f"Il tuo rischio di osteoporosi è: {result}")
