import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Fissa il seme per garantire la riproducibilità
np.random.seed(42)

# Simula il caricamento del modello e i dati
# Sostituisci con il tuo modello addestrato e dataset
x_columns = [
    "Age", "Sex", "BMI", "Alcohol_Consumption", "Smoking", 
    "Genetic_Risk", "Physical_Activity", "Diabetes", "Hypertension"
]

# Crea un Logistic Regression come esempio
log_reg = LogisticRegression(max_iter=1000, random_state=42)  # Aggiungi random_state

# Crea un DataFrame di esempio con valori casuali per addestrare il modello
# Aumenta la varianza dei dati per simulare una distribuzione più realistica
sample_data = pd.DataFrame({
    "Age": np.random.randint(18, 80, size=100),
    "Sex": np.random.randint(0, 2, size=100),  # 0 = femmina, 1 = maschio
    "BMI": np.random.uniform(18.5, 40.0, size=100),  # BMI variabili tra 18.5 e 40.0
    "Alcohol_Consumption": np.random.randint(0, 21, size=100),  # Alcol a settimana (0-20 unità)
    "Smoking": np.random.randint(0, 2, size=100),  # 0 = No, 1 = Sì
    "Genetic_Risk": np.random.randint(0, 3, size=100),  # 0 = Basso, 1 = Medio, 2 = Alto
    "Physical_Activity": np.random.randint(0, 10, size=100),  # Ore settimanali di attività fisica
    "Diabetes": np.random.randint(0, 2, size=100),  # 0 = No, 1 = Sì
    "Hypertension": np.random.randint(0, 2, size=100)  # 0 = No, 1 = Sì
})
sample_data['Diagnosis'] = np.random.randint(0, 2, size=100)  # Diagnosi casuale (0 = sano, 1 = malato)

X = sample_data.drop('Diagnosis', axis=1)
y = sample_data['Diagnosis']
log_reg.fit(X, y)

def predict_risk(age, sex, weight, height, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension):
    bmi = weight / (height ** 2)
    data = np.array([[age, sex, bmi, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension]])
    probabilities = log_reg.predict_proba(data)
    risk_percentage = probabilities[0, 1] * 100
    return f"{risk_percentage:.2f}%"

# Streamlit App
st.title("Liver Disease Risk Prediction")

st.header("Inserisci i tuoi dati:")
age = st.slider("Età", 18, 100, 30)
sex = st.selectbox("Sesso", ("Maschio", "Femmina"))
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Altezza (m)", min_value=1.0, max_value=2.5, value=1.75)
alcohol_consumption = st.slider("Consumo di Alcol (unità a settimana)", 0, 100, 10)
smoking = st.selectbox("Fumatore", ("No", "Sì"))
genetic_risk = st.selectbox("Rischio Genetico", ("Basso", "Medio", "Alto"))
physical_activity = st.slider("Attività Fisica (ore a settimana)", 0, 40, 5)
diabetes = st.selectbox("Diabete", ("No", "Sì"))
hypertension = st.selectbox("Ipertensione", ("No", "Sì"))

if st.button("Calcola Rischio"):
    sex_encoded = 1 if sex == "Maschio" else 0
    smoking_encoded = 1 if smoking == "Sì" else 0
    genetic_risk_encoded = {"Basso": 0, "Medio": 1, "Alto": 2}[genetic_risk]
    diabetes_encoded = 1 if diabetes == "Sì" else 0
    hypertension_encoded = 1 if hypertension == "Sì" else 0

    result = predict_risk(age, sex_encoded, weight, height, alcohol_consumption, smoking_encoded, genetic_risk_encoded, physical_activity, diabetes_encoded, hypertension_encoded)
    st.success(f"Il tuo rischio di malattia epatica è: {result}")



