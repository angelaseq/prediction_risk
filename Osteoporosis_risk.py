import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Fissa il seme per garantire la riproducibilità
np.random.seed(42)

# Carica i dati (inserisci il percorso corretto per il file CSV)
df = pd.read_csv("Osteoporosis.csv")

# Codifica le variabili categoriche
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Normalizza le colonne numeriche
scaler = MinMaxScaler()

# Normalizza solo la colonna 'Age'
df['Age'] = scaler.fit_transform(df[['Age']])

# Dividi i dati in X (features) e y (target)
x = df.drop('Osteoporosis', axis=1)
y = df['Osteoporosis']

# Suddividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crea e allena il modello di regressione logistica
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Funzione per fare una previsione del rischio
def predict_risk(Age, Gender, HormonalChanges, FamilyHistory, BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures):
    # Codifica variabili categoriche
    Gender = 1 if Gender == "Femmina" else 0
    HormonalChanges = 1 if HormonalChanges == "Postmenopausa" else 0
    FamilyHistory = 1 if FamilyHistory == "Sì" else 0
    
    # Codifica il BodyWeight
    if BodyWeight == "Normale":
        BodyWeight = 0
    elif BodyWeight == "Sottopeso":
        BodyWeight = -1
    else:
        BodyWeight = 1  # Sovrappeso
    
    # Codifica l'assunzione di calcio e vitamina D
    CalciumIntake = 1 if CalciumIntake == "Adeguato" else 0
    VitaminDIntake = 1 if VitaminDIntake == "Sufficiente" else 0
    
    # Codifica l'attività fisica
    PhysicalActivity = 1 if PhysicalActivity == "Attivo" else 0
    
    # Codifica il fumo
    Smoking = 1 if Smoking == "Sì" else 0
    
    # Codifica il consumo di alcol
    AlcoholConsumption = {"No": 0, "Moderato": 1, "Alto": 2}[AlcoholConsumption]
    
    # Codifica altre patologie
    MedicalConditions = 1 if MedicalConditions == "Artrite reumatoide" else 0  # Aggiungi altre patologie
    
    # Codifica l'assunzione di farmaci
    Medications = 1 if Medications == "Corticosteroidi" else 0
    
    # Codifica fratture precedenti
    PriorFractures = 1 if PriorFractures == "Sì" else 0
    
    # Normalizza l'età dell'utente (applica lo stesso scaler usato durante l'allenamento del modello)
    age_normalized = scaler.transform([[Age]])[0][0]
    
    # Crea il vettore di input per la previsione
    data = np.zeros(len(x.columns))
    data[0] = age_normalized  # Età normalizzata
    data[1] = Gender
    data[2] = HormonalChanges
    data[3] = FamilyHistory
    data[4] = BodyWeight
    data[5] = CalciumIntake
    data[6] = VitaminDIntake
    data[7] = PhysicalActivity
    data[8] = Smoking
    data[9] = AlcoholConsumption
    data[10] = MedicalConditions
    data[11] = Medications
    data[12] = PriorFractures
    
    # Calcola la probabilità di rischio usando il modello
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
FamilyHistory = st.selectbox("Storia Familiare di Osteoporosi", ["Si", "No"])
BodyWeight = st.selectbox("Peso", ["Normale", "Sottopeso", "Sovrappeso"])
CalciumIntake = st.selectbox("Assunzione di Calcio", ["Adeguato", "Basso"])
VitaminDIntake = st.selectbox("Assunzione di Vitamina D", ["Sufficiente", "Insufficiente"])
PhysicalActivity = st.selectbox("Attività Fisica", ["Attivo", "Sedentario"])
Smoking = st.selectbox("Fumo", ["Si", "No"])
AlcoholConsumption = st.selectbox("Consumo di Alcol", ["No", "Moderato", "Alto"])
MedicalConditions = st.selectbox("Altre patologie", ["No", "Artrite reumatoide", "Ipertiroidismo"])
Medications = st.selectbox("Assunzione di Farmaci", ["Corticosteroidi", "No"])
PriorFractures = st.selectbox("Fratture Precedenti", ["Si", "No"])

# Bottone per calcolare il rischio
if st.button('Calcola rischio di osteoporosi'):
    result = predict_risk(Age, Gender, HormonalChanges, FamilyHistory, BodyWeight, CalciumIntake, VitaminDIntake, PhysicalActivity, Smoking, AlcoholConsumption, MedicalConditions, Medications, PriorFractures)
    st.success(f"Il tuo rischio di osteoporosi è: {result}")


