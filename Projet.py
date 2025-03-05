import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Chargement des données
def load_data():
   df = pd.read_csv(r"C:\Users\Koureissi\Desktop\M2 IA SCHOOL\cours_streamlit\automobile_data.csv", delimiter=';')
   return df  # Return the dataframe

# Call the function to load the data
df = load_data()  # Now df is defined globally

# Print DataFrame info
print(df.shape)  # Check if DataFrame has rows and columns
print(df.head())  # Preview the first few rows

def preprocess_data(df):
    df = df.dropna()  # Suppression des valeurs manquantes
    df = pd.get_dummies(df, drop_first=True)  # Encodage des variables catégorielles
    return df

def train_models(X_train, y_train):
    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = model
    return results

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return mae, rmse

# Interface Streamlit
st.title("Prédiction des prix des voitures 🚗")

# Chargement des données
df = load_data()
st.write("### Aperçu des données :", df.head())

# Nettoyage et prétraitement
df = preprocess_data(df)
X = df.drop(columns=["price"])  # Supposons que la colonne cible est 'price'
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement des modèles
models = train_models(X_train, y_train)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculer RMSE manuellement
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    return mae, rmse

# Évaluation des modèles
st.write("## Évaluation des modèles")
for name, model in models.items():
    mae, rmse = evaluate_model(model, X_test, y_test)
    st.write(f"**{name}** → MAE: {mae:.2f}, RMSE: {rmse:.2f}")


# Sélection du modèle
selected_model_name = st.selectbox("Choisissez un modèle pour la prédiction :", list(models.keys()))
selected_model = models[selected_model_name]

# Interface de prédiction
st.write("## Faites une prédiction")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

if st.button("Prédire le prix"):
    input_df = pd.DataFrame([input_data])
    prediction = selected_model.predict(input_df)[0]
    st.success(f"Prix estimé du véhicule : {prediction:.2f} €")

st.write("Application réalisée avec Streamlit 🚀")