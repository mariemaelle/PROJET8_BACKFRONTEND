##################################################################################################
### API CREDIT SCORING
##################################################################################################

"""
Sections principales:
1. Importation des bibliothèques et initialisation de l'API
2. Chargement des modèles et données
3. Fonction
4. Endpoint
"""

###############################################################################################################
# Import des bibliothèques et initialisation de l'API
#--------------------------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle
import os
import shap 

app = FastAPI()

# Définir les chemins relatifs à partir du dossier "api"
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Revenir au dossier "backend"




###############################################################################################################
# CHARGEMENT DES MODELES ET DONNEES

# Charger le modèle de machine learning
model_path = os.path.join(base_path, "model", "lightgbm_classifier_model", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Charger les données clients en CSV
data_path = os.path.join(base_path, "data", "sample_client_api.csv")
df = pd.read_csv(data_path)

# Charger les features importances en CSV
feature_importance_path = os.path.join(base_path, "data", "feature_importance.csv")
feature_importance_df = pd.read_csv(feature_importance_path)

# Charger le fichier contenant les descriptions des colonnes en CSV
file_path = os.path.join(base_path,'data', 'HomeCredit_columns_description.csv')
df_columns_description = pd.read_csv(file_path, encoding='ISO-8859-1')

# Utiliser le seuil pour la décision de prêt
THRESHOLD = 0.36



###############################################################################################################
# FONCTION

# Fonction utilitaire pour obtenir le top 10 des features
def get_top_10_features():
    return feature_importance_df['Feature'].head(10).tolist()




###############################################################################################################
# ENDPOINTS

#------------------------------------------------------------------------------------------------
# ENDPOINT: récupère données clients, calcule la probabilité de défaut, calcule les valeurs SHAP

@app.get("/client/{client_id}")
def get_client_info(client_id: int):
    # Rechercher les données du client par son ID
    client_data = df[df["SK_ID_CURR"] == client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")

    # Préparation des données pour la prédiction
    features = client_data.drop(columns=["SK_ID_CURR", "TARGET"])
    
    # Faire une prédiction avec le modèle chargé
    probability = model.predict_proba(features)[:, 1][0]
    decision = "Crédit accordé" if probability < THRESHOLD else "Crédit non accordé"

    # Extraire le modèle LightGBM depuis le pipeline
    lgbm_model = model.named_steps['lgbm']

    # Utiliser SHAP pour calculer les valeurs locales des features
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(features)
    
    # Créer un dictionnaire des valeurs SHAP associées aux noms des features
    shap_dict = {
        "features": list(features.columns),
        "shap_values": shap_values[0].tolist()  # shap_values[0] pour la classe positive car dans les classifications binaires, shap ne renvoie qu'une série de valeurs
    }

    # Créer un dictionnaire des valeurs des features du client
    client_feature_values = features.iloc[0].replace([np.inf, -np.inf], np.nan).to_dict()  # Récupère les valeurs des features du client

    # Remplacer les NaN par None (compatible avec JSON)
    client_feature_values = {k: (None if pd.isna(v) else v) for k, v in client_feature_values.items()}

    return {
        "client_id": client_id,
        "probability_of_default": probability,
        "decision": decision,
        "shap_values": shap_dict,
        "client_feature_values": client_feature_values
    }




#------------------------------------------------------------------------------------------------
# ENDPOINT: récupère la liste du top 10 des features importances

# Récupère la liste du top 10 features importances
@app.get("/feature-importance")
def get_feature_importance():
    try:
        # Retourner les 10 features les plus importantes
        top_10_features = feature_importance_df.head(10).to_dict(orient="records")
        return {"top_10_feature_importance": top_10_features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#------------------------------------------------------------------------------------------------
# ENDPOINT: récupère les données du top 10 des features importances pour visualisation

# Endpoint pour récupérer les 10 features les plus importantes et leurs données, ainsi que la target
@app.get("/feature-data")
def get_feature_data():
    try:
        # Obtenir les 10 features les plus importantes
        top_10_features = get_top_10_features()

        # Extraire les valeurs des features pour tout le dataset
        feature_data = {}

        # Remplacer les valeurs infinies et NaN par None, compatible avec JSON
        for feature in top_10_features:
            # Remplacer inf, -inf et NaN par None
            values = df[feature].replace([np.inf, -np.inf, np.nan], None)
            feature_data[feature] = values.tolist()  # Convertir en liste compatible JSON

        # Inclure la variable target dans la réponse (sans drop de valeurs)
        target = df['TARGET'].replace([np.inf, -np.inf, np.nan], None).tolist()

        return {
            "top_10_features": top_10_features,
            "feature_data": feature_data,
            "target": target  # Inclure target dans la réponse
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#------------------------------------------------------------------------------------------------
# ENDPOINT: récupère les descriptions des features pour visualisation

# récupérer les descriptions des features
@app.get("/column-description")
def get_column_description():
    try:
        # Extraire les colonnes "Row" et "Description"
        columns_data = df_columns_description[['Row', 'Description']].dropna().to_dict(orient='records')
        return {"columns_description": columns_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#------------------------------------------------------------------------------------------------
# ENDPOINT: vérifier que l'API fonctionne

@app.get("/")
def read_root():
    return {"message": "API is running!"}