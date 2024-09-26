import sys
import os
# Ajouter le chemin du dossier "BACKEND_FRONTEND" au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.main import app  # Importe l'application FastAPI

# Créer un client de test pour simuler les requêtes HTTP à l'API
client = TestClient(app)

# Test 1: Vérifier que l'API fonctionne
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running!"}

# Test 2: Vérifier les informations du client
def test_get_client_info():
    # Utiliser un ID de client valide présent dans le fichier CSV 
    client_id = 346699
    response = client.get(f"/client/{client_id}")
    assert response.status_code == 200

    # Vérifier la structure de la réponse
    json_response = response.json()
    assert "client_id" in json_response
    assert "probability_of_default" in json_response
    assert "decision" in json_response
    assert "shap_values" in json_response
    assert "client_feature_values" in json_response

# Test 3: Vérifier les features importantes
def test_get_feature_importance():
    response = client.get("/feature-importance")
    assert response.status_code == 200

    # Vérifier que la réponse contient bien les top 10 features
    json_response = response.json()
    assert "top_10_feature_importance" in json_response
    assert len(json_response["top_10_feature_importance"]) == 10