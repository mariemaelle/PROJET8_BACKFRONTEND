# PROJET8_BACKFRONTEND
# Credit Scoring App

## Description
Le **Credit Scoring App** est un projet comportant une API et un tableau de bord interactif conçu pour aider à la prise de 
décision dans l'octroi de crédits pour un client sélectionné. L'API utilise un modèle de machine 
learning entrainé sur un historique de plus de 300000 clients de la société **Prêt à dépenser** pour prédire les 
probabilités de défaut de paiement. L'interface interagit avec l'API et permet ainsi de visualiser :
 - les résulats de l'API, 
 - les caractéristiques principales qui ont conduit à la décision d'octroi de crédit (acceptée ou refusée),
 - la distribution globale des principales caractéristiques conduisant au calcul de la probabilité de défaut,
 - où se situe le client choisi par rapport aux autres clients pour comprendre les résultats et potentiellement réévaluer la décision finale.

## Fonctionnalités principales
- Sélection du client à partir d'un identifiant unique et décision d'octroi de crédit.
- Visualisation des probabilités de défaut et de l'importance des caractéristiques ayant conduit à la prédiction.
- Visualisation des distributions des valeurs des caractéristiques et diagramme de dispersion par couple de caractéristiques les plus importantes du modèle.
- Visualisation de l'importance des 10 caractéristiques les plus importantes du modèle de prédiction.
- Onglet de sélection des caractéristiques pour obtenir leur description.
- Interface web interactive basée sur **Streamlit** et API construite avec **FastAPI** et déployée avec gcloud app engine.

## Prérequis
- Python 3.9 ou version supérieure
- Google Cloud SDK (pour le déploiement sur Google Cloud)
- GitHub (pour le déploiement avec Streamlit cloud)
- Streamlit Cloud (pour le déploiement de l'interface avec Streamlit Cloud)
- Bibliothèques Python mentionnées dans le `requirements.txt` à la racine pour l'API et dans le sous-dossier Streamlit_app pour le déploiement de l'interface web.

## Utilisation

### Commandes pour utilisation en local
*API*
pip install -r requirements.txt        # Pour installer les dépendances de l'API FastAPI
uvicorn api.main_projet8:app --reload  # Pour exécuter l'API en local
http://127.0.0.1:8000/docs             # Pour visualiser la documentation de l'API

*Streamlit Interface*
streamlit run streamlit_app/dashboard_projet8.py     # Pour lancer l'interface streamlit en local
http://localhost:8501                                # Accès à l'interface streamlit

### Utilisation de l'interface
- Entrez l'ID d'un client pour voir sa décision de crédit et sa probabilité de défaut
- Naviguez à travers les onglets pour voir l'analyse détaillée des caractéristiques importantes
- Pour interpréter les caractéristiques, choisir une caractéristique dans l'onglet "Description" et lire son contenu.

### Déploiement de l'API sur Google Cloud App Engine
ID du projet: projet8-credit-risk
gcloud auth login                                 # S'authentifier avec Google CLI
gcloud config set project projet8-credit-risk     # Configure mon projet gcloud
gcloud projects list                              # Trouver l'ID du projet dans ma liste
gcloud config list                                # Vérifie que la session est active et bien configurée
gcloud services enable appengine.googleapi.com    # Vérfier mes services activés sur google app engine 
                                                  # (autorisations nécessaires à modifier directement dans la console gcloud: 
                                                  # IAM --> choisir le compte de service --> modifier --> Ajouter un autre role --> Administrateur storage)
gcloud app deploy                                 # Déploiement de l'API après avoir créé son dossier de travail
gcloud app logs tail -s default                   # Pour comprendre les bugs de déploiement

### Déploiement de l'interface Streamlit
- Dans le code Streamlit: s'assurer de modifier l'URL de l'API déployée sur le cloud
- Pusher les modifications sur GitHub
- S'assurer que le dépôt GitHub est public
- Suivre les instructions sur Streamlit Cloud


