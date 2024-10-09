##################################################################################################
### INTERFACE CREDIT SCORING
##################################################################################################

"""
Sections principales:
1. Importation des bibliothèques et lien vers l'API
2. Choix du client et décision d'octroi de crédit
3. Onglet 1: Visualisations de la décision et de l'importance locale des caractéristiques
4. Onglet 2: Visualisation de l'importance globale des caractéristiques dans le modèle
5. Onglet 3: Visualisation de la distribution des caractéristiques
6. Onglet 4: Visualisation bi-variée des caractéristiques
7. Onglet 5: Description des caractéristiques
"""





###################################################################################################
# Import des bibliothèques et lien vers l'API
#--------------------------------------------------------------------------------------------------

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import tempfile
import base64
import math
import plotly.graph_objects as go

# Adresse de l'API
api_url = "http://127.0.0.1:8000"

# Threshold pour la décision
THRESHOLD = 0.36

#-------------------------------------------------------------------------------------------------
# Création des onglets

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Choisissez une section:", 
                             ["Accueil",
                              "Le client: décision d'octroi de crédit", 
                              "Distribution des Caractéristiques", 
                              "Analyse Bi-variée",
                              "Le modèle global", 
                              "Description des caractéristiques"])





###################################################################################################
# Choix du client et décision d'octroi de crédit
#--------------------------------------------------------------------------------------------------

# Description des Sous-parties:
# - Introduction au tableau de bord et Initialisation des variables
# - Bouton d'obtention des informations du client
# - Affichage de la décision d'octroi de crédit après avoir cliqué sur le bouton
# - Panneau latéral pour les 10 caractéristqiues principales

#--------------------------------------------------------------------------------------------------
# Introduction au tableau de bord et initialisation des variables

if selection == "Accueil":
    st.title("Credit Scoring")
    st.markdown(""" **Bienvenue sur le tableau de bord interactif d’octroi de crédit de la société Prêt à dépenser!**  
    En entrant l'ID d'un client, vous obtenez une décision d'octroi de crédit basée sur un modèle de machine learning 
    entraîné sur les données historiques de plus de 300 000 clients. Les différentes sections de ce tableau de bord vous 
    permettent d’avoir une vue détaillée des facteurs déterminants dans la décision d’octroi de crédit et de faciliter 
    une potentielle réévaluation de la décision.
            """)






###################################################################################################
# Onglet 1 : Décision d'octroi de crédit et feature importance locale
#--------------------------------------------------------------------------------------------------

# Description: Visualisation de :
# - la décision d'octroi de crédit sous forme de compteur avec indication de la valeur seuil de décision.
# - la feature importance locale (valeurs shap)

if selection == "Le client: décision d'octroi de crédit":
    st.header("Décision d'octroi de crédit du client")

    client_id = st.text_input("Entrez l'ID du Client")

    # Initialiser les variables pour éviter l'erreur si le bouton n'est pas cliqué
    data = None
    probability = None
    decision = None
    shap_values = None
    features = None
    client_values = {}  # Initialisation pour éviter les erreurs
    top_10_features = []  # Initialisation pour récupérer le top 10 des features


    ##################################################################################################
    # Récupérer le top 10 des features via l'API
    feature_importance_endpoint = f"{api_url}/feature-importance"
    feature_response = requests.get(feature_importance_endpoint)

    if feature_response.status_code == 200:
        # Récupérer les 10 features les plus importantes
        feature_data = feature_response.json()
        top_10_features = [feature["Feature"] for feature in feature_data["top_10_feature_importance"]]


    #--------------------------------------------------------------------------------------------------
    # Bouton d'obtention des informations du client
    if st.button("Obtenir les Informations du Client"):
        st.session_state['client_id'] = client_id  # Stocker client_id dans session_state
        # Construire l'URL de l'API pour ce client
        endpoint = f"{api_url}/client/{client_id}"
        
        # Envoyer une requête GET à l'API
        response = requests.get(endpoint)

        if response.status_code == 200:
            # Récupérer les données JSON renvoyées par l'API
            data = response.json()
            probability = data['probability_of_default']
            decision = data["decision"]
            
            # Récupérer les valeurs SHAP
            shap_endpoint = f"{api_url}/client/{client_id}"
            shap_response = requests.get(shap_endpoint)
            
            if shap_response.status_code == 200:
                shap_data = shap_response.json()
                shap_values = np.array(shap_data["shap_values"]["shap_values"])
                features = shap_data["shap_values"]["features"]
                
                # Récupérer les valeurs du client uniquement pour les 10 features importantes
                client_values = {k: v for k, v in shap_data.get("client_feature_values", {}).items() if k in top_10_features}
            else:
                shap_values = None
                features = None
                client_values = {}
        else:
            st.error("Erreur lors de la récupération des informations du client.")
            data = None
            probability = None
            decision = None
            shap_values = None
            features = None
            client_values = {}



    # -------------------------------------------------------------------------------------
    # Affichage de la décision d'octroi de crédit après avoir cliqué sur le bouton

    if data:
        decision_color = "#008BFB" if decision == "Crédit accordé" else "#FF005E"
        st.markdown(f"""
            <div style='display: inline-block; padding: 10px 20px; border: 2px solid {decision_color}; 
                        border-radius: 10px; color: {decision_color}; font-size: 24px; font-weight: bold;'>
                {decision}
            </div>
        """, unsafe_allow_html=True)

    # Afficher les informations du client et la décision
    if data:
        # Visualisation de la probabilité sous forme de compteur
        st.write("### Visualisation de la Probabilité de Défaut")
        st.markdown(""" Le compteur indique la probabilité (en pourcentage) que le client puisse faire défaut,
        c'est à dire qu'il ne rembourse pas son prêt.
        Nous considérons que si la probabilité de défaut dépasse 36%, le risque de ne pas rembourser le crédit
        est trop important pour la société. Si le client, indiqué par la bande noire sur le compteur, est situé
        en zone bleue, il remboursera probablement son prêt. S'il se situe en zone rose, le client ne remboursera probablement
        pas son prêt. Le nombre noir
        indiqué au milieu du graphique indique la valeur de la probabilité de défaut du client. Le nombre inscrit
        en vert indique la différence entre la valeur seuil de risque et la valeur de probabilité de défaut du client.  
        ---  
        Si le client se situe proche du seuil de décision, vous pouvez regarder en détail les caractéristiques 
        du client qui ont contribué à cette décision grâce à la figure des contributions des caractéristiques 
        ci-dessous et réévaluer votre décision. """)

        

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de Défaut", 'font': {'size': 24}},
            delta={'reference': 36, 'font': {'size': 20}},
            number={'font': {'size': 60, 'color': 'black'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 20}},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 36], 'color': '#008BFB'},  # Couleur pour en dessous du seuil
                    {'range': [36, 100], 'color': '#FF005E'}  # Couleur pour au-dessus du seuil
                ],
                'threshold': {
                    'line': {'color': "darkblue", 'width': 4},
                    'thickness': 0.75,
                    'value': 36
                }
            }
        ))

        # Afficher le compteur dans Streamlit
        st.plotly_chart(fig_gauge)

        #------------------------------------------------------------------------------------------
        # VISUALISATION DES IMPORTANCES LOCALES

        # Afficher le plot SHAP si les valeurs sont disponibles
        if shap_values is not None:
            st.write("### Contribution des caractéristiques du client")
            st.markdown("""  
            La figure montre les principales caractéristiques du client qui ont contribué à la décision d'octroi de crédit.
            Les caractéristiques roses ont contribué à augmenter la probabilité de défaut du client
            tandis que les caractéristiques bleues ont contribué en la faveur de l'octroi du crédit au client.""")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                shap.force_plot(0, shap_values, features, matplotlib=True, show=False)
                plt.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
                plt.close()

            st.image(tmpfile.name)
        else:
            st.error("Erreur lors de la récupération des valeurs SHAP.")











###################################################################################################
# Onglet 3 : Distribution des caractéristiques
#-----------------------------------------------------------------------------------------------

# Description: Visualisation des caractéristiques issues du top 10 des features importances globales seulement.
# Après avoir choisi une caractéristique à analyser, il y a 3 figures:
# - Figure de la distribution de tous les clients
# - Figure de la distribution des clients ayant remboursé leur prêt
# - Figure de la distribution des clients n'ayant pas remboursé leur prêt.
# La situation du client concerné par l'analyse est indiquée par une ligne verticale.


elif selection == "Distribution des Caractéristiques":
    st.header("Distribution des Caractéristiques")
    st.markdown("""
        Choisissez une des 10 caractéristiques les plus importantes du modèle
        afin de visualiser la distribution de tous les clients, celle des clients ayant remboursé leur
        crédit, et celle des clients en défaut. La ligne verticale indique la position du client par 
        rapport à l'ensemble des autres clients.
    """)

    #---------------------------------------------------------------------------------------------
    # Vérifier si 'client_id' est dans st.session_state
    if 'client_id' not in st.session_state:
        st.error("Veuillez entrer un ID client dans l'onglet 'Le client'.")
    else:
        client_id = st.session_state['client_id']  # Récupérer client_id depuis session_state
    #---------------------------------------------------------------------------------------------

    # Récupérer les données des 10 features les plus importantes via l'API
    feature_data_endpoint = f"{api_url}/feature-data"
    response = requests.get(feature_data_endpoint)

    if response.status_code == 200:
        # Récupérer les données JSON renvoyées par l'API
        data = response.json()
        top_10_features = data["top_10_features"]
        feature_data = data["feature_data"]
        target_data = data["target"]

        # Menu déroulant pour sélectionner une feature à visualiser
        selected_feature = st.selectbox("Sélectionnez une variable à visualiser", top_10_features)

        # Obtenir les données de la feature sélectionnée
        feature_values = feature_data[selected_feature]

        # Filtrer les valeurs None avant de tracer l'histogramme
        cleaned_feature_values = [value for value in feature_values if value is not None]

        # Obtenir la valeur de la feature pour le client sélectionné (récupéré dans l'onglet 1)
        client_value = None
        if client_id:
            client_endpoint = f"{api_url}/client/{client_id}"
            client_response = requests.get(client_endpoint)

            if client_response.status_code == 200:
                client_data = client_response.json()

                # Récupérer la valeur de la feature sélectionnée pour le client
                try:
                    client_value = client_data["client_feature_values"][selected_feature]
                except KeyError:
                    st.error(f"La feature sélectionnée '{selected_feature}' n'existe pas dans les données du client.")
                    client_value = None
            else:
                st.error("Erreur lors de la récupération des informations du client.")

        # Afficher l'histogramme si des valeurs valides sont disponibles
        if cleaned_feature_values:
            # Créer une figure pour la distribution globale
            fig_global, ax_global = plt.subplots(figsize=(10, 6))
            ax_global.hist(cleaned_feature_values, bins=30, color='skyblue', edgecolor='black')
            ax_global.set_title(f"Distribution de {selected_feature} (Tous les clients)")
            ax_global.set_xlabel(selected_feature)
            ax_global.set_ylabel("Fréquence")

            # Ajouter une ligne verticale pour représenter la valeur du client dans la distribution globale
            if client_value is not None:
                ax_global.axvline(x=client_value, color='red', linestyle='--', label=f"Client ID {client_id}", linewidth=2)
                ax_global.legend()

            # Afficher la figure de la distribution globale dans Streamlit
            st.pyplot(fig_global)

            # Créer une figure avec deux sous-plots pour TARGET=0 et TARGET=1
            fig, (ax_target_0, ax_target_1) = plt.subplots(1, 2, figsize=(15, 6))

            # Distribution pour TARGET=0
            target_0_values = [value for value, target in zip(feature_values, target_data) if target == 0 and value is not None]
            ax_target_0.hist(target_0_values, bins=30, color='#008BFB', edgecolor='black')
            ax_target_0.set_title(f"Distribution de {selected_feature} (clients au crédit remboursé)")
            ax_target_0.set_xlabel(selected_feature)
            ax_target_0.set_ylabel("Fréquence")

            if client_value is not None:
                ax_target_0.axvline(x=client_value, color='red', linestyle='--', label=f"Client ID {client_id}", linewidth=2)
                ax_target_0.legend()

            # Distribution pour TARGET=1
            target_1_values = [value for value, target in zip(feature_values, target_data) if target == 1 and value is not None]
            ax_target_1.hist(target_1_values, bins=30, color='#FF005E', edgecolor='black')
            ax_target_1.set_title(f"Distribution de {selected_feature} (clients en défaut)")
            ax_target_1.set_xlabel(selected_feature)
            ax_target_1.set_ylabel("Fréquence")

            if client_value is not None:
                ax_target_1.axvline(x=client_value, color='red', linestyle='--', label=f"Client ID {client_id}", linewidth=2)
                ax_target_1.legend()

            # Ajustement de la mise en page pour les sous-plots
            plt.tight_layout()

            # Afficher les sous-plots dans Streamlit
            st.pyplot(fig)
        else:
            st.error(f"Aucune donnée valide disponible pour la feature '{selected_feature}'.")

    else:
        st.error("Erreur lors de la récupération des données de features.")










###################################################################################################
# Onglet 4 : Analyse bi-variée
#-----------------------------------------------------------------------------------------------

# Description: Visualisation des caractéristiques issues du top 10 des features importances globales seulement.
# Après avoir choisi deux caractéristiques à analyser, il y a 3 figures:
# - Figure de la distribution de tous les clients
# - Figure de la distribution des clients ayant remboursé leur prêt
# - Figure de la distribution des clients n'ayant pas remboursé leur prêt.
# La situation du client concerné par l'analyse est indiquée par une croix rouge.


elif selection == "Analyse Bi-variée":
    st.header("Analyse Bi-variée des Caractéristiques")
    st.write("""Sélectionnez deux caractéristiques parmi les 10 caractéristiques les plus importantes 
    pour visualiser un scatter plot de tous les clients, des clients ayant remboursé leur prêt ou des 
    clients en défaut. La croix rouge indique la position du client par rapport à l'ensemble
    des autres clients.""")

    #---------------------------------------------------------------------------------------------
    # Vérifier si 'client_id' est dans st.session_state
    if 'client_id' not in st.session_state:
        st.error("Veuillez entrer un ID client dans l'onglet 'Le client'.")
    else:
        client_id = st.session_state['client_id']  # Récupérer client_id depuis session_state
    #---------------------------------------------------------------------------------------------

    # Récupérer les données des 10 features les plus importantes via l'API
    feature_data_endpoint = f"{api_url}/feature-data"
    response = requests.get(feature_data_endpoint)

    if response.status_code == 200:
        # Récupérer les données JSON renvoyées par l'API
        data = response.json()
        top_10_features = data["top_10_features"]
        feature_data = data["feature_data"]
        target_data = data["target"]  # Récupérer les données target

        # Sélectionner deux features parmi les top 10
        selected_feature_x = st.selectbox("Sélectionnez la première feature (axe X)", top_10_features, key="x_feature")
        selected_feature_y = st.selectbox("Sélectionnez la deuxième feature (axe Y)", top_10_features, key="y_feature")

        # Obtenir les valeurs des deux features sélectionnées
        feature_values_x = feature_data[selected_feature_x]
        feature_values_y = feature_data[selected_feature_y]

        # Filtrer les valeurs None dans les deux features
        cleaned_values = [
            (x, y, target) for x, y, target in zip(feature_values_x, feature_values_y, target_data)
            if x is not None and y is not None
        ]

        if cleaned_values:
            # Séparer les valeurs X et Y après le nettoyage
            cleaned_feature_values_x, cleaned_feature_values_y, cleaned_target = zip(*cleaned_values)

            # Initialiser le scatter plot global
            fig_global, ax_global = plt.subplots(figsize=(10, 6))
            ax_global.scatter(cleaned_feature_values_x, cleaned_feature_values_y, color='skyblue', edgecolor='black', alpha=0.7)
            ax_global.set_title(f"Scatter Plot entre {selected_feature_x} et {selected_feature_y} (Tous les clients)")
            ax_global.set_xlabel(selected_feature_x)
            ax_global.set_ylabel(selected_feature_y)

            # Ajouter un point pour le client spécifique dans le graphique global
            if client_id:
                client_endpoint = f"{api_url}/client/{client_id}"
                client_response = requests.get(client_endpoint)

                if client_response.status_code == 200:
                    client_data = client_response.json()

                    # Obtenir les valeurs X et Y pour le client spécifique
                    try:
                        client_value_x = client_data["client_feature_values"][selected_feature_x]
                        client_value_y = client_data["client_feature_values"][selected_feature_y]

                        # Ajouter un point pour le client spécifique
                        ax_global.scatter(client_value_x, client_value_y, color='red', label=f"Client ID {client_id}", s=200, edgecolor='black', marker='X')
                        ax_global.legend()

                    except KeyError:
                        st.error(f"Les features sélectionnées '{selected_feature_x}' et/ou '{selected_feature_y}' n'existent pas dans les données du client.")
                else:
                    st.error("Erreur lors de la récupération des informations du client.")

            # Afficher le scatter plot global dans Streamlit
            st.pyplot(fig_global)

            # Créer deux sous-plots pour TARGET=0 et TARGET=1
            fig, (ax_target_0, ax_target_1) = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter plot pour TARGET=0 avec des points bleus
            target_0_values = [
                (x, y) for x, y, target in zip(cleaned_feature_values_x, cleaned_feature_values_y, cleaned_target)
                if target == 0
            ]
            if target_0_values:
                target_0_x, target_0_y = zip(*target_0_values)
                ax_target_0.scatter(target_0_x, target_0_y, color='#008BFB', edgecolor='black', alpha=0.7)
            ax_target_0.set_title(f"Scatter Plot entre {selected_feature_x} et {selected_feature_y} (clients au crédit remboursé)")
            ax_target_0.set_xlabel(selected_feature_x)
            ax_target_0.set_ylabel(selected_feature_y)

            if client_value_x is not None and client_value_y is not None:
                ax_target_0.scatter(client_value_x, client_value_y, color='red', label=f"Client ID {client_id}", s=200, edgecolor='black', marker='X')
                ax_target_0.legend()

            # Scatter plot pour TARGET=1 avec des points roses
            target_1_values = [
                (x, y) for x, y, target in zip(cleaned_feature_values_x, cleaned_feature_values_y, cleaned_target)
                if target == 1
            ]
            if target_1_values:
                target_1_x, target_1_y = zip(*target_1_values)
                ax_target_1.scatter(target_1_x, target_1_y, color='#FF005E', edgecolor='black', alpha=0.7)
            ax_target_1.set_title(f"Scatter Plot entre {selected_feature_x} et {selected_feature_y} (clients en défaut)")
            ax_target_1.set_xlabel(selected_feature_x)
            ax_target_1.set_ylabel(selected_feature_y)

            if client_value_x is not None and client_value_y is not None:
                ax_target_1.scatter(client_value_x, client_value_y, color='red', label=f"Client ID {client_id}", s=200, edgecolor='black', marker='X')
                ax_target_1.legend()

            # Ajuster la mise en page et afficher les sous-plots
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.error(f"Aucune donnée valide disponible pour les features '{selected_feature_x}' et '{selected_feature_y}'.")

    else:
        st.error("Erreur lors de la récupération des données de features.")










###################################################################################################
# Onglet 2 : Importance Globale des Features dans le modèle
#-----------------------------------------------------------------------------------------------

# Description: Visualisation des importances des caractéristiques issues du top 10 des features importances globales
# du modèle seulement seulement.


elif selection == "Le modèle global":
    st.header("Modèle Global: Importance des Caractéristiques")
    st.write("""L'histogramme représente les 10 plus importantes caractéristiques qui ont contribuées
    à l'élaboration du modèle de prédiction de défaut du client. """)

    # Construire l'URL de l'API pour récupérer la feature importance
    feature_importance_endpoint = f"{api_url}/feature-importance"
    
    # Envoyer une requête GET pour obtenir les 10 features les plus importantes
    feature_response = requests.get(feature_importance_endpoint)
    
    if feature_response.status_code == 200:
        # Récupérer les données JSON renvoyées par l'API
        feature_data = feature_response.json()
        feature_importance = pd.DataFrame(feature_data["top_10_feature_importance"])

        # Afficher le tableau des features importantes
        # st.write("### Top 10 Features les Plus Importantes")
        # st.write(feature_importance)

        # Visualiser les features importantes sous forme de graphique
        # st.write("### Top 10 des features importances globales")
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top 10 Features les Plus Importantes")
        st.pyplot(plt)
        plt.close()
    else:
        st.error("Erreur lors de la récupération des features importantes.")










###################################################################################################
# Onglet 5 : Description des caractéristiques
#-----------------------------------------------------------------------------------------------

# Description: Pour aider à l'interprétation des figures, nous utilisons la description du contenu 
# des colonnes en anglais.


elif selection == "Description des caractéristiques":
    st.header("Description des caractéristiques")
    st.write("Sélectionnez une caractéristique pour voir sa description en anglais.")

    # Récupérer les données de description via l'API
    description_endpoint = f"{api_url}/column-description"
    response = requests.get(description_endpoint)

    if response.status_code == 200:
        # Extraire les données
        data = response.json()
        columns_description = data['columns_description']

        # Créer un dictionnaire pour associer les noms des variables à leurs descriptions
        columns_dict = {item['Row']: item['Description'] for item in columns_description}

        # Créer un menu déroulant pour choisir une variable
        selected_variable = st.selectbox("Sélectionnez une variable", list(columns_dict.keys()))

        # Afficher la description de la variable sélectionnée
        st.write(f"### Description de {selected_variable}")
        st.write(columns_dict[selected_variable])
    else:
        st.error("Erreur lors de la récupération des données de description des colonnes.")