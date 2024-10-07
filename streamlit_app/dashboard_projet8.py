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

st.title("Dashboard de Scoring de Crédit")

# Menu déroulant pour sélectionner un client
client_id = st.text_input("Entrez l'ID du Client")

# Threshold pour la décision
THRESHOLD = 0.36

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(["Décision et Importance Locale", "Importance Globale", "Distribution des Variables", "Analyse Bi-variée"])

#--------------------------------------------------------------------------------------------------
# Onglet 1 : Décision d'octroi de crédit et feature importance locale
#--------------------------------------------------------------------------------------------------

with tab1:
    st.header("Décision d'octroi de crédit et Importance Locale")

    # Bouton pour déclencher la requête à l'API
    if st.button("Obtenir les Informations du Client"):
        # Construire l'URL de l'API pour ce client
        endpoint = f"{api_url}/client/{client_id}"
        
        # Envoyer une requête GET à l'API
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            # Récupérer les données JSON renvoyées par l'API
            data = response.json()
            probability = data['probability_of_default']
            
            # Afficher les résultats
            st.write("### Décision d'octroi de crédit")
            # st.write(f"**Probabilité de Défaut** : {probability:.2f}")
            
            # Afficher la décision avec une couleur
            decision = data["decision"]
            decision_color = "#008BFB" if decision == "Crédit accordé" else "#FF005E"

            # Utiliser un style CSS pour ajouter une bordure autour de la décision
            st.markdown(f"""
                <div style='display: inline-block; padding: 10px 20px; border: 2px solid {decision_color}; 
                            border-radius: 10px; color: {decision_color}; font-size: 24px; font-weight: bold;'>
                    {decision}
                </div>
            """, unsafe_allow_html=True)

            #----------------------------------------------------------------
            # VISUALISATION DE LA PROBABILITE DE DEFAUT

            # Visualisation de la probabilité sous forme de compteur
            st.write("### Visualisation de la Probabilité de Défaut")
            st.write("""La jauge représente la probabilité qu'un client rembourse (bleu) ou pas 
            son prêt (rose). La valeur du client est indiquée en noir.""")

            # Création du compteur avec Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilité de Défaut", 'font': {'size': 24}},
                delta={'reference': THRESHOLD * 100, 'font': {'size': 20}},
                number={'font': {'size': 60, 'color': 'black'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 20}},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, THRESHOLD * 100], 'color': '#008BFB'},  # Couleur pour en dessous du seuil (bleu)
                        {'range': [THRESHOLD * 100, 100], 'color': '#FF005E'}  # Couleur pour au-dessus du seuil (rose)
                    ],
                    'threshold': {
                        'line': {'color': "darkblue", 'width': 4},
                        'thickness': 0.75,
                        'value': THRESHOLD * 100
                    }
                }
            ))

            # Afficher le compteur dans Streamlit
            st.plotly_chart(fig_gauge)

            #----------------------------------------------------------
            # FEATURE IMPORTANCE LOCALE

            # Obtenir les valeurs SHAP pour ce client
            shap_endpoint = f"{api_url}/client/{client_id}"
            shap_response = requests.get(shap_endpoint)

            # # Vérifier la réponse brute de l'API
            # st.write("### Réponse brute SHAP:")
            # st.json(shap_response.json())  # Afficher le contenu brut de la réponse
            
            if shap_response.status_code == 200:
                # Récupérer les valeurs SHAP
                shap_data = shap_response.json()
                features = shap_data["shap_values"]["features"]
                shap_values = np.array(shap_data["shap_values"]["shap_values"])
                
                # Créer un explainer SHAP avec force_plot et sauvegarder dans un fichier temporaire
                st.write("### Contribution des features à la décision (SHAP)")
                st.write("""
                La figure force plot indique les features qui ont majoritairement contribué à la 
                prédiction de la probabilité de défaut. Les features indiquées en roses 
                sont en défaveur d'un octroi de crédit tandis que les features 
                en bleu sont en faveur d'un octroi de crédit. La taille des flèches est 
                proportionnelle à leur importance dans la décision finale.""")

                # Sauvegarder le graphique SHAP sous forme d'image dans un fichier temporaire
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    shap.force_plot(0, shap_values, features, matplotlib=True, show=False)
                    plt.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
                    plt.close()
                
                # Lire l'image et l'afficher dans Streamlit
                st.image(tmpfile.name)

            else:
                st.error("Erreur lors de la récupération des valeurs SHAP.")

        else:
            st.error("Client non trouvé ou erreur avec l'API.")

#-----------------------------------------------------------------------------------------------
# Onglet 2 : Importance Globale des Features
#-----------------------------------------------------------------------------------------------
with tab2:
    st.header("Importance Globale des Features (Top 10)")
    st.write("""La figure ci-dessous représente les 10 features les plus importantes qui ont contribuées
    à l'élaboration du modèle de prédiction de défaut du client.""")

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

#-----------------------------------------------------------------------------------------------
# Onglet 3 : Distribution des variables
#-----------------------------------------------------------------------------------------------

with tab3:
    st.header("Distribution des Variables")
    st.markdown("""
        Le menu déroulant vous permet de sélectionner une des 10 features les plus importantes 
        afin de visualiser sa distribution parmi tous les autres clients de la base de données.
    """)

    # Récupérer les données des 10 features les plus importantes via l'API
    feature_data_endpoint = f"{api_url}/feature-data"
    response = requests.get(feature_data_endpoint)

    if response.status_code == 200:
        # Récupérer les données JSON renvoyées par l'API
        data = response.json()
        top_10_features = data["top_10_features"]
        feature_data = data["feature_data"]

        # Menu déroulant pour sélectionner une feature à visualiser
        selected_feature = st.selectbox("Sélectionnez une variable à visualiser", top_10_features)

        # Obtenir les données de la feature sélectionnée
        feature_values = feature_data[selected_feature]

        # Filtrer les valeurs None avant de tracer l'histogramme
        cleaned_feature_values = [value for value in feature_values if value is not None]

        # Obtenir la valeur de la feature pour le client sélectionné (récupéré dans l'onglet 1)
        client_value = None
        if client_id:
            # Utiliser les données du client récupérées dans l'onglet 1
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
            # Afficher la distribution globale sous forme d'un histogramme
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(cleaned_feature_values, bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribution de {selected_feature} (Tous les clients)")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Fréquence")

            # Ajouter une ligne verticale pour représenter la valeur du client
            if client_value is not None:
                ax.axvline(x=client_value, color='red', linestyle='--', label=f"Client ID {client_id}", linewidth=2)
                ax.legend()

            # Afficher l'histogramme dans Streamlit
            st.pyplot(fig)
        else:
            st.error(f"Aucune donnée valide disponible pour la feature '{selected_feature}'.")

    else:
        st.error("Erreur lors de la récupération des données de features.")

#-----------------------------------------------------------------------------------------------
# Onglet 4 : Analyse bi-variée
#-----------------------------------------------------------------------------------------------

with tab4:
    st.header("Analyse Bi-variée")
    st.write("Sélectionnez deux features parmi les 10 features les plus importantes pour visualiser un scatter plot.")

    # Récupérer les données des 10 features les plus importantes via l'API
    feature_data_endpoint = f"{api_url}/feature-data"
    response = requests.get(feature_data_endpoint)

    if response.status_code == 200:
        # Récupérer les données JSON renvoyées par l'API
        data = response.json()
        top_10_features = data["top_10_features"]
        feature_data = data["feature_data"]

        # Sélectionner deux features parmi les top 10
        selected_feature_x = st.selectbox("Sélectionnez la première feature (axe X)", top_10_features, key="x_feature")
        selected_feature_y = st.selectbox("Sélectionnez la deuxième feature (axe Y)", top_10_features, key="y_feature")

        # Obtenir les valeurs des deux features sélectionnées
        feature_values_x = feature_data[selected_feature_x]
        feature_values_y = feature_data[selected_feature_y]

        # Filtrer les valeurs None dans les deux features
        cleaned_values = [
            (x, y) for x, y in zip(feature_values_x, feature_values_y)
            if x is not None and y is not None
        ]

        if cleaned_values:
            # Séparer les valeurs X et Y après le nettoyage
            cleaned_feature_values_x, cleaned_feature_values_y = zip(*cleaned_values)

            # Afficher le scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(cleaned_feature_values_x, cleaned_feature_values_y, color='blue', edgecolor='black', alpha=0.7)
            ax.set_title(f"Scatter Plot entre {selected_feature_x} et {selected_feature_y}")
            ax.set_xlabel(selected_feature_x)
            ax.set_ylabel(selected_feature_y)

            # Afficher le scatter plot dans Streamlit
            st.pyplot(fig)
        else:
            st.error(f"Aucune donnée valide disponible pour les features '{selected_feature_x}' et '{selected_feature_y}'.")

    else:
        st.error("Erreur lors de la récupération des données de features.")
