import os
import sqlite3
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Risque Crédit", layout="wide")

DECISION_THRESHOLD = 0.30


@st.cache_resource
def get_db_connection():
    return sqlite3.connect("data/proceed/credit_risk.db", check_same_thread=False)


conn = get_db_connection()

st.title("📊 Dashboard Risque Crédit & Explicabilité LightGBM")

# Configuration des onglets
tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Score Client Individuel",
        "✏️ Édition & Prédiction Dynamic",
        "📈 Feature Importance Globale",
    ]
)


# --- RECUPERATION DE LA LISTE DES CLIENTS ---
@st.cache_data
def get_client_list():
    df_ids = pd.read_sql_query(
        "SELECT DISTINCT SK_ID_CURR FROM predictions ORDER BY SK_ID_CURR", conn
    )
    return df_ids["SK_ID_CURR"].tolist()


client_list = get_client_list()


# --- TAB 1 : SCORE CLIENT (LISTE DÉROULANTE) ---
with tab1:
    st.header("Analyse d'une demande de crédit")

    # Remplacement du number_input (+/-) par une Selectbox (liste déroulante)
    sk_id = st.selectbox("Saisir / Sélectionner SK_ID_CURR :", client_list)

    if st.button("Évaluer le client"):
        query = f"SELECT * FROM predictions WHERE SK_ID_CURR = {sk_id}"
        client_data = pd.read_sql_query(query, conn)

        if not client_data.empty:
            prob = client_data["TARGET"].iloc[0]
            decision = client_data["DECISION"].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Probabilité de Défaillance",
                    value=f"{prob * 100:.2f} %",
                )
            with col2:
                if decision == "APPROVED":
                    st.success(f"Décision : **{decision}**")
                else:
                    st.error(f"Décision : **{decision}**")
        else:
            st.warning(f"Client {sk_id} introuvable dans la base.")


# --- TAB 2 : ÉDITION CLIENTS & PREDICTION DYNAMIQUE ---
with tab2:
    st.header("Édition des données client & Simulation de Prédiction")

    selected_client = st.selectbox(
        "Sélectionner un client :", client_list, key="edit_select"
    )

    top_n_features = st.slider(
        "Nombre de variables clés à afficher/éditer :", 5, 50, 15
    )

    if st.button("Afficher"):
        # 1. Obtenir toutes les colonnes réellement existantes dans client_features
        db_cols = pd.read_sql_query(
            "SELECT * FROM client_features LIMIT 1", conn
        ).columns.tolist()

        # 2. Obtenir les top features depuis feature_importance
        query_top_fi = f"SELECT feature FROM feature_importance ORDER BY importance DESC LIMIT {top_n_features}"
        df_top_fi = pd.read_sql_query(query_top_fi, conn)
        top_cols = df_top_fi["feature"].tolist()

        # 3. Filtrer uniquement les colonnes FI qui existent VRAIMENT dans client_features
        valid_cols = [c for c in top_cols if c in db_cols]

        # Si aucune feature transformée n'est trouvée dans la BDD brute, prendre les N premières colonnes disponibles
        if not valid_cols:
            st.warning(
                "Les features du modèle sont issues du feature engineering. Affichage des variables brutes disponibles."
            )
            valid_cols = [
                c for c in db_cols if c not in ["SK_ID_CURR", "TARGET", "DECISION"]
            ][:top_n_features]

        # 4. Construire la liste finale des colonnes à charger
        mandatory = [c for c in ["SK_ID_CURR", "TARGET", "DECISION"] if c in db_cols]
        cols_to_select = mandatory + [c for c in valid_cols if c not in mandatory]

        # Échappement des noms de colonnes avec des guillemets pour éviter tout problème de syntaxe SQL
        cols_str = ", ".join([f'"{c}"' for c in cols_to_select])

        # 5. Exécution de la requête ciblée
        query_client = f"SELECT {cols_str} FROM client_features WHERE SK_ID_CURR = {selected_client}"
        df_res = pd.read_sql_query(query_client, conn)

        if df_res.empty:
            st.error(f"Aucune donnée trouvée pour le client {selected_client}.")
        else:
            st.session_state.edited_data = df_res

    # Affichage du tableau modifiable
    if "edited_data" in st.session_state and st.session_state.edited_data is not None:
        st.subheader(f"Valeurs pour le client {selected_client}")

        edited_df = st.data_editor(
            st.session_state.edited_data,
            num_rows="fixed",
            use_container_width=True,
        )

        st.markdown("---")
        #### Appel à FastAPI

        if st.button("Prédiction", key="predict_api"):
            try:
                # Charger toutes les variables du client
                full_client_df = pd.read_sql_query(
                    f"""
                    SELECT *
                    FROM client_features
                    WHERE SK_ID_CURR = {int(selected_client)}
                    """,
                    conn,
                )

                if full_client_df.empty:
                    st.error("Aucune donnée trouvée pour ce client.")
                    st.stop()

                # Remplacer les valeurs visibles par les valeurs modifiées
                for column in edited_df.columns:
                    if column in full_client_df.columns:
                        full_client_df.loc[0, column] = edited_df.loc[0, column]

                # Retirer les colonnes non utilisées comme variables
                features_df = full_client_df.drop(
                    columns=["SK_ID_CURR", "TARGET", "DECISION"],
                    errors="ignore",
                )

                # Conversion des NaN en None sans convertir les catégories en nombres
                features_dict = features_df.iloc[0].replace({np.nan: None}).to_dict()

                payload = {
                    "SK_ID_CURR": int(selected_client),
                    "features": features_dict,
                }

                API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

                if response.ok:
                    result = response.json()
                    score_pred = float(result["probability"])
                    decision_sim = result["decision"]

                    st.subheader("Résultat de la prédiction")
                    score_percent = score_pred * 100
                    gauge_fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=score_percent,
                            number={"suffix": "%", "valueformat": ".2f"},
                            title={"text": "Probabilité de défaillance"},
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickmode": "array",
                                    "tickvals": [0, DECISION_THRESHOLD * 100, 100],
                                    "ticktext": ["0 %", "30 %", "100 %"],
                                },
                                "steps": [
                                    {
                                        "range": [0, DECISION_THRESHOLD * 100],
                                        "color": "#d9eee7",
                                    },
                                    {
                                        "range": [DECISION_THRESHOLD * 100, 100],
                                        "color": "#f7dfdc",
                                    },
                                ],
                                "bar": {
                                    "color": "#d95d39"
                                    if score_pred >= DECISION_THRESHOLD
                                    else "#258f73"
                                },
                                "threshold": {
                                    "line": {"color": "#252525", "width": 4},
                                    "thickness": 0.8,
                                    "value": DECISION_THRESHOLD * 100,
                                },
                            },
                        )
                    )
                    gauge_fig.update_layout(height=280, margin={"t": 50, "b": 20})
                    st.plotly_chart(
                        gauge_fig,
                        use_container_width=True,
                        key="prediction_gauge",
                    )

                    if decision_sim == "APPROVED":
                        st.success(f"Décision : **{decision_sim}**")
                    else:
                        st.error(f"Décision : **{decision_sim}**")
                else:
                    st.error(
                        f"Erreur FastAPI ({response.status_code}) : {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI est inaccessible. Lancez : "
                    "`python -m uvicorn app.main:app --reload`"
                )
            except Exception as error:
                st.error(f"Erreur lors de la prédiction : {error}")


####

# --- TAB 3 : FEATURE IMPORTANCE  ---
with tab3:
    st.header("Facteurs d'influence du modèle")

    top_n = st.slider("Nombre de variables à afficher :", 5, 50, 20)

    try:
        query_fi = (
            f"SELECT * FROM feature_importance ORDER BY importance DESC LIMIT {top_n}"
        )
        df_fi = pd.read_sql_query(query_fi, conn)

        if not df_fi.empty:
            fig = px.bar(
                df_fi,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top {top_n} des variables les plus importantes",
                labels={
                    "importance": "Score d'importance",
                    "feature": "Variable",
                },
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}, height=15 * top_n
            )

            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(df_fi, hide_index=True)
        else:
            st.info("Aucune donnée trouvée dans la table feature_importance.")
    except Exception as e:
        st.error(f"Erreur de lecture SQLite : {e}")
