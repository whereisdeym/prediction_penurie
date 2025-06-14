import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image
import base64
# Ajout pour les animations Lottie
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# URLs d'animations Lottie fiables et publiques
LOTTIE_ACCUEIL = "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json"  # Pilule pharmacie
LOTTIE_ALERTE = "https://assets2.lottiefiles.com/packages/lf20_jzv1zqsu.json"   # Alerte rouge
LOTTIE_SUCCES = "https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json"   # Check vert
LOTTIE_FAQ = "https://assets2.lottiefiles.com/packages/lf20_3rwasyjy.json"      # Robot santé

# Chargement des animations
lottie_accueil = load_lottieurl(LOTTIE_ACCUEIL)
lottie_alerte = load_lottieurl(LOTTIE_ALERTE)
lottie_succes = load_lottieurl(LOTTIE_SUCCES)
lottie_faq = load_lottieurl(LOTTIE_FAQ)

# --- Thème visuel et configuration ---
st.set_page_config(page_title="Gestion de Stock Médicaments - Prévisions", layout="wide", page_icon="💊")

# Palette de couleurs
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
DANGER_COLOR = "#d62728"
BG_COLOR = "#f5f7fa"

# --- Logo et image d'accueil ---
# (Remplacer par le chemin de ton logo si tu en as un)
# logo = Image.open("logo_pharmacie.png")
# st.image(logo, width=120)

# Animation d'accueil
col_anim, col_txt = st.columns([1,3])
with col_anim:
    if lottie_accueil:
        st_lottie(lottie_accueil, height=120, key="accueil")
    else:
        st.info("Animation non disponible (connexion ou URL invalide).")
with col_txt:
    st.markdown(f"""
    <div style='background-color:{PRIMARY_COLOR};padding:1.2rem;border-radius:10px'>
    <h1 style='color:white;'>💊 Gestion de Stock de Médicaments</h1>
    <p style='color:white;font-size:1.2rem;'>Bienvenue ! Cette application vous aide à <b>anticiper les risques de rupture</b> et à <b>optimiser la gestion de vos stocks</b> en pharmacie.<br>
    <em>Explorez, analysez, et prenez les bonnes décisions, même sans connaissances techniques !</em></p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Sidebar modernisée ---
st.sidebar.header("🔧 Paramètres d'analyse")
st.sidebar.info("Choisissez le médicament et la période à analyser. Les résultats sont expliqués pas à pas.")

MEDICAMENT_CIBLE = st.sidebar.selectbox(
    "Quel médicament souhaitez-vous suivre ?",
    ["Paracétamol", "Ibuprofène", "Amoxicilline", "Oméprazole", "Doliprane"],
    help="Sélectionnez le médicament pour lequel vous souhaitez voir l'évolution du stock et des ruptures."
)

date_debut = st.sidebar.date_input("Date de début de l'analyse", pd.to_datetime('2020-01-01'), help="Choisissez la date de début de la période d'analyse.")
date_fin = st.sidebar.date_input("Date de fin de l'analyse", pd.to_datetime('2023-01-01'), help="Choisissez la date de fin de la période d'analyse.")

# --- Génération des données (inchangé) ---
base_sales_min = 50
base_sales_max = 300
stock_initial_min = 200
stock_initial_max = 1000

def generer_donnees():
    dates = pd.date_range(start=date_debut, end=date_fin, freq='D')
    np.random.seed(42)
    base_sales = np.random.randint(base_sales_min, base_sales_max, size=len(dates))
    seasonal_factor = 1 + 0.3 * np.sin(np.pi * dates.dayofyear / 182.5)
    trend_factor = 1 + 0.1 * (dates - dates[0]).days / 365
    winter_boost = np.where((dates.month.isin([12, 1, 2])), np.random.randint(0, 200, size=len(dates)), 0)
    ventes = (base_sales * seasonal_factor * trend_factor + winter_boost).astype(int)
    stock_initial = np.random.randint(stock_initial_min, stock_initial_max, size=len(dates))
    stocks = np.zeros(len(dates), dtype=int)
    stocks[0] = stock_initial[0]
    reappro_quantity = np.random.randint(200, 500, size=len(dates))
    reappro_schedule = np.zeros(len(dates), dtype=int)
    reappro_schedule[::7] = 1
    for i in range(1, len(dates)):
        stocks[i] = max(0, stocks[i-1] - ventes[i-1]) + (reappro_quantity[i] if reappro_schedule[i] else 0)
    ruptures = np.where(stocks < ventes, 1, 0)
    medicaments = ["Paracétamol", "Ibuprofène", "Amoxicilline", "Oméprazole", "Doliprane"]
    delai_livraison = np.random.randint(1, 10, size=len(dates))
    prix = np.round(np.random.uniform(5.0, 20.0, size=len(dates)), 2)
    Type_Produit = np.random.choice(['Médicament', 'Dispositif médical', 'Complément alimentaire'], size=len(dates))
    Catégorie = np.random.choice(['Critique', 'Important', 'Standard'], size=len(dates))
    Fournisseur = np.random.choice(['Fournisseur A', 'Fournisseur B', 'Fournisseur C'], size=len(dates))
    Demande_Prévue = np.random.randint(40, 600, size=len(dates))
    saisons = np.where(dates.month.isin([12, 1, 2]), 'Hiver',
                      np.where(dates.month.isin([3, 4, 5]), 'Printemps',
                              np.where(dates.month.isin([6, 7, 8]), 'Été', 'Automne')))
    problemes_production = np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])
    retards_transport = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    data = pd.DataFrame({
        'Date': dates,
        'Stock': stocks,
        'Ventes': ventes,
        'Rupture_Stock': ruptures,
        'Délai_livraison': delai_livraison,
        'Prix': prix,
        'Saison': saisons,
        'Type_Produit': Type_Produit,
        'Catégorie': Catégorie,
        'Fournisseur': Fournisseur,
        'Demande_Prévue': Demande_Prévue,
        'Problèmes_Production': problemes_production,
        'Retards_Transport': retards_transport,
        'Nom_Medicament': np.random.choice(medicaments, size=len(dates))
    })
    return data

data = generer_donnees()
data_med = data[data['Nom_Medicament'] == MEDICAMENT_CIBLE].copy()

# --- Tabs pour navigation ---
tabs = st.tabs(["Vue d'ensemble", "Visualisations", "Prédiction", "Conseils & FAQ"])

# --- Vue d'ensemble ---
with tabs[0]:
    st.subheader(f"Vue d'ensemble de {MEDICAMENT_CIBLE}")
    st.info(f"""
    Vous consultez l'évolution du stock et des ruptures pour **{MEDICAMENT_CIBLE}** entre le {date_debut.strftime('%d/%m/%Y')} et le {date_fin.strftime('%d/%m/%Y')}.
    
    - **Stock** : Quantité de médicaments disponible chaque jour.
    - **Ventes** : Nombre de médicaments vendus chaque jour.
    - **Rupture** : Jours où la pharmacie n'a pas pu répondre à la demande.
    """)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jours sans stock", int(data_med['Rupture_Stock'].sum()), help="Nombre de jours où le médicament était indisponible.")
    with col2:
        st.metric("% de jours sans stock", f"{(data_med['Rupture_Stock'].sum() / len(data_med) * 100):.2f}%", help="Pourcentage de jours où il y a eu une rupture.")
    with col3:
        st.metric("Stock moyen", f"{data_med['Stock'].mean():.0f}", help="Quantité moyenne de médicaments en stock chaque jour.")
    with col4:
        st.metric("Ventes moyennes", f"{data_med['Ventes'].mean():.0f}", help="Nombre moyen de médicaments vendus par jour.")
    st.caption("Ces indicateurs vous donnent une vision rapide de la situation du médicament sélectionné sur la période choisie.")
    st.divider()
    # Badge d'alerte si risque élevé
    rupture_pct = data_med['Rupture_Stock'].sum() / len(data_med) * 100
    if rupture_pct > 20:
        col_anim, col_txt = st.columns([1,3])
        with col_anim:
            if lottie_alerte:
                st_lottie(lottie_alerte, height=80, key="alerte")
            else:
                st.info("Animation non disponible (connexion ou URL invalide).")
        with col_txt:
            st.error(f"⚠️ Attention : {rupture_pct:.1f}% de jours sans stock. Risque élevé de rupture !")
    elif rupture_pct > 10:
        st.warning(f"🔶 {rupture_pct:.1f}% de jours sans stock. Soyez vigilant.")
    else:
        col_anim, col_txt = st.columns([1,3])
        with col_anim:
            if lottie_succes:
                st_lottie(lottie_succes, height=80, key="succes")
            else:
                st.info("Animation non disponible (connexion ou URL invalide).")
        with col_txt:
            st.success(f"✅ Situation sous contrôle : {rupture_pct:.1f}% de jours sans stock.")

# --- Visualisations interactives ---
with tabs[1]:
    st.subheader("Visualisations interactives")
    st.info("Vous pouvez explorer ici l'évolution du stock, des ventes et des ruptures de façon interactive.")
    choix_graphique = st.radio(
        "Que souhaitez-vous visualiser ?",
        ["Évolution du stock et des ventes", "Distribution des ventes", "Ruptures par saison"],
        horizontal=True
    )
    if choix_graphique == "Évolution du stock et des ventes":
        st.markdown("""
        **Ce graphique montre comment le stock et les ventes évoluent chaque jour.**
        - Si la courbe du stock descend trop bas, cela peut indiquer un risque de rupture.
        - La courbe des ventes permet de voir les périodes de forte demande.
        """)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_med['Date'], y=data_med['Stock'], mode='lines', name='Stock', line=dict(color=PRIMARY_COLOR)))
        fig.add_trace(go.Scatter(x=data_med['Date'], y=data_med['Ventes'], mode='lines', name='Ventes', line=dict(color=SECONDARY_COLOR)))
        fig.update_layout(title=f'Évolution du stock et des ventes - {MEDICAMENT_CIBLE}', xaxis_title='Date', yaxis_title='Quantité', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    elif choix_graphique == "Distribution des ventes":
        st.markdown("""
        **Ce graphique montre la répartition des ventes quotidiennes.**
        - Il permet de voir si les ventes sont stables ou s'il y a des pics de demande.
        - Une distribution large indique des variations importantes d'un jour à l'autre.
        """)
        fig = px.histogram(data_med, x='Ventes', nbins=50, color_discrete_sequence=[SECONDARY_COLOR])
        fig.update_layout(title='Distribution des ventes journalières', xaxis_title='Ventes par jour', yaxis_title='Nombre de jours', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    elif choix_graphique == "Ruptures par saison":
        st.markdown("""
        **Ce graphique montre la part de jours avec rupture de stock selon la saison.**
        - Il permet d'identifier les périodes à risque (ex : hiver).
        - Plus la part orange est grande, plus il y a eu de jours sans stock.
        """)
        season_counts = data_med.groupby(['Saison', 'Rupture_Stock']).size().unstack()
        season_percentages = season_counts.div(season_counts.sum(axis=1), axis=0) * 100
        fig = px.bar(season_percentages, barmode='stack', color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR])
        fig.update_layout(title='Pourcentage de ruptures de stock par saison', xaxis_title='Saison', yaxis_title='Pourcentage', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Vous pouvez changer de graphique à tout moment pour explorer d'autres aspects de la gestion du stock.")

# --- Prédiction et analyse avancée ---
with tabs[2]:
    st.subheader("Analyse Prédictive des Ruptures de Stock")
    st.write("Nous allons utiliser plusieurs modèles de machine learning pour prédire les ruptures de stock futures.")
    # Préparation des données pour le machine learning
    data_med['Jour_Semaine'] = data_med['Date'].dt.dayofweek
    data_med['Mois'] = data_med['Date'].dt.month
    data_med['Jour_Mois'] = data_med['Date'].dt.day
    data_med['Trimestre'] = data_med['Date'].dt.quarter
    data_med['Stock_J-1'] = data_med['Stock'].shift(1).fillna(0)
    data_med['Ventes_J-1'] = data_med['Ventes'].shift(1).fillna(0)
    data_med['Ventes_Moy_7J'] = data_med['Ventes'].rolling(window=7).mean().fillna(0)
    data_med['Couverture_Stock'] = data_med['Stock'] / data_med['Ventes'].replace(0, 1)
    le = LabelEncoder()
    data_med['Saison_encoded'] = le.fit_transform(data_med['Saison'])
    data_med['Type_Produit_encoded'] = le.fit_transform(data_med['Type_Produit'])
    data_med['Catégorie_encoded'] = le.fit_transform(data_med['Catégorie'])
    data_med['Fournisseur_encoded'] = le.fit_transform(data_med['Fournisseur'])
    features = ['Stock', 'Ventes', 'Délai_livraison', 'Prix', 'Saison_encoded',
               'Type_Produit_encoded', 'Catégorie_encoded', 'Fournisseur_encoded',
               'Jour_Semaine', 'Mois', 'Trimestre', 'Stock_J-1', 'Ventes_J-1',
               'Ventes_Moy_7J', 'Couverture_Stock', 'Problèmes_Production', 'Retards_Transport']
    X = data_med[features]
    y = data_med['Rupture_Stock']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    st.markdown("**Évaluation des modèles**")
    model_results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_results.append({'Modèle': name, 'Précision': accuracy})
        with st.expander(f"Résultats détaillés - {name}"):
            st.write(f"Précision: {accuracy:.4f}")
            st.write("Rapport de classification:")
            st.text(classification_report(y_test, y_pred))
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues',
                            labels=dict(x="Prédit", y="Réel", color="Nombre"))
            fig.update_layout(title=f'Matrice de confusion - {name}')
            st.plotly_chart(fig, use_container_width=True)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting']),
            ('lr', models['LogisticRegression'])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    st.success(f"Précision du modèle d'ensemble : {accuracy_ensemble:.4f}")
    model_results.append({'Modèle': 'Ensemble', 'Précision': accuracy_ensemble})
    results_df = pd.DataFrame(model_results)
    fig = px.bar(results_df, x='Modèle', y='Précision', color='Modèle', color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, '#9467bd'])
    fig.update_layout(title='Comparaison des performances des modèles', yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    # Importance des caractéristiques
    st.markdown("**Importance des caractéristiques (Random Forest)**")
    importance = pd.DataFrame({
        'Caractéristique': features,
        'Importance': models['RandomForest'].feature_importances_
    })
    importance = importance.sort_values('Importance', ascending=True)
    fig = px.bar(importance.tail(10), y='Caractéristique', x='Importance', orientation='h', color='Importance', color_continuous_scale='Blues')
    fig.update_layout(title='Top 10 des caractéristiques les plus importantes')
    st.plotly_chart(fig, use_container_width=True)
    # Prédiction future
    st.markdown("**Prédiction des ruptures de stock futures**")
    jours_prediction = st.slider("Nombre de jours à prédire", 7, 90, 30)
    derniere_date = data_med['Date'].max()
    dates_futures = pd.date_range(start=derniere_date + pd.Timedelta(days=1), periods=jours_prediction, freq='D')
    future_data = pd.DataFrame({'Date': dates_futures})
    future_data['Jour_Semaine'] = future_data['Date'].dt.dayofweek
    future_data['Mois'] = future_data['Date'].dt.month
    future_data['Jour_Mois'] = future_data['Date'].dt.day
    future_data['Trimestre'] = future_data['Date'].dt.quarter
    for feature in features:
        if feature not in ['Jour_Semaine', 'Mois', 'Jour_Mois', 'Trimestre']:
            future_data[feature] = data_med[feature].mean()
    future_data['Saison_encoded'] = le.fit_transform(
        np.where(future_data['Mois'].isin([12, 1, 2]), 'Hiver',
                 np.where(future_data['Mois'].isin([3, 4, 5]), 'Printemps',
                         np.where(future_data['Mois'].isin([6, 7, 8]), 'Été', 'Automne')))
    )
    future_data['Type_Produit_encoded'] = le.fit_transform(['Médicament'] * len(future_data))
    future_data['Catégorie_encoded'] = le.fit_transform(['Standard'] * len(future_data))
    future_data['Fournisseur_encoded'] = le.fit_transform(['Fournisseur A'] * len(future_data))
    X_future = future_data[features]
    X_future_scaled = scaler.transform(X_future)
    predictions = ensemble.predict(X_future_scaled)
    probabilities = ensemble.predict_proba(X_future_scaled)[:, 1]
    prediction_df = pd.DataFrame({
        'Date': dates_futures,
        'Rupture_Prévue': predictions,
        'Probabilité_Rupture': probabilities
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Probabilité_Rupture'], mode='lines+markers', name='Probabilité de rupture', line=dict(color=DANGER_COLOR)))
    fig.update_layout(title='Prédiction des probabilités de rupture de stock', xaxis_title='Date', yaxis_title='Probabilité', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    # Résumé automatique
    risque_eleve = prediction_df['Probabilité_Rupture'].max() > 0.5
    if risque_eleve:
        col_anim, col_txt = st.columns([1,3])
        with col_anim:
            if lottie_alerte:
                st_lottie(lottie_alerte, height=80, key="alerte_pred")
            else:
                st.info("Animation non disponible (connexion ou URL invalide).")
        with col_txt:
            st.error("⚠️ Risque élevé de rupture détecté dans les prochains jours ! Pensez à anticiper vos commandes.")
    else:
        col_anim, col_txt = st.columns([1,3])
        with col_anim:
            if lottie_succes:
                st_lottie(lottie_succes, height=80, key="succes_pred")
            else:
                st.info("Animation non disponible (connexion ou URL invalide).")
        with col_txt:
            st.success("✅ Aucun risque majeur de rupture détecté dans la période prédite.")
    st.write("Tableau des prédictions :")
    st.dataframe(prediction_df)
    st.download_button(
        label="⬇️ Télécharger les données",
        data=data_med.to_csv(index=False).encode('utf-8'),
        file_name=f'simulation_{MEDICAMENT_CIBLE}.csv',
        mime='text/csv'
    )
    st.caption("Cliquez pour télécharger la simulation complète de vos stocks.")

# --- Conseils & FAQ ---
with tabs[3]:
    st.subheader("Conseils personnalisés & FAQ")
    rupture_pct = data_med['Rupture_Stock'].sum() / len(data_med) * 100
    col_anim, col_txt = st.columns([1,3])
    with col_anim:
        if lottie_faq:
            st_lottie(lottie_faq, height=80, key="faq")
        else:
            st.info("Animation non disponible (connexion ou URL invalide).")
    with col_txt:
        if rupture_pct > 20:
            st.error("💡 Conseil : Augmentez la fréquence de vos commandes et surveillez les périodes de forte demande (hiver, épidémies).")
        elif rupture_pct > 10:
            st.warning("💡 Conseil : Analysez les causes des ruptures (retards, problèmes de production) et ajustez vos seuils de réapprovisionnement.")
        else:
            st.success("👍 Votre gestion de stock est efficace ! Continuez à surveiller les tendances.")
    st.markdown("""
    ### FAQ
    **Q : Comment sont calculées les prévisions ?**
    > Les prévisions utilisent plusieurs modèles de machine learning entraînés sur vos données historiques.

    **Q : Que faire en cas de risque de rupture ?**
    > Anticipez vos commandes, contactez vos fournisseurs et surveillez les périodes à risque (hiver, promotions).

    **Q : Puis-je télécharger mes données ?**
    > Oui, utilisez le bouton de téléchargement dans l'onglet Prédiction.

    **Q : Comment améliorer la précision des prévisions ?**
    > Plus vous avez de données fiables et à jour, plus les prévisions seront précises.
    """)
    st.info("Pour toute question ou suggestion, contactez votre responsable informatique ou le support technique.")

st.divider()
st.caption("Cette solution IA vous est proposée par Ornela Mafi AWAGA pour la prévention des risques de rupture de stock.") 