import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configuration de la page
st.set_page_config(page_title="Simulation de Gestion de Stock", layout="wide")
st.title("Simulation de Gestion de Stock de Médicaments")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres de la simulation")

# Paramètres configurables
MEDICAMENT_CIBLE = st.sidebar.selectbox(
    "Médicament à analyser",
    ["Paracétamol", "Ibuprofène", "Amoxicilline", "Oméprazole", "Doliprane"]
)

date_debut = st.sidebar.date_input("Date de début", pd.to_datetime('2020-01-01'))
date_fin = st.sidebar.date_input("Date de fin", pd.to_datetime('2023-01-01'))

# Paramètres de simulation
base_sales_min = st.sidebar.slider("Ventes minimales", 50, 100, 50)
base_sales_max = st.sidebar.slider("Ventes maximales", 200, 500, 300)
stock_initial_min = st.sidebar.slider("Stock initial minimum", 100, 500, 200)
stock_initial_max = st.sidebar.slider("Stock initial maximum", 500, 2000, 1000)

def generer_donnees():
    dates = pd.date_range(start=date_debut, end=date_fin, freq='D')
    np.random.seed(42)
    
    # Génération des ventes avec saisonnalité
    base_sales = np.random.randint(base_sales_min, base_sales_max, size=len(dates))
    seasonal_factor = 1 + 0.3 * np.sin(np.pi * dates.dayofyear / 182.5)
    trend_factor = 1 + 0.1 * (dates - dates[0]).days / 365
    winter_boost = np.where((dates.month.isin([12, 1, 2])), np.random.randint(0, 200, size=len(dates)), 0)
    ventes = (base_sales * seasonal_factor * trend_factor + winter_boost).astype(int)

    # Simulation des stocks
    stock_initial = np.random.randint(stock_initial_min, stock_initial_max, size=len(dates))
    stocks = np.zeros(len(dates), dtype=int)
    stocks[0] = stock_initial[0]

    reappro_quantity = np.random.randint(200, 500, size=len(dates))
    reappro_schedule = np.zeros(len(dates), dtype=int)
    reappro_schedule[::7] = 1

    for i in range(1, len(dates)):
        stocks[i] = max(0, stocks[i-1] - ventes[i-1]) + (reappro_quantity[i] if reappro_schedule[i] else 0)

    # Calcul des ruptures de stock
    ruptures = np.where(stocks < ventes, 1, 0)

    # Autres variables
    medicaments = ["Paracétamol", "Ibuprofène", "Amoxicilline", "Oméprazole", "Doliprane"]
    delai_livraison = np.random.randint(1, 10, size=len(dates))
    prix = np.round(np.random.uniform(5.0, 20.0, size=len(dates)), 2)
    Type_Produit = np.random.choice(['Médicament', 'Dispositif médical', 'Complément alimentaire'], size=len(dates))
    Catégorie = np.random.choice(['Critique', 'Important', 'Standard'], size=len(dates))
    Fournisseur = np.random.choice(['Fournisseur A', 'Fournisseur B', 'Fournisseur C'], size=len(dates))
    Demande_Prévue = np.random.randint(40, 600, size=len(dates))

    # Saisons
    saisons = np.where(dates.month.isin([12, 1, 2]), 'Hiver',
                      np.where(dates.month.isin([3, 4, 5]), 'Printemps',
                              np.where(dates.month.isin([6, 7, 8]), 'Été', 'Automne')))

    # Problèmes de production et transport
    problemes_production = np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])
    retards_transport = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])

    # Création du DataFrame
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

# Génération des données
data = generer_donnees()
data_med = data[data['Nom_Medicament'] == MEDICAMENT_CIBLE].copy()

# Affichage des métriques clés
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nombre total de ruptures", data_med['Rupture_Stock'].sum())
with col2:
    st.metric("Taux de rupture (%)", f"{(data_med['Rupture_Stock'].sum() / len(data_med) * 100):.2f}%")
with col3:
    st.metric("Stock moyen", f"{data_med['Stock'].mean():.0f}")
with col4:
    st.metric("Ventes moyennes", f"{data_med['Ventes'].mean():.0f}")

# Visualisations
st.subheader("Évolution du stock et des ventes")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_med['Date'], data_med['Stock'], label='Stock', alpha=0.7)
ax.plot(data_med['Date'], data_med['Ventes'], label='Ventes', alpha=0.7)
ax.set_title(f'Évolution du stock et des ventes - {MEDICAMENT_CIBLE}')
ax.legend()
st.pyplot(fig)

# Distribution des ventes
st.subheader("Distribution des ventes")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=data_med, x='Ventes', bins=50, kde=True, ax=ax)
ax.set_title('Distribution des ventes journalières')
st.pyplot(fig)

# Ruptures par saison
st.subheader("Ruptures de stock par saison")
season_counts = data_med.groupby(['Saison', 'Rupture_Stock']).size().unstack()
season_percentages = season_counts.div(season_counts.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 6))
season_percentages.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Pourcentage de ruptures de stock par saison')
ax.set_xlabel('Saison')
ax.set_ylabel('Pourcentage')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%')
plt.xticks(rotation=45)
st.pyplot(fig)

# Section d'analyse prédictive
st.header("Analyse Prédictive des Ruptures de Stock")

# Préparation des données pour le machine learning
st.subheader("Préparation des données")
st.write("Nous allons utiliser plusieurs modèles de machine learning pour prédire les ruptures de stock futures.")

# Ajout des caractéristiques temporelles
data_med['Jour_Semaine'] = data_med['Date'].dt.dayofweek
data_med['Mois'] = data_med['Date'].dt.month
data_med['Jour_Mois'] = data_med['Date'].dt.day
data_med['Trimestre'] = data_med['Date'].dt.quarter

# Caractéristiques décalées
data_med['Stock_J-1'] = data_med['Stock'].shift(1).fillna(0)
data_med['Ventes_J-1'] = data_med['Ventes'].shift(1).fillna(0)
data_med['Ventes_Moy_7J'] = data_med['Ventes'].rolling(window=7).mean().fillna(0)
data_med['Couverture_Stock'] = data_med['Stock'] / data_med['Ventes'].replace(0, 1)

# Encodage des variables catégorielles
le = LabelEncoder()
data_med['Saison_encoded'] = le.fit_transform(data_med['Saison'])
data_med['Type_Produit_encoded'] = le.fit_transform(data_med['Type_Produit'])
data_med['Catégorie_encoded'] = le.fit_transform(data_med['Catégorie'])
data_med['Fournisseur_encoded'] = le.fit_transform(data_med['Fournisseur'])

# Sélection des caractéristiques
features = ['Stock', 'Ventes', 'Délai_livraison', 'Prix', 'Saison_encoded',
           'Type_Produit_encoded', 'Catégorie_encoded', 'Fournisseur_encoded',
           'Jour_Semaine', 'Mois', 'Trimestre', 'Stock_J-1', 'Ventes_J-1',
           'Ventes_Moy_7J', 'Couverture_Stock', 'Problèmes_Production', 'Retards_Transport']

X = data_med[features]
y = data_med['Rupture_Stock']

# Normalisation des données
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Création et entraînement des modèles
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42)
}

# Évaluation des modèles individuels
st.subheader("Évaluation des modèles")
model_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_results.append({'Modèle': name, 'Précision': accuracy})
    
    # Affichage des résultats pour chaque modèle
    with st.expander(f"Résultats détaillés - {name}"):
        st.write(f"Précision: {accuracy:.4f}")
        st.write("Rapport de classification:")
        st.text(classification_report(y_test, y_pred))
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matrice de confusion - {name}')
        ax.set_xlabel('Prédit')
        ax.set_ylabel('Réel')
        st.pyplot(fig)

# Création et entraînement du modèle d'ensemble
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

# Affichage des résultats de l'ensemble
st.subheader("Modèle d'ensemble")
st.write(f"Précision du modèle d'ensemble: {accuracy_ensemble:.4f}")

# Graphique comparatif des performances
model_results.append({'Modèle': 'Ensemble', 'Précision': accuracy_ensemble})
results_df = pd.DataFrame(model_results)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Modèle', y='Précision', data=results_df, ax=ax)
ax.set_title('Comparaison des performances des modèles')
ax.set_ylim(0, 1)
for i, v in enumerate(results_df['Précision']):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
st.pyplot(fig)

# Importance des caractéristiques (Random Forest)
st.subheader("Importance des caractéristiques")
importance = pd.DataFrame({
    'Caractéristique': features,
    'Importance': models['RandomForest'].feature_importances_
})
importance = importance.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=importance.tail(10), y='Caractéristique', x='Importance', ax=ax)
ax.set_title('Top 10 des caractéristiques les plus importantes')
st.pyplot(fig)

# Prédiction future
st.subheader("Prédiction des ruptures de stock futures")
jours_prediction = st.slider("Nombre de jours à prédire", 7, 90, 30)

# Génération de données futures pour la prédiction
derniere_date = data_med['Date'].max()
dates_futures = pd.date_range(start=derniere_date + pd.Timedelta(days=1), periods=jours_prediction, freq='D')

# Création d'un DataFrame pour les prédictions futures
future_data = pd.DataFrame({'Date': dates_futures})
future_data['Jour_Semaine'] = future_data['Date'].dt.dayofweek
future_data['Mois'] = future_data['Date'].dt.month
future_data['Jour_Mois'] = future_data['Date'].dt.day
future_data['Trimestre'] = future_data['Date'].dt.quarter

# Utilisation des moyennes pour les autres caractéristiques
for feature in features:
    if feature not in ['Jour_Semaine', 'Mois', 'Jour_Mois', 'Trimestre']:
        future_data[feature] = data_med[feature].mean()

# Encodage des variables catégorielles pour les données futures
future_data['Saison_encoded'] = le.fit_transform(
    np.where(future_data['Mois'].isin([12, 1, 2]), 'Hiver',
             np.where(future_data['Mois'].isin([3, 4, 5]), 'Printemps',
                     np.where(future_data['Mois'].isin([6, 7, 8]), 'Été', 'Automne')))
)
future_data['Type_Produit_encoded'] = le.fit_transform(['Médicament'] * len(future_data))
future_data['Catégorie_encoded'] = le.fit_transform(['Standard'] * len(future_data))
future_data['Fournisseur_encoded'] = le.fit_transform(['Fournisseur A'] * len(future_data))

# Normalisation des données futures
X_future = future_data[features]
X_future_scaled = scaler.transform(X_future)

# Prédiction avec le modèle d'ensemble
predictions = ensemble.predict(X_future_scaled)
probabilities = ensemble.predict_proba(X_future_scaled)[:, 1]

# Affichage des prédictions
prediction_df = pd.DataFrame({
    'Date': dates_futures,
    'Rupture_Prévue': predictions,
    'Probabilité_Rupture': probabilities
})

# Graphique des prédictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(prediction_df['Date'], prediction_df['Probabilité_Rupture'], label='Probabilité de rupture', color='red')
ax.set_title('Prédiction des probabilités de rupture de stock')
ax.set_xlabel('Date')
ax.set_ylabel('Probabilité')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)

# Tableau des prédictions
st.write("Tableau des prédictions:")
st.dataframe(prediction_df)

# Bouton pour télécharger les données
st.download_button(
    label="Télécharger les données",
    data=data_med.to_csv(index=False).encode('utf-8'),
    file_name=f'simulation_{MEDICAMENT_CIBLE}.csv',
    mime='text/csv'
) 