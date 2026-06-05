# Prédiction des Maladies Cardiaques — Projet ML

> Système complet de prédiction de maladies cardiaques basé sur le Machine Learning,  
> incluant exploration des données, entraînement de modèles, API REST et interface web interactive.

**Validation du projet — Samedi 06 juin 2026**

---

## Problématique

Les maladies cardiovasculaires représentent la première cause de mortalité mondiale. Un diagnostic tardif ou manqué peut être fatal. Ce projet vise à **aider les professionnels de santé** en fournissant un outil d'aide à la décision capable de prédire, à partir de données cliniques courantes, la probabilité qu'un patient soit atteint d'une maladie cardiaque.

---

## Démarche — Pipeline complet

Le projet suit une démarche MLOps en 7 étapes, de la collecte des données jusqu'au déploiement :

```
Semaine 1 → Collecte & EDA       : scraping + 6 visualisations statistiques
Semaine 2 → Prétraitement         : StandardScaler + SMOTE
Semaine 3 → Modélisation          : 3 modèles, GridSearchCV, MLflow tracking
Semaine 4 → API REST              : FastAPI, 3 endpoints, Swagger auto-généré
Semaine 5 → Interface web         : React + Vite, dark glass-morphism UI
Semaine 6 → Conteneurisation      : Docker multi-stage, docker-compose 3 services
Semaine 7 → CI/CD                 : GitHub Actions (lint + build Docker)
```

---

## Dataset

| Attribut   | Détail |
|------------|--------|
| Source     | UCI Machine Learning Repository — Cleveland Heart Disease |
| Fichier    | `data/heart_disease.csv` |
| Taille     | 303 patients · 13 features · 1 cible binaire |
| Cible      | `target` — 1 = Maladie cardiaque · 0 = Sain |
| Équilibre  | 54 % malades · 46 % sains |
| Qualité    | 0 valeur manquante · 1 doublon supprimé |

### Features cliniques (13 variables d'entrée)

| Feature      | Description                                         |
|--------------|-----------------------------------------------------|
| `age`        | Âge du patient (années)                             |
| `sex`        | Sexe (1 = Homme, 0 = Femme)                        |
| `cp`         | Type de douleur thoracique (0–3)                    |
| `trestbps`   | Pression artérielle au repos (mm Hg)                |
| `chol`       | Cholestérol sérique (mg/dl)                         |
| `fbs`        | Glycémie à jeun > 120 mg/dl (1 = Oui)             |
| `restecg`    | Résultats ECG au repos (0–2)                        |
| `thalach`    | Fréquence cardiaque maximale atteinte               |
| `exang`      | Angine induite par l'effort (1 = Oui)              |
| `oldpeak`    | Dépression ST à l'effort                           |
| `slope`      | Pente du segment ST (0–2)                           |
| `ca`         | Nombre de vaisseaux colorés (0–3)                  |
| `thal`       | Thalassémie (0 = Normal, 1 = Fixe, 2 = Réversible) |

---

## Analyse exploratoire (EDA)

Six visualisations sont générées automatiquement dans `data/` :

| Fichier                     | Contenu                                              |
|-----------------------------|------------------------------------------------------|
| `target_imbalance.png`      | Distribution des classes (54 / 46 %)                |
| `categorical_impact.png`    | Impact de chaque variable catégorielle sur la cible |
| `numerical_boxplots.png`    | Boxplots des variables continues                    |
| `multivariate_pairplot.png` | Relations pairwise entre features (200 patients)    |
| `correlation_heatmap.png`   | Matrice de corrélation feature ↔ cible              |
| `age_distribution.png`      | Distribution des âges selon le statut cardiaque     |

---

## Prétraitement

Le pipeline de prétraitement est entièrement intégré dans le modèle final (sérialisé avec `joblib`) :

1. **StandardScaler** — normalise les 13 features (μ = 0, σ = 1)
2. **SMOTE** — rééquilibrage synthétique des classes avant entraînement
3. **Train/Test Split** — 80 / 20 avec stratification

---

## Modélisation & Résultats

### Comparaison des modèles

| Modèle                                | Accuracy | F1-Score | Sélectionné |
|---------------------------------------|----------|----------|:-----------:|
| **Random Forest + SMOTE**             | **82 %** | **0.85** | ✅           |
| XGBoost + SMOTE                       | 80 %     | 0.83     |             |
| Stacking Ensemble (RF + XGB + LogReg) | 80 %     | 0.83     |             |

> **Critère de sélection : F1-Score** — pénalise davantage les faux négatifs (patients malades non détectés), ce qui est critique en contexte médical.

### Optimisation

- **GridSearchCV** avec validation croisée 5-fold sur le F1-score
- Random Forest : `n_estimators` ∈ [100, 200], `max_depth` ∈ [10, ∞], `min_samples_split` ∈ [2, 5]
- XGBoost : `learning_rate` ∈ [0.05, 0.1], `n_estimators` ∈ [100, 200], `max_depth` ∈ [3, 6]

### Tracking MLflow

Chaque expérience est tracée dans MLflow (`http://localhost:5000`) :
- Hyperparamètres, accuracy, F1-score, matrice de confusion, courbe ROC
- Modèles versionnés et comparables visuellement
- Meilleur modèle exporté : `data/best_model_pipeline.pkl`

---

## Architecture globale

```
┌────────────────────────────────────────────────────────┐
│                     UTILISATEUR                        │
└────────────────────┬───────────────────────────────────┘
                     │ http://localhost:5173
┌────────────────────▼───────────────────────────────────┐
│             FRONTEND — React 19 + Vite                 │
│   HeartForm      → prédiction patient individuel       │
│   BatchUpload    → import CSV de patients              │
│   Résultat       → jauge de probabilité animée         │
└────────────────────┬───────────────────────────────────┘
                     │ POST /predict  |  POST /predict_batch
┌────────────────────▼───────────────────────────────────┐
│             BACKEND — FastAPI + Uvicorn                │
│   GET  /health        → statut de l'API               │
│   POST /predict       → prédiction 1 patient          │
│   POST /predict_batch → prédiction par lot (CSV)      │
│   GET  /docs          → documentation Swagger         │
└──────┬──────────────────────────┬──────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────────────────┐
│   MODÈLE    │          │   MLFLOW UI                 │
│  Pipeline   │          │   http://localhost:5000     │
│  RF + SMOTE │          │   Expériences & Métriques   │
└─────────────┘          └─────────────────────────────┘
```

---

## Démonstration

### Prérequis

- Python 3.9+
- Node.js 18+

### Option A — Lancement local (recommandé pour la démo)

```bash
# 1. Environnement Python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. EDA + entraînement (~10 min la première fois)
python code/data_exploration.py
python code/modeling.py

# 3. Lancer les 3 services dans 3 terminaux séparés
venv\Scripts\python.exe code\app.py
venv\Scripts\mlflow.exe ui --port 5000
cd frontend && npm install && npm run dev
```

### Option B — Docker (un seul terminal)

```bash
docker-compose up --build
```

### URLs de la démonstration

| Service       | URL                        | Rôle                              |
|---------------|----------------------------|-----------------------------------|
| Interface web | http://localhost:5173      | Formulaire patient + résultats    |
| API Swagger   | http://localhost:8000/docs | Documentation + tests interactifs |
| MLflow        | http://localhost:5000      | Comparaison des expériences       |

---

## Exemple de prédiction API

### Requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 0, "trestbps": 130,
    "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
  }'
```

### Réponse

```json
{
  "heart_disease_prediction": 1,
  "heart_disease_probability": 0.68,
  "status": "Disease Detected"
}
```

---

## Stack technique

| Couche           | Technologies                                                          |
|------------------|-----------------------------------------------------------------------|
| Données / ML     | pandas · numpy · scikit-learn · XGBoost · imbalanced-learn (SMOTE)   |
| Tracking         | MLflow 2.0+                                                           |
| Backend          | FastAPI · Uvicorn · Pydantic · joblib                                 |
| Frontend         | React 19 · Vite 5 · Axios · lucide-react · CSS3 (glass-morphism)     |
| Conteneurisation | Docker · docker-compose · Nginx (multi-stage build)                   |
| CI/CD            | GitHub Actions (flake8 lint + Docker builds)                          |

---

## Structure du projet

```
Heart-Disease-Prediction/
├── data/
│   ├── heart_disease.csv           ← dataset UCI (303 patients)
│   ├── best_model_pipeline.pkl     ← pipeline entraîné (RF + SMOTE + Scaler)
│   └── *.png                       ← visualisations EDA et évaluations modèles
│
├── code/
│   ├── scraping.py                 ← scraping actualités cardio (Medical News Today)
│   ├── data_exploration.py         ← EDA avancée (6 visualisations)
│   ├── modeling.py                 ← Pipeline + GridSearch + MLflow tracking
│   └── app.py                      ← FastAPI : /health  /predict  /predict_batch
│
├── notebooks/
│   ├── 01_eda.ipynb                ← EDA interactive
│   └── 02_preprocessing.ipynb     ← Pipeline prétraitement illustré
│
├── frontend/
│   └── src/
│       ├── App.jsx                 ← dashboard dark glass-morphism
│       └── components/
│           ├── HeartForm.jsx       ← formulaire 13 champs cliniques
│           └── BatchUpload.jsx     ← upload CSV de patients
│
├── .github/workflows/main.yml      ← CI/CD GitHub Actions
├── docker-compose.yml              ← orchestration 3 services
├── Dockerfile.backend
├── Dockerfile.frontend
└── requirements.txt
```

---

## Détail des fichiers — dossier `code/`

### `scraping.py` — Collecte de données web

**Rôle :** Scraper automatiquement des actualités médicales sur les maladies cardiaques depuis *Medical News Today*.

**Ce qu'il fait :**
- Envoie une requête HTTP vers la page dédiée aux maladies cardiaques
- Parse le HTML avec **BeautifulSoup** pour extraire titres, dates et liens
- Prévoir un fallback si la structure HTML du site change
- Sauvegarde le résultat dans `data/health_news.csv`

**Pourquoi :** Montrer qu'on peut enrichir l'analyse avec des données en temps réel issues du web, et pas seulement un dataset statique.

---

### `data_exploration.py` — Analyse exploratoire (EDA)

**Rôle :** Comprendre et visualiser la structure du dataset avant toute modélisation.

**Ce qu'il fait :**
- Charge `data/heart_disease.csv` et affiche les statistiques de base (shape, types, valeurs manquantes, doublons)
- Génère **6 graphiques** sauvegardés dans `data/` :

| Graphique                   | Ce qu'il révèle                                                         |
|-----------------------------|-------------------------------------------------------------------------|
| `target_imbalance.png`      | Légère majorité de malades (54 %) — justifie l'usage de SMOTE          |
| `categorical_impact.png`    | Quelles catégories (sexe, douleur thoracique…) sont liées à la maladie |
| `numerical_boxplots.png`    | Distribution des valeurs continues selon le statut du patient           |
| `multivariate_pairplot.png` | Relations entre variables deux à deux                                   |
| `correlation_heatmap.png`   | Features les plus corrélées avec la cible                               |
| `age_distribution.png`      | Les patients malades sont légèrement plus âgés                          |

**Pourquoi :** L'EDA guide les choix de modélisation — elle confirme que le dataset est propre, légèrement déséquilibré, et que des variables comme `cp`, `thalach` et `ca` sont très discriminantes.

---

### `modeling.py` — Entraînement des modèles

**Rôle :** Entraîner, comparer et sélectionner le meilleur modèle ML, avec tracking complet des expériences.

**Ce qu'il fait :**

1. **Préparation** — charge le dataset, applique `StandardScaler` via un `ColumnTransformer`
2. **Pipeline imbalanced-learn** — chaîne `Scaler → SMOTE → Classifieur` en un seul objet
3. **3 modèles testés** avec `GridSearchCV` (5-fold, optimisé sur le F1-score) :
   - `RandomForestClassifier` — forêt d'arbres de décision
   - `XGBClassifier` — gradient boosting XGBoost
   - `StackingClassifier` — combinaison RF + XGBoost avec LogisticRegression en méta-modèle
4. **MLflow tracking** — logue automatiquement : hyperparamètres, accuracy, F1, matrice de confusion, courbe ROC
5. **Sélection** — compare les F1 des 3 modèles et sauvegarde le meilleur dans `data/best_model_pipeline.pkl`

> **Pourquoi le F1 plutôt que l'accuracy :** En médecine, rater un patient malade (faux négatif) est plus grave que surdiagnostiquer (faux positif). Le F1 pénalise davantage les faux négatifs.

---

### `app.py` — API REST (FastAPI)

**Rôle :** Exposer le modèle entraîné comme un service web interrogeable en temps réel.

**Ce qu'il fait :**
- Charge le pipeline `best_model_pipeline.pkl` au démarrage du serveur
- Active le CORS pour que le frontend React puisse appeler l'API
- Expose **3 endpoints** :

| Endpoint         | Méthode | Utilité                                                                     |
|------------------|---------|-----------------------------------------------------------------------------|
| `/health`        | GET     | Vérifie que l'API tourne et que le modèle est bien chargé                   |
| `/predict`       | POST    | Reçoit les 13 données cliniques d'un patient en JSON, retourne la prédiction + probabilité |
| `/predict_batch` | POST    | Reçoit un fichier CSV de plusieurs patients, retourne le CSV enrichi avec les prédictions |

- **Validation automatique** avec Pydantic — si un champ manque ou est du mauvais type, l'API renvoie une erreur claire
- **Swagger auto-généré** sur `/docs` — permet de tester l'API directement dans le navigateur sans code

> **Pourquoi FastAPI :** Asynchrone, rapide, et génère la documentation interactive automatiquement — idéal pour une démo.

---

## Référence

Dataset : [UCI Machine Learning Repository — Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
Projet réalisé dans le cadre du module **Intelligence Artificielle et Introduction au Machine Learning**.
