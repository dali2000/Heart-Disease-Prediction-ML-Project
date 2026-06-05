# Heart Disease Prediction — ML Project

> Système de prédiction de maladies cardiaques basé sur le Machine Learning, déployé via une API FastAPI et une interface React.

---

## Objectif métier

Prédire si un patient souffre d'une maladie cardiaque à partir de ses données cliniques, afin d'aider les professionnels de santé à **prioriser les examens** et **réduire les diagnostics manqués**.

---

## Dataset

| Attribut | Détail |
|---|---|
| Source | UCI Machine Learning Repository — Cleveland Heart Disease |
| Fichier | `data/heart_disease.csv` |
| Taille | 303 patients · 13 features · 1 cible |
| Cible | `target` — 1 = Maladie cardiaque · 0 = Sain |
| Équilibre | 54% malades · 46% sains |

### Features

| Feature | Description |
|---|---|
| `age` | Âge du patient (années) |
| `sex` | Sexe (1 = Homme, 0 = Femme) |
| `cp` | Type de douleur thoracique (0–3) |
| `trestbps` | Pression artérielle au repos (mm Hg) |
| `chol` | Cholestérol sérique (mg/dl) |
| `fbs` | Glycémie à jeun > 120 mg/dl (1 = Oui) |
| `restecg` | Résultats ECG au repos (0–2) |
| `thalach` | Fréquence cardiaque maximale atteinte |
| `exang` | Angine induite par l'effort (1 = Oui) |
| `oldpeak` | Dépression ST à l'effort |
| `slope` | Pente du segment ST (0–2) |
| `ca` | Nombre de vaisseaux colorés (0–3) |
| `thal` | Thalassémie (0 = Normal, 1 = Fixe, 2 = Réversible) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   UTILISATEUR                       │
└───────────────────┬─────────────────────────────────┘
                    │ http://localhost:5173
┌───────────────────▼─────────────────────────────────┐
│              FRONTEND — React + Vite                │
│   HeartForm (prédiction individuelle)               │
│   BatchUpload (prédiction CSV)                      │
└───────────────────┬─────────────────────────────────┘
                    │ POST /predict · POST /predict_batch
┌───────────────────▼─────────────────────────────────┐
│              BACKEND — FastAPI                      │
│   GET  /health        → statut API                 │
│   POST /predict       → 1 patient                  │
│   POST /predict_batch → fichier CSV                │
└──────┬────────────────────────┬────────────────────┘
       │                        │
┌──────▼──────┐        ┌────────▼────────────────────┐
│   MODÈLE    │        │   MLFLOW UI                 │
│  best_model │        │   http://localhost:5000      │
│  _pipeline  │        │   Expériences + Métriques   │
│  .pkl       │        └─────────────────────────────┘
└─────────────┘
```

---

## Résultats des modèles

| Modèle | Accuracy | F1-Score | Meilleur |
|---|---|---|---|
| **RandomForest + SMOTE** | **82%** | **0.85** | ✅ |
| XGBoost + SMOTE | 80% | 0.83 | |
| Stacking Ensemble | 80% | 0.83 | |

> Le meilleur modèle est sauvegardé automatiquement dans `data/best_model_pipeline.pkl`.

---

## Stack technique

- **Python 3.9** — pandas · numpy · scikit-learn · XGBoost · imbalanced-learn
- **MLflow** — tracking des expériences, comparaison des modèles
- **FastAPI + Uvicorn** — API REST avec validation Pydantic
- **React 19 + Vite** — interface utilisateur dark glass-morphism
- **Docker + docker-compose** — conteneurisation 3 services
- **GitHub Actions** — CI/CD (lint + build Docker)

---

## Installation locale

### Prérequis

- Python 3.9+
- Node.js 18+
- Git

### 1. Cloner le projet

```bash
git clone <repo-url>
cd Heart-Disease-Prediction
```

### 2. Environnement Python

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Pipeline ML

```bash
# Scraping actualités santé (optionnel)
python code/scraping.py

# Analyse exploratoire → génère les PNG dans data/
python code/data_exploration.py

# Entraînement des modèles + tracking MLflow (~10 min)
python code/modeling.py
```

### 4. Lancer tous les services

```powershell
start cmd /k "title BACKEND && venv\Scripts\python.exe code\app.py"
start cmd /k "title MLFLOW  && venv\Scripts\mlflow.exe ui --port 5000"
start cmd /k "title FRONTEND && cd frontend && npm run dev"
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## Docker

```bash
# Lancer les 3 services (backend + mlflow + frontend)
docker-compose up --build

# Arrêter
docker-compose down
```

| Service | Port |
|---|---|
| Backend | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Frontend | http://localhost:5173 |

---

## Structure du projet

```
Heart-Disease-Prediction/
├── data/
│   ├── heart_disease.csv          ← dataset UCI (303 patients)
│   ├── best_model_pipeline.pkl    ← meilleur modèle entraîné
│   ├── target_imbalance.png
│   ├── categorical_impact.png
│   ├── numerical_boxplots.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_*.png
│   └── roc_curve_*.png
│
├── code/
│   ├── scraping.py                ← scraping actualités cardio
│   ├── data_exploration.py        ← EDA avancée (6 visualisations)
│   ├── modeling.py                ← Pipeline + SMOTE + GridSearch + MLflow
│   └── app.py                     ← FastAPI /health /predict /predict_batch
│
├── notebooks/
│   ├── 01_eda.ipynb               ← EDA interactive
│   └── 02_preprocessing.ipynb    ← Pipeline preprocessing illustré
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                ← dashboard dark glass-morphism
│   │   ├── index.css              ← thème sombre
│   │   └── components/
│   │       ├── HeartForm.jsx      ← formulaire 13 champs cliniques
│   │       └── BatchUpload.jsx    ← upload CSV de patients
│   ├── package.json
│   └── vite.config.js
│
├── .github/
│   └── workflows/main.yml         ← CI/CD GitHub Actions
│
├── docker-compose.yml             ← 3 services orchestrés
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── README.md
```

---

## API — Endpoints

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
**Body :**
```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 130,
  "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
  "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}
```
**Réponse :**
```json
{
  "heart_disease_prediction": 1,
  "heart_disease_probability": 0.68,
  "status": "Disease Detected"
}
```

### `POST /predict_batch`
Upload d'un fichier `.csv` avec les mêmes colonnes → retourne le CSV enrichi avec `Disease_Prediction` et `Disease_Probability`.

---

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/main.yml`) :
1. **Lint** — vérifie le code Python avec `flake8`
2. **Build Backend** — construit l'image Docker backend
3. **Build Frontend** — construit l'image Docker frontend

---

## Auteur

Projet réalisé dans le cadre du module **Intelligence Artificielle et Introduction au ML**.  
Dataset : [UCI Heart Disease — Cleveland](https://archive.ics.uci.edu/ml/datasets/heart+disease)
