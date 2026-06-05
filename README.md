# Heart Disease Prediction — ML Project

Projet ML complet de prédiction de maladies cardiaques, de l'EDA au déploiement.

## Objectif métier

Prédire si un patient souffre d'une maladie cardiaque à partir de données cliniques,
pour aider les professionnels de santé à prioriser les examens complémentaires.

## Dataset

| Attribut | Valeur |
|---|---|
| Source | UCI Machine Learning Repository — Cleveland Heart Disease |
| Taille | 303 patients, 13 features + 1 cible |
| Cible | `target` (1 = Maladie, 0 = Sain) |
| Équilibre | ~54% malades / ~46% sains |

**Features** : age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

## Architecture

```
React (Vite)      FastAPI           MLflow
port 5173    →   port 8000    ←→   port 5000
  HeartForm       /predict          tracking
  BatchUpload     /predict_batch    artifacts
```

## Stack technique

- Python 3.9 · pandas · scikit-learn · XGBoost · LightGBM
- MLflow (tracking expériences + modèles)
- FastAPI + uvicorn (API REST)
- React 19 + Vite (frontend)
- Docker + docker-compose (3 services)

## Lancer le projet

### Installation locale

```bash
git clone <repo-url>
cd Heart-Disease-Prediction

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux/Mac

pip install -r requirements.txt
```

### Données

Télécharger `heart.csv` depuis Kaggle → renommer en `heart_disease.csv` → placer dans `data/`
(Kaggle dataset : *Heart Disease UCI*)

### Pipeline ML

```bash
# 1. Scraping actualités santé (optionnel)
python code/scraping.py

# 2. EDA
python code/data_exploration.py

# 3. Entraînement + MLflow (~10-20 min)
python code/modeling.py

# 4. Interface MLflow
mlflow ui --port 5000
# → http://localhost:5000

# 5. API
python code/app.py
# → http://localhost:8000/docs

# 6. Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Docker

```bash
docker-compose up --build

# Backend  → http://localhost:8000
# MLflow   → http://localhost:5000
# Frontend → http://localhost:5173
```

## Structure du projet

```
Heart-Disease-Prediction/
├── data/
│   ├── heart_disease.csv       ← dataset UCI
│   ├── best_model_pipeline.pkl ← meilleur modèle
│   └── *.png                   ← visualisations EDA
├── code/
│   ├── scraping.py             ← scraping actualités santé
│   ├── data_exploration.py     ← EDA avancée
│   ├── modeling.py             ← Pipeline + SMOTE + MLflow
│   └── app.py                  ← FastAPI
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
├── frontend/src/
│   ├── App.jsx                 ← dark glass-morphism UI
│   └── components/
│       ├── HeartForm.jsx       ← formulaire patient
│       └── BatchUpload.jsx     ← prédiction CSV
├── .github/workflows/main.yml  ← CI/CD
├── docker-compose.yml
├── Dockerfile.backend
└── Dockerfile.frontend
```

## Résultats attendus

| Modèle | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| RandomForest + SMOTE | ~84% | ~83% | ~90% |
| XGBoost + SMOTE | ~86% | ~85% | ~92% |
| Stacking Ensemble | ~87% | ~86% | ~93% |
