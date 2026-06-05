# Lancer le projet — Guide étape par étape

---

## Prérequis à installer une seule fois

- [Python 3.9+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)

---

## Installation complète (première fois)

### 1. Créer l'environnement Python

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Installer les dépendances frontend

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction\frontend
npm install
```

### 3. Entraîner le modèle (~10 min)

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction
venv\Scripts\activate
python code\data_exploration.py
python code\modeling.py
```

À la fin tu dois voir :
```
Best model saved to data/best_model_pipeline.pkl | F1=0.85
```

---

## Lancement — 3 terminaux

### Terminal 1 — Backend

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction
venv\Scripts\activate
python code\app.py
```

Résultat attendu :
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — MLflow

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction
venv\Scripts\activate
mlflow ui --port 5000
```

Résultat attendu :
```
[INFO] Listening at: http://127.0.0.1:5000
```

### Terminal 3 — Frontend

```powershell
cd C:\Users\medal\Desktop\Heart-Disease-Prediction\frontend
npm run dev
```

Résultat attendu :
```
➜  Local:   http://localhost:5173/
```

---

## Vérifier que tout tourne

### Backend
Ouvrir http://localhost:8000/docs → tu dois voir la page Swagger avec 3 endpoints (`/health`, `/predict`, `/predict_batch`)

Tester `/health` → cliquer **Try it out → Execute**, résultat attendu :
```json
{ "status": "ok", "model_loaded": true }
```

Tester `/predict` → coller ce body et cliquer **Execute** :
```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 130,
  "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
  "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}
```
Résultat attendu :
```json
{ "heart_disease_prediction": 1, "heart_disease_probability": 0.68, "status": "Disease Detected" }
```

### Frontend
Ouvrir http://localhost:5173 → tu dois voir le formulaire patient avec les 13 champs  
Remplir les champs et cliquer **Predict** → la jauge de probabilité doit s'afficher en rouge ou vert

### MLflow
Ouvrir http://localhost:5000 → tu dois voir l'expérience `HeartDisease_Optimized` avec les 3 runs (RandomForest, XGBoost, Stacking)  
Cliquer sur un run → vérifier que les métriques, matrices de confusion et courbes ROC sont bien loggées

---

## Arrêter les services

Dans chaque terminal : `Ctrl + C`
