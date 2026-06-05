# Prédiction des Maladies Cardiaques
### Projet Machine Learning — Validation 06 juin 2026

---

## Problématique

Les maladies cardiovasculaires sont la **première cause de mortalité mondiale**.  
Un diagnostic tardif peut être fatal.

**Notre objectif :** construire un outil capable de prédire, à partir de données cliniques simples, si un patient est à risque — pour aider les médecins à prioriser leurs examens.

---

## Le Dataset

Source : **UCI Machine Learning Repository** (Cleveland Heart Disease)

- 303 patients
- 13 variables cliniques (âge, tension, cholestérol, ECG, fréquence cardiaque…)
- Cible : 0 = Sain · 1 = Malade
- Aucune valeur manquante

---

## Les étapes du projet

### Étape 1 — Collecte & Exploration des données
- Scraping d'actualités médicales (Medical News Today)
- Analyse statistique du dataset
- Génération de 6 visualisations : distribution des classes, corrélations, boxplots, pairplot…

### Étape 2 — Prétraitement
- **StandardScaler** : normalisation des 13 variables
- **SMOTE** : rééquilibrage des classes par génération de données synthétiques
- Pipeline intégré directement dans le modèle final

### Étape 3 — Modélisation
Comparaison de 3 modèles avec **GridSearchCV** (optimisation sur le F1-score) :

| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| **Random Forest + SMOTE** | **82 %** | **0.85** ✅ |
| XGBoost + SMOTE | 80 % | 0.83 |
| Stacking Ensemble | 80 % | 0.83 |

**Vérification des résultats :**  
Le choix de Random Forest est confirmé par deux graphes générés automatiquement. Les **courbes ROC** (`roc_curve_*.png`) montrent que Random Forest obtient l'aire sous la courbe (AUC) la plus élevée, ce qui signifie qu'il discrimine mieux les patients malades des patients sains. Les **matrices de confusion** (`confusion_matrix_*.png`) confirment qu'il produit le moins de faux négatifs parmi les trois modèles — c'est-à-dire moins de patients malades passés inaperçus, ce qui est le critère le plus critique dans un contexte médical.

Tracking complet des expériences avec **MLflow** (métriques, courbes ROC, matrices de confusion).

### Étape 4 — API REST
Exposition du modèle via **FastAPI** :
- `GET /health` — vérification du statut
- `POST /predict` — prédiction pour un patient
- `POST /predict_batch` — prédiction depuis un fichier CSV
- Documentation interactive auto-générée sur `/docs`

### Étape 5 — Interface Web
Application **React + Vite** avec un design dark glass-morphism :
- Formulaire avec les 13 champs cliniques
- Jauge de probabilité animée (rouge = maladie / vert = sain)
- Upload CSV pour prédire sur plusieurs patients à la fois

### Étape 6 — Déploiement
- **Docker** : 3 services conteneurisés (backend, MLflow, frontend)
- **GitHub Actions** : pipeline CI/CD automatique (lint + build à chaque push)

---

## Démonstration

| Service | URL |
|---------|-----|
| Interface web | http://localhost:5173 |
| API Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
