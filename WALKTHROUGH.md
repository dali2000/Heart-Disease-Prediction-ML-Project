# Guide Complet — Heart Disease Prediction
## Explication de chaque partie du projet

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Semaine 1 — Scraping & EDA](#2-semaine-1--scraping--eda)
3. [Semaine 2 — Preprocessing](#3-semaine-2--preprocessing)
4. [Semaine 3 — Modélisation & MLflow](#4-semaine-3--modélisation--mlflow)
5. [Semaine 4 — API FastAPI](#5-semaine-4--api-fastapi)
6. [Semaine 5 — Frontend React](#6-semaine-5--frontend-react)
7. [Semaine 6 — Docker](#7-semaine-6--docker)
8. [Semaine 7 — CI/CD & Déploiement](#8-semaine-7--cicd--déploiement)

---

## 1. Vue d'ensemble

### Problème métier

Les maladies cardiovasculaires sont la **première cause de mortalité mondiale** (OMS). Un diagnostic tardif coûte des vies. Ce projet construit un système de prédiction automatique qui, à partir de **13 mesures cliniques simples** (âge, tension, cholestérol…), détermine si un patient est à risque de maladie cardiaque.

### Dataset utilisé

**UCI Heart Disease Dataset (Cleveland)** — l'un des datasets médicaux les plus utilisés en ML.

- **303 patients**, **13 features**, **1 variable cible**
- Variable cible : `target` → `1` = Maladie présente, `0` = Sain
- Aucune valeur manquante, 1 doublon supprimé

### Pipeline global

```
Données brutes
     ↓
Scraping (actualités médicales)
     ↓
EDA (data_exploration.py)        ← comprendre les données
     ↓
Preprocessing (dans modeling.py) ← nettoyer + normaliser
     ↓
Modélisation (modeling.py)       ← entraîner + comparer
     ↓
Évaluation + MLflow              ← tracker les expériences
     ↓
API FastAPI (app.py)             ← exposer le modèle
     ↓
Frontend React                   ← interface utilisateur
     ↓
Docker                           ← conteneuriser
     ↓
CI/CD GitHub Actions             ← automatiser
```

---

## 2. Semaine 1 — Scraping & EDA

### `code/scraping.py` — Collecte de données web

**Objectif** : Collecter automatiquement des articles d'actualité sur les maladies cardiaques depuis Medical News Today pour enrichir le contexte métier.

**Comment ça marche :**

```python
def scrape_health_news(url="https://www.medicalnewstoday.com/categories/heart-disease"):
```

1. On envoie une requête HTTP avec un `User-Agent` de navigateur (pour éviter d'être bloqué)
2. `BeautifulSoup` parse le HTML de la page
3. On extrait les balises `<li>` contenant les articles
4. On récupère : titre, date, lien
5. Le résultat est sauvegardé dans `data/health_news.csv`

**Pourquoi scraper ?** Dans un projet réel, les actualités peuvent être analysées avec du NLP pour identifier les tendances (ex : nouvelles thérapies, facteurs de risque émergents). Ici c'est une étape de collecte de données complémentaire.

**Lancer :**
```bash
python code/scraping.py
```

---

### `code/data_exploration.py` — Analyse Exploratoire (EDA)

**Objectif** : Comprendre la structure des données avant de modéliser. Répondre à : *Quelles features sont discriminantes ? Y a-t-il un déséquilibre de classes ? Des outliers ?*

**Les 6 analyses produites :**

#### 1. Distribution de la variable cible (`target_imbalance.png`)
```python
df['target'].value_counts(normalize=True).plot(kind='bar')
```
→ Vérifie si les classes sont équilibrées. Ici ~54/46 — relativement équilibré, mais on applique SMOTE quand même.

#### 2. Impact des variables catégorielles (`categorical_impact.png`)
```python
sns.countplot(x=feature, hue='target', data=df)
```
→ Pour chaque variable catégorielle (sexe, type de douleur, angine…), on voit si elle discrimine les malades des sains. Par exemple : les hommes ont un taux de maladie plus élevé dans ce dataset.

#### 3. Boxplots des variables numériques (`numerical_boxplots.png`)
```python
sns.boxplot(y=df[feature], x=df['target'])
```
→ Compare la distribution de chaque feature continue (âge, cholestérol, fréquence cardiaque…) entre les deux classes. Identifie les outliers et les features les plus séparables.

#### 4. Pairplot multivarié (`multivariate_pairplot.png`)
```python
sns.pairplot(df.sample(200), hue='target')
```
→ Visualise les relations entre toutes les features numériques en même temps. Les nuages de points bien séparés indiquent de bonnes features.

#### 5. Heatmap de corrélation (`correlation_heatmap.png`)
```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
```
→ Identifie les corrélations entre features (multicolinéarité) et avec la cible. `thalach` (fréquence cardiaque max) est négativement corrélée à la maladie — contre-intuitif mais médically expliqué.

#### 6. Distribution par âge (`age_distribution.png`)
→ Histogramme superposé : les malades ont tendance à être plus âgés (55-65 ans).

**Lancer :**
```bash
python code/data_exploration.py
# → 6 PNG générés dans data/
```

---

## 3. Semaine 2 — Preprocessing

### Preprocessing intégré dans `code/modeling.py`

Le preprocessing est encapsulé dans un **Pipeline scikit-learn** — ce qui garantit qu'il est appliqué de façon identique à l'entraînement et à l'inférence.

#### Étape 1 : Nettoyage
```python
df = df.drop_duplicates()          # supprime le doublon (1 trouvé)
X = df.drop('target', axis=1)      # features
y = df['target']                   # cible
```

#### Étape 2 : StandardScaler (normalisation)
```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features)
])
```
**Pourquoi normaliser ?** Les features ont des échelles très différentes : `age` (29–77) vs `chol` (126–564). Sans normalisation, les algorithmes basés sur des distances (et même les arbres dans certains cas) seront biaisés par les grandes valeurs.

`StandardScaler` centre chaque feature autour de 0 avec un écart-type de 1 :
```
x_scaled = (x - mean) / std
```

#### Étape 3 : SMOTE (rééquilibrage)
```python
('smote', SMOTE(random_state=42))
```
**Pourquoi SMOTE ?** Même si les classes sont à 54/46, on applique SMOTE pour garantir un entraînement robuste. SMOTE (**Synthetic Minority Over-sampling Technique**) génère de nouveaux exemples synthétiques de la classe minoritaire en interpolant entre voisins existants — il ne duplique pas simplement des données.

#### Étape 4 : Train/Test split (80/20 stratifié)
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
`stratify=y` garantit que les deux classes sont représentées proportionnellement dans train ET test — essentiel pour une évaluation fiable.

### `notebooks/02_preprocessing.ipynb`
Ce notebook illustre chaque étape visuellement avec des graphiques avant/après SMOTE.

---

## 4. Semaine 3 — Modélisation & MLflow

### `code/modeling.py` — Entraînement & Tracking

**3 modèles entraînés, tous dans le même Pipeline :**

```
ImbPipeline([
    preprocessor  →  SMOTE  →  classifier
])
```

L'avantage du Pipeline : **tout est encapsulé**. Quand on sauvegarde le modèle, on sauvegarde aussi le scaler et le SMOTE — pas besoin de les appliquer séparément en production.

#### Modèle 1 : RandomForest + SMOTE
```python
RandomForestClassifier(random_state=42)
params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, None],
    'classifier__min_samples_split': [2, 5]
}
```
**Principe** : Ensemble de N arbres de décision entraînés sur des sous-ensembles aléatoires des données. La prédiction finale est un vote majoritaire. Robuste aux outliers, peu de réglages nécessaires.

**Résultat** : Accuracy 82%, F1 **0.85** ← meilleur modèle

#### Modèle 2 : XGBoost + SMOTE
```python
XGBClassifier(eval_metric='logloss', random_state=42)
params = {
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6]
}
```
**Principe** : Gradient Boosting — entraîne les arbres séquentiellement, chaque arbre corrige les erreurs du précédent. Plus puissant que RF sur données tabulaires, mais plus sensible aux hyperparamètres.

**Résultat** : Accuracy 80%, F1 0.83

#### Modèle 3 : Stacking Ensemble
```python
StackingClassifier(
    estimators=[('rf', RandomForest), ('xgb', XGBoost)],
    final_estimator=LogisticRegression(),
    cv=5
)
```
**Principe** : Combine les prédictions de RF et XGBoost (base learners) et les donne en entrée à une Régression Logistique (méta-modèle). L'idée est d'exploiter les forces complémentaires de chaque modèle.

**Résultat** : Accuracy 80%, F1 0.83

#### GridSearchCV — Optimisation des hyperparamètres
```python
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
```
`GridSearchCV` teste toutes les combinaisons d'hyperparamètres avec une **validation croisée à 5 folds**. On optimise le **F1-score** plutôt que l'accuracy car le coût d'un faux négatif (manquer une maladie) est élevé.

#### MLflow — Tracking des expériences

```python
with mlflow.start_run(run_name=model_name):
    mlflow.log_params(best_params)       # hyperparamètres
    mlflow.log_metric("accuracy", acc)   # métriques
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(cm_path)         # matrice de confusion
    mlflow.log_artifact(roc_path)        # courbe ROC
    mlflow.sklearn.log_model(model, "model")  # modèle sérialisé
```

**Pourquoi MLflow ?** Il permet de :
- Comparer les runs côte à côte dans une UI web
- Reproduire n'importe quelle expérience passée
- Promouvoir le meilleur modèle en production
- Éviter de perdre des résultats expérimentaux

Voir les résultats : **http://localhost:5000** → Expérience `HeartDisease_Optimized`

#### Sauvegarde du meilleur modèle
```python
joblib.dump(best_overall_model, "data/best_model_pipeline.pkl")
```
Le meilleur modèle (selon le F1-score) est sérialisé avec `joblib`. Ce fichier `.pkl` contient **l'intégralité du Pipeline** : scaler + SMOTE + classifier.

**Lancer :**
```bash
python code/modeling.py
```

---

## 5. Semaine 4 — API FastAPI

### `code/app.py` — Serveur d'inférence

**Pourquoi FastAPI ?**
- Validation automatique des inputs via Pydantic
- Documentation Swagger auto-générée (`/docs`)
- Performances proches de Node.js (async)
- Typage Python natif

#### Chargement du modèle au démarrage
```python
MODEL_PATH = "data/best_model_pipeline.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
```
Le modèle est chargé **une seule fois** au démarrage du serveur, puis gardé en mémoire — pas rechargé à chaque requête.

#### Endpoint 1 : `GET /health`
```python
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}
```
Utilisé par Docker et les outils de monitoring pour vérifier que le service est vivant et que le modèle est bien chargé.

#### Endpoint 2 : `POST /predict` — Prédiction individuelle
```python
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    # ... 13 champs au total
```
**Pydantic** valide automatiquement le JSON entrant — si un champ manque ou a le mauvais type, l'API retourne une erreur 422 avec un message clair.

```python
input_df = pd.DataFrame([data.dict()])
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]
```
Le modèle reçoit un DataFrame pandas (même format qu'à l'entraînement) et retourne la prédiction et la probabilité.

#### Endpoint 3 : `POST /predict_batch` — Prédiction en masse
```python
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    predictions = model.predict(df)
```
Accepte un fichier CSV avec plusieurs patients → retourne le CSV enrichi avec `Disease_Prediction` et `Disease_Probability` pour chaque ligne.

#### CORS — Communication avec le frontend
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```
Sans CORS, le navigateur bloquerait les requêtes du frontend (port 5173) vers l'API (port 8000) car ils sont sur des origines différentes.

**Lancer :**
```bash
python code/app.py
# Documentation : http://localhost:8000/docs
```

---

## 6. Semaine 5 — Frontend React

### Architecture du frontend

```
frontend/src/
├── App.jsx               ← composant racine + gestion de l'état global
├── index.css             ← thème dark glass-morphism
└── components/
    ├── HeartForm.jsx     ← formulaire 13 champs cliniques
    └── BatchUpload.jsx   ← upload et affichage CSV
```

### Design — Dark Glass-Morphism

```css
.glass-card {
  background: rgba(30, 41, 59, 0.7);   /* fond semi-transparent */
  backdrop-filter: blur(12px);           /* flou de fond */
  border: 1px solid rgba(255,255,255,0.1); /* bordure subtile */
}
```
Ce style "verre givré" sur fond sombre est inspiré des dashboards médicaux modernes. Il crée une hiérarchie visuelle claire sans surcharger l'interface.

### `App.jsx` — Gestion de l'état

```javascript
const [prediction, setPrediction] = useState(null);
const [loading, setLoading] = useState(false);

const handlePredict = async (data) => {
    setLoading(true);
    const response = await axios.post('http://localhost:8000/predict', data);
    setPrediction(response.data);
    setLoading(false);
};
```

**3 états gérés :**
- `loading` → affiche "Processing..." pendant l'appel API
- `prediction` → affiche le résultat une fois reçu
- `null` → affiche le message d'invitation initiale

**Layout :**
- Grille 2 colonnes : formulaire à gauche (2/3), résultats à droite (1/3)
- Sur mobile (< 968px) : 1 colonne verticale

### `HeartForm.jsx` — Formulaire clinique

```javascript
const [formData, setFormData] = useState({
    age: 55, sex: 1, cp: 0, trestbps: 130, ...
});
```

Chaque champ est typé et labellisé avec sa signification médicale. Les variables catégorielles (cp, slope, thal…) utilisent des `<select>` avec les options expliquées en français pour faciliter la saisie.

### `BatchUpload.jsx` — Upload CSV

1. L'utilisateur clique sur la zone **dropzone** → sélectionne un CSV
2. Clic sur "Upload & Process" → POST multipart vers `/predict_batch`
3. Le tableau affiche les 5 premiers résultats avec le % de risque et le statut coloré

### Affichage du résultat — Cercle de probabilité

```jsx
<div className="probability-circle"
  style={{ borderColor: prediction === 1 ? '#ef4444' : '#22c55e' }}>
  <span>{(probability * 100).toFixed(1)}%</span>
  <span>Disease Risk</span>
</div>
```
Le cercle change de couleur (rouge = maladie, vert = sain) et affiche le pourcentage de risque. L'effet glow (`box-shadow`) amplifie l'impact visuel.

**Lancer :**
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 7. Semaine 6 — Docker

### Pourquoi Docker ?

Docker garantit que l'application fonctionne **identiquement** sur n'importe quelle machine — peu importe l'OS, la version de Python ou de Node installée. On empaquète le code + ses dépendances dans des conteneurs isolés.

### `Dockerfile.backend`

```dockerfile
FROM python:3.9-slim           # image Python légère (~150MB)
WORKDIR /app
RUN apt-get install build-essential  # compilateurs pour certaines libs ML
COPY requirements.txt .
RUN pip install -r requirements.txt  # installe les dépendances
COPY code/ ./code/
COPY data/ ./data/             # inclut le modèle .pkl
EXPOSE 8000
CMD ["python", "code/app.py"]  # démarre l'API
```

**Build multi-étapes non nécessaire** ici — le backend Python n'a pas d'étape de compilation séparée.

### `Dockerfile.frontend`

```dockerfile
# Étape 1 : Builder
FROM node:18-alpine AS build
RUN npm install
RUN npm run build              # génère dist/

# Étape 2 : Serveur web
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

**Pourquoi 2 étapes ?** L'image finale ne contient que Nginx + les fichiers statiques compilés (~20MB), pas Node.js ni les node_modules (~200MB). C'est plus léger et plus sécurisé.

### `docker-compose.yml` — Orchestration 3 services

```yaml
services:
  backend:   # FastAPI - port 8000
  mlflow:    # MLflow UI - port 5000
  frontend:  # Nginx/React - port 5173
```

**`depends_on`** : le backend attend que MLflow soit prêt, le frontend attend le backend.

**Volume** `./data:/app/data` : le dossier `data/` (qui contient le modèle `.pkl`) est monté dans le conteneur — ça évite de reconstruire l'image si on réentraîne le modèle.

**Lancer tout avec Docker :**
```bash
docker-compose up --build    # première fois (build des images)
docker-compose up            # les fois suivantes
docker-compose down          # arrêter
```

---

## 8. Semaine 7 — CI/CD & Déploiement

### `.github/workflows/main.yml` — Pipeline CI/CD

Le pipeline s'exécute automatiquement à chaque `git push` sur `master` :

```
Push sur master
      ↓
lint-python (flake8)     ← vérifie la syntaxe Python
      ↓ (si OK)
build-docker-backend     ← construit l'image backend
build-docker-frontend    ← construit l'image frontend (en parallèle)
```

#### Job 1 : Lint Python
```yaml
- name: Lint with flake8
  run: flake8 code/ --max-line-length=120
```
`flake8` vérifie les erreurs de style PEP8 et les erreurs de syntaxe. Si le code ne passe pas le lint, les builds Docker ne se lancent pas.

#### Jobs 2 & 3 : Build Docker
```yaml
- name: Build backend image
  run: docker build -f Dockerfile.backend -t heart-disease-backend .
```
Vérifie que les Dockerfiles sont valides et que les images se construisent sans erreur.

### Options de déploiement

#### Option A — Render (recommandé, gratuit)
1. Créer un compte sur render.com
2. "New Web Service" → connecter le repo GitHub
3. Build command : `pip install -r requirements.txt`
4. Start command : `python code/app.py`
5. Pour le frontend : "New Static Site" → Build: `npm run build` → Publish: `dist/`

#### Option B — Railway
```bash
railway login
railway up
```

#### Option C — VPS (Ubuntu)
```bash
scp -r . user@vps:/app
ssh user@vps "cd /app && docker-compose up -d"
```

---

## Commandes de référence rapide

```bash
# Environnement
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Pipeline ML complet
python code/scraping.py          # collecter actualités
python code/data_exploration.py  # générer visualisations
python code/modeling.py          # entraîner + sauvegarder modèle

# Lancer tous les services
start cmd /k "title BACKEND  && venv\Scripts\python.exe code\app.py"
start cmd /k "title MLFLOW   && venv\Scripts\mlflow.exe ui --port 5000"
start cmd /k "title FRONTEND && cd frontend && npm run dev"

# Docker
docker-compose up --build

# URLs
# Frontend  → http://localhost:5173
# API Docs  → http://localhost:8000/docs
# MLflow    → http://localhost:5000
```

---

*Projet réalisé dans le cadre du module Intelligence Artificielle et Introduction au ML.*  
*Dataset : UCI Heart Disease — Cleveland (303 patients, 13 features)*
