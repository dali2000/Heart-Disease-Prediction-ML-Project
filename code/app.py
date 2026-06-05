from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import os

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "data/best_model_pipeline.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PatientData(BaseModel):
    age: int
    sex: int          # 1 = Male, 0 = Female
    cp: int           # Chest pain type (0-3)
    trestbps: float   # Resting blood pressure (mm Hg)
    chol: float       # Serum cholesterol (mg/dl)
    fbs: int          # Fasting blood sugar > 120 mg/dl (1 = True)
    restecg: int      # Resting ECG results (0-2)
    thalach: float    # Max heart rate achieved
    exang: int        # Exercise-induced angina (1 = Yes)
    oldpeak: float    # ST depression by exercise
    slope: int        # Slope of peak exercise ST segment (0-2)
    ca: int           # Number of major vessels (0-3)
    thal: int         # Thalassemia (0-3)

@app.get("/health")
def health_check():
    """
    Health check endpoint to ensure API is running and model is loaded.
    """
    if model:
        return {"status": "ok", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    """
    Real-time prediction for a single patient.
    Returns: disease prediction (0/1), probability, and status label.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")

    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "heart_disease_prediction": int(prediction),
        "heart_disease_probability": float(probability),
        "status": "Disease Detected" if prediction == 1 else "No Disease"
    }

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from a CSV file.
    Returns predictions with 'Disease_Prediction' and 'Disease_Probability' columns.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                         'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400,
                                detail=f"CSV must contain columns: {required_cols}")

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df['Disease_Prediction'] = predictions
        df['Disease_Probability'] = probabilities

        return df.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
