"""
FastAPI REST API for Income Prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing import DataPreprocessor

app = FastAPI(title="Adult Income Prediction API", version="1.0.0")

# Load model and preprocessor
model = None
preprocessor = None

class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    income_class: str

@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, preprocessor
    
    try:
        model_path = Path('models/best_model.joblib')
        if not model_path.exists():
            # Try alternative paths
            model_path = Path('models/baseline_logistic_regression.joblib')
            if not model_path.exists():
                model_path = Path('models/baseline_random_forest.joblib')
        
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: Model file not found. Please train a model first.")
        
        preprocessor_path = Path('models/preprocessor.joblib')
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(str(preprocessor_path))
            print(f"Preprocessor loaded from {preprocessor_path}")
        else:
            print("Warning: Preprocessor file not found.")
            
    except Exception as e:
        print(f"Error loading model/preprocessor: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Adult Income Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make income predictions",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict income class"""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Please ensure models are trained."
        )
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([{
            'age': request.age,
            'workclass': request.workclass,
            'fnlwgt': request.fnlwgt,
            'education': request.education,
            'education-num': request.education_num,
            'marital-status': request.marital_status,
            'occupation': request.occupation,
            'relationship': request.relationship,
            'race': request.race,
            'sex': request.sex,
            'capital-gain': request.capital_gain,
            'capital-loss': request.capital_loss,
            'hours-per-week': request.hours_per_week,
            'native-country': request.native_country
        }])
        
        # Add the engineered features that the model expects
        # Age groups
        input_data['age_group'] = pd.cut(input_data['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])

        # Capital features
        input_data['capital-total'] = input_data['capital-gain'] - input_data['capital-loss']
        input_data['has_capital'] = (input_data['capital-total'] > 0).astype(int)

        # Hours per week groups
        input_data['hours_group'] = pd.cut(input_data['hours-per-week'], bins=[0, 35, 40, 50, 100], labels=[0, 1, 2, 3])

        # Education level grouping
        education_mapping = {
            'Preschool': 0, '1st-4th': 0, '5th-6th': 0, '7th-8th': 0, '9th': 0,
            '10th': 0, '11th': 0, '12th': 0, 'HS-grad': 1, 'Prof-school': 2,
            'Assoc-acdm': 1, 'Assoc-voc': 1, 'Some-college': 1, 'Bachelors': 2,
            'Masters': 2, 'Doctorate': 2
        }
        input_data['education_level'] = input_data['education'].map(education_mapping)
        
        # Use the preprocessor's encode_categorical method directly instead of full preprocess
        X_processed = preprocessor.encode_categorical(input_data, fit=False)

        # Then scale the numerical features
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 0:
            X_processed[numerical_cols] = preprocessor.scaler.transform(X_processed[numerical_cols])
        
        # Ensure feature order matches training
        if preprocessor.feature_names:
            X_processed = X_processed[preprocessor.feature_names]
        
        # Predict
        prediction = model.predict(X_processed.values)[0]
        
        # Get probability (handle different model types)
        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X_processed.values)[0]
            elif hasattr(model, 'fitted_pipeline_') and hasattr(model.fitted_pipeline_.steps[-1][1], 'predict_proba'):
                # TPOT model
                probability = model.fitted_pipeline_.steps[-1][1].predict_proba(X_processed.values)[0]
            else:
                # Fallback: use prediction as probability
                probability = [1.0 - float(prediction), float(prediction)]
        except:
            # Fallback if predict_proba fails
            probability = [1.0 - float(prediction), float(prediction)]
        
        # Decode prediction
        income_class = preprocessor.target_encoder.inverse_transform([prediction])[0]
        prob_high_income = probability[1] if len(probability) > 1 else probability[0]
        
        return PredictionResponse(
            prediction=income_class,
            probability=float(prob_high_income),
            income_class=">50K" if income_class == ">50K" else "<=50K"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)