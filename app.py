from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import json
import pandas as pd
import joblib
from feature_engineering import engineer_features

# -----------------------------
# Load model and feature list
# -----------------------------
model = joblib.load("final_model.joblib")

with open("feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(
    title="Klarna Pay Later Default Model - AOB",
    description="Predicts probability of default for a Pay Later loan",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class LoanApplication(BaseModel):
    loan_amount: float
    existing_klarna_debt: float
    num_active_loans: int 
    days_since_first_loan: int

    new_exposure_7d: float
    new_exposure_14d: float

    num_failed_payments_3m: int
    num_failed_payments_6m: int
    num_failed_payments_1y: int

    num_confirmed_payments_3m: int
    num_confirmed_payments_6m: int

    amount_repaid_3m: float
    amount_repaid_6m: float
    amount_repaid_1y: float

    merchant_group: str
    merchant_category: str

    repayment_to_loan_ratio: float
    failed_ratio_3m_1y: float
    confirmed_ratio_3m_6m: float
    loan_to_existing_debt: float
    loan_vs_avg: float
      
# -----------------------------
# Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict_default(application: LoanApplication):

    df = pd.DataFrame([application.dict()])
    df = engineer_features(df)

    df = df[FEATURE_COLUMNS]

    pd_default = model.predict_proba(df)[:, 1][0]

    return {"probability_of_default": float(pd_default)}


# --- Batch prediction endpoint ---
@app.post("/predict_batch")
def predict_default_batch(applications: List[LoanApplication]):
    # Convert list of Pydantic objects to DataFrame
    df = pd.DataFrame([app.dict() for app in applications])
    
    # Engineer features
    df = engineer_features(df)
    
    # Select model columns
    df = df[FEATURE_COLUMNS]
    
    # Predict probabilities
    probabilities = model.predict_proba(df)[:, 1].tolist()
    
    return {"probabilities_of_default": probabilities}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # 1️⃣ Read CSV into DataFrame
    df = pd.read_csv(file.file)
    
    # 2️⃣ Engineer features
    df = engineer_features(df)
    
    # 3️⃣ Keep only model columns
    df_model = df[FEATURE_COLUMNS]
    
    # 4️⃣ Predict probabilities
    probabilities = model.predict_proba(df_model)[:, 1].tolist()
    
    # 5️⃣ Return probabilities (optionally include IDs if in CSV)
    return {"probabilities_of_default": probabilities}
