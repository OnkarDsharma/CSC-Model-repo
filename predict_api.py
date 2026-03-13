import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURE_COLUMNS = [
    "service_type",
    "age",
    "gender",
    "caste",
    "district",
    "annual_income",
    "average_income_last_3_years",
    "missing_documents_count",
    "missing_fields_count",
    "field_mismatch_count",
    "document_quality_score",
    "age_eligible",
    "income_eligible",
    "district_valid",
]

MODEL_PATH = Path("risk_model.pkl")
META_PATH = Path("risk_model_meta.json")


class RiskInput(BaseModel):
    service_type: str = Field(..., examples=["income_certificate"])
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., examples=["male", "female"])
    caste: str = Field(..., examples=["OBC", "SC", "ST", "GENERAL", "EWS"])
    district: str = Field(..., examples=["raipur"])
    annual_income: int = Field(..., ge=0)
    average_income_last_3_years: int = Field(..., ge=0)
    missing_documents_count: int = Field(..., ge=0)
    missing_fields_count: int = Field(..., ge=0)
    field_mismatch_count: int = Field(..., ge=0)
    document_quality_score: float = Field(..., ge=0.0, le=1.0)
    age_eligible: int = Field(..., ge=0, le=1)
    income_eligible: int = Field(..., ge=0, le=1)
    district_valid: int = Field(..., ge=0, le=1)


class RiskOutput(BaseModel):
    risk_probability: float
    risk_score: float
    risk_level: str
    rejected_prediction: int
    threshold_used: float
    main_contributing_factors: List[str]


_model = None
_threshold = 0.5


def load_model_and_threshold() -> None:
    global _model
    global _threshold

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    _model = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        _threshold = float(meta.get("selected_threshold", 0.5))


def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def contributing_factors(payload: Dict) -> List[str]:
    factors = []

    if payload["missing_documents_count"] > 0:
        factors.append(("missing_documents", min(0.46, payload["missing_documents_count"] * 0.18)))
    if payload["field_mismatch_count"] > 0:
        factors.append(("field_mismatch", min(0.24, payload["field_mismatch_count"] * 0.08)))
    if payload["missing_fields_count"] > 0:
        factors.append(("missing_fields", min(0.18, payload["missing_fields_count"] * 0.06)))
    if payload["document_quality_score"] < 0.8:
        factors.append(("document_quality", max(0.0, (1.0 - payload["document_quality_score"]) * 0.35)))
    if payload["age_eligible"] == 0:
        factors.append(("age_eligible", 0.24))
    if payload["income_eligible"] == 0:
        factors.append(("income_eligible", 0.22))
    if payload["district_valid"] == 0:
        factors.append(("district_valid", 0.16))

    if not factors:
        return ["no_major_issue_detected"]

    factors.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in factors[:3]]


def predict_row(payload: Dict) -> Dict:
    if _model is None:
        load_model_and_threshold()

    input_df = pd.DataFrame([[payload[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probability = float(_model.predict_proba(input_df)[0, 1])
    score = round(probability * 100, 2)
    decision = int(probability >= _threshold)

    return {
        "risk_probability": round(probability, 4),
        "risk_score": score,
        "risk_level": risk_level(probability),
        "rejected_prediction": decision,
        "threshold_used": round(_threshold, 6),
        "main_contributing_factors": contributing_factors(payload),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model_and_threshold()
    yield


app = FastAPI(title="CSC Rejection Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-risk", response_model=RiskOutput)
def predict_risk(payload: RiskInput) -> Dict:
    try:
        return predict_row(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict rejection risk from a JSON file.")
    parser.add_argument("--input", required=True, help="Path to JSON input payload.")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    result = predict_row(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
