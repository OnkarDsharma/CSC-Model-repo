# CSC Rejection Risk Dataset

This workspace contains a schema-driven synthetic dataset generator for 6 Chhattisgarh eDistrict services:

1. income_certificate
2. domicile_certificate
3. sc_st_certificate
4. obc_certificate
5. land_use_information
6. birth_certificate_correction

## Files

- `schemas/service_registry.json`: Master service registry with sections, fields, and required documents.
- `data_generator.py`: Synthetic data generator with rejection-risk logic.
- `dataset.csv`: Generated dataset (1000 samples by default).
- `datasets/*.csv`: Service-specific split datasets.

## Dataset Link (Local Workspace)

- `./dataset.csv`
- `./datasets/income_certificate.csv`
- `./datasets/domicile_certificate.csv`
- `./datasets/sc_st_certificate.csv`
- `./datasets/obc_certificate.csv`
- `./datasets/land_use_information.csv`
- `./datasets/birth_certificate_correction.csv`

## Generated Columns

- application_id
- service_type
- service_category
- age
- gender
- caste
- district
- annual_income
- average_income_last_3_years
- missing_documents_count
- missing_fields_count
- field_mismatch_count
- document_quality_score
- age_eligible
- income_eligible
- district_valid
- risk_probability
- risk_score
- risk_level
- main_contributing_factors
- rejected
- uploaded_documents
- application_fields

## How Rejection Risk Is Simulated

Rules increase rejection probability when:

- missing_documents_count > 0
- missing_fields_count > 0
- field_mismatch_count > 0
- document_quality_score is low
- age_eligible = 0
- income_eligible = 0
- district_valid = 0

The final label is binary:

- `0` = accepted
- `1` = rejected

## Regenerate Dataset

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" data_generator.py --samples 1000 --output dataset.csv --seed 42
```

You can change `--samples` to any value (for example, 500 to 2000).

## Train The Model

Run training with feature engineering and stratified split:

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl
```

Recommended tuned run (faster):

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl --search randomized --n-iter 30
```

Optional exhaustive tuning (slow):

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl --search grid
```

Train without tuning:

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl --search none
```

## Threshold Tuning (Recall-Focused)

Tune decision threshold using precision-recall curve:

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl --search randomized --n-iter 30 --tune-threshold
```

Balanced threshold tuning with minimum precision constraint:

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" train_model.py --data dataset.csv --model-output risk_model.pkl --search randomized --n-iter 30 --tune-threshold --min-precision 0.60
```

Threshold metadata is saved to `risk_model_meta.json`.

## Runtime Risk Prediction

Use trained model inference to generate risk score from input form/document features.

### CLI Prediction

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" predict_api.py --input sample_request.json
```

### Start API Server

```powershell
Set-Location "d:/ml model csc"
& "d:/ml model csc/.venv/Scripts/python.exe" -m uvicorn predict_api:app --host 0.0.0.0 --port 8000
```

### Endpoint

- `POST /predict-risk`

Example request body:

```json
{
	"service_type": "income_certificate",
	"age": 58,
	"gender": "male",
	"caste": "OBC",
	"district": "raipur",
	"annual_income": 120000,
	"average_income_last_3_years": 110000,
	"missing_documents_count": 1,
	"missing_fields_count": 0,
	"field_mismatch_count": 1,
	"document_quality_score": 0.8,
	"age_eligible": 0,
	"income_eligible": 1,
	"district_valid": 1
}
```

Example response:

```json
{
	"risk_probability": 0.3632,
	"risk_score": 36.32,
	"risk_level": "medium",
	"rejected_prediction": 0,
	"threshold_used": 0.451725,
	"main_contributing_factors": [
		"age_eligible",
		"missing_documents",
		"field_mismatch"
	]
}
```

What `train_model.py` applies:

- Numerical features: `age`, `annual_income`, `average_income_last_3_years`, `missing_documents_count`, `missing_fields_count`, `field_mismatch_count`, `document_quality_score`
- Binary flags: `age_eligible`, `income_eligible`, `district_valid`
- Categorical encoding:
	- `service_type` one-hot
	- `gender` one-hot
	- `caste` one-hot
	- `district` one-hot
- Stratified train/test split using target `rejected`

Model artifact:

- `risk_model.pkl`

## Deploy On Render

This repository is ready for Render deployment using:

- `requirements.txt`
- `runtime.txt`
- `render.yaml`

### Option A: Blueprint Deploy (recommended)

1. Open Render dashboard.
2. Click **New** -> **Blueprint**.
3. Select this GitHub repo.
4. Render reads `render.yaml` and creates service automatically.

### Option B: Manual Web Service

1. Open Render dashboard.
2. Click **New** -> **Web Service**.
3. Select this GitHub repo.
4. Use these settings:
	- Build Command: `pip install -r requirements.txt`
	- Start Command: `uvicorn predict_api:app --host 0.0.0.0 --port $PORT`

### Verify Deployment

- Health: `GET /health`
- Prediction: `POST /predict-risk`

Example:

- `https://<your-render-url>/health`
- `https://<your-render-url>/predict-risk`

### For Teammate Backend

Share your Render base URL. Teammate backend should call:

- `POST https://<your-render-url>/predict-risk`

Then teammate frontend shows returned fields:

- `risk_score`
- `risk_level`
- `main_contributing_factors`
