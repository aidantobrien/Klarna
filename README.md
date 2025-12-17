# Klarna

## Running the API Locally
Run the Klarna API locally in a few easy steps.

## Prerequisites

- Python 3.0+ installed
- `pip` package manager

1. Clone the repository:
```bash
git clone https://github.com/aidantobrien/Klarna_AOB.git
cd Klarna_AOB
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the API
```bash
python -m uvicorn app:app --reload
```

4. Open API docs
```bash
http://127.0.0.1:8000/docs
```

5. Hitting the API

Standard JSON
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "loan_amount": 0,
  "existing_klarna_debt": 0,
  "days_since_first_loan": 0,
  "new_exposure_7d": 0,
  "new_exposure_14d": 0,
  "num_failed_payments_3m": 0,
  "num_failed_payments_6m": 0,
  "num_failed_payments_1y": 0,
  "num_confirmed_payments_3m": 0,
  "num_confirmed_payments_6m": 0,
  "amount_repaid_3m": 0,
  "amount_repaid_6m": 0,
  "amount_repaid_1y": 0,
  "merchant_group": "string",
  "merchant_category": "string",
  "repayment_to_loan_ratio": 0,
  "failed_ratio_3m_1y": 0,
  "confirmed_ratio_3m_6m": 0,
  "loan_to_existing_debt": 0,
  "loan_vs_avg": 0
}'
```
Batch JSON in list format 
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch" \
-H "Content-Type: application/json" \
-d '[
  {
    "loan_amount": 0,
    "existing_klarna_debt": 0,
    "days_since_first_loan": 0,
    "new_exposure_7d": 0,
    "new_exposure_14d": 0,
    "num_failed_payments_3m": 0,
    "num_failed_payments_6m": 0,
    "num_failed_payments_1y": 0,
    "num_confirmed_payments_3m": 0,
    "num_confirmed_payments_6m": 0,
    "amount_repaid_3m": 0,
    "amount_repaid_6m": 0,
    "amount_repaid_1y": 0,
    "merchant_group": "string",
    "merchant_category": "string",
    "repayment_to_loan_ratio": 0,
    "failed_ratio_3m_1y": 0,
    "confirmed_ratio_3m_6m": 0,
    "loan_to_existing_debt": 0,
    "loan_vs_avg": 0
  },
  {
    "loan_amount": 100,
    "existing_klarna_debt": 50,
    "days_since_first_loan": 200,
    "new_exposure_7d": 0,
    "new_exposure_14d": 5,
    "num_failed_payments_3m": 0,
    "num_failed_payments_6m": 1,
    "num_failed_payments_1y": 1,
    "num_confirmed_payments_3m": 1,
    "num_confirmed_payments_6m": 2,
    "amount_repaid_3m": 50,
    "amount_repaid_6m": 100,
    "amount_repaid_1y": 150,
    "merchant_group": "groupA",
    "merchant_category": "categoryA",
    "repayment_to_loan_ratio": 0.5,
    "failed_ratio_3m_1y": 0.2,
    "confirmed_ratio_3m_6m": 0.8,
    "loan_to_existing_debt": 2,
    "loan_vs_avg": 1.2
  }
]'
```

CSV 
```bash 
curl -X POST "http://127.0.0.1:8000/predict_csv" \
-H "Content-Type: multipart/form-data" \
-F "file=@data.csv"
```

