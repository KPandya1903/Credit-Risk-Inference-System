# Quick Start Guide

## Setup (One-time)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Verify installation
pip list | grep -E "xgboost|flask|scikit-learn"
```

## Run the System

### Option 1: Using the startup script
```bash
./run_api.sh
```

### Option 2: Manual steps
```bash
# Start API server
source venv/bin/activate
python app/api.py
```

The API will be running at `http://localhost:5000`

## Test the System

### 1. Quick API Test
```bash
# In a new terminal
source venv/bin/activate
python example_usage.py
```

### 2. Run Unit Tests
```bash
pytest tests/test_api.py -v
```

### 3. Run Stress Test (1,000 requests)
```bash
# Make sure API is running first!
python tests/stress_test.py
```

## Common Commands

```bash
# Regenerate data
python data/generate_data.py

# Retrain model
python train_model.py

# Simple API test
python tests/stress_test.py simple

# Custom stress test (500 requests, 5 workers)
python tests/stress_test.py 500 5
```

## API Examples

### Using curl
```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "income": 75000,
    "employment_length": 10,
    "credit_score": 720,
    "num_credit_lines": 5,
    "credit_utilization": 0.3,
    "debt_to_income": 0.35,
    "total_debt": 25000,
    "savings": 15000,
    "loan_amount": 20000,
    "loan_term": 36,
    "num_late_payments": 0,
    "num_bankruptcies": 0,
    "inquiries_last_6m": 1,
    "employment_status": "employed",
    "home_ownership": "mortgage",
    "loan_purpose": "debt_consolidation"
  }'
```

### Using Python
```python
import requests

# Make prediction
response = requests.post('http://localhost:5000/predict', json={
    'age': 45,
    'income': 75000,
    # ... other features
})

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

## Performance Benchmarks

Based on test set (150 samples):

**Model Performance:**
- Accuracy: 81.33%
- AUC-ROC: 91.22%
- Precision: 70.73%
- Recall: 64.44%

**API Performance:**
- Target Latency: < 200ms
- Expected Median: ~15-50ms
- Expected P95: ~100-200ms
- Throughput: > 100 req/s

## Troubleshooting

**API won't start:**
- Check if port 5000 is available: `lsof -i :5000`
- Ensure virtual environment is activated
- Verify model file exists: `ls models/credit_risk_model.pkl`

**Model file missing:**
```bash
python train_model.py
```

**Tests failing:**
- Ensure API is running for stress tests
- Check that all dependencies are installed
- Verify data file exists: `ls data/credit_data.csv`

## File Locations

- Dataset: `data/credit_data.csv`
- Model: `models/credit_risk_model.pkl`
- Metrics: `models/metrics.json`
- API: `app/api.py`
- Tests: `tests/`

## Next Steps

1. Review model metrics in `models/metrics.json`
2. Run stress tests to verify performance
3. Customize features in `data/generate_data.py`
4. Add your own training data
5. Deploy to production (see README.md)
