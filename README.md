# Credit Risk Inference System

A production-ready machine learning system for credit risk assessment with real-time inference capabilities. Built with Python, XGBoost, Flask, and scikit-learn.

## Overview

This system provides automated credit risk scoring for loan applications with the following features:

- **ML Pipeline**: Scikit-learn Pipeline with StandardScaler, categorical encoding, and XGBoost classifier
- **Performance**: AUC-ROC > 0.91, Accuracy > 81%
- **Low Latency**: API responses < 200ms (median)
- **Scalability**: Handles 1,000+ daily requests with concurrent processing
- **REST API**: Flask-based API with `/predict` endpoint

## Tech Stack

- **Core ML**: Python 3.11, XGBoost 2.0.0, scikit-learn 1.3.0
- **API Framework**: Flask 2.3.3 with CORS support
- **Database**: PostgreSQL (psycopg2-binary)
- **Data Processing**: pandas, numpy
- **Testing**: pytest, requests

## Project Structure

```
CRA/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── credit_data.csv         # Generated dataset (1,000 rows)
├── models/
│   ├── credit_risk_model.pkl   # Trained model pipeline
│   └── metrics.json            # Performance metrics
├── app/
│   ├── __init__.py
│   └── api.py                  # Flask REST API
├── tests/
│   ├── test_api.py             # Unit tests
│   └── stress_test.py          # Performance testing (1,000 requests)
├── train_model.py              # Model training script
├── requirements.txt            # Dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository (if applicable)
cd CRA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```bash
# Generate synthetic credit dataset
python data/generate_data.py

# Train XGBoost model
python train_model.py
```

**Expected Output:**
- Dataset: 1,000 samples, 30% default rate
- Test Set AUC-ROC: ~0.91
- Test Set Accuracy: ~0.81

### 3. Run API Server

```bash
# Start Flask API
python app/api.py
```

The API will be available at `http://localhost:5000`

### 4. Test the API

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Single Prediction
```bash
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

**Response:**
```json
{
  "status": "success",
  "prediction": 0,
  "risk_score": 24.56,
  "risk_level": "low",
  "probability": {
    "low_risk": 0.7544,
    "high_risk": 0.2456
  },
  "processing_time_ms": 12.45,
  "timestamp": 1704999999.123
}
```

### 5. Run Tests

#### Unit Tests
```bash
pytest tests/test_api.py -v
```

#### Stress Test (1,000 Requests)
```bash
# Ensure API is running first
python tests/stress_test.py
```

**Expected Performance:**
- Median latency: < 200ms
- Throughput: > 100 requests/second
- Success rate: > 99%

#### Simple API Test
```bash
python tests/stress_test.py simple
```

## API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1704999999.123
}
```

### `POST /predict`
Single prediction endpoint

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": 0,
  "risk_score": 24.56,
  "risk_level": "low",
  "probability": {
    "low_risk": 0.7544,
    "high_risk": 0.2456
  },
  "processing_time_ms": 12.45,
  "timestamp": 1704999999.123
}
```

### `POST /predict/batch`
Batch prediction endpoint for multiple applications

**Request Body:**
```json
{
  "applications": [
    { /* application 1 */ },
    { /* application 2 */ }
  ]
}
```

## Features

### Dataset Generation
- 1,000 synthetic credit applications
- 17 input features + 2 derived features
- Realistic distributions based on financial data
- 30% default rate with risk-based logic

### Feature Engineering
- **Numerical Features**: age, income, credit_score, debt_to_income, etc.
- **Categorical Features**: employment_status, home_ownership, loan_purpose
- **Derived Features**: monthly_payment, payment_to_income ratio

### Model Pipeline
1. **Preprocessing**:
   - StandardScaler for numerical features
   - LabelEncoder for categorical features
   - ColumnTransformer for combined preprocessing

2. **Classifier**:
   - XGBoost with optimized hyperparameters
   - 100 estimators, max_depth=6, learning_rate=0.1
   - L1/L2 regularization

3. **Evaluation Metrics**:
   - AUC-ROC: 0.9122
   - Accuracy: 0.8133
   - Precision: 0.7073
   - Recall: 0.6444
   - F1 Score: 0.6744

### API Optimization
- Model loaded once at startup (not per request)
- Efficient pandas DataFrame operations
- Response time tracking
- Error handling and validation

## Performance Metrics

### Model Performance (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 81.33% |
| AUC-ROC | 91.22% |
| Precision | 70.73% |
| Recall | 64.44% |
| F1 Score | 67.44% |

### API Performance (1,000 Requests)
| Metric | Target | Actual |
|--------|--------|--------|
| Median Latency | < 200ms | ~15-50ms* |
| P95 Latency | < 500ms | ~100-200ms* |
| Throughput | > 50 req/s | > 100 req/s* |
| Success Rate | > 95% | > 99% |

*Actual values depend on hardware

## Development

### Adding New Features
1. Update `data/generate_data.py` to include new features
2. Modify `train_model.py` to handle new features in pipeline
3. Update API input validation in `app/api.py`
4. Retrain model: `python train_model.py`

### Retraining Model
```bash
# Generate new dataset
python data/generate_data.py

# Train model
python train_model.py

# Restart API server
python app/api.py
```

### Running in Production

#### Using Gunicorn (Recommended)
```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app.api:app
```

#### Using Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.api:app"]
```

## Future Enhancements

- [ ] PostgreSQL integration for data storage
- [ ] Model versioning and A/B testing
- [ ] Real-time monitoring and alerting
- [ ] Feature importance visualization
- [ ] Batch processing for large datasets
- [ ] SHAP values for model explainability
- [ ] API authentication and rate limiting
- [ ] Docker containerization
- [ ] CI/CD pipeline

## License

MIT License

## Contact

For questions or support, please contact your ML Engineering team.
