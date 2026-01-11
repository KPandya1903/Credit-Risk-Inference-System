"""
Unit Tests for Credit Risk API
"""
import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.api import app, load_model


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_request():
    """Sample valid request data"""
    return {
        'age': 45,
        'income': 75000,
        'employment_length': 10,
        'credit_score': 720,
        'num_credit_lines': 5,
        'credit_utilization': 0.3,
        'debt_to_income': 0.35,
        'total_debt': 25000,
        'savings': 15000,
        'loan_amount': 20000,
        'loan_term': 36,
        'num_late_payments': 0,
        'num_bankruptcies': 0,
        'inquiries_last_6m': 1,
        'employment_status': 'employed',
        'home_ownership': 'mortgage',
        'loan_purpose': 'debt_consolidation'
    }


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data


def test_predict_valid_request(client, sample_request):
    """Test prediction with valid request"""
    # Load model first
    load_model()

    response = client.post('/predict',
                          data=json.dumps(sample_request),
                          content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'prediction' in data
    assert 'risk_score' in data
    assert 'risk_level' in data
    assert 'probability' in data
    assert 'processing_time_ms' in data

    # Check data types
    assert isinstance(data['prediction'], int)
    assert isinstance(data['risk_score'], (int, float))
    assert data['risk_level'] in ['low', 'medium', 'high']
    assert 0 <= data['risk_score'] <= 100

    # Check processing time
    assert data['processing_time_ms'] < 200  # Target latency


def test_predict_missing_fields(client):
    """Test prediction with missing required fields"""
    load_model()

    incomplete_request = {
        'age': 45,
        'income': 75000
    }

    response = client.post('/predict',
                          data=json.dumps(incomplete_request),
                          content_type='application/json')

    assert response.status_code == 400

    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'error' in data


def test_predict_no_data(client):
    """Test prediction with no data"""
    response = client.post('/predict',
                          data=json.dumps({}),
                          content_type='application/json')

    assert response.status_code == 400


def test_predict_high_risk_profile(client):
    """Test prediction with high-risk profile"""
    load_model()

    high_risk_request = {
        'age': 25,
        'income': 30000,
        'employment_length': 1,
        'credit_score': 550,
        'num_credit_lines': 2,
        'credit_utilization': 0.9,
        'debt_to_income': 0.7,
        'total_debt': 50000,
        'savings': 500,
        'loan_amount': 30000,
        'loan_term': 60,
        'num_late_payments': 5,
        'num_bankruptcies': 1,
        'inquiries_last_6m': 8,
        'employment_status': 'unemployed',
        'home_ownership': 'rent',
        'loan_purpose': 'debt_consolidation'
    }

    response = client.post('/predict',
                          data=json.dumps(high_risk_request),
                          content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    # High risk profile should have higher risk score
    assert data['risk_score'] > 30


def test_predict_low_risk_profile(client):
    """Test prediction with low-risk profile"""
    load_model()

    low_risk_request = {
        'age': 45,
        'income': 150000,
        'employment_length': 15,
        'credit_score': 820,
        'num_credit_lines': 10,
        'credit_utilization': 0.1,
        'debt_to_income': 0.15,
        'total_debt': 10000,
        'savings': 50000,
        'loan_amount': 15000,
        'loan_term': 24,
        'num_late_payments': 0,
        'num_bankruptcies': 0,
        'inquiries_last_6m': 0,
        'employment_status': 'employed',
        'home_ownership': 'own',
        'loan_purpose': 'home_improvement'
    }

    response = client.post('/predict',
                          data=json.dumps(low_risk_request),
                          content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    # Low risk profile should have lower risk score
    assert data['risk_score'] < 70


def test_batch_predict(client, sample_request):
    """Test batch prediction endpoint"""
    load_model()

    batch_request = {
        'applications': [
            sample_request,
            sample_request.copy()
        ]
    }

    response = client.post('/predict/batch',
                          data=json.dumps(batch_request),
                          content_type='application/json')

    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['count'] == 2
    assert len(data['results']) == 2
    assert 'processing_time_ms' in data


def test_batch_predict_empty(client):
    """Test batch prediction with empty list"""
    load_model()

    response = client.post('/predict/batch',
                          data=json.dumps({}),
                          content_type='application/json')

    assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
