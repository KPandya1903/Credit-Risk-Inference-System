"""
Example Usage of Credit Risk API
Demonstrates how to interact with the API programmatically
"""
import requests
import json


API_URL = 'http://localhost:5000'


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f'{API_URL}/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ Model Loaded: {data['model_loaded']}\n")
            return True
        else:
            print(f"✗ API Health Check Failed: {response.status_code}\n")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API: {e}")
        print("Please make sure the API is running: python app/api.py\n")
        return False


def predict_single_application(application_data):
    """Make a single prediction"""
    try:
        response = requests.post(
            f'{API_URL}/predict',
            json=application_data,
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def main():
    """Main example"""
    print("="*60)
    print("Credit Risk API - Example Usage")
    print("="*60)
    print()

    # Check API health
    print("1. Checking API health...")
    if not check_api_health():
        return

    # Example 1: Low-risk applicant
    print("2. Example 1: Low-Risk Applicant")
    print("-" * 60)
    low_risk_app = {
        'age': 45,
        'income': 120000,
        'employment_length': 15,
        'credit_score': 780,
        'num_credit_lines': 8,
        'credit_utilization': 0.2,
        'debt_to_income': 0.25,
        'total_debt': 15000,
        'savings': 40000,
        'loan_amount': 20000,
        'loan_term': 36,
        'num_late_payments': 0,
        'num_bankruptcies': 0,
        'inquiries_last_6m': 1,
        'employment_status': 'employed',
        'home_ownership': 'own',
        'loan_purpose': 'home_improvement'
    }

    result = predict_single_application(low_risk_app)
    if result:
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Prediction: {'HIGH RISK' if result['prediction'] == 1 else 'LOW RISK'}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print()

    # Example 2: High-risk applicant
    print("3. Example 2: High-Risk Applicant")
    print("-" * 60)
    high_risk_app = {
        'age': 28,
        'income': 35000,
        'employment_length': 2,
        'credit_score': 580,
        'num_credit_lines': 3,
        'credit_utilization': 0.85,
        'debt_to_income': 0.65,
        'total_debt': 45000,
        'savings': 1000,
        'loan_amount': 25000,
        'loan_term': 60,
        'num_late_payments': 4,
        'num_bankruptcies': 0,
        'inquiries_last_6m': 6,
        'employment_status': 'employed',
        'home_ownership': 'rent',
        'loan_purpose': 'debt_consolidation'
    }

    result = predict_single_application(high_risk_app)
    if result:
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Prediction: {'HIGH RISK' if result['prediction'] == 1 else 'LOW RISK'}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print()

    # Example 3: Batch predictions
    print("4. Example 3: Batch Predictions")
    print("-" * 60)

    batch_request = {
        'applications': [low_risk_app, high_risk_app]
    }

    try:
        response = requests.post(
            f'{API_URL}/predict/batch',
            json=batch_request,
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Processed {data['count']} applications")
            print(f"Total Processing Time: {data['processing_time_ms']:.2f}ms")
            print(f"Average Time per Application: {data['processing_time_ms']/data['count']:.2f}ms")
            print()

            for i, result in enumerate(data['results'], 1):
                print(f"Application {i}:")
                print(f"  Risk Score: {result['risk_score']:.2f}")
                print(f"  Prediction: {'HIGH RISK' if result['prediction'] == 1 else 'LOW RISK'}")
            print()

    except requests.exceptions.RequestException as e:
        print(f"Batch request failed: {e}")
        print()

    print("="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
