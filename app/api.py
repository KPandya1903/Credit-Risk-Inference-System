"""
Flask REST API for Credit Risk Inference
Provides /predict endpoint for real-time risk scoring
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import time
import os
import sys

# Add parent directory to path to import train_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global model variable
MODEL = None
MODEL_PATH = 'models/credit_risk_model.pkl'


def load_model():
    """Load the trained model"""
    global MODEL
    if MODEL is None:
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                MODEL_PATH
            )
            MODEL = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    return MODEL


def prepare_input(data):
    """Prepare input data for prediction"""
    # Expected features
    required_features = [
        'age', 'income', 'employment_length', 'credit_score',
        'num_credit_lines', 'credit_utilization', 'debt_to_income',
        'total_debt', 'savings', 'loan_amount', 'loan_term',
        'num_late_payments', 'num_bankruptcies', 'inquiries_last_6m',
        'employment_status', 'home_ownership', 'loan_purpose'
    ]

    # Validate required features
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Create DataFrame
    df = pd.DataFrame([data])

    # Calculate derived features
    df['monthly_payment'] = (df['loan_amount'] / df['loan_term'] * 1.05).astype(int)
    df['payment_to_income'] = (df['monthly_payment'] * 12 / df['income']).clip(0, 1)

    # Encode categorical features
    model_data = MODEL
    for col in model_data['categorical_features']:
        if col in df.columns:
            try:
                df[col] = model_data['label_encoders'][col].transform(df[col])
            except ValueError as e:
                # Handle unseen categories by using the most frequent category
                df[col] = 0

    return df


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'timestamp': time.time()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict credit risk for a loan application

    Expected JSON format:
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
    """
    start_time = time.time()

    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400

        # Load model if not loaded
        model_data = load_model()
        pipeline = model_data['pipeline']

        # Prepare input
        df = prepare_input(data)

        # Make prediction
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0]

        # Calculate risk score (0-100)
        risk_score = float(probability[1] * 100)

        # Determine risk level
        if risk_score < 30:
            risk_level = 'low'
        elif risk_score < 60:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # in milliseconds

        # Build response
        response = {
            'status': 'success',
            'prediction': int(prediction),
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'probability': {
                'low_risk': round(float(probability[0]), 4),
                'high_risk': round(float(probability[1]), 4)
            },
            'processing_time_ms': round(processing_time, 2),
            'timestamp': time.time()
        }

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 400

    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple applications
    """
    start_time = time.time()

    try:
        # Parse request
        data = request.get_json()

        if not data or 'applications' not in data:
            return jsonify({
                'error': 'No applications provided',
                'status': 'error'
            }), 400

        applications = data['applications']

        # Load model if not loaded
        model_data = load_model()
        pipeline = model_data['pipeline']

        # Process each application
        results = []
        for app in applications:
            try:
                df = prepare_input(app)
                prediction = pipeline.predict(df)[0]
                probability = pipeline.predict_proba(df)[0]
                risk_score = float(probability[1] * 100)

                results.append({
                    'prediction': int(prediction),
                    'risk_score': round(risk_score, 2),
                    'probability_high_risk': round(float(probability[1]), 4)
                })
            except Exception as e:
                results.append({
                    'error': str(e),
                    'status': 'error'
                })

        processing_time = (time.time() - start_time) * 1000

        return jsonify({
            'status': 'success',
            'count': len(results),
            'results': results,
            'processing_time_ms': round(processing_time, 2)
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500


if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")

    # Run the app
    print("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
