#!/bin/bash

# Credit Risk API Startup Script

echo "=========================================="
echo "Credit Risk Inference System"
echo "=========================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "models/credit_risk_model.pkl" ]; then
    echo "Model not found! Training model first..."
    python train_model.py
    echo ""
fi

# Start API server
echo "Starting Flask API server..."
echo "API will be available at: http://localhost:5000"
echo ""
echo "Endpoints:"
echo "  - GET  /health"
echo "  - POST /predict"
echo "  - POST /predict/batch"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python app/api.py
