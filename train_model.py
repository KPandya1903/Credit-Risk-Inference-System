"""
Credit Risk Model Training Script
Trains XGBoost classifier with preprocessing pipeline
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)
import xgboost as xgb
import os
from datetime import datetime


class CreditRiskModel:
    """Credit Risk Model with preprocessing pipeline"""

    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.label_encoders = {}

    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Define feature groups
        self.categorical_features = ['employment_status', 'home_ownership', 'loan_purpose']
        self.numerical_features = [
            'age', 'income', 'employment_length', 'credit_score',
            'num_credit_lines', 'credit_utilization', 'debt_to_income',
            'total_debt', 'savings', 'loan_amount', 'loan_term',
            'num_late_payments', 'num_bankruptcies', 'inquiries_last_6m',
            'monthly_payment', 'payment_to_income'
        ]

        # Encode categorical features
        df_encoded = df.copy()
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])

        return df_encoded

    def build_pipeline(self):
        """Build scikit-learn pipeline with XGBoost"""
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', 'passthrough', self.categorical_features)
            ])

        # XGBoost classifier with optimized hyperparameters
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )

        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_classifier)
        ])

        return self.pipeline

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print("Training XGBoost model...")

        # Build pipeline if not exists
        if self.pipeline is None:
            self.build_pipeline()

        # Train without early stopping (will use pipeline directly)
        self.pipeline.fit(X_train, y_train)

        print("Training completed!")
        return self.pipeline

    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"Performance Metrics - {dataset_name} Set")
        print('='*60)

        # Predictions
        y_pred = self.pipeline.predict(X)
        y_pred_proba = self.pipeline.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Low Risk', 'High Risk']))

        return metrics

    def save_model(self, filepath='models/credit_risk_model.pkl'):
        """Save trained model and encoders"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'pipeline': self.pipeline,
            'label_encoders': self.label_encoders,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }

        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")

    def load_model(self, filepath='models/credit_risk_model.pkl'):
        """Load trained model and encoders"""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.label_encoders = model_data['label_encoders']
        self.categorical_features = model_data['categorical_features']
        self.numerical_features = model_data['numerical_features']
        print(f"Model loaded from: {filepath}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("Credit Risk Model Training Pipeline")
    print("="*60)

    # Load data
    print("\n1. Loading data...")
    data_path = 'data/credit_data.csv'
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']

    # Split data
    print("\n2. Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Initialize model
    print("\n3. Building model pipeline...")
    model = CreditRiskModel()

    # Prepare features
    X_train_prep = model.prepare_features(X_train)
    X_val_prep = model.prepare_features(X_val)
    X_test_prep = model.prepare_features(X_test)

    # Build and train model
    print("\n4. Training model...")
    model.build_pipeline()
    model.train(X_train_prep, y_train, X_val_prep, y_val)

    # Evaluate on train, validation, and test sets
    print("\n5. Evaluating model...")
    train_metrics = model.evaluate(X_train_prep, y_train, "Train")
    val_metrics = model.evaluate(X_val_prep, y_val, "Validation")
    test_metrics = model.evaluate(X_test_prep, y_test, "Test")

    # Save model
    print("\n6. Saving model...")
    model.save_model('models/credit_risk_model.pkl')

    # Save metrics
    metrics_data = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'timestamp': datetime.now().isoformat()
    }

    metrics_path = 'models/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nKey Metrics (Test Set):")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
