"""
Credit Risk Data Generator
Generates synthetic credit risk dataset with realistic features
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os


def generate_credit_data(n_samples=1000, random_state=42):
    """
    Generate synthetic credit risk dataset

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(random_state)

    # Feature generation with realistic distributions
    data = {
        # Demographic features
        'age': np.random.normal(45, 12, n_samples).clip(18, 75).astype(int),
        'income': np.random.lognormal(10.8, 0.5, n_samples).clip(20000, 300000).astype(int),
        'employment_length': np.random.exponential(7, n_samples).clip(0, 40).astype(int),

        # Credit features
        'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850).astype(int),
        'num_credit_lines': np.random.poisson(5, n_samples).clip(0, 20),
        'credit_utilization': np.random.beta(2, 5, n_samples).clip(0, 1),

        # Financial features
        'debt_to_income': np.random.beta(2, 5, n_samples).clip(0, 0.8),
        'total_debt': np.random.lognormal(10.5, 0.8, n_samples).clip(0, 200000).astype(int),
        'savings': np.random.lognormal(9, 1.5, n_samples).clip(0, 100000).astype(int),

        # Loan features
        'loan_amount': np.random.lognormal(9.5, 0.7, n_samples).clip(1000, 100000).astype(int),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),

        # Behavioral features
        'num_late_payments': np.random.poisson(1.5, n_samples).clip(0, 10),
        'num_bankruptcies': np.random.binomial(1, 0.05, n_samples),
        'inquiries_last_6m': np.random.poisson(2, n_samples).clip(0, 10),
    }

    # Categorical features
    data['employment_status'] = np.random.choice(
        ['employed', 'self_employed', 'unemployed', 'retired'],
        n_samples,
        p=[0.65, 0.20, 0.10, 0.05]
    )

    data['home_ownership'] = np.random.choice(
        ['mortgage', 'rent', 'own', 'other'],
        n_samples,
        p=[0.45, 0.35, 0.15, 0.05]
    )

    data['loan_purpose'] = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement',
         'business', 'medical', 'other'],
        n_samples,
        p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    )

    df = pd.DataFrame(data)

    # Calculate derived features
    df['monthly_payment'] = (df['loan_amount'] / df['loan_term'] * 1.05).astype(int)
    df['payment_to_income'] = (df['monthly_payment'] * 12 / df['income']).clip(0, 1)

    # Generate target variable with realistic risk logic
    risk_score = (
        # Positive factors (reduce risk)
        - (df['credit_score'] - 680) / 100 * 0.3
        - (df['income'] - 60000) / 100000 * 0.2
        - df['employment_length'] / 40 * 0.15
        - (df['savings'] / 50000).clip(0, 1) * 0.2

        # Negative factors (increase risk)
        + df['debt_to_income'] * 2.0
        + df['credit_utilization'] * 1.5
        + df['num_late_payments'] / 10 * 1.0
        + df['num_bankruptcies'] * 1.5
        + df['inquiries_last_6m'] / 10 * 0.5
        + df['payment_to_income'] * 1.0

        # Add some randomness
        + np.random.normal(0, 0.3, n_samples)
    )

    # Convert to binary classification (1 = high risk, 0 = low risk)
    threshold = np.percentile(risk_score, 70)
    df['default'] = (risk_score > threshold).astype(int)

    return df


def save_data(df, filename='credit_data.csv'):
    """Save dataset to CSV file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"\nDefault rate: {df['default'].mean():.2%}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())
    return filepath


if __name__ == "__main__":
    print("Generating synthetic credit risk dataset...")
    df = generate_credit_data(n_samples=1000, random_state=42)
    save_data(df)

    print("\n" + "="*50)
    print("Feature distributions:")
    print("="*50)
    print(f"\nEmployment Status:")
    print(df['employment_status'].value_counts())
    print(f"\nHome Ownership:")
    print(df['home_ownership'].value_counts())
    print(f"\nLoan Purpose:")
    print(df['loan_purpose'].value_counts())
