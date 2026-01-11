"""
Stress Testing Script for Credit Risk API
Simulates 1,000 daily requests to test latency and throughput
"""
import requests
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# API Configuration
API_URL = 'http://localhost:5000/predict'
HEALTH_URL = 'http://localhost:5000/health'


def generate_sample_request():
    """Generate a random sample request"""
    return {
        'age': int(np.random.normal(45, 12)),
        'income': int(np.random.lognormal(10.8, 0.5)),
        'employment_length': int(np.random.exponential(7)),
        'credit_score': int(np.random.normal(680, 80)),
        'num_credit_lines': int(np.random.poisson(5)),
        'credit_utilization': float(np.random.beta(2, 5)),
        'debt_to_income': float(np.random.beta(2, 5)),
        'total_debt': int(np.random.lognormal(10.5, 0.8)),
        'savings': int(np.random.lognormal(9, 1.5)),
        'loan_amount': int(np.random.lognormal(9.5, 0.7)),
        'loan_term': int(np.random.choice([12, 24, 36, 48, 60])),
        'num_late_payments': int(np.random.poisson(1.5)),
        'num_bankruptcies': int(np.random.binomial(1, 0.05)),
        'inquiries_last_6m': int(np.random.poisson(2)),
        'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed', 'retired']),
        'home_ownership': np.random.choice(['mortgage', 'rent', 'own', 'other']),
        'loan_purpose': np.random.choice([
            'debt_consolidation', 'credit_card', 'home_improvement',
            'business', 'medical', 'other'
        ])
    }


def make_request(request_id):
    """Make a single prediction request"""
    try:
        sample = generate_sample_request()
        start_time = time.time()

        response = requests.post(API_URL, json=sample, timeout=5)

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # in milliseconds

        if response.status_code == 200:
            data = response.json()
            return {
                'request_id': request_id,
                'status': 'success',
                'latency_ms': latency,
                'api_processing_time_ms': data.get('processing_time_ms', 0),
                'risk_score': data.get('risk_score', 0),
                'prediction': data.get('prediction', 0)
            }
        else:
            return {
                'request_id': request_id,
                'status': 'failed',
                'latency_ms': latency,
                'error': response.text
            }

    except Exception as e:
        return {
            'request_id': request_id,
            'status': 'error',
            'error': str(e),
            'latency_ms': -1
        }


def check_health():
    """Check if API is healthy"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Health Check: {data['status']}")
            print(f"✓ Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API Health Check Error: {str(e)}")
        return False


def run_stress_test(num_requests=1000, num_workers=10):
    """
    Run stress test with concurrent requests

    Args:
        num_requests: Total number of requests to make
        num_workers: Number of concurrent workers
    """
    print("="*70)
    print(f"Credit Risk API Stress Test")
    print("="*70)
    print(f"Total Requests: {num_requests}")
    print(f"Concurrent Workers: {num_workers}")
    print(f"Target Latency: <200ms")
    print("="*70)

    # Check API health
    print("\nChecking API health...")
    if not check_health():
        print("\n⚠️  API is not healthy. Please start the API server first.")
        print("Run: python app/api.py")
        return

    print("\n🚀 Starting stress test...\n")

    results = []
    start_time = time.time()

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all requests
        futures = {executor.submit(make_request, i): i for i in range(num_requests)}

        # Collect results
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress indicator
            if completed % 100 == 0:
                print(f"Progress: {completed}/{num_requests} requests completed")

    end_time = time.time()
    total_time = end_time - start_time

    # Analyze results
    analyze_results(results, total_time, num_requests)


def analyze_results(results, total_time, num_requests):
    """Analyze and display test results"""
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Success rate
    success_count = len(df[df['status'] == 'success'])
    success_rate = (success_count / len(results)) * 100

    print(f"\n📊 Overall Metrics:")
    print(f"  Total Requests: {num_requests}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Throughput: {num_requests/total_time:.2f} requests/second")

    # Latency analysis (only successful requests)
    successful = df[df['status'] == 'success']

    if len(successful) > 0:
        latencies = successful['latency_ms']
        api_times = successful['api_processing_time_ms']

        print(f"\n⚡ Latency Metrics (Total Round-trip):")
        print(f"  Mean: {latencies.mean():.2f} ms")
        print(f"  Median: {latencies.median():.2f} ms")
        print(f"  Min: {latencies.min():.2f} ms")
        print(f"  Max: {latencies.max():.2f} ms")
        print(f"  P95: {latencies.quantile(0.95):.2f} ms")
        print(f"  P99: {latencies.quantile(0.99):.2f} ms")

        print(f"\n🔧 API Processing Time (Server-side):")
        print(f"  Mean: {api_times.mean():.2f} ms")
        print(f"  Median: {api_times.median():.2f} ms")
        print(f"  Min: {api_times.min():.2f} ms")
        print(f"  Max: {api_times.max():.2f} ms")
        print(f"  P95: {api_times.quantile(0.95):.2f} ms")
        print(f"  P99: {api_times.quantile(0.99):.2f} ms")

        # Check if meeting latency requirement
        target_latency = 200
        within_target = (latencies <= target_latency).sum()
        target_percentage = (within_target / len(latencies)) * 100

        print(f"\n🎯 Latency Target (<{target_latency}ms):")
        print(f"  Requests within target: {within_target}/{len(latencies)} ({target_percentage:.2f}%)")

        if latencies.median() < target_latency:
            print(f"  ✓ PASS: Median latency ({latencies.median():.2f}ms) < {target_latency}ms")
        else:
            print(f"  ✗ FAIL: Median latency ({latencies.median():.2f}ms) >= {target_latency}ms")

        # Prediction distribution
        print(f"\n📈 Prediction Distribution:")
        print(f"  Low Risk (0): {(successful['prediction'] == 0).sum()}")
        print(f"  High Risk (1): {(successful['prediction'] == 1).sum()}")

        # Risk score statistics
        print(f"\n💯 Risk Score Statistics:")
        print(f"  Mean: {successful['risk_score'].mean():.2f}")
        print(f"  Median: {successful['risk_score'].median():.2f}")
        print(f"  Min: {successful['risk_score'].min():.2f}")
        print(f"  Max: {successful['risk_score'].max():.2f}")

    # Error analysis
    errors = df[df['status'] != 'success']
    if len(errors) > 0:
        print(f"\n⚠️  Errors Encountered:")
        error_types = errors['error'].value_counts()
        for error, count in error_types.items():
            print(f"  {error}: {count}")

    print("\n" + "="*70)
    print("Stress Test Completed!")
    print("="*70)

    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'tests/stress_test_results_{timestamp}.csv'
    df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")


def run_simple_test():
    """Run a simple single request test"""
    print("Running simple API test...")

    if not check_health():
        print("\n⚠️  API is not healthy. Please start the API server first.")
        return

    sample = generate_sample_request()
    print(f"\nSample Request:")
    print(json.dumps(sample, indent=2))

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=sample, timeout=5)
        latency = (time.time() - start_time) * 1000

        print(f"\nResponse Status: {response.status_code}")
        print(f"Latency: {latency:.2f} ms")

        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'simple':
            run_simple_test()
        else:
            num_requests = int(sys.argv[1])
            num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            run_stress_test(num_requests, num_workers)
    else:
        # Default: 1,000 requests with 10 concurrent workers
        run_stress_test(num_requests=1000, num_workers=10)
