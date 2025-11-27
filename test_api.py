"""
Test script for the FastAPI
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    """Test predict endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Sample data
    sample_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=sample_data,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print(f"\n✅ Prediction: {result['income_class']}")
            print(f"   Probability: {result['probability']*100:.2f}%")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("FASTAPI TEST SUITE")
    print("=" * 80)
    
    # Test health
    health_ok = test_health()
    
    # Test predict
    if health_ok:
        predict_ok = test_predict()
        
        if predict_ok:
            print("\n" + "=" * 80)
            print("✅ All tests passed!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ Predict test failed")
            print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Health check failed. Make sure the API is running.")
        print("   Start it with: python api.py")
        print("=" * 80)

