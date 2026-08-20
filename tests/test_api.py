import pytest
from fastapi.testclient import TestClient
from app.main import app

# The correct API Key we established in main.py
HEADERS = {"X-API-Key": "advance-soft-logics-beastmode"}

# A perfect 30-feature payload mimicking a real medical record
VALID_PAYLOAD = {
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0, 
    "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001, "mean_concave_points": 0.1471, 
    "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871, 
    "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4, 
    "smoothness_error": 0.006399, "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587, 
    "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, 
    "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0, 
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119, "worst_concave_points": 0.2654, 
    "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
}

# Using a pytest fixture to explicitly trigger FastAPI's startup event
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Test 1: Verify the server is up and the model is loaded in RAM."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"

def test_valid_prediction(client):
    """Test 2: Verify the API returns a prediction and confidence score for valid data."""
    response = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence_score" in response.json()

def test_missing_field_bad_request(client):
    """Test 3: Bad Input Category 1 - Missing a required field."""
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["mean_radius"]
    
    response = client.post("/predict", json=invalid_payload, headers=HEADERS)
    assert response.status_code == 400
    assert "Invalid input detected" in response.json()["detail"]

def test_wrong_data_type_bad_request(client):
    """Test 4: Bad Input Category 2 - Passing a string instead of a float."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["mean_radius"] = "this_is_a_string"
    
    response = client.post("/predict", json=invalid_payload, headers=HEADERS)
    assert response.status_code == 400
    assert "Invalid input detected" in response.json()["detail"]

def test_invalid_api_key(client):
    """Test 5: Verify the API blocks unauthorized requests."""
    response = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401