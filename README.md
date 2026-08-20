# AI/ML Internship Week 5: Production ML Inference API

## Project Overview
This repository transitions a tuned Random Forest model from a static Jupyter Notebook training environment into a live, production-ready inference system using FastAPI. It enforces strict separation of concerns, utilizing Pydantic for input validation, custom middleware for HTTP 400 error handling, and structured JSON logging. A Streamlit frontend is provided for seamless client interaction.

## Architecture
* **`app/main.py`**: The FastAPI application, handling startup model loading, inference routing, and structured logging.
* **`app/schemas.py`**: Strict Pydantic models enforcing data types for all 30 Breast Cancer dataset features.
* **`app/model_assets/`**: Contains the serialized `best_rf_model.pkl` engine.
* **`tests/test_api.py`**: An automated Pytest suite verifying health, predictions, and validation errors.
* **`frontend.py`**: A minimal Streamlit dashboard acting as the client UI.

## Setup Instructions
1. Clone the repository: `git clone https://github.com/Abdul-Moiz-Asif/asl-internship-aiml-week5-abdulmoizasif.git`
2. Create and activate a virtual environment: `python -m venv .venv` | `.\.venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Boot the FastAPI Server: `uvicorn app.main:app --reload`
5. Launch the UI (in a separate terminal): `streamlit run frontend.py`

## API Contract & Documentation

### Authentication
All inference endpoints require an API Key passed in the headers:
`X-API-Key: advance-soft-logics-beastmode`

### 1. Health Check (`GET /health`)
Verifies if the API is online and the ML model is loaded into RAM.
* **Response:** `200 OK`
  ```json
  {"status": "Healthy", "model_version": "RandomForest_v1.0_Tuned", "message": "API is ready for inference."}