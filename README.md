# house-prices-advanced-regression-techniques

## Run Backend API

1. Train the model:

	`python backend/xgb_pipeline.py`

2. Start FastAPI:

	`python -m uvicorn backend.api:app --reload`

3. Open docs:

	`http://127.0.0.1:8000/docs`

## Run Streamlit Frontend

1. Start the frontend:

	`python -m streamlit run frontend/app.py`

2. Open browser:

	`http://localhost:8501`
