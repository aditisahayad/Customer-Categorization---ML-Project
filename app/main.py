"""
main.py
=======
FastAPI application for Customer Categorization prediction.

Endpoints:
    GET  /          → Welcome message
    POST /predict   → Predict customer cluster from input features

The app loads saved ML models (preprocessor, PCA, classifier) at startup,
connects to MongoDB for storing predictions, and serves the frontend UI.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schema import CustomerInput, PredictionOutput
from app.database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Cluster label mapping
# -------------------------------------------------------------------
CLUSTER_LABELS = {
    0: "Low Value Customer",
    1: "Medium Value Customer",
    2: "High Value Customer",
}

# -------------------------------------------------------------------
# Model paths
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# -------------------------------------------------------------------
# Global model references
# -------------------------------------------------------------------
model = None
pca_model = None
preprocessor = None


def load_model(filepath: str):
    """Load a pickled model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler:
    - On startup: load models and connect to MongoDB
    - On shutdown: close MongoDB connection
    """
    global model, pca_model, preprocessor

    # Startup
    logger.info("[APP] Loading ML models...")
    try:
        model = load_model(MODEL_PATH)
        pca_model = load_model(PCA_PATH)
        preprocessor = load_model(PREPROCESSOR_PATH)
        logger.info("[APP] All models loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"[APP] {e}")
        logger.error("[APP] Run 'python save_models.py' first to generate model files.")

    # Connect to MongoDB
    db.connect()

    yield

    # Shutdown
    db.close()
    logger.info("[APP] Application shutdown complete.")


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Customer Categorization API",
    description=(
        "A production-ready ML API that predicts customer clusters "
        "based on demographic and spending data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint. Returns a greeting message and API info.
    """
    return {
        "message": "Welcome to the Customer Categorization API!",
        "description": "Use POST /predict to classify a customer into a cluster.",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/ui", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict the customer cluster based on input features.

    **Input fields:**
    - Age: Customer's age
    - Income: Annual household income
    - Total_Spending: Total amount spent
    - Children: Number of children
    - Education: 0=Basic, 1=Diploma, 2=Graduation, 3=Master, 4=PhD

    **Returns:**
    - cluster: Predicted cluster number (0, 1, or 2)
    - category: Human-readable label (Low / Medium / High Value Customer)
    """
    # Ensure models are loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML models are not loaded. Run 'python save_models.py' first.",
        )

    try:
        # Prepare input as DataFrame (same column order as training)
        input_data = pd.DataFrame([{
            "Age": customer.Age,
            "Income": customer.Income,
            "Total_Spending": customer.Total_Spending,
            "Children": customer.Children,
            "Education": customer.Education,
        }])

        # The model pipeline includes scaling + classification
        predicted_cluster = int(model.predict(input_data)[0])

        # Map cluster to label
        category = CLUSTER_LABELS.get(predicted_cluster, "Unknown")

        # Store prediction in MongoDB
        db.store_prediction(
            input_data=customer.model_dump(),
            cluster=predicted_cluster,
            category=category,
        )

        return PredictionOutput(cluster=predicted_cluster, category=category)

    except Exception as e:
        logger.error(f"[APP] Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
