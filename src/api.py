
import sys
import os

from fastapi import FastAPI
from typing import Optional

# Allow imports from the src folder
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from preprocessing import load_data, preprocess_data
from recommendation import RecommendationSystem


# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data"
)

customers, products, interactions = load_data(DATA_PATH)

interactions = preprocess_data(interactions)


# ---------------------------------------------------------
# Create recommendation system
# ---------------------------------------------------------

recommender = RecommendationSystem(
    customers,
    products,
    interactions
)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Intelligent Product Recommendation System",
    description="Hybrid product recommendation API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


@app.get("/")
def root():
    return {
        "message": "Intelligent Product Recommendation API is running"
    }


# ---------------------------------------------------------
# Existing User Recommendation
# ---------------------------------------------------------

@app.get("/recommend/{customer_id}")
def recommend_customer(
    customer_id: int,
    top_n: int = 10
):

    recommendations = recommender.hybrid_recommendations(
        customer_id=customer_id,
        top_n=top_n,
        alpha=0.6,
        beta=0.4
    )

    if recommendations.empty:
        return {
            "customer_id": customer_id,
            "message": "Customer not found or no interaction history.",
            "recommendations": []
        }

    return {
        "customer_id": customer_id,
        "recommendations": recommendations.to_dict(
            orient="records"
        )
    }


# ---------------------------------------------------------
# Cold Start Recommendation
# ---------------------------------------------------------

@app.get("/cold-start")
def cold_start(
    preferred_category: Optional[str] = None,
    top_n: int = 10
):

    recommendations = recommender.cold_start_recommendations(
        preferred_category=preferred_category,
        top_n=top_n
    )

    return {
        "preferred_category": preferred_category,
        "recommendations": recommendations.to_dict(
            orient="records"
        )
    }
