
import pandas as pd


def load_data(data_path="../data"):
    customers = pd.read_csv(f"{data_path}/customers.csv")
    products = pd.read_csv(f"{data_path}/products.csv")
    interactions = pd.read_csv(f"{data_path}/interactions.csv")

    return customers, products, interactions


def preprocess_data(interactions):
    interactions = interactions.copy()

    interactions["interaction_date"] = pd.to_datetime(
        interactions["interaction_date"],
        errors="coerce"
    )

    interaction_weights = {
        "view": 1,
        "click": 2,
        "wishlist": 3,
        "cart": 4,
        "purchase": 5
    }

    interactions["interaction_strength"] = (
        interactions["interaction_type"].map(interaction_weights)
    )

    return interactions
