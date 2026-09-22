
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack


class RecommendationSystem:

    def __init__(self, customers, products, interactions):
        self.customers = customers
        self.products = products
        self.interactions = interactions

        # -----------------------------
        # User-Item Matrix
        # -----------------------------
        self.user_item_matrix = (
            interactions
            .pivot_table(
                index="customer_id",
                columns="product_id",
                values="interaction_strength",
                aggfunc="sum",
                fill_value=0
            )
        )

        # -----------------------------
        # Collaborative Filtering
        # -----------------------------
        self.item_similarity = cosine_similarity(
            self.user_item_matrix.T
        )

        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        # -----------------------------
        # Content-Based Filtering
        # -----------------------------
        encoder = OneHotEncoder(handle_unknown="ignore")

        categorical_features = encoder.fit_transform(
            products[["category", "brand"]]
        )

        scaler = MinMaxScaler()

        price_feature = scaler.fit_transform(
            products[["price"]]
        )

        product_feature_matrix = hstack([
            categorical_features,
            price_feature
        ])

        content_similarity = cosine_similarity(
            product_feature_matrix
        )

        self.content_similarity_df = pd.DataFrame(
            content_similarity,
            index=products["product_id"],
            columns=products["product_id"]
        )

        # -----------------------------
        # Popularity Baseline
        # -----------------------------
        self.product_popularity = (
            interactions
            .groupby("product_id")
            .agg(
                interaction_count=("interaction_id", "count"),
                total_strength=("interaction_strength", "sum")
            )
            .reset_index()
            .sort_values(
                "total_strength",
                ascending=False
            )
        )

    # ==========================================================
    # Collaborative Filtering
    # ==========================================================

    def collaborative_recommendations(self, customer_id, top_n=10):

        if customer_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        history = self.user_item_matrix.loc[customer_id]

        interacted_products = history[
            history > 0
        ].index.tolist()

        if not interacted_products:
            return pd.DataFrame()

        recommendation_scores = {}

        for product_id in interacted_products:

            similar_products = self.item_similarity_df[
                product_id
            ]

            for similar_product_id, similarity in similar_products.items():

                if similar_product_id in interacted_products:
                    continue

                score = (
                    similarity *
                    history[product_id]
                )

                recommendation_scores[similar_product_id] = (
                    recommendation_scores.get(
                        similar_product_id, 0
                    ) + score
                )

        recommendations = (
            pd.Series(recommendation_scores)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        recommendations.columns = [
            "product_id",
            "recommendation_score"
        ]

        recommendations = recommendations.merge(
            self.products,
            on="product_id",
            how="left"
        )

        return recommendations[
            [
                "product_id",
                "product_name",
                "category",
                "brand",
                "price",
                "recommendation_score"
            ]
        ]

    # ==========================================================
    # Content-Based Filtering
    # ==========================================================

    def content_recommendations(self, customer_id, top_n=10):

        if customer_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        history = self.user_item_matrix.loc[customer_id]

        interacted_products = history[
            history > 0
        ].index.tolist()

        if not interacted_products:
            return pd.DataFrame()

        recommendation_scores = {}

        for product_id in interacted_products:

            similar_products = self.content_similarity_df[
                product_id
            ]

            for similar_product_id, similarity in similar_products.items():

                if similar_product_id in interacted_products:
                    continue

                score = (
                    similarity *
                    history[product_id]
                )

                recommendation_scores[similar_product_id] = (
                    recommendation_scores.get(
                        similar_product_id, 0
                    ) + score
                )

        recommendations = (
            pd.Series(recommendation_scores)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        recommendations.columns = [
            "product_id",
            "content_score"
        ]

        recommendations = recommendations.merge(
            self.products,
            on="product_id",
            how="left"
        )

        return recommendations[
            [
                "product_id",
                "product_name",
                "category",
                "brand",
                "price",
                "content_score"
            ]
        ]

    # ==========================================================
    # Hybrid Recommendation
    # ==========================================================

    def hybrid_recommendations(
        self,
        customer_id,
        top_n=10,
        alpha=0.6,
        beta=0.4
    ):

        cf = self.collaborative_recommendations(
            customer_id,
            top_n=50
        )

        content = self.content_recommendations(
            customer_id,
            top_n=50
        )

        if cf.empty and content.empty:
            return pd.DataFrame()

        cf_scores = dict(
            zip(
                cf["product_id"],
                cf["recommendation_score"]
            )
        )

        content_scores = dict(
            zip(
                content["product_id"],
                content["content_score"]
            )
        )

        all_products = set(cf_scores) | set(content_scores)

        max_cf = max(cf_scores.values()) if cf_scores else 1
        max_content = (
            max(content_scores.values())
            if content_scores
            else 1
        )

        final_scores = []

        for product_id in all_products:

            cf_score = (
                cf_scores.get(product_id, 0) /
                max_cf
            )

            content_score = (
                content_scores.get(product_id, 0) /
                max_content
            )

            hybrid_score = (
                alpha * cf_score +
                beta * content_score
            )

            final_scores.append({
                "product_id": product_id,
                "cf_score": cf_score,
                "content_score": content_score,
                "hybrid_score": hybrid_score
            })

        recommendations = pd.DataFrame(
            final_scores
        )

        recommendations = (
            recommendations
            .sort_values(
                "hybrid_score",
                ascending=False
            )
            .head(top_n)
        )

        recommendations = recommendations.merge(
            self.products,
            on="product_id",
            how="left"
        )

        recommendations.insert(
            0,
            "rank",
            range(1, len(recommendations) + 1)
        )

        return recommendations[
            [
                "rank",
                "product_id",
                "product_name",
                "category",
                "brand",
                "price",
                "cf_score",
                "content_score",
                "hybrid_score"
            ]
        ]

    # ==========================================================
    # Cold Start
    # ==========================================================

    def cold_start_recommendations(
        self,
        preferred_category=None,
        top_n=10
    ):

        recommendations = self.product_popularity.merge(
            self.products,
            on="product_id",
            how="left"
        )

        if preferred_category:
            recommendations = recommendations[
                recommendations["category"].str.lower()
                == preferred_category.lower()
            ]

        recommendations = (
            recommendations
            .sort_values(
                "total_strength",
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        recommendations["rank"] = range(
            1,
            len(recommendations) + 1
        )

        return recommendations[
            [
                "rank",
                "product_id",
                "product_name",
                "category",
                "brand",
                "price",
                "total_strength"
            ]
        ]
