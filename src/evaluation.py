
import numpy as np


def precision_at_k(recommended, actual, k=10):
    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = len(set(recommended) & {actual})

    return hits / k


def recall_at_k(recommended, actual, k=10):
    recommended = recommended[:k]

    if actual in recommended:
        return 1.0

    return 0.0


def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall /
        (precision + recall)
    )


def ndcg_at_k(recommended, actual, k=10):
    recommended = recommended[:k]

    if actual not in recommended:
        return 0.0

    rank = recommended.index(actual) + 1

    return 1 / np.log2(rank + 1)


def evaluate_recommendations(
    test_data,
    recommendation_function,
    k=10
):
    precisions = []
    recalls = []
    ndcgs = []

    for _, row in test_data.iterrows():

        customer_id = row["customer_id"]
        actual_product = row["product_id"]

        recommendations = recommendation_function(
            customer_id,
            k
        )

        precision = precision_at_k(
            recommendations,
            actual_product,
            k
        )

        recall = recall_at_k(
            recommendations,
            actual_product,
            k
        )

        ndcg = ndcg_at_k(
            recommendations,
            actual_product,
            k
        )

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    ndcg = np.mean(ndcgs)

    f1 = f1_at_k(
        precision,
        recall
    )

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "F1@10": f1,
        "NDCG@10": ndcg
    }
