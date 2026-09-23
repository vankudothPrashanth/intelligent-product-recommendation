
import requests
import pandas as pd
import gradio as gr


API_URL = "https://intelligent-product-recommendation-hjj2.onrender.com"


def recommendation_ui(customer_id, category):

    # ---------------------------------------------------------
    # Existing User
    # ---------------------------------------------------------

    if customer_id is not None and str(customer_id).strip() != "":
        try:
            customer_id = int(customer_id)

            response = requests.get(
                f"{API_URL}/recommend/{customer_id}",
                params={"top_n": 10}
            )

            if response.status_code != 200:
                return "API Error", pd.DataFrame()

            result = response.json()

            if result["recommendations"]:
                df = pd.DataFrame(
                    result["recommendations"]
                )

                return (
                    "Existing User - Hybrid Recommendations",
                    df
                )

            return (
                "Customer not found or no interaction history.",
                pd.DataFrame()
            )

        except (ValueError, TypeError):
            return (
                "Invalid Customer ID",
                pd.DataFrame()
            )

        except requests.RequestException:
            return (
                "Could not connect to FastAPI.",
                pd.DataFrame()
            )

    # ---------------------------------------------------------
    # New User / Cold Start
    # ---------------------------------------------------------

    try:
        response = requests.get(
            f"{API_URL}/cold-start",
            params={
                "preferred_category": category,
                "top_n": 10
            }
        )

        if response.status_code != 200:
            return "API Error", pd.DataFrame()

        result = response.json()

        df = pd.DataFrame(
            result["recommendations"]
        )

        return (
            "New User - Cold Start Recommendations",
            df
        )

    except requests.RequestException:
        return (
            "Could not connect to FastAPI.",
            pd.DataFrame()
        )


# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------

with gr.Blocks(
    title="Intelligent Product Recommendation System"
) as demo:

    gr.Markdown(
        "# 🛍️ Intelligent Product Recommendation System"
    )

    gr.Markdown(
        "Enter a Customer ID for personalized hybrid "
        "recommendations or select a category for "
        "new-user cold-start recommendations."
    )

    with gr.Row():

        customer_id = gr.Textbox(
            label="Customer ID",
            placeholder="Enter Customer ID (1-1500)"
        )

        category = gr.Dropdown(
            choices=[
                "Electronics",
                "Sports",
                "Home",
                "Fashion",
                "Books",
                "Beauty"
            ],
            label="Preferred Category",
            value="Electronics"
        )

    submit_button = gr.Button(
        "Get Recommendations"
    )

    result_title = gr.Textbox(
        label="Recommendation Type"
    )

    result_table = gr.Dataframe(
        label="Recommended Products",
        interactive=False
    )

    submit_button.click(
        fn=recommendation_ui,
        inputs=[
            customer_id,
            category
        ],
        outputs=[
            result_title,
            result_table
        ]
    )


if __name__ == "__main__":
    demo.launch()
