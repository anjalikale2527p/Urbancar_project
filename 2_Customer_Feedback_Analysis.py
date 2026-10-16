
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="UrbanKart Customer Feedback Analysis",
    page_icon="💬",
    layout="wide"
)

# File paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "hugging_face.csv"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# Load sentiment model once
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME
    )


# Analyze reviews
def analyze_reviews(texts, model):
    predictions = []

    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        batch_predictions = model(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )

        predictions.extend(batch_predictions)

    return predictions


# Header
st.title("💬 UrbanKart Customer Feedback Analysis")
st.write(
    "Analyze customer reviews using Hugging Face "
    "sentiment analysis."
)

st.divider()


# Tabs
tab1, tab2 = st.tabs(
    ["📊 Dataset Analysis", "✍️ Analyze a Review"]
)


# ---------------------------------------------
# TAB 1: DATASET ANALYSIS
# ---------------------------------------------
with tab1:

    st.subheader("Customer Reviews Dataset")

    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found: {DATA_PATH.name}. "
            "Place hugging_face.csv in the same folder "
            "as this Python file."
        )
    else:
        try:
            df = load_data()

            if "review_text" not in df.columns:
                st.error(
                    "The dataset must contain a "
                    "'review_text' column."
                )
            else:
                st.metric("Total Customer Reviews", len(df))

                st.write("Dataset preview")
                st.dataframe(df.head(10), use_container_width=True)

                number_of_reviews = st.number_input(
                    "Number of reviews to analyze",
                    min_value=1,
                    max_value=max(1, len(df)),
                    value=min(100, max(1, len(df))),
                    step=1
                )

                if st.button(
                    "Analyze Dataset",
                    type="primary"
                ):
                    valid_df = df.head(
                        int(number_of_reviews)
                    ).copy()

                    valid_df["review_text"] = (
                        valid_df["review_text"]
                        .fillna("")
                        .astype(str)
                    )

                    valid_df = valid_df[
                        valid_df["review_text"].str.strip() != ""
                    ].copy()

                    if valid_df.empty:
                        st.warning(
                            "No non-empty reviews were found."
                        )
                    else:
                        model = load_model()

                        with st.spinner(
                            "Analyzing customer reviews..."
                        ):
                            predictions = analyze_reviews(
                                valid_df["review_text"].tolist(),
                                model
                            )

                        valid_df["predicted_sentiment"] = [
                            item["label"].lower()
                            for item in predictions
                        ]

                        valid_df["confidence_score"] = [
                            round(float(item["score"]), 4)
                            for item in predictions
                        ]

                        st.session_state["results"] = valid_df

                if "results" in st.session_state:
                    results = st.session_state["results"]

                    st.divider()
                    st.subheader("Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Reviews Analyzed",
                            len(results)
                        )

                    with col2:
                        positive = (
                            results["predicted_sentiment"]
                            == "positive"
                        ).sum()
                        st.metric("Positive Reviews", int(positive))

                    with col3:
                        negative = (
                            results["predicted_sentiment"]
                            == "negative"
                        ).sum()
                        st.metric("Negative Reviews", int(negative))

                    st.subheader("Sentiment Distribution")

                    sentiment_counts = (
                        results["predicted_sentiment"]
                        .value_counts()
                    )

                    st.bar_chart(sentiment_counts)

                    st.subheader("Predicted Reviews")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    if "sentiment_label" in results.columns:
                        st.subheader("Actual vs Predicted")

                        st.dataframe(
                            results[
                                [
                                    "sentiment_label",
                                    "predicted_sentiment"
                                ]
                            ].value_counts().reset_index(
                                name="count"
                            ),
                            use_container_width=True
                        )

                    st.download_button(
                        "Download Results as CSV",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="urbankart_sentiment_results.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Unable to load or analyze data: {error}")


# ---------------------------------------------
# TAB 2: SINGLE REVIEW ANALYSIS
# ---------------------------------------------
with tab2:

    st.subheader("Analyze an Individual Review")

    review = st.text_area(
        "Enter customer review",
        placeholder=(
            "Example: The product quality was excellent "
            "and delivery was fast!"
        ),
        height=150
    )

    if st.button(
        "Predict Sentiment",
        type="primary"
    ):

        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                model = load_model()

                with st.spinner("Predicting sentiment..."):
                    prediction = model(
                        review,
                        truncation=True,
                        max_length=512
                    )[0]

                st.success(
                    f"Predicted sentiment: "
                    f"{prediction['label'].upper()}"
                )

                st.metric(
                    "Confidence Score",
                    f"{prediction['score']:.2%}"
                )

                single_result = pd.DataFrame([{
                    "review_text": review,
                    "predicted_sentiment": prediction["label"].lower(),
                    "confidence_score": round(
                        float(prediction["score"]), 4
                    )
                }])

                st.download_button(
                    "Download Single Review Result",
                    data=single_result.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="single_review_result.csv",
                    mime="text/csv"
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
