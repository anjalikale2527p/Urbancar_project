
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from transformers import pipeline

# -------------------- APP CONFIGURATION --------------------
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent

REVENUE_DATA = PROJECT_DIR / "UrbanKart_Customer_Analytics_Master.csv"
REVENUE_MODEL = PROJECT_DIR / "best_revenue_model.joblib"
FEEDBACK_DATA = PROJECT_DIR / "hugging_face.csv"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

FEATURES = [
    "Age", "Income", "Customer_Tenure_Months",
    "Acquisition_Channel", "Website_Visits_Monthly",
    "App_Sessions_Monthly", "Purchase_Frequency_Monthly",
    "Average_Order_Value", "Discount_Usage_Rate",
    "Orders_Last_12M", "Days_Since_Last_Purchase",
    "Return_Rate", "Support_Tickets_12M",
    "Customer_Satisfaction_Score", "Payment_Delay_Count",
    "Loyalty_Points", "Marketing_Response_Rate"
]

# -------------------- LOADERS --------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_revenue_model(path):
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def analyze_reviews(texts, model):
    predictions = []
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]
        predictions.extend(
            model(batch, batch_size=16, truncation=True, max_length=512)
        )
    return predictions


# -------------------- NAVIGATION --------------------
st.sidebar.title("🛍️ UrbanKart")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Executive Dashboard",
        "Revenue Prediction",
        "Feedback Analysis",
        "Analyze a Review"
    ]
)

# -------------------- HOME --------------------
if page == "Home":
    st.title("🛍️ UrbanKart Customer Intelligence Hub")
    st.write(
        "One platform for customer analytics, annual revenue "
        "prediction, and AI-powered customer feedback analysis."
    )

    col1, col2 = st.columns(2)
    col1.subheader("📊 Customer Analytics")
    col1.write("Explore revenue, customer counts, and customer segments.")
    col2.subheader("🤖 AI Analysis")
    col2.write("Predict revenue and analyze customer review sentiment.")

# -------------------- EXECUTIVE DASHBOARD --------------------
elif page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")

    if not REVENUE_DATA.exists():
        st.error(f"File not found: {REVENUE_DATA.name}")
        st.stop()

    df = load_csv(str(REVENUE_DATA))

    required = [
        "Annual_Revenue",
        "Preferred_Category",
        "Customer_Value_Segment"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing dataset columns: {missing}")
        st.stop()

    revenue = pd.to_numeric(df["Annual_Revenue"], errors="coerce")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Total Annual Revenue", f"{revenue.sum():,.2f}")
    c3.metric(
        "Average Annual Revenue",
        f"{revenue.mean():,.2f}" if revenue.notna().any() else "N/A"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Preferred Category")
        chart_df = df.assign(_revenue=revenue)
        st.bar_chart(
            chart_df.groupby("Preferred_Category")["_revenue"]
            .sum().sort_values(ascending=False)
        )

    with right:
        st.subheader("Customer Value Segments")
        st.bar_chart(df["Customer_Value_Segment"].value_counts())

    st.subheader("Customer Data")
    st.dataframe(df.head(20), use_container_width=True)

# -------------------- REVENUE PREDICTION --------------------
elif page == "Revenue Prediction":
    st.title("💰 Annual Revenue Prediction")

    if not REVENUE_DATA.exists() or not REVENUE_MODEL.exists():
        st.error(
            "Please ensure both UrbanKart_Customer_Analytics_Master.csv "
            "and best_revenue_model.joblib are in the project folder."
        )
        st.stop()

    df = load_csv(str(REVENUE_DATA))
    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        st.error(f"Missing feature columns: {missing}")
        st.stop()

    channels = df["Acquisition_Channel"].dropna().unique().tolist()
    if not channels:
        st.error("No acquisition channel options are available.")
        st.stop()

    with st.form("revenue_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Income", min_value=0.0, value=300000.0)
            tenure = st.number_input("Customer Tenure (months)", min_value=0, value=12)
            channel = st.selectbox("Acquisition Channel", channels)
            visits = st.number_input("Monthly Website Visits", min_value=0.0, value=10.0)
            sessions = st.number_input("Monthly App Sessions", min_value=0.0, value=10.0)
            frequency = st.number_input("Monthly Purchase Frequency", min_value=0.0, value=3.0)
            order_value = st.number_input("Average Order Value", min_value=0.0, value=2000.0)
            discount = st.number_input("Discount Usage Rate (0-1)", 0.0, 1.0, 0.2)

        with right:
            orders = st.number_input("Orders in Last 12 Months", min_value=0, value=12)
            days = st.number_input("Days Since Last Purchase", min_value=0, value=30)
            returns = st.number_input("Return Rate (0-1)", 0.0, 1.0, 0.05)
            tickets = st.number_input("Support Tickets in Last 12 Months", min_value=0, value=1)
            satisfaction = st.number_input("Customer Satisfaction (0-5)", 0.0, 5.0, 4.0)
            delays = st.number_input("Payment Delay Count", min_value=0, value=0)
            points = st.number_input("Loyalty Points", min_value=0, value=500)
            response = st.number_input("Marketing Response Rate (0-1)", 0.0, 1.0, 0.2)

        submitted = st.form_submit_button("Predict Annual Revenue")

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "Customer_Tenure_Months": tenure,
            "Acquisition_Channel": channel,
            "Website_Visits_Monthly": visits,
            "App_Sessions_Monthly": sessions,
            "Purchase_Frequency_Monthly": frequency,
            "Average_Order_Value": order_value,
            "Discount_Usage_Rate": discount,
            "Orders_Last_12M": orders,
            "Days_Since_Last_Purchase": days,
            "Return_Rate": returns,
            "Support_Tickets_12M": tickets,
            "Customer_Satisfaction_Score": satisfaction,
            "Payment_Delay_Count": delays,
            "Loyalty_Points": points,
            "Marketing_Response_Rate": response
        }])[FEATURES]

        try:
            model = load_revenue_model(str(REVENUE_MODEL))
            result = model.predict(input_df)[0]
            st.success(f"Estimated Annual Revenue: {result:,.2f}")
        except Exception as e:
            st.error(f"Revenue prediction failed: {e}")
            st.info(
                "If this mentions RemainderColsList or sklearn, "
                "the saved model needs a compatible scikit-learn version."
            )

# -------------------- FEEDBACK DATASET ANALYSIS --------------------
elif page == "Feedback Analysis":
    st.title("💬 Customer Feedback Analysis")

    if not FEEDBACK_DATA.exists():
        st.error(f"File not found: {FEEDBACK_DATA.name}")
        st.stop()

    df = load_csv(str(FEEDBACK_DATA))

    if "review_text" not in df.columns:
        st.error("The dataset must contain a review_text column.")
        st.stop()

    st.metric("Total Customer Reviews", len(df))
    st.dataframe(df.head(10), use_container_width=True)

    if df.empty:
        st.warning("The feedback dataset is empty.")
        st.stop()

    count = st.number_input(
        "Number of reviews to analyze",
        min_value=1,
        max_value=len(df),
        value=min(100, len(df))
    )

    if st.button("Analyze Dataset", type="primary"):
        selected = df.head(int(count)).copy()
        selected["review_text"] = selected["review_text"].fillna("").astype(str)
        selected = selected[selected["review_text"].str.strip() != ""]

        if selected.empty:
            st.warning("No non-empty reviews were found.")
        else:
            model = load_sentiment_model()
            with st.spinner("Analyzing reviews..."):
                predictions = analyze_reviews(
                    selected["review_text"].tolist(), model
                )

            selected["predicted_sentiment"] = [
                p["label"].lower() for p in predictions
            ]
            selected["confidence_score"] = [
                round(float(p["score"]), 4) for p in predictions
            ]
            st.session_state["feedback_results"] = selected

    if "feedback_results" in st.session_state:
        results = st.session_state["feedback_results"]

        st.subheader("Analysis Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews Analyzed", len(results))
        c2.metric(
            "Positive Reviews",
            int((results["predicted_sentiment"] == "positive").sum())
        )
        c3.metric(
            "Negative Reviews",
            int((results["predicted_sentiment"] == "negative").sum())
        )

        st.subheader("Sentiment Distribution")
        st.bar_chart(results["predicted_sentiment"].value_counts())
        st.dataframe(results, use_container_width=True)

        if "sentiment_label" in results.columns:
            st.subheader("Actual vs Predicted")
            comparison = results.groupby(
                ["sentiment_label", "predicted_sentiment"],
                dropna=False
            ).size().reset_index(name="count")
            st.dataframe(comparison, use_container_width=True)

        st.download_button(
            "Download Results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="urbankart_sentiment_results.csv",
            mime="text/csv"
        )

# -------------------- SINGLE REVIEW PREDICTION --------------------
elif page == "Analyze a Review":
    st.title("✍️ Analyze an Individual Review")

    review = st.text_area(
        "Enter customer review",
        placeholder="The product quality was excellent and delivery was fast!",
        height=150
    )

    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                model = load_sentiment_model()
                with st.spinner("Predicting sentiment..."):
                    prediction = model(
                        review, truncation=True, max_length=512
                    )[0]

                sentiment = prediction["label"].lower()
                score = float(prediction["score"])

                st.success(f"Predicted Sentiment: {sentiment.upper()}")
                st.metric("Confidence Score", f"{score:.2%}")

                result = pd.DataFrame([{
                    "review_text": review,
                    "predicted_sentiment": sentiment,
                    "confidence_score": round(score, 4)
                }])

                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download Review Result",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="single_review_result.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Sentiment prediction failed: {e}")
