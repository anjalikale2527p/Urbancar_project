
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "UrbanKart_Customer_Analytics_Master.csv"
MODEL_PATH = PROJECT_DIR / "urbankart_best_regression_model.joblib"

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


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("📈 Revenue Prediction")
st.caption("UrbanKart customer analytics and annual revenue estimation")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Rename your existing best_revenue_model.joblib file "
        "to urbankart_best_regression_model.joblib."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

required_columns = FEATURES + [
    "Annual_Revenue",
    "Preferred_Category",
    "Customer_Value_Segment"
]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Required dataset columns are missing: {missing}")
    st.stop()

st.header("Executive Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Total Annual Revenue", f"{df['Annual_Revenue'].sum():,.2f}")
c3.metric("Average Annual Revenue", f"{df['Annual_Revenue'].mean():,.2f}")

left, right = st.columns(2)

with left:
    st.subheader("Revenue by Preferred Category")
    st.bar_chart(
        df.groupby("Preferred_Category")["Annual_Revenue"]
        .sum().sort_values(ascending=False)
    )

with right:
    st.subheader("Customer Value Segments")
    st.bar_chart(df["Customer_Value_Segment"].value_counts())

st.subheader("Customer Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.header("Annual Revenue Prediction")
st.write("Enter a customer's profile to estimate annual revenue.")

with st.form("revenue_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Income", min_value=0.0, value=300000.0)
        tenure = st.number_input("Customer Tenure (months)", min_value=0, value=12)

        channels = sorted(
            df["Acquisition_Channel"].dropna().unique().tolist()
        )
        acquisition = st.selectbox("Acquisition Channel", channels)

        website_visits = st.number_input("Monthly Website Visits", min_value=0.0, value=10.0)
        app_sessions = st.number_input("Monthly App Sessions", min_value=0.0, value=10.0)
        purchase_frequency = st.number_input("Monthly Purchase Frequency", min_value=0.0, value=3.0)
        average_order = st.number_input("Average Order Value", min_value=0.0, value=2000.0)
        discount_rate = st.number_input("Discount Usage Rate", 0.0, 1.0, 0.2)

    with col2:
        orders = st.number_input("Orders in Last 12 Months", min_value=0, value=12)
        days_since_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)
        return_rate = st.number_input("Return Rate", 0.0, 1.0, 0.05)
        support_tickets = st.number_input("Support Tickets in Last 12 Months", min_value=0, value=1)
        satisfaction = st.number_input("Customer Satisfaction Score", 0.0, 5.0, 4.0)
        payment_delays = st.number_input("Payment Delay Count", min_value=0, value=0)
        loyalty_points = st.number_input("Loyalty Points", min_value=0, value=500)
        marketing_response = st.number_input("Marketing Response Rate", 0.0, 1.0, 0.2)

    submitted = st.form_submit_button("Predict Annual Revenue", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Customer_Tenure_Months": tenure,
        "Acquisition_Channel": acquisition,
        "Website_Visits_Monthly": website_visits,
        "App_Sessions_Monthly": app_sessions,
        "Purchase_Frequency_Monthly": purchase_frequency,
        "Average_Order_Value": average_order,
        "Discount_Usage_Rate": discount_rate,
        "Orders_Last_12M": orders,
        "Days_Since_Last_Purchase": days_since_purchase,
        "Return_Rate": return_rate,
        "Support_Tickets_12M": support_tickets,
        "Customer_Satisfaction_Score": satisfaction,
        "Payment_Delay_Count": payment_delays,
        "Loyalty_Points": loyalty_points,
        "Marketing_Response_Rate": marketing_response
    }])[FEATURES]

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Annual Revenue: {prediction:,.2f}")
        st.caption("This is a model estimate, not a guaranteed revenue outcome.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
