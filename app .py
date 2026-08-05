import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Bank Customer Churn Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "European_Bank.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def engineer(data):
    d = data.copy()
    d["Balance_to_Salary"] = d["Balance"] / (d["EstimatedSalary"].abs() + 1)
    d["Product_Engagement"] = d["NumOfProducts"] * d["IsActiveMember"]
    d["Age_Tenure_Interaction"] = d["Age"] * d["Tenure"]
    d["Product_Density"] = d["NumOfProducts"] / (d["Age"] + 1)
    return d

st.title("🏦 Bank Customer Churn Risk Scoring")
st.caption("Predictive modeling and risk scoring for retail-bank customer churn")

try:
    bundle = load_model()
    model = bundle["model"]
    best_model = bundle.get("best_model", "Gradient Boosting")
except Exception as e:
    st.error(f"Model file could not be loaded: {e}")
    st.stop()

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Churn Prediction", "📋 Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['Exited'].mean()*100:.1f}%")
    c3.metric("Active Members", f"{df['IsActiveMember'].mean()*100:.1f}%")
    c4.metric("Average Balance", f"€{df['Balance'].mean():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age", color="Exited", nbins=25,
                           barmode="overlay", title="Age Distribution by Churn")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        churn_geo = df.groupby("Geography", as_index=False)["Exited"].mean()
        churn_geo["Exited"] *= 100
        fig = px.bar(churn_geo, x="Geography", y="Exited",
                     title="Churn Rate by Geography",
                     labels={"Exited": "Churn Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        product = df.groupby("NumOfProducts", as_index=False)["Exited"].mean()
        product["Exited"] *= 100
        fig = px.bar(product, x="NumOfProducts", y="Exited",
                     title="Churn Rate by Number of Products",
                     labels={"Exited": "Churn Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        active = df.groupby("IsActiveMember", as_index=False)["Exited"].mean()
        active["Member Status"] = active["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        active["Exited"] *= 100
        fig = px.bar(active, x="Member Status", y="Exited",
                     title="Churn Rate by Activity",
                     labels={"Exited": "Churn Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)

    with c1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", sorted(df["Geography"].dropna().unique()))
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        age = st.number_input("Age", 18, 100, 40)
    with c2:
        tenure = st.number_input("Tenure (years)", 0, 20, 5)
        balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
        products = st.number_input("Number of Products", 1, 10, 1)
        has_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
    with c3:
        active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")
        salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

    if st.button("Calculate Churn Risk", type="primary", use_container_width=True):
        customer = pd.DataFrame([{
            "Year": 2025,
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }])
        customer = engineer(customer)
        probability = float(model.predict_proba(customer)[0, 1])
        prediction = int(probability >= 0.50)

        if probability < 0.30:
            risk = "Low Risk"
        elif probability < 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        a, b, c = st.columns(3)
        a.metric("Churn Probability", f"{probability*100:.1f}%")
        b.metric("Risk Category", risk)
        c.metric("Model", best_model)

        if prediction:
            st.warning("The model flags this customer as likely to churn.")
            st.write("Suggested action: proactive retention contact, service review, and targeted engagement.")
        else:
            st.success("The model does not flag this customer as likely to churn.")
            st.write("Suggested action: maintain engagement and monitor future risk signals.")

        st.progress(probability)

with tab3:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "Download dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="European_Bank.csv",
        mime="text/csv"
    )
