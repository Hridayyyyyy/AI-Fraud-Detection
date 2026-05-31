import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide"
)

st.title("💳 AI Fraud Detection Dashboard")

st.markdown("""
Upload transaction and identity datasets to detect fraudulent transactions using Machine Learning.
""")

@st.cache_resource
def load_model():

    model = joblib.load("fraud_detection_model.pkl")

    return model

model = load_model()

def preprocess_data(df):

    df = df.copy()

    if "isFraud" in df.columns:
        df = df.drop("isFraud", axis=1)

    missing_percent = (
        df.isnull().sum() / len(df)
    ) * 100

    cols_to_drop = missing_percent[
        missing_percent > 50
    ].index

    df = df.drop(columns=cols_to_drop)

    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    df[num_cols] = df[num_cols].fillna(
        df[num_cols].median()
    )

    cat_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in cat_cols:

        df[col] = df[col].fillna("Unknown")

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

    df = df.fillna(0)

    return df

st.header("📂 Upload Files")

transaction_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"],
    key="transaction"
)

identity_file = st.file_uploader(
    "Upload Identity CSV",
    type=["csv"],
    key="identity"
)

if transaction_file is not None and identity_file is not None:

    try:

        transaction_df = pd.read_csv(
            transaction_file
        )

        identity_df = pd.read_csv(
            identity_file
        )

        input_df = pd.merge(
            transaction_df,
            identity_df,
            on="TransactionID",
            how="left"
        )

        st.success(
            "✅ Files Uploaded & Merged Successfully"
        )

        st.subheader("📄 Uploaded Dataset")

        st.dataframe(input_df.head())

        total_transactions = len(input_df)

        st.metric(
            "Total Transactions",
            total_transactions
        )

        processed_df = preprocess_data(
            input_df
        )

        processed_df = processed_df.select_dtypes(
            include=[np.number]
        )

        processed_df = processed_df.fillna(0)

        predictions = model.predict(
            processed_df
        )

        probabilities = model.predict_proba(
            processed_df
        )[:, 1]

        results_df = input_df.copy()

        results_df["Fraud Prediction"] = predictions

        results_df["Fraud Probability"] = probabilities

        results_df["Risk Level"] = pd.cut(
            results_df["Fraud Probability"],
            bins=[0, 0.4, 0.75, 1],
            labels=[
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ]
        )

        fraud_count = int(
            (
                results_df[
                    "Fraud Prediction"
                ] == 1
            ).sum()
        )

        normal_count = (
            total_transactions - fraud_count
        )

        fraud_percent = (
            fraud_count / total_transactions
        ) * 100

        st.header("📊 Prediction Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transactions Processed",
            total_transactions
        )

        c2.metric(
            "Fraud Transactions",
            fraud_count
        )

        c3.metric(
            "Fraud Percentage",
            f"{fraud_percent:.2f}%"
        )

        pie_df = pd.DataFrame({

            "Type": [
                "Legitimate",
                "Fraud"
            ],

            "Count": [
                normal_count,
                fraud_count
            ]
        })

        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            title="Fraud Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "📈 Risk Level Distribution"
        )

        risk_counts = results_df[
            "Risk Level"
        ].value_counts()

        risk_fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            labels={
                "x": "Risk Level",
                "y": "Count"
            },
            title="Risk Level Counts"
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True
        )

        st.subheader(
            "🚨 Fraudulent Transactions"
        )

        fraud_cases = results_df[
            results_df[
                "Fraud Prediction"
            ] == 1
        ]

        st.dataframe(fraud_cases.head(500))

        st.subheader(
            "🔍 Search Transaction"
        )

        selected_id = st.selectbox(
            "Select Transaction ID",
            results_df["TransactionID"]
        )

        selected_txn = results_df[
            results_df[
                "TransactionID"
            ] == selected_id
        ]

        st.dataframe(selected_txn)

        csv = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

else:

    st.info(
        "Upload both Transaction CSV and Identity CSV to begin fraud detection."
    )