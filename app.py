import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "fraud_detection_model.pkl"
DEMO_PATH = "demo_transactions.csv"


@st.cache_resource
def load_model():

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["model"],
        artifact.get("threshold", 0.5),
        artifact["features"]
    )


@st.cache_data
def load_demo_data():

    return pd.read_csv(DEMO_PATH)


model, threshold, features = load_model()
demo_df = load_demo_data()


# =========================================================
# HEADER
# =========================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    *Machine Learning based transaction risk assessment*

    This system uses an *XGBoost classification model*
    to estimate the probability that a transaction is fraudulent.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Detection Mode")

mode = st.sidebar.radio(
    "Choose mode",
    [
        "🧪 Demo Transaction",
        "🔬 Advanced Model Input"
    ]
)


# =========================================================
# DEMO MODE
# =========================================================

if mode == "🧪 Demo Transaction":

    st.subheader("🧪 Transaction Simulator")

    st.info(
        "Select a real transaction from the benchmark dataset. "
        "The underlying anonymized model features are sent to "
        "the trained ML model automatically."
    )

    # Transaction selector
    selected_id = st.selectbox(
        "Select Transaction",
        demo_df["Demo_ID"].tolist()
    )

    transaction = demo_df[
        demo_df["Demo_ID"] == selected_id
    ].iloc[0]

    # -----------------------------------------------------
    # Transaction information
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        amount = transaction["Amount"]

        st.metric(
            "💰 Transaction Amount",
            f"${amount:,.2f}"
        )

    with col2:

        total_seconds = transaction["Time"]

        hour = int(
            (total_seconds / 3600) % 24
        )

        minute = int(
            (total_seconds % 3600) / 60
        )

        st.metric(
            "🕐 Transaction Time",
            f"{hour:02d}:{minute:02d}"
        )

    with col3:

        st.metric(
            "🆔 Transaction ID",
            f"TXN-{int(selected_id):04d}"
        )

    st.write("")

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Transaction",
        use_container_width=True
    ):

        # Prepare model input
        input_data = pd.DataFrame(
            [transaction]
        )

        # Remove target and UI-only columns
        input_data = input_data.drop(
            columns=[
                "Demo_ID",
                "Class",
                "Time"
            ],
            errors="ignore"
        )

        # Make sure exact training columns are used
        input_data = input_data[features]

        # Prediction probability
        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= threshold
        )

        st.divider()

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 Potential Fraud Detected"
            )

            risk = "HIGH"

        else:

            st.success(
                "✅ Transaction Appears Genuine"
            )

            if probability >= 0.20:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )

        # -------------------------------------------------
        # Probability bar
        # -------------------------------------------------

        st.write("### Fraud Risk Score")

        st.progress(
            min(float(probability), 1.0)
        )

        # -------------------------------------------------
        # Dataset label
        # -------------------------------------------------

        actual = int(
            transaction["Class"]
        )

        if actual == 1:

            st.caption(
                "Dataset reference: This selected transaction "
                "is labeled as FRAUD in the benchmark dataset."
            )

        else:

            st.caption(
                "Dataset reference: This selected transaction "
                "is labeled as GENUINE in the benchmark dataset."
            )


# =========================================================
# ADVANCED MODE
# =========================================================

else:

    st.subheader("🔬 Advanced Model Input")

    st.warning(
        "V1–V28 are anonymized mathematical features from "
        "the original fraud-detection dataset. Their original "
        "business meanings are not available."
    )

    input_data = {}

    # -----------------------------------------------------
    # V1-V28
    # -----------------------------------------------------

    st.write("### Anonymized Model Features")

    cols = st.columns(4)

    for i in range(1, 29):

        with cols[(i - 1) % 4]:

            input_data[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}"
            )

    st.write("### Transaction Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        amount = st.number_input(
            "💰 Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

    with col2:

        hour = st.slider(
            "🕐 Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with col3:

        minute = st.slider(
            "Minute",
            min_value=0,
            max_value=59,
            value=0
        )

    # Convert hour into cyclic features
    total_hour = hour + minute / 60

    hour_sin = np.sin(
        2 * np.pi * total_hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * total_hour / 24
    )

    input_data["Amount"] = amount
    input_data["Hour_sin"] = hour_sin
    input_data["Hour_cos"] = hour_cos

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Transaction",
        use_container_width=True
    ):

        X = pd.DataFrame(
            [input_data]
        )

        X = X[features]

        probability = model.predict_proba(
            X
        )[0][1]

        prediction = int(
            probability >= threshold
        )

        st.divider()

        if prediction == 1:

            st.error(
                "🚨 Potential Fraud Detected"
            )

            risk = "HIGH"

        else:

            st.success(
                "✅ Transaction Appears Genuine"
            )

            if probability >= 0.20:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )

        st.write("### Fraud Risk Score")

        st.progress(
            min(float(probability), 1.0)
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Powered by XGBoost • Threshold Optimization • "
    "Machine Learning Fraud Detection"
)




