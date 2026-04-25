import streamlit as st
import os
import pickle
import pandas as pd

# Resolve model path relative to this file's location
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/model.pkl"))

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

def make_feature_df(payload: dict) -> pd.DataFrame:
    d = payload.copy()

    d['total_charges'] = d['tenure'] * d['monthly_charges']
    d['avg_charge_per_month'] = d['total_charges'] / (d['tenure'] + 1e-6)

    t = d['tenure']
    if t <= 12:
        d['tenure_bucket'] = '0-12'
    elif t <= 24:
        d['tenure_bucket'] = '12-24'
    elif t <= 48:
        d['tenure_bucket'] = '24-48'
    else:
        d['tenure_bucket'] = '48+'

    svc_map = {
        "Multiple Lines":     "multiple_lines",
        "Online Security":    "online_security",
        "Online Backup":      "online_backup",
        "Device Protection":  "device_protection",
        "Tech Support":       "tech_support",
        "Streaming TV":       "streaming_tv",
        "Streaming Movies":   "streaming_movies",
    }
    for col in svc_map.values():
        d[col] = "No"
    
    for svc in d.pop('services_list', []):
        key = svc_map.get(svc)
        if key:
            d[key] = "Yes"

    d['services_count'] = sum(1 for col in svc_map.values() if d[col] == "Yes")

    cols = [
        'tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month',
        'services_count', 'contract', 'gender', 'senior_citizen', 'partner',
        'dependents', 'internet_service', 'paperless_billing', 'payment_method',
        'multiple_lines', 'online_security', 'online_backup',
        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
        'tenure_bucket'
    ]
    return pd.DataFrame([d], columns=cols)

def prediction_form():
    st.header("🔮 Customer Churn Prediction")
    st.markdown(
        "Fill in the customer’s profile below. "
        "Hover over any input for guidance on how to choose."
    )
    with st.form("prediction_form"):
        # --- Numeric Inputs ---
        tenure = st.number_input(
            "⏳ Tenure (months)",
            min_value=0, max_value=100, value=12,
            help="How many months this customer has been with the company."
        )
        monthly_charges = st.number_input(
            "💰 Monthly Charges (₹)",
            min_value=0.0, value=700.0, format="%.2f",
            help="What the customer pays every month, on average."
        )

        # --- Core Categorical Inputs ---
        contract = st.selectbox(
            "📄 Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help=(
                "Month-to-month contracts often see higher churn.\n"
                "One- and two-year contracts generally retain customers longer."
            )
        )
        gender = st.selectbox(
            "👤 Gender",
            ["Female", "Male"],
            help="Select the customer’s reported gender."
        )
        senior_citizen = st.selectbox(
            "🧓 Senior Citizen",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether the customer is a senior citizen (1 = Yes)."
        )
        partner = st.selectbox(
            "💑 Has Partner?",
            ["Yes", "No"],
            help="Whether the customer has a partner."
        )
        dependents = st.selectbox(
            "👨‍👩‍👧 Has Dependents?",
            ["Yes", "No"],
            help="Whether the customer has dependents."
        )

        # --- Network & Billing Inputs ---
        internet_service = st.selectbox(
            "🌐 Internet Service",
            ["DSL", "Fiber optic", "No"],
            help=(
                "Type of internet service the customer uses. "
                "Customers with Fiber optic historically churn more."
            )
        )
        paperless_billing = st.selectbox(
            "🧾 Paperless Billing?",
            ["Yes", "No"],
            help="Whether the customer uses paperless billing."
        )
        payment_method = st.selectbox(
            "💳 Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            help=(
                "How the customer pays their bill. "
                "Electronic check users are often at higher churn risk."
            )
        )

        # --- Services Multiselect ---
        services_list = st.multiselect(
            "🛎️ Optional Services",
            ["Multiple Lines", "Online Security", "Online Backup",
             "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"],
            help=(
                "Choose all optional services the customer subscribes to. "
                "More services generally indicate lower churn risk."
            )
        )

        submitted = st.form_submit_button("Predict Churn")
        if submitted:
            payload = {
                "tenure": tenure,
                "monthly_charges": monthly_charges / 10.0,
                "contract": contract,
                "gender": gender,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "internet_service": internet_service,
                "paperless_billing": paperless_billing,
                "payment_method": payment_method,
                "services_list": services_list
            }
            try:
                model = load_model()
                if model is None:
                    st.error("Prediction failed: Could not load the machine learning model. Make sure model.pkl exists in the backend directory.")
                else:
                    df = make_feature_df(payload)
                    proba = model.predict_proba(df)[0, 1]
                    threshold = 0.5
                    prediction = "Yes" if proba >= threshold else "No"
                    
                    st.success(
                        f"**Prediction:** {prediction}  \n"
                        f"**Churn Probability:** {proba:.2%}"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
