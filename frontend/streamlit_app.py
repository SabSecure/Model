import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict_fraud"

st.title("🛡️ Fraud Detection System")

st.sidebar.header("Enter Transaction Details")

# User inputs
transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
transaction_channel = st.sidebar.selectbox("Transaction Channel", ["mobile", "web", "POS"])
payment_mode = st.sidebar.selectbox("Payment Mode", ["UPI", "Credit Card", "Debit Card", "Net Banking"])
gateway_bank = st.sidebar.selectbox("Payment Gateway Bank", ["HDFC", "ICICI", "SBI", "Axis"])
payer_browser = st.sidebar.selectbox("Payer Browser", ["Chrome", "Firefox", "Edge", "Safari"])
payer_email = st.sidebar.text_input("Payer Email", "user@example.com")
transaction_id = st.sidebar.text_input("Transaction ID", "ANON_123")

transaction_data = {
    "transaction_amount": transaction_amount,
    "transaction_channel": transaction_channel,
    "transaction_payment_mode_anonymous": payment_mode,
    "payment_gateway_bank_anonymous": gateway_bank,
    "payer_browser_anonymous": payer_browser,
    "payer_email_anonymous": payer_email,
    "transaction_id_anonymous": transaction_id,
    "payee_ip_anonymous": "127.0.0.1",  # Default placeholder value
    "payer_mobile_anonymous": "1234567890",  # Default placeholder value
    "payee_id_anonymous": "PAYEE_001"  # Default placeholder value
}

if st.sidebar.button("Check Fraud"):
    try:
        response = requests.post(API_URL, json=transaction_data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        result = response.json()
        fraud_prob = result["fraud_probability"]
        is_fraud = "🚨 Fraud Detected!" if result["is_fraud"] else "✅ Legitimate Transaction"
        st.metric(label="Fraud Probability", value=f"{fraud_prob:.2%}")
        st.subheader(is_fraud)
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except ValueError:
        st.error("Invalid response from the API. Please check the API server.")
