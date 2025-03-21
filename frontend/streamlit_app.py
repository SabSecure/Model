import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import sqlite3
import serial

API_URL = "http://127.0.0.1:5000"

# Initialize serial communication with Arduino
arduino_port = "COM8"  # Replace with your Arduino port
baud_rate = 9600
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
except Exception as e:
    st.error(f"Failed to connect to Arduino: {e}")

# Custom UI Styling
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.markdown("""
    <style>
        .stTabs {background-color: #f8f9fa; border-radius: 10px; padding: 10px;}
        .stButton > button {width: 100%; border-radius: 10px;}
        .stDataFrame {border-radius: 10px; overflow: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
selected = option_menu(
    menu_title=None,
    options=["Fraud Dashboard", "Predict Fraud", "Transactions", "Edit Rule Book"],
    icons=["bar-chart", "search", "table", "pencil-square"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Fraud Dashboard":
    st.title("Fraud Detection Overview")
    
    # Fetch fraud statistics
    try:
        response = requests.get(f"{API_URL}/fraud_summary")
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Transactions", value=data["total_transactions"])
        col2.metric(label="Fraudulent Transactions", value=data["fraud_transactions"], delta=f"{data['fraud_percentage']:.2f}%")
        col3.metric(label="Detection Accuracy", value=f"{data['detection_accuracy']}%")
        
        # Fraud Trends Chart
        fraud_df = pd.DataFrame(data["fraud_trends"])
        fig = px.line(fraud_df, x="date", y="fraud_count", title="Fraud Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Transactions Table
        st.subheader("Recent Transactions")
        transactions_df = pd.DataFrame(data["recent_transactions"])
        st.dataframe(transactions_df, use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch fraud data: {e}")
    except ValueError as e:
        st.error(f"Invalid response from the API: {e}")

elif selected == "Predict Fraud":
    st.title("Transaction Fraud Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        transaction_data = {
            "transaction_id_anonymous": st.text_input("Transaction ID"),
            "transaction_amount": st.number_input("Transaction Amount", min_value=0.0),
            "transaction_channel": st.selectbox("Transaction Channel", ["mobile", "web", "POS"]),
            "transaction_payment_mode_anonymous": st.text_input("Payment Mode"),
            "payment_gateway_bank_anonymous": st.text_input("Payment Gateway Bank"),
        }
    with col2:
        transaction_data.update({
            "payer_browser_anonymous": st.text_input("Payer Browser"),
            "payer_email_anonymous": st.text_input("Payer Email"),
            "payee_ip_anonymous": st.text_input("Payee IP"),
            "payer_mobile_anonymous": st.text_input("Payer Mobile"),
            "payee_id_anonymous": st.text_input("Payee ID"),
        })
    
    if st.button("Predict Fraud"):
        response = requests.post(f"{API_URL}/predict_fraud", json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Results")
            st.metric(label="Transaction ID", value=result['transaction_id'])
            st.metric(label="Prediction", value="Fraud" if result['is_fraud'] else "Not Fraud")
            st.write("Fraud Reason:", result['fraud_reason'])
            st.write("Fraud Score:", result['fraud_score'])
            
            # Blink LED on Arduino if fraud is detected
            if result['is_fraud']:
                try:
                    arduino.write(b'1')  # Send signal to Arduino to blink LED
                except Exception as e:
                    st.error(f"Failed to send signal to Arduino: {e}")
        else:
            st.error("Error: " + response.json()["error"])

elif selected == "Transactions":
    st.title("Transactions Table")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(r"C:\Users\RAJAT\Model\backend\fraud_detection.db")
    query = "SELECT * FROM transactions"
    transactions_df = pd.read_sql(query, conn)
    conn.close()
    
    # Filters
    st.sidebar.header("Filter Transactions")
    transaction_id_filter = st.sidebar.text_input("Transaction ID")
    min_amount = st.sidebar.number_input("Min Transaction Amount", min_value=0.0, value=0.0)
    max_amount = st.sidebar.number_input("Max Transaction Amount", min_value=0.0, value=transactions_df["amount"].max())
    is_fraud_filter = st.sidebar.selectbox("Is Fraud", options=["All", "True", "False"])
    
    # Apply filters
    if transaction_id_filter:
        transactions_df = transactions_df[transactions_df["transaction_id"].str.contains(transaction_id_filter, case=False)]
    transactions_df = transactions_df[(transactions_df["amount"] >= min_amount) & (transactions_df["amount"] <= max_amount)]
    if is_fraud_filter != "All":
        transactions_df = transactions_df[transactions_df["is_fraud"] == (is_fraud_filter == "True")]
    
    # Sort options
    st.sidebar.header("Sort Transactions")
    sort_by = st.sidebar.selectbox("Sort By", options=transactions_df.columns)
    sort_order = st.sidebar.radio("Sort Order", options=["Ascending", "Descending"])
    
    # Apply sorting
    transactions_df = transactions_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
    
    st.dataframe(transactions_df, use_container_width=True)

elif selected == "Edit Rule Book":
    st.title("Rule Book Management")
    
    rule_book = {
        "high_transaction_amount": st.number_input("High Transaction Amount", min_value=0.0, value=100000.0),
        "same_payer_payee": st.checkbox("Same Payer and Payee", value=True),
        "high_atm_withdrawal": st.number_input("High ATM Withdrawal Amount", min_value=0.0, value=50000.0),
        "unrecognized_bank": st.text_input("Unrecognized Bank", value="UNKNOWN"),
        "disposable_email_domain": st.text_input("Disposable Email Domain", value="@disposable.com"),
        "private_network_ip_prefixes": st.text_input("Private Network IP Prefixes", value="192.168,10."),
        "channel_rule": {"mobile": st.number_input("Mobile Channel Amount", min_value=0.0, value=2000.0),
                          "web": st.number_input("Web Channel Amount", min_value=0.0, value=5000.0)},
        "amount_rule": st.number_input("Amount Rule", min_value=0.0, value=2000.0),
        "email_ip_rule": {"email": st.number_input("Max Transactions per Email", min_value=0, value=3),
                           "ip": st.number_input("Max Transactions per IP", min_value=0, value=3)},
        "transaction_frequency_rule": {"time_window": st.number_input("Transaction Frequency Time Window (minutes)", min_value=0, value=60),
                                       "max_transactions": st.number_input("Max Transactions in Time Window", min_value=0, value=5)},
        "payee_id_rule": st.number_input("Max Transactions per Payee ID", min_value=0, value=3),
        "payer_payee_pattern_rule": st.checkbox("Flag Unusual Payer-Payee Patterns", value=True)
    }
    
    if st.button("Update Rule Book"):
        response = requests.post(f"{API_URL}/update_rule_book", json=rule_book)
        if response.status_code == 200:
            result = response.json()
            st.success("Rule book updated successfully!")
            st.write("Updated Rule Book:")
            st.table(pd.DataFrame.from_dict(rule_book, orient='index', columns=['Value']))
        else:
            st.error("Error updating rule book: " + response.json()["error"])
