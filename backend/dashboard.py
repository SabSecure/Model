import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import random
import pickle
import os

# Set page config
st.set_page_config(layout="wide", page_title="Transaction and Fraud Monitoring Dashboard")

# API URL
API_URL = "http://127.0.0.1:5000"

# Mock data for demo purposes
def generate_mock_data(num_rows=100):
    channels = ["mobile", "web", "POS"]
    payment_modes = ["UPI", "Credit Card", "Debit Card", "NetBanking", "Wallet"]
    banks = ["HDFC", "ICICI", "SBI", "Axis", "Yes Bank", "UNKNOWN"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
    
    data = {
        "transaction_id": [f"TXN{i:06d}" for i in range(1, num_rows+1)],
        "timestamp": [(datetime.now() - timedelta(days=random.randint(0, 30), 
                                               hours=random.randint(0, 23), 
                                               minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S") 
                     for _ in range(num_rows)],
        "transaction_amount": [random.uniform(100, 50000) for _ in range(num_rows)],
        "transaction_channel": [random.choice(channels) for _ in range(num_rows)],
        "transaction_payment_mode": [random.choice(payment_modes) for _ in range(num_rows)],
        "payment_gateway_bank": [random.choice(banks) for _ in range(num_rows)],
        "payer_browser": [random.choice(browsers) for _ in range(num_rows)],
        "payer_id": [f"P{random.randint(1000, 9999)}" for _ in range(num_rows)],
        "payee_id": [f"M{random.randint(1000, 9999)}" for _ in range(num_rows)],
        "is_fraud_predicted": [random.choice([True, False, False, False, False]) for _ in range(num_rows)],
        "is_fraud_reported": [random.choice([True, False, False, False, False, False]) for _ in range(num_rows)],
        "fraud_score": [random.uniform(0, 1) for _ in range(num_rows)],
        "fraud_source": [random.choice(["Rule Engine", "ML Model", "Manual Review", ""]) for _ in range(num_rows)],
        "fraud_reason": [random.choice(["High Amount", "Suspicious IP", "Unusual Pattern", "Multiple Attempts", ""]) for _ in range(num_rows)]
    }
    
    # Make fraud_source and fraud_reason empty for non-fraud transactions
    for i in range(num_rows):
        if not data["is_fraud_predicted"][i] and not data["is_fraud_reported"][i]:
            data["fraud_source"][i] = ""
            data["fraud_reason"][i] = ""
            data["fraud_score"][i] = 0.0
    
    return pd.DataFrame(data)

# Use session state to maintain data across reruns
if 'data' not in st.session_state:
    st.session_state.data = generate_mock_data(200)
    st.session_state.data['timestamp'] = pd.to_datetime(st.session_state.data['timestamp'])
    st.session_state.data = st.session_state.data.sort_values('timestamp', ascending=False)

if 'rule_book' not in st.session_state:
    st.session_state.rule_book = {
        "high_transaction_amount": 100000.0,
        "same_payer_payee": True,
        "high_atm_withdrawal": 50000.0,
        "unrecognized_bank": "UNKNOWN",
        "disposable_email_domain": "@disposable.com",
        "private_network_ip_prefixes": "192.168,10.",
        "channel_rule": {"mobile": 2000.0, "web": 5000.0},
        "amount_rule": 2000.0,
        "email_ip_rule": {"email": 3, "ip": 3},
        "browser_payment_rule": [("Chrome", "UNKNOWN"), ("Firefox", "UNKNOWN")],
        "transaction_frequency_rule": {"time_window": 60, "max_transactions": 5},
        "payee_id_rule": 3,
        "payer_payee_pattern_rule": True
    }

if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {
        "train_accuracy": 0.94,
        "test_accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.83,
        "f1_score": 0.86,
        "confusion_matrix": [[156, 12], [8, 74]]
    }

# Load the real model and label encoders
@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(r"c:\Users\Dell\Downloads", 'Model', 'fraud_model.pkl')
        encoders_path = os.path.join(r"c:\Users\Dell\Downloads", 'Model', 'label_encoders.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, label_encoders = load_models()

# Function to predict fraud (mock)
def predict_fraud(transaction_data):
    try:
        # Create DataFrame with single transaction
        df = pd.DataFrame([transaction_data])
        
        # Process data using label encoders
        for column in label_encoders:
            if column in df.columns:
                df[column] = label_encoders[column].transform(df[column])
        
        # Make prediction
        is_fraud = bool(model.predict(df)[0])
        fraud_score = float(model.predict_proba(df)[0][1])
        
        # Determine fraud source and reason
        fraud_source = "ML Model" if is_fraud else ""
        fraud_reason = "High Risk Score" if is_fraud else ""
        
        return {
            "transaction_id": transaction_data["transaction_id_anonymous"],
            "is_fraud": is_fraud,
            "fraud_source": fraud_source,
            "fraud_reason": fraud_reason,
            "fraud_score": fraud_score
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            "transaction_id": transaction_data["transaction_id_anonymous"],
            "is_fraud": False,
            "fraud_source": "Error",
            "fraud_reason": str(e),
            "fraud_score": 0.0
        }

# App header
st.title("Transaction and Fraud Monitoring Dashboard")

# Create tabs for different sections
tabs = st.tabs(["Dashboard Overview", "Transaction Analysis", "Evaluation Metrics", "Rule Book", "Predict New Transaction"])

# Dashboard Overview Tab
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Fraud Overview")
        
        # Summary metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        total_transactions = len(st.session_state.data)
        predicted_frauds = st.session_state.data['is_fraud_predicted'].sum()
        reported_frauds = st.session_state.data['is_fraud_reported'].sum()
        
        metrics_col1.metric("Total Transactions", f"{total_transactions:,}")
        metrics_col2.metric("Predicted Frauds", f"{int(predicted_frauds):,}")
        metrics_col3.metric("Reported Frauds", f"{int(reported_frauds):,}")
        metrics_col4.metric("Fraud Rate", f"{predicted_frauds/total_transactions:.2%}")
        
        # Time series chart
        st.subheader("Fraud Trends Over Time")
        
        # Group by date
        time_data = st.session_state.data.copy()
        time_data['date'] = time_data['timestamp'].dt.date
        time_agg = time_data.groupby('date').agg({
            'is_fraud_predicted': 'sum',
            'is_fraud_reported': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        time_agg.columns = ['date', 'predicted_frauds', 'reported_frauds', 'total_transactions']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_agg['date'], y=time_agg['predicted_frauds'], 
                                mode='lines+markers', name='Predicted Frauds'))
        fig.add_trace(go.Scatter(x=time_agg['date'], y=time_agg['reported_frauds'], 
                                mode='lines+markers', name='Reported Frauds'))
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Count',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Distribution")
        
        # Distribution by channel
        channel_data = st.session_state.data.groupby('transaction_channel').agg({
            'is_fraud_predicted': 'sum',
            'is_fraud_reported': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        fig = px.bar(channel_data, x='transaction_channel', 
                    y=['is_fraud_predicted', 'is_fraud_reported'],
                    title='Fraud by Channel',
                    labels={'value': 'Count', 'transaction_channel': 'Channel', 'variable': 'Type'},
                    height=300)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by payment mode
        payment_data = st.session_state.data.groupby('transaction_payment_mode').agg({
            'is_fraud_predicted': 'sum',
            'is_fraud_reported': 'sum'
        }).reset_index()
        
        fig = px.bar(payment_data, x='transaction_payment_mode', 
                    y=['is_fraud_predicted', 'is_fraud_reported'],
                    title='Fraud by Payment Mode',
                    labels={'value': 'Count', 'transaction_payment_mode': 'Payment Mode', 'variable': 'Type'},
                    height=300)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

# Transaction Analysis Tab
with tabs[1]:
    st.subheader("Transaction Data")
    
    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        # Date range filter
        min_date = st.session_state.data['timestamp'].min().date()
        max_date = st.session_state.data['timestamp'].max().date()
        date_range = st.date_input("Select Date Range", 
                                  value=(min_date, max_date),
                                  min_value=min_date,
                                  max_value=max_date)
    
    with filter_col2:
        # Payer ID filter
        payer_ids = st.session_state.data['payer_id'].unique().tolist()
        selected_payer_ids = st.multiselect("Filter by Payer ID", payer_ids)
    
    with filter_col3:
        # Payee ID filter
        payee_ids = st.session_state.data['payee_id'].unique().tolist()
        selected_payee_ids = st.multiselect("Filter by Payee ID", payee_ids)
    
    with filter_col4:
        # Transaction ID search
        search_txn_id = st.text_input("Search Transaction ID")
    
    # Apply filters
    filtered_data = st.session_state.data.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['timestamp'].dt.date >= start_date) & 
            (filtered_data['timestamp'].dt.date <= end_date)
        ]
    
    if selected_payer_ids:
        filtered_data = filtered_data[filtered_data['payer_id'].isin(selected_payer_ids)]
    
    if selected_payee_ids:
        filtered_data = filtered_data[filtered_data['payee_id'].isin(selected_payee_ids)]
    
    if search_txn_id:
        filtered_data = filtered_data[filtered_data['transaction_id'].str.contains(search_txn_id, case=False)]
    
    # Display data table
    st.dataframe(filtered_data[['transaction_id', 'timestamp', 'transaction_amount', 
                             'transaction_channel', 'transaction_payment_mode', 
                             'payment_gateway_bank', 'payer_id', 'payee_id',
                             'is_fraud_predicted', 'is_fraud_reported', 'fraud_score', 
                             'fraud_source', 'fraud_reason']], height=400)
    
    # Dynamic graphs based on dimensions
    st.subheader("Fraud Analysis by Dimension")
    
    # Dimension selection
    dim_col1, dim_col2 = st.columns(2)
    
    with dim_col1:
        dimension = st.selectbox("Select Dimension", 
                               ["Transaction Channel", "Transaction Payment Mode", 
                                "Payment Gateway Bank", "Payer ID", "Payee ID"])
    
    with dim_col2:
        top_n = st.slider("Show Top N Values", min_value=3, max_value=20, value=10)
    
    # Map selection to column
    dimension_map = {
        "Transaction Channel": "transaction_channel",
        "Transaction Payment Mode": "transaction_payment_mode",
        "Payment Gateway Bank": "payment_gateway_bank",
        "Payer ID": "payer_id",
        "Payee ID": "payee_id"
    }
    
    selected_col = dimension_map[dimension]
    
    # Aggregate by selected dimension
    dim_data = filtered_data.groupby(selected_col).agg({
        'is_fraud_predicted': 'sum',
        'is_fraud_reported': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    dim_data.columns = [selected_col, 'predicted_frauds', 'reported_frauds', 'total_transactions']
    
    # Calculate fraud rate
    dim_data['predicted_fraud_rate'] = dim_data['predicted_frauds'] / dim_data['total_transactions'] * 100
    dim_data['reported_fraud_rate'] = dim_data['reported_frauds'] / dim_data['total_transactions'] * 100
    
    # Sort and get top N
    dim_data = dim_data.sort_values('total_transactions', ascending=False).head(top_n)
    
    # Create two charts side by side
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Absolute numbers
        fig = px.bar(dim_data, x=selected_col, y=['predicted_frauds', 'reported_frauds'],
                    title=f'Fraud Count by {dimension}',
                    labels={'value': 'Count', selected_col: dimension, 'variable': 'Type'},
                    height=400)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Fraud rate
        fig = px.bar(dim_data, x=selected_col, y=['predicted_fraud_rate', 'reported_fraud_rate'],
                    title=f'Fraud Rate (%) by {dimension}',
                    labels={'value': 'Rate (%)', selected_col: dimension, 'variable': 'Type'},
                    height=400)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

# Evaluation Metrics Tab
with tabs[2]:
    st.subheader("Model Evaluation")
    
    # Time period selection for evaluation
    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        eval_period = st.selectbox("Select Evaluation Period", 
                                 ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"])
    
    # Metrics in card format
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    metrics_col1.metric("Training Accuracy", f"{st.session_state.evaluation_metrics['train_accuracy']:.2%}")
    metrics_col2.metric("Testing Accuracy", f"{st.session_state.evaluation_metrics['test_accuracy']:.2%}")
    metrics_col3.metric("Precision", f"{st.session_state.evaluation_metrics['precision']:.2%}")
    metrics_col4.metric("Recall", f"{st.session_state.evaluation_metrics['recall']:.2%}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    conf_mat = st.session_state.evaluation_metrics['confusion_matrix']
    conf_mat_labels = ['Not Fraud', 'Fraud']
    
    conf_mat_fig = px.imshow(conf_mat,
                           x=conf_mat_labels,
                           y=conf_mat_labels,
                           color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           text_auto=True)
    conf_mat_fig.update_layout(
        width=500,
        height=500
    )
    
    precision_recall_col1, precision_recall_col2 = st.columns([1, 1])
    
    with precision_recall_col1:
        st.plotly_chart(conf_mat_fig)
    
    with precision_recall_col2:
        st.subheader("Model Performance Over Time")
        
        # Mock precision-recall data over time
        dates = [datetime.now() - timedelta(days=i*7) for i in range(10)]
        precision_vals = [0.89, 0.87, 0.88, 0.86, 0.85, 0.89, 0.91, 0.88, 0.87, 0.89]
        recall_vals = [0.83, 0.81, 0.82, 0.84, 0.80, 0.82, 0.85, 0.83, 0.81, 0.83]
        f1_vals = [0.86, 0.84, 0.85, 0.85, 0.82, 0.85, 0.88, 0.85, 0.84, 0.86]
        
        perf_data = pd.DataFrame({
            'date': dates,
            'precision': precision_vals,
            'recall': recall_vals,
            'f1_score': f1_vals
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf_data['date'], y=perf_data['precision'], 
                                mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=perf_data['date'], y=perf_data['recall'], 
                                mode='lines+markers', name='Recall'))
        fig.add_trace(go.Scatter(x=perf_data['date'], y=perf_data['f1_score'], 
                                mode='lines+markers', name='F1 Score'))
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Score',
            yaxis=dict(range=[0.7, 1.0]),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# Rule Book Tab
with tabs[3]:
    st.subheader("Fraud Detection Rule Book")
    
    # Two column layout for rule configuration
    rule_col1, rule_col2 = st.columns(2)
    
    with rule_col1:
        st.session_state.rule_book["high_transaction_amount"] = st.number_input(
            "High Transaction Amount Threshold", 
            min_value=0.0, 
            value=st.session_state.rule_book["high_transaction_amount"]
        )
        
        st.session_state.rule_book["same_payer_payee"] = st.checkbox(
            "Flag Same Payer and Payee", 
            value=st.session_state.rule_book["same_payer_payee"]
        )
        
        st.session_state.rule_book["high_atm_withdrawal"] = st.number_input(
            "High ATM Withdrawal Threshold", 
            min_value=0.0, 
            value=st.session_state.rule_book["high_atm_withdrawal"]
        )
        
        st.session_state.rule_book["unrecognized_bank"] = st.text_input(
            "Unrecognized Bank Flag", 
            value=st.session_state.rule_book["unrecognized_bank"]
        )
        
        st.session_state.rule_book["disposable_email_domain"] = st.text_input(
            "Disposable Email Domain", 
            value=st.session_state.rule_book["disposable_email_domain"]
        )
        
        st.session_state.rule_book["private_network_ip_prefixes"] = st.text_input(
            "Private Network IP Prefixes", 
            value=st.session_state.rule_book["private_network_ip_prefixes"]
        )
    
    with rule_col2:
        st.subheader("Channel Rules")
        st.session_state.rule_book["channel_rule"]["mobile"] = st.number_input(
            "Mobile Channel Amount Threshold", 
            min_value=0.0, 
            value=st.session_state.rule_book["channel_rule"]["mobile"]
        )
        
        st.session_state.rule_book["channel_rule"]["web"] = st.number_input(
            "Web Channel Amount Threshold", 
            min_value=0.0, 
            value=st.session_state.rule_book["channel_rule"]["web"]
        )
        
        st.subheader("Frequency Rules")
        st.session_state.rule_book["transaction_frequency_rule"]["time_window"] = st.number_input(
            "Transaction Frequency Time Window (minutes)", 
            min_value=0, 
            value=st.session_state.rule_book["transaction_frequency_rule"]["time_window"]
        )
        
        st.session_state.rule_book["transaction_frequency_rule"]["max_transactions"] = st.number_input(
            "Max Transactions in Time Window", 
            min_value=0, 
            value=st.session_state.rule_book["transaction_frequency_rule"]["max_transactions"]
        )
        
        st.session_state.rule_book["payee_id_rule"] = st.number_input(
            "Max Transactions per Payee ID", 
            min_value=0, 
            value=st.session_state.rule_book["payee_id_rule"]
        )
    
    # Rule effectiveness visualization
    st.subheader("Rule Effectiveness")
    
    # Mock data for rule effectiveness
    rule_names = [
        "High Transaction Amount", 
        "Same Payer/Payee", 
        "High ATM Withdrawal", 
        "Unrecognized Bank",
        "Disposable Email",
        "Channel Rules",
        "Frequency Rules"
    ]
    
    true_positives = [15, 8, 12, 7, 10, 18, 14]
    false_positives = [5, 3, 8, 9, 4, 7, 6]
    
    rule_data = pd.DataFrame({
        'rule': rule_names,
        'true_positives': true_positives,
        'false_positives': false_positives
    })
    
    rule_data['precision'] = rule_data['true_positives'] / (rule_data['true_positives'] + rule_data['false_positives'])
    rule_data['total_triggers'] = rule_data['true_positives'] + rule_data['false_positives']
    
    rule_fig = px.bar(rule_data, x='rule', y=['true_positives', 'false_positives'],
                    title='Rule Triggers by Type',
                    labels={'value': 'Count', 'rule': 'Rule', 'variable': 'Type'},
                    color_discrete_map={'true_positives': 'green', 'false_positives': 'red'},
                    height=400)
    rule_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    
    precision_fig = px.bar(rule_data, x='rule', y='precision',
                         title='Rule Precision',
                         labels={'precision': 'Precision', 'rule': 'Rule'},
                         color='precision',
                         color_continuous_scale='RdYlGn',
                         height=400)
    precision_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    
    rule_col1, rule_col2 = st.columns(2)
    
    with rule_col1:
        st.plotly_chart(rule_fig, use_container_width=True)
    
    with rule_col2:
        st.plotly_chart(precision_fig, use_container_width=True)
    
    # Update button
    if st.button("Update Rule Book"):
        st.success("Rule book updated successfully!")
        st.balloons()

# Predict New Transaction Tab
with tabs[4]:
    st.subheader("Predict New Transaction")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        transaction_id = st.text_input("Transaction ID", value="TXN" + datetime.now().strftime("%Y%m%d%H%M%S"))
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        transaction_channel = st.selectbox("Transaction Channel", ["mobile", "web", "POS"])
    
    with pred_col2:
        payment_mode = st.text_input("Payment Mode", value="Credit Card")
        payment_gateway_bank = st.text_input("Payment Gateway Bank", value="HDFC")
        payer_browser = st.text_input("Payer Browser", value="Chrome")
    
    with pred_col3:
        payer_email = st.text_input("Payer Email", value="user@example.com")
        payee_ip = st.text_input("Payee IP", value="192.168.1.1")
        payer_mobile = st.text_input("Payer Mobile", value="+919876543210")
        payee_id = st.text_input("Payee ID", value="M1234")
    
    transaction_data = {
        "transaction_id_anonymous": transaction_id,
        "transaction_amount": transaction_amount,
        "transaction_channel": transaction_channel,
        "transaction_payment_mode_anonymous": payment_mode,
        "payment_gateway_bank_anonymous": payment_gateway_bank,
        "payer_browser_anonymous": payer_browser,
        "payer_email_anonymous": payer_email,
        "payee_ip_anonymous": payee_ip,
        "payer_mobile_anonymous": payer_mobile,
        "payee_id_anonymous": payee_id
    }
    
    if st.button("Predict", type="primary"):
        # Mock API call for prediction
        try:
            # Uncomment to use actual API
            # response = requests.post(f"{API_URL}/predict_fraud", json=transaction_data)
            # if response.status_code == 200:
            #     result = response.json()
            
            # For demo, use mock prediction
            result = predict_fraud(transaction_data)
            
            # Create a visually appealing card for results
            st.subheader("Prediction Results")
            result_cols = st.columns(4)
            
            with result_cols[0]:
                st.metric("Transaction ID", result["transaction_id"])
            
            with result_cols[1]:
                fraud_text = "⚠️ FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE"
                st.metric("Prediction", fraud_text)
            
            with result_cols[2]:
                st.metric("Fraud Score", f"{result['fraud_score']:.2%}")
            
            with result_cols[3]:
                st.metric("Fraud Source", result["fraud_source"] if result["fraud_source"] else "N/A")
            
            # Additional details
            if result["is_fraud"]:
                st.info(f"Fraud Reason: {result['fraud_reason']}")
                
                # Recommendation
                st.warning("Recommendation: This transaction should be flagged for manual review based on the fraud score and detection rules.")
            else:
                st.success("This transaction appears to be legitimate and can be processed.")
            
            # Add to session data for demo purposes
            new_row = pd.DataFrame({
                'transaction_id': [transaction_id],
                'timestamp': [datetime.now()],
                'transaction_amount': [transaction_amount],
                'transaction_channel': [transaction_channel],
                'transaction_payment_mode': [payment_mode],
                'payment_gateway_bank': [payment_gateway_bank],
                'payer_browser': [payer_browser],
                'payer_id': ["P" + transaction_id[-4:]],
                'payee_id': [payee_id],
                'is_fraud_predicted': [result["is_fraud"]],
                'is_fraud_reported': [False],
                'fraud_score': [result["fraud_score"]],
                'fraud_source': [result["fraud_source"]],
                'fraud_reason': [result["fraud_reason"]]
            })
            
            st.session_state.data = pd.concat([new_row, st.session_state.data]).reset_index(drop=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("© 2025 SabPaisa Fraud Monitoring System")
