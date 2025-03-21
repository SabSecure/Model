from fastapi import FastAPI
import joblib
import sqlite3
import pandas as pd
from rule_engine import apply_fraud_rules  # Import Rule-Based System

app = FastAPI()

# Load AI Model & Preprocessing Tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
imputer = joblib.load("imputer.pkl")

# Connect to SQLite Database
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# Fraud Detection API (Combining AI + Rules)
@app.post("/detect-fraud/")
def detect_fraud(transaction: dict):
    # Apply Rule-Based System
    rule_result = apply_fraud_rules(transaction)

    # If rule-based system detects fraud, return result immediately
    if rule_result["is_fraud"]:
        return {
            "transaction_id": transaction["transaction_id_anonymous"],
            "is_fraud": True,
            "fraud_source": "rule",
            "fraud_reason": ", ".join(rule_result["fraud_reasons"]),
            "fraud_score": rule_result["fraud_score"],
        }

    # Apply AI Model for Fraud Detection
    df = pd.DataFrame([transaction])
    df = df.drop(columns=["transaction_id_anonymous"])  # Remove ID column
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)  # Handle NaN
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)  # Normalize Data

    fraud_prediction = model.predict(df)[0]
    fraud_score = model.predict_proba(df)[0][1]  # AI Confidence Score

    # Store in Database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO transactions (transaction_id, is_fraud_predicted, fraud_source, fraud_reason, fraud_score)
        VALUES (?, ?, ?, ?, ?)""",
        (transaction["transaction_id_anonymous"], fraud_prediction, "model", "AI prediction", fraud_score),
    )
    conn.commit()
    conn.close()

    return {
        "transaction_id": transaction["transaction_id_anonymous"],
        "is_fraud": bool(fraud_prediction),
        "fraud_source": "model",
        "fraud_reason": "AI prediction",
        "fraud_score": round(fraud_score, 2),
    }
