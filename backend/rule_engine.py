import json
import sqlite3
import datetime

# Load Rules from JSON
def load_rules():
    with open("rulebook.json", "r") as file:
        return json.load(file)

rules = load_rules()

# Apply Fraud Rules
def apply_fraud_rules(transaction):
    fraud_score = 0.0
    fraud_reasons = []

    # Rule 1: High Transaction Amount
    if rules["high_transaction_amount"]["enabled"]:
        if transaction["transaction_amount"] > rules["high_transaction_amount"]["threshold"]:
            fraud_score += rules["high_transaction_amount"]["fraud_score"]
            fraud_reasons.append(rules["high_transaction_amount"]["reason"])

    # Rule 2: Multiple Transactions in Short Time
    if rules["multiple_transactions_short_time"]["enabled"]:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE payer_mobile_anonymous = ? 
            AND timestamp >= datetime('now', ?)
            """,
            (transaction["payer_mobile_anonymous"], "-" + rules["multiple_transactions_short_time"]["time_window"]),
        )
        recent_txn_count = cursor.fetchone()[0]
        if recent_txn_count > rules["multiple_transactions_short_time"]["limit"]:
            fraud_score += rules["multiple_transactions_short_time"]["fraud_score"]
            fraud_reasons.append(rules["multiple_transactions_short_time"]["reason"])

    # Rule 3: Blacklisted Payer/Payee
    if rules["blacklisted_users"]["enabled"]:
        if transaction["payer_mobile_anonymous"] in rules["blacklisted_users"]["blacklist"] or \
           transaction["payee_id_anonymous"] in rules["blacklisted_users"]["blacklist"]:
            fraud_score += rules["blacklisted_users"]["fraud_score"]
            fraud_reasons.append(rules["blacklisted_users"]["reason"])

    # Rule 4: High-Risk Payment Mode
    if rules["high_risk_payment_mode"]["enabled"]:
        if transaction["transaction_payment_mode_anonymous"] in rules["high_risk_payment_mode"]["modes"]:
            fraud_score += rules["high_risk_payment_mode"]["fraud_score"]
            fraud_reasons.append(rules["high_risk_payment_mode"]["reason"])

    # Rule 5: Unusual Location (Dummy Check)
    if rules["unusual_location"]["enabled"]:
        fraud_score += rules["unusual_location"]["fraud_score"]
        fraud_reasons.append(rules["unusual_location"]["reason"])

    # Determine if transaction is fraud based on threshold
    is_fraud = fraud_score >= 1.0  # Threshold: 1.0

    return {
        "is_fraud": is_fraud,
        "fraud_score": round(fraud_score, 2),
        "fraud_reasons": fraud_reasons,
    }
