from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from database import create_app, db, insert_transaction, insert_fraud_report

# Create and configure the Flask app
app = create_app()
CORS(app)

# Load trained fraud detection model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

FEATURE_COLUMNS = [
    "transaction_amount", "transaction_channel", "transaction_payment_mode_anonymous",
    "payment_gateway_bank_anonymous", "payer_browser_anonymous", "payer_email_anonymous",
    "payee_ip_anonymous", "payer_mobile_anonymous", "transaction_id_anonymous",
    "payee_id_anonymous"
]

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict_fraud", methods=["POST"])
def predict_fraud():
    try:
        data = request.json
        if not data:
            print("Error: No input data provided")
            return jsonify({"error": "No input data provided"}), 400
        
        # Log incoming data for debugging
        print(f"Received data: {data}")

        fraud_reason = "Model"

        # ✅ Rule-Based Checks Before AI Model
        if data["transaction_amount"] > 100000:
            fraud_reason = "Rule: High transaction amount"
            is_fraud = 1
            fraud_prob = 1.0  # High risk
        elif data["payer_mobile_anonymous"] == data["payee_id_anonymous"]:
            fraud_reason = "Rule: Same payer and payee"
            is_fraud = 1
            fraud_prob = 1.0
        elif data["transaction_channel"] == "ATM" and data["transaction_amount"] > 50000:
            fraud_reason = "Rule: High ATM withdrawal"
            is_fraud = 1
            fraud_prob = 1.0
        elif data["payment_gateway_bank_anonymous"] == "UNKNOWN":
            fraud_reason = "Rule: Unrecognized bank used"
            is_fraud = 1
            fraud_prob = 1.0
        elif data["payer_email_anonymous"].endswith("@disposable.com"):
            fraud_reason = "Rule: Temporary/disposable email used"
            is_fraud = 1
            fraud_prob = 1.0
        elif data["payee_ip_anonymous"].startswith("192.168") or data["payee_ip_anonymous"].startswith("10."):
            fraud_reason = "Rule: Private network IP address"
            is_fraud = 1
            fraud_prob = 1.0
        else:
            # ✅ AI Model Prediction
            try:
                # Encode categorical fields
                categorical_fields = ["transaction_channel", "transaction_payment_mode_anonymous", 
                                      "payment_gateway_bank_anonymous", "payer_browser_anonymous"]
                encoded_data = data.copy()
                for field in categorical_fields:
                    encoded_data[field] = float(hash(data[field]) % 1000)  # Convert hash to float

                # Handle email, IP, and other fields separately
                encoded_data["payer_email_anonymous"] = float(hash(data["payer_email_anonymous"]) % 1000)
                encoded_data["payee_ip_anonymous"] = float(hash(data["payee_ip_anonymous"]) % 1000)
                encoded_data["payer_mobile_anonymous"] = float(hash(data["payer_mobile_anonymous"]) % 1000)
                encoded_data["transaction_id_anonymous"] = float(hash(data["transaction_id_anonymous"]) % 1000)
                encoded_data["payee_id_anonymous"] = float(hash(data["payee_id_anonymous"]) % 1000)

                # Prepare features for the model
                features = np.array([encoded_data[col] for col in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
                fraud_prob = model.predict_proba(features)[0][1]
                is_fraud = int(fraud_prob > 0.2)
            except KeyError as e:
                print(f"Error: Missing feature in input data - {e}")
                return jsonify({"error": f"Missing feature: {e}"}), 400
            except Exception as e:
                print(f"Error during model prediction: {e}")
                return jsonify({"error": "Error during model prediction"}), 500

        # ✅ Store transaction in the database
        try:
            # Ensure the correct transaction_id is passed
            transaction_data = {
                "transaction_id": data["transaction_id_anonymous"],  # Correctly map transaction_id
                "transaction_amount": data["transaction_amount"]
            }
            insert_transaction(transaction_data, is_fraud, fraud_prob)
        except Exception as e:
            print(f"Error inserting transaction into database: {e}")
            return jsonify({"error": "Database insertion failed"}), 500

        return jsonify({
            "fraud_probability": fraud_prob,
            "is_fraud": is_fraud,
            "fraud_source": fraud_reason
        })

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in /predict_fraud: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/report_fraud", methods=["POST"])
def report_fraud():
    try:
        data = request.json
        if not all(k in data for k in ("transaction_id", "reporting_entity_id", "fraud_details")):
            return jsonify({"error": "Missing required fields"}), 400
        
        insert_fraud_report(data["transaction_id"], data["reporting_entity_id"], data["fraud_details"])
        return jsonify({"message": "Fraud report saved successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        data = request.json.get("transactions", [])
        if not data:
            return jsonify({"error": "No transactions provided"}), 400
        
        results = {}
        for transaction in data:
            features = np.array([transaction[col] for col in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
            fraud_prob = model.predict_proba(features)[0][1]
            is_fraud = int(fraud_prob > 0.2)

            # Store each transaction result
            insert_transaction(transaction, is_fraud, fraud_prob)
            results[transaction["transaction_id_anonymous"]] = {
                "is_fraud": is_fraud,
                "fraud_score": fraud_prob
            }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
