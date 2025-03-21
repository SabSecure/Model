from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import datetime
import os

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app():
    """Factory function to create and configure the Flask app."""
    app = Flask(__name__)

    # Database Configuration
    # Ensure the database file is saved in a persistent location
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///d:/Model/backend/fraud_detection.db")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize SQLAlchemy with the app
    db.init_app(app)

    # Create Database and Tables
    with app.app_context():
        db.create_all()  # This creates the database and tables

    return app

# Define Tables
class Transaction(db.Model):
    __tablename__ = "transactions"
    transaction_id = db.Column(db.String, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    is_fraud = db.Column(db.Boolean, nullable=False)
    fraud_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class FraudReport(db.Model):
    __tablename__ = "fraud_reports"
    report_id = db.Column(db.String, primary_key=True)
    transaction_id = db.Column(db.String, nullable=False)
    reporting_entity_id = db.Column(db.String, nullable=False)
    fraud_details = db.Column(db.String, nullable=False)
    reported_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

def insert_transaction(data, is_fraud, fraud_score):
    """ Inserts transaction details into the database """
    try:
        # Check if the transaction already exists
        existing_transaction = Transaction.query.filter_by(transaction_id=data["transaction_id"]).first()
        if existing_transaction:
            print(f"Transaction with ID {data['transaction_id']} already exists. Skipping insertion.")
            return  # Skip insertion if the transaction already exists

        transaction = Transaction(
            transaction_id=data["transaction_id"],
            amount=data["transaction_amount"],
            is_fraud=is_fraud,
            fraud_score=fraud_score
        )
        db.session.add(transaction)
        db.session.commit()
        print(f"Transaction with ID {data['transaction_id']} saved successfully.")
    except Exception as e:
        print(f"Error inserting transaction: {e}")
        db.session.rollback()
        raise

def insert_fraud_report(transaction_id, reporting_entity_id, fraud_details):
    """ Inserts a fraud report into the database """
    try:
        fraud_report = FraudReport(
            report_id=f"REPORT_{transaction_id}",
            transaction_id=transaction_id,
            reporting_entity_id=reporting_entity_id,
            fraud_details=fraud_details
        )
        db.session.add(fraud_report)
        db.session.commit()
        print(f"Fraud report for transaction ID {transaction_id} saved successfully.")
    except Exception as e:
        print(f"Error inserting fraud report: {e}")
        db.session.rollback()
        raise
