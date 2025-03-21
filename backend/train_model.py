import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv("transactions_train.csv")

# 🔹 Step 1: Drop Unnecessary Columns
df = df.drop(columns=["transaction_date"])  # Drop date column

# 🔹 Step 2: Encode Categorical Features
categorical_cols = [
    "transaction_channel",
    "transaction_payment_mode_anonymous",
    "payment_gateway_bank_anonymous",
    "payer_browser_anonymous",
    "payer_email_anonymous",
    "payee_ip_anonymous",
    "payer_mobile_anonymous",
    "transaction_id_anonymous",
    "payee_id_anonymous"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le  # Store encoders for later use

# 🔹 Step 3: Separate Features and Target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# 🔹 Step 4: Handle Missing Values (Fix NaNs)
imputer = SimpleImputer(strategy="median")  # Replace NaN with median value
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Apply imputer

# 🔹 Step 5: Normalize the Data (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 6: Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Step 7: Fix Class Imbalance Using SMOTE
print(f"Original Class Distribution:\n{y_train.value_counts()}")
smote = SMOTE(sampling_strategy=0.05, random_state=42)  
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"New Class Distribution:\n{y_train.value_counts()}")

# 🔹 Step 8: Train the Model with Class Weights to Avoid Overfitting
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, min_samples_split=10, class_weight="balanced")
model.fit(X_train, y_train)

# 🔹 Step 9: Evaluate the Model
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print Accuracy & Classification Report
print(f"✅ Training Accuracy: {train_accuracy:.4f}")
print(f"✅ Testing Accuracy: {test_accuracy:.4f}")
print("\n🔹 Classification Report on Test Data:\n", classification_report(y_test, y_test_pred))

# 🔹 Step 10: Save the Model
joblib.dump(model, "backend/model.pkl")
joblib.dump(scaler, "backend/scaler.pkl")
joblib.dump(label_encoders, "backend/label_encoders.pkl")
joblib.dump(imputer, "backend/imputer.pkl")  # Save Imputer for future use

print("\n✅ Model, Scaler, Encoders, and Imputer saved successfully!")
