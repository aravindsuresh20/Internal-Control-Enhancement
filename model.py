import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("audit_data.csv")

# Select features and target
features = ['Audit_Risk', 'Inherent_Risk', 'Score', 'TOTAL', 'Money_Value']
X = df[features]
y = df['Risk']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "risk_model.pkl")

# Confirmation message
print("Model got saved successfully.")
