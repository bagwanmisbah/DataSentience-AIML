# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("data/Crop_recommendation.csv")

# Encode label
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
joblib.dump(model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model and label encoder saved.")
