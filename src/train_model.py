import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------- Load dataset --------
DATA_PATH = "data/landmarks.csv"
MODEL_PATH = "models/gesture_model.pkl"

import os
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nClass counts:")
print(df["label"].value_counts())

# -------- Split features and labels --------
X = df.drop("label", axis=1)
y = df["label"]

# -------- Train/test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# -------- Train model --------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------- Predictions --------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# -------- Evaluation --------
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\nTrain Accuracy:", round(train_acc, 4))
print("Test Accuracy:", round(test_acc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# -------- Save model --------
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")