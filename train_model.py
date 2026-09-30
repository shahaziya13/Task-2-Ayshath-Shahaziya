import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

df = pd.read_csv("data/flood_risk_dataset.csv")

print("Dataset Shape:", df.shape)

X = df.drop("flood_risk", axis=1)
y = df["flood_risk"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("================================")
print("MODEL PERFORMANCE")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print()
print("Saving model...")

joblib.dump(
    model,
    "models/floodguard_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print()
print("================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")
print("models/floodguard_model.pkl")
print("models/label_encoder.pkl")
