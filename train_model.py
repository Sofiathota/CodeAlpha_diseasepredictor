import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle

# Load dataset
data = pd.read_csv("disease_data.csv")

# Shuffle dataset
data = shuffle(data, random_state=42)

# Encode disease labels
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Features and target
X = data.drop("disease", axis=1)
y = data["disease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Better Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump((model, encoder), file)

print("Model saved successfully!")