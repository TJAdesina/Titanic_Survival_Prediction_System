import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# ========================
# 1. Load Dataset
# ========================
DATA_PATH = "titanic.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())

# ========================
# 2. Feature Selection
# ========================
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

df = df[features + [target]]

# ========================
# 3. Handle Missing Values
# ========================

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ========================
# 4. Encode Categorical Variables
# ========================

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# ========================
# 5. Split Features and Target
# ========================

X = df[features]
y = df[target]

# ========================
# 6. Train/Test Split
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 7. Feature Scaling
# ========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# 8. Train Model
# ========================

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# ========================
# 9. Evaluate Model
# ========================

y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========================
# 10. Save Model, Scaler, Encoders
# ========================

os.makedirs("saved_model", exist_ok=True)

joblib.dump(model, "saved_model/titanic_model.pkl")
joblib.dump(scaler, "saved_model/scaler.pkl")
joblib.dump(le_sex, "saved_model/sex_encoder.pkl")
joblib.dump(le_embarked, "saved_model/embarked_encoder.pkl")

print("\nModel and preprocessors saved successfully!")

# ========================
# 11. Reload & Test Prediction
# ========================

loaded_model = joblib.load("saved_model/titanic_model.pkl")
loaded_scaler = joblib.load("saved_model/scaler.pkl")

sample_input = np.array([[3, 1, 25, 7.25, 2]])  # example passenger

sample_input_scaled = loaded_scaler.transform(sample_input)
prediction = loaded_model.predict(sample_input_scaled)

print("\nReloaded model prediction test:", prediction[0])
