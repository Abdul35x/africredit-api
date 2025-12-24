print("Script is starting...")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading German Credit dataset...")
data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame

# ---------------------------------------------------------
# 2. CLEANING & FEATURE SELECTION
# ---------------------------------------------------------
print("Preprocessing data...")
features = ['duration', 'credit_amount', 'installment_commitment', 
            'residence_since', 'age', 'existing_credits', 'num_dependents']
target = 'class' 
X = df[features]
y = df[target]

# ---------------------------------------------------------
# 3. LABEL ENCODING (The "Why" is explained below)
# ---------------------------------------------------------
y = y.map({'good': 1, 'bad': 0})

# ---------------------------------------------------------
# 4. SPLITTING DATA
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. TRAINING THE MODEL
# ---------------------------------------------------------
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. SAVING THE ASSET
# ---------------------------------------------------------
joblib.dump(model, 'model/credit_model.pkl')
print("Success! Model saved to 'model/credit_model.pkl'")
print(f"Model Accuracy on Test Data: {model.score(X_test, y_test):.2f}")