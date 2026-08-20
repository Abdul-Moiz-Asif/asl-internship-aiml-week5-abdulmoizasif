import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load the exact dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Build the Week 4 tuned pipeline
rf_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)
rf_pipe.fit(X, y)

# 3. Save it directly to our new architecture
filepath = "app/model_assets/best_rf_model.pkl"
joblib.dump(rf_pipe, filepath)
print(f"SUCCESS: Model regenerated and saved to {filepath}")
