import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "European_Bank.csv"

def engineer(data):
    d = data.copy()
    d["Balance_to_Salary"] = d["Balance"] / (d["EstimatedSalary"].abs() + 1)
    d["Product_Engagement"] = d["NumOfProducts"] * d["IsActiveMember"]
    d["Age_Tenure_Interaction"] = d["Age"] * d["Tenure"]
    d["Product_Density"] = d["NumOfProducts"] / (d["Age"] + 1)
    return d

df = pd.read_csv(DATA_PATH)
X = engineer(df.drop(columns=["Exited", "CustomerId", "Surname"]))
y = df["Exited"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = [c for c in X.columns if c not in categorical]

def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )
}

rows = []
pipelines = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })
    pipelines[name] = pipe

metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
metrics.to_csv("model_metrics.csv", index=False)

best_name = metrics.iloc[0]["Model"]
joblib.dump({
    "model": pipelines[best_name],
    "features": list(X.columns),
    "target": "Exited",
    "best_model": best_name
}, "churn_model.joblib")

print(metrics.to_string(index=False))
print(f"\nSaved best model: {best_name}")
