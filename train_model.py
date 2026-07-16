import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_card_applications.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def generate_dataset(n_rows: int = 30) -> pd.DataFrame:
    """Generate a balanced dataset of 30 credit applications."""
    rng = np.random.default_rng(42)
    
    # Create realistic income distribution
    income = np.array([35000, 45000, 55000, 65000, 75000, 85000, 95000, 50000, 60000, 70000,
                      55000, 48000, 72000, 68000, 55000, 62000, 58000, 51000, 64000, 47000,
                      52000, 59000, 66000, 71000, 44000, 53000, 61000, 54000, 57000, 63000])
    
    # Debt to income ratios (more favorable)
    debt_to_income = np.array([0.25, 0.30, 0.35, 0.20, 0.40, 0.28, 0.32, 0.38, 0.22, 0.45,
                              0.26, 0.33, 0.29, 0.24, 0.36, 0.31, 0.27, 0.41, 0.23, 0.37,
                              0.34, 0.25, 0.39, 0.21, 0.42, 0.28, 0.30, 0.35, 0.26, 0.33])
    
    # Credit scores
    credit_score = np.array([700, 720, 680, 750, 650, 710, 730, 670, 760, 640,
                            695, 725, 705, 740, 660, 715, 690, 655, 735, 645,
                            710, 700, 750, 680, 720, 695, 745, 665, 755, 685])
    
    # Age
    age = np.array([28, 35, 42, 31, 45, 29, 38, 50, 32, 52,
                   26, 40, 33, 44, 48, 30, 36, 53, 27, 41,
                   34, 46, 25, 39, 49, 28, 43, 37, 51, 30])
    
    # Employment years
    employment_years = np.array([3, 5, 8, 2, 10, 4, 6, 12, 3, 15,
                                2, 7, 5, 9, 11, 3, 6, 14, 4, 8,
                                5, 10, 2, 7, 13, 3, 9, 5, 12, 4])
    
    # Loan balance
    loan_balance = np.array([8000, 12000, 15000, 6000, 20000, 10000, 14000, 18000, 7000, 22000,
                            9000, 13000, 11000, 16000, 19000, 8500, 12500, 21000, 6500, 17000,
                            10500, 14500, 5500, 15500, 23000, 9500, 16500, 11500, 19500, 13500])
    
    # Credit inquiries (fewer)
    num_inquiries = np.array([1, 0, 2, 1, 3, 0, 1, 2, 0, 4,
                             1, 2, 1, 0, 2, 1, 1, 3, 0, 2,
                             1, 2, 0, 1, 2, 1, 0, 2, 1, 1])
    
    # Prior defaults (minimal)
    prior_defaults = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Income types
    income_type = np.array(["Salary", "Salary", "Business", "Salary", "Salary", "Salary", "Business", "Retired", "Salary", "Salary",
                           "Salary", "Salary", "Business", "Salary", "Salary", "Salary", "Business", "Retired", "Salary", "Salary",
                           "Salary", "Salary", "Business", "Salary", "Salary", "Salary", "Business", "Salary", "Retired", "Salary"])
    
    # Employment status
    employment_status = np.array(["Employed", "Employed", "Self-employed", "Employed", "Employed", "Employed", "Self-employed", "Retired", "Employed", "Employed",
                                "Employed", "Self-employed", "Employed", "Employed", "Employed", "Employed", "Self-employed", "Retired", "Employed", "Employed",
                                "Employed", "Employed", "Self-employed", "Employed", "Employed", "Employed", "Employed", "Employed", "Retired", "Employed"])
    
    # Co-applicant
    has_coapplicant = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
                               0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
                               1, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    
    # Create approval based on realistic criteria (mostly approved)
    approval = []
    for i in range(n_rows):
        score = 0
        if credit_score[i] >= 700:
            score += 2
        if debt_to_income[i] <= 0.35:
            score += 2
        if income[i] >= 50000:
            score += 2
        if employment_years[i] >= 3:
            score += 1
        if num_inquiries[i] <= 1:
            score += 1
        if prior_defaults[i] == 0:
            score += 1
        if has_coapplicant[i]:
            score += 1
        
        approval.append(1 if score >= 5 else 0)
    
    df = pd.DataFrame({
        "income": income,
        "debt_to_income": debt_to_income,
        "credit_score": credit_score,
        "age": age,
        "employment_years": employment_years,
        "loan_balance": loan_balance,
        "num_inquiries": num_inquiries,
        "prior_defaults": prior_defaults,
        "income_type": income_type,
        "employment_status": employment_status,
        "has_coapplicant": has_coapplicant,
        "approval": approval,
    })
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "income",
        "debt_to_income",
        "credit_score",
        "age",
        "employment_years",
        "loan_balance",
        "num_inquiries",
        "prior_defaults",
        "has_coapplicant",
    ]
    categorical_features = ["income_type", "employment_status"]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )


def build_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=220, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
    else:
        models["xgboost"] = GradientBoostingClassifier(random_state=42)
    return models


def train_and_save_model(data_path: Path = DATA_PATH) -> dict:
    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        generate_dataset().to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["approval"])
    y = df["approval"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    preprocessor = build_preprocessor()
    results = []

    for name, model in build_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append((name, accuracy, pipeline))

    best_name, best_accuracy, best_pipeline = max(results, key=lambda x: x[1])
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")

    metadata = {
        "model_name": best_name,
        "accuracy": round(float(best_accuracy), 4),
        "features": list(X.columns),
        "target": "approval",
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Training complete")
    for name, accuracy, _ in results:
        print(f"{name}: {accuracy:.4f}")
    print(f"Best model: {best_name} with accuracy {best_accuracy:.4f}")
    return metadata


if __name__ == "__main__":
    train_and_save_model()
