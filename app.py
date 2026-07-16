import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from credit_logic import score_application
from database import get_applications, init_db, save_application

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

app = Flask(__name__)
DB_PATH = ROOT / "data" / "loan_approval.db"


def ensure_model_ready():
    if not MODEL_PATH.exists():
        import train_model

        train_model.train_and_save_model()
    return MODEL_PATH, METADATA_PATH


@app.route("/", methods=["GET", "POST"])
def index():
    model_path, metadata_path = ensure_model_ready()
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    init_db(DB_PATH)

    result = None
    applications = get_applications(DB_PATH)
    if request.method == "POST":
        credit_score = float(request.form.get("credit_score", 0))
        cibil_score_value = request.form.get("cibil_score")
        cibil_score = int(cibil_score_value) if cibil_score_value not in (None, "") else int(credit_score)
        payload = {
            "income": float(request.form.get("income", 0)),
            "debt_to_income": float(request.form.get("debt_to_income", 0)),
            "credit_score": credit_score,
            "age": int(request.form.get("age", 0)),
            "employment_years": float(request.form.get("employment_years", 0)),
            "loan_balance": float(request.form.get("loan_balance", 0)),
            "num_inquiries": int(request.form.get("num_inquiries", 0)),
            "prior_defaults": int(request.form.get("prior_defaults", 0)),
            "income_type": request.form.get("income_type", "Salary"),
            "employment_status": request.form.get("employment_status", "Employed"),
            "has_coapplicant": int(request.form.get("has_coapplicant", 0)),
            "cibil_score": cibil_score,
        }
        risk_result = score_application(payload)
        result = {
            "prediction": risk_result["approval"],
            "probability": round(risk_result["score"], 1),
            "model": metadata.get("model_name", "unknown"),
            "score": risk_result["score"],
            "cibil_score": risk_result["cibil_score"],
            "reason": "Your CIBIL score and repayment profile are strong enough for approval." if risk_result["approval"] == "Approved" else "Your profile shows elevated risk due to low credit quality, debt burden, or prior defaults.",
        }

        save_application(
            DB_PATH,
            {
                "full_name": request.form.get("full_name", "").strip(),
                "email": request.form.get("email", "").strip(),
                "phone": request.form.get("phone", "").strip(),
                "address": request.form.get("address", "").strip(),
                **payload,
                "approval_status": risk_result["approval"],
                "prediction_score": risk_result["score"],
            },
        )
        applications = get_applications(DB_PATH)

    return render_template("index.html", result=result, metadata=metadata, applications=applications)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
