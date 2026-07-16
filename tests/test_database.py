import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from database import get_applications, init_db, save_application


def test_save_and_fetch_applications(tmp_path):
    db_path = tmp_path / "loan_applications.db"
    init_db(db_path)

    application_id = save_application(
        db_path,
        {
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "9876543210",
            "address": "12 Main Street",
            "income": 60000,
            "debt_to_income": 0.35,
            "credit_score": 780,
            "cibil_score": 760,
            "age": 32,
            "employment_years": 5,
            "loan_balance": 12000,
            "num_inquiries": 1,
            "prior_defaults": 0,
            "income_type": "Salary",
            "employment_status": "Employed",
            "has_coapplicant": 0,
            "approval_status": "Approved",
            "prediction_score": 82.5,
        },
    )

    applications = get_applications(db_path)

    assert application_id is not None
    assert len(applications) == 1
    assert applications[0]["full_name"] == "Jane Doe"
    assert applications[0]["approval_status"] == "Approved"
