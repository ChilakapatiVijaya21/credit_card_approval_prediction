import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_logic import score_application


def test_higher_cibil_score_increases_approval_score():
    low = score_application({
        "cibil_score": 300,
        "debt_to_income": 0.45,
        "income": 40000,
        "loan_balance": 30000,
        "num_inquiries": 4,
        "prior_defaults": 1,
        "employment_years": 1,
        "has_coapplicant": 0,
        "income_type": "Salary",
        "employment_status": "Employed",
    })
    high = score_application({
        "cibil_score": 800,
        "debt_to_income": 0.45,
        "income": 40000,
        "loan_balance": 30000,
        "num_inquiries": 4,
        "prior_defaults": 1,
        "employment_years": 1,
        "has_coapplicant": 0,
        "income_type": "Salary",
        "employment_status": "Employed",
    })

    assert high["score"] > low["score"]
    assert high["approval"] == "Approved" or high["approval"] == "Rejected"


def test_invalid_values_are_safely_normalized():
    result = score_application({
        "cibil_score": "bad",
        "debt_to_income": "bad",
        "income": "",
        "loan_balance": None,
        "num_inquiries": "x",
        "prior_defaults": "oops",
        "employment_years": "",
        "has_coapplicant": "",
        "income_type": "",
        "employment_status": "",
    })

    assert result["score"] >= 0
    assert result["approval"] in {"Approved", "Rejected"}


def test_credit_score_field_is_used_when_cibil_score_is_missing():
    result = score_application({
        "credit_score": 750,
        "debt_to_income": 0.45,
        "income": 40000,
        "loan_balance": 30000,
        "num_inquiries": 4,
        "prior_defaults": 1,
        "employment_years": 1,
        "has_coapplicant": 0,
    })

    assert result["cibil_score"] == 750


def test_moderate_profile_is_approved():
    result = score_application({
        "cibil_score": 700,
        "debt_to_income": 0.35,
        "income": 60000,
        "loan_balance": 12000,
        "num_inquiries": 2,
        "prior_defaults": 0,
        "employment_years": 4,
        "has_coapplicant": 0,
    })

    assert result["approval"] == "Approved"
