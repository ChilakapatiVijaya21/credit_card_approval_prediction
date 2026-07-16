from __future__ import annotations

from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def score_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Score a credit application with balanced approval logic."""
    cibil_value = payload.get("cibil_score")
    if cibil_value in (None, ""):
        cibil_value = payload.get("credit_score")
    cibil_score = _safe_int(cibil_value, 650)
    debt_to_income = _safe_float(payload.get("debt_to_income"), 0.3)
    income = _safe_float(payload.get("income"), 50000.0)
    loan_balance = _safe_float(payload.get("loan_balance"), 10000.0)
    num_inquiries = _safe_int(payload.get("num_inquiries"), 0)
    prior_defaults = _safe_int(payload.get("prior_defaults"), 0)
    employment_years = _safe_float(payload.get("employment_years"), 3.0)
    has_coapplicant = _safe_int(payload.get("has_coapplicant"), 0)

    # Simple, fair scoring logic
    score = 0.0
    
    # Credit score: main factor (0-90 points)
    score += (cibil_score - 300) / 600 * 90
    
    # Income stability (0-20 points)
    if income >= 50000:
        score += 20
    elif income >= 30000:
        score += 15
    elif income >= 20000:
        score += 10
    
    # Debt-to-income ratio (0-20 points)
    if debt_to_income <= 0.3:
        score += 20
    elif debt_to_income <= 0.5:
        score += 15
    elif debt_to_income <= 0.7:
        score += 10
    
    # Employment stability (0-15 points)
    if employment_years >= 5:
        score += 15
    elif employment_years >= 2:
        score += 10
    elif employment_years >= 1:
        score += 5
    
    # Inquiries penalty (-10 points)
    score -= num_inquiries * 2
    
    # Prior defaults penalty (-30 points each)
    score -= prior_defaults * 30
    
    # Co-applicant bonus (5 points)
    score += has_coapplicant * 5
    
    score = max(score, 0.0)
    score = min(score, 100.0)
    
    # Approval threshold: 40/100
    approval = "Approved" if score >= 40 else "Rejected"
    
    return {
        "cibil_score": cibil_score,
        "score": round(score, 2),
        "approval": approval,
    }
