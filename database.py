from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "loan_approval.db"


def init_db(db_path: Optional[Path | str] = None) -> Path:
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            income REAL,
            debt_to_income REAL,
            credit_score INTEGER,
            cibil_score INTEGER,
            age INTEGER,
            employment_years REAL,
            loan_balance REAL,
            num_inquiries INTEGER,
            prior_defaults INTEGER,
            income_type TEXT,
            employment_status TEXT,
            has_coapplicant INTEGER,
            approval_status TEXT,
            prediction_score REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    connection.close()
    return path


def save_application(db_path: Optional[Path | str], payload: Dict[str, Any]) -> int:
    path = init_db(db_path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            INSERT INTO loan_applications (
                full_name, email, phone, address, income, debt_to_income,
                credit_score, cibil_score, age, employment_years, loan_balance,
                num_inquiries, prior_defaults, income_type, employment_status,
                has_coapplicant, approval_status, prediction_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("full_name", ""),
                payload.get("email", ""),
                payload.get("phone", ""),
                payload.get("address", ""),
                payload.get("income"),
                payload.get("debt_to_income"),
                payload.get("credit_score"),
                payload.get("cibil_score"),
                payload.get("age"),
                payload.get("employment_years"),
                payload.get("loan_balance"),
                payload.get("num_inquiries"),
                payload.get("prior_defaults"),
                payload.get("income_type"),
                payload.get("employment_status"),
                payload.get("has_coapplicant"),
                payload.get("approval_status"),
                payload.get("prediction_score"),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def get_applications(db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    path = init_db(db_path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM loan_applications ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
