"""Thread-safe SQLite Clinical User & Diagnostic Study Store (`data/cxr_clinical.db`).

Persists:
1. `users` — Individual user / patient profiles, MRNs, demographics, clinical histories, and UI preferences.
2. `studies` — Complete diagnostic history per user (14-pathology scores, PSPNet CTR, Gemini Decision 1,
   real AI dictation transcripts, ACR reports, and physician attestation/addendum updates).
3. `copilot_messages` — Persistent grounded Gemini Radiology Copilot consultation transcripts per study.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cxr_clinical.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


SEED_USERS: list[dict[str, Any]] = [
    {
        "id": "usr-eleanor-vance",
        "mrn": "MRN-40821",
        "full_name": "Eleanor Vance",
        "email": "eleanor.vance@clinical.local",
        "role": "Patient",
        "age": 68,
        "sex": "Female",
        "setting": "Inpatient Cardiology Service",
        "indication": "Progressive dyspnea on exertion, 3-pillow orthopnea, bilateral pitting ankle edema",
        "comorbidities": "Congestive Heart Failure (HFrEF, EF 32%), Chronic Kidney Disease",
        "smoking": "Never smoker",
        "notes": "Serial cardiothoracic monitoring for HFrEF volume status.",
    },
    {
        "id": "usr-marcus-chen",
        "mrn": "MRN-51904",
        "full_name": "Marcus Chen",
        "email": "marcus.chen@clinical.local",
        "role": "Patient",
        "age": 64,
        "sex": "Male",
        "setting": "Pulmonology Outpatient Clinic",
        "indication": "Chronic dry cough, 14 lb unintentional weight loss over 3 months, mild hemoptysis",
        "comorbidities": "COPD (GOLD Stage II)",
        "smoking": "Active smoker (35 pack-years)",
        "notes": "High-risk thoracic nodule & Fleischner 2017 surveillance.",
    },
    {
        "id": "usr-arthur-reyes",
        "mrn": "MRN-62318",
        "full_name": "Arthur Reyes",
        "email": "arthur.reyes@clinical.local",
        "role": "Patient",
        "age": 72,
        "sex": "Male",
        "setting": "Emergency Department (ED)",
        "indication": "High fever (39.1°C), productive cough with purulent sputum, pleuritic right chest pain",
        "comorbidities": "Type 2 Diabetes, Hypertension",
        "smoking": "Former smoker (quit 12y ago)",
        "notes": "Acute ED evaluation for community-acquired pneumonia.",
    },
    {
        "id": "usr-priya-nair",
        "mrn": "MRN-73105",
        "full_name": "Priya Nair",
        "email": "priya.nair@clinical.local",
        "role": "Patient",
        "age": 45,
        "sex": "Female",
        "setting": "Ambulatory / Outpatient",
        "indication": "Routine preoperative cardiopulmonary clearance",
        "comorbidities": "None documented",
        "smoking": "Never smoker",
        "notes": "Baseline pre-operative screening cohort.",
    },
]


class ClinicalStore:
    """Thread-safe SQLite repository for users, studies, AI transcripts, and Copilot history."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        resolved = db_path or os.environ.get("CXR_DB_PATH") or _DEFAULT_DB_PATH
        self.db_path = Path(resolved)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()
        self._seed_default_users()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA cache_size = -16000;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    mrn TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT DEFAULT '',
                    role TEXT DEFAULT 'Patient',
                    age INTEGER,
                    sex TEXT DEFAULT 'Unspecified',
                    setting TEXT DEFAULT 'Ambulatory / Outpatient',
                    indication TEXT DEFAULT 'Thoracic diagnostic evaluation',
                    comorbidities TEXT DEFAULT '',
                    smoking TEXT DEFAULT '',
                    preferred_theme TEXT DEFAULT 'dark',
                    preferred_threshold REAL DEFAULT 0.50,
                    preferred_colormap TEXT DEFAULT 'turbo',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    study_name TEXT NOT NULL,
                    filename TEXT DEFAULT '',
                    sample_id TEXT,
                    triage_level TEXT DEFAULT 'ROUTINE',
                    critical_alert INTEGER DEFAULT 0,
                    top_pathology TEXT,
                    top_score REAL,
                    ctr REAL,
                    ctr_status TEXT,
                    report_source TEXT DEFAULT 'template',
                    dictation_transcript TEXT DEFAULT '',
                    predictions_json TEXT DEFAULT '[]',
                    anatomy_json TEXT DEFAULT '{}',
                    decision_summary_json TEXT DEFAULT '{}',
                    report_json TEXT DEFAULT '{}',
                    ensemble_info_json TEXT DEFAULT '{}',
                    timings_json TEXT DEFAULT '{}',
                    physician_name TEXT DEFAULT 'Dr. S. Patel, MD',
                    physician_creds TEXT DEFAULT 'Board-Certified Thoracic Radiologist',
                    physician_notes TEXT DEFAULT '',
                    signed_at TEXT,
                    vision_latency_ms REAL DEFAULT 0.0,
                    total_latency_ms REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_studies_user_created
                    ON studies(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS copilot_messages (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provider TEXT,
                    interaction_id TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (study_id) REFERENCES studies(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_copilot_study_created
                    ON copilot_messages(study_id, created_at ASC);
                """
            )

    def _seed_default_users(self) -> None:
        with self._lock, self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count > 0:
                return
            now = _utc_now()
            for u in SEED_USERS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO users (
                        id, mrn, full_name, email, role, age, sex, setting,
                        indication, comorbidities, smoking, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        u["id"],
                        u["mrn"],
                        u["full_name"],
                        u["email"],
                        u["role"],
                        u["age"],
                        u["sex"],
                        u["setting"],
                        u["indication"],
                        u["comorbidities"],
                        u["smoking"],
                        u["notes"],
                        now,
                        now,
                    ),
                )

    # ------------------------------------------------------------------
    # Users CRUD
    # ------------------------------------------------------------------

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.*,
                    COUNT(s.id) AS study_count,
                    MAX(s.created_at) AS last_study_at,
                    (
                        SELECT s2.triage_level
                        FROM studies s2
                        WHERE s2.user_id = u.id
                        ORDER BY s2.created_at DESC
                        LIMIT 1
                    ) AS latest_triage,
                    (
                        SELECT s2.ctr
                        FROM studies s2
                        WHERE s2.user_id = u.id AND s2.ctr IS NOT NULL
                        ORDER BY s2.created_at DESC
                        LIMIT 1
                    ) AS latest_ctr
                FROM users u
                LEFT JOIN studies s ON s.user_id = u.id
                GROUP BY u.id
                ORDER BY COALESCE(MAX(s.created_at), u.updated_at) DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                return None
            user = dict(row)
            studies = self.list_user_studies(user_id, include_payload=False)
            user["study_count"] = len(studies)
            user["studies"] = studies
            user["ctr_history"] = [
                {
                    "study_id": st["id"],
                    "study_name": st["study_name"],
                    "ctr": st["ctr"],
                    "created_at": st["created_at"],
                }
                for st in reversed(studies)
                if st.get("ctr") is not None
            ]
            return user

    def create_user(self, data: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now()
        user_id = data.get("id") or f"usr-{uuid.uuid4().hex[:8]}"
        mrn = (data.get("mrn") or f"MRN-{uuid.uuid4().int % 90000 + 10000}").strip()
        full_name = (data.get("full_name") or "New Patient").strip()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    id, mrn, full_name, email, role, age, sex, setting,
                    indication, comorbidities, smoking, preferred_theme,
                    preferred_threshold, preferred_colormap, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    mrn,
                    full_name,
                    (data.get("email") or "").strip(),
                    (data.get("role") or "Patient").strip(),
                    data.get("age"),
                    (data.get("sex") or "Unspecified").strip(),
                    (data.get("setting") or "Ambulatory / Outpatient").strip(),
                    (data.get("indication") or "Thoracic diagnostic evaluation").strip(),
                    (data.get("comorbidities") or "").strip(),
                    (data.get("smoking") or "").strip(),
                    data.get("preferred_theme") or "dark",
                    float(data.get("preferred_threshold", 0.50)),
                    data.get("preferred_colormap") or "turbo",
                    (data.get("notes") or "").strip(),
                    now,
                    now,
                ),
            )
        return self.get_user(user_id)  # type: ignore[return-value]

    def update_user(self, user_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        allowed = {
            "full_name",
            "mrn",
            "email",
            "role",
            "age",
            "sex",
            "setting",
            "indication",
            "comorbidities",
            "smoking",
            "preferred_theme",
            "preferred_threshold",
            "preferred_colormap",
            "notes",
        }
        updates = {k: v for k, v in patch.items() if k in allowed and v is not None}
        if not updates:
            return self.get_user(user_id)

        updates["updated_at"] = _utc_now()
        cols = ", ".join(f"{k} = ?" for k in updates.keys())
        vals = list(updates.values()) + [user_id]

        with self._lock, self._connect() as conn:
            cur = conn.execute(f"UPDATE users SET {cols} WHERE id = ?", vals)
            if cur.rowcount == 0:
                return None
        return self.get_user(user_id)

    def delete_user(self, user_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Diagnostic Studies & Real AI Transcripts CRUD
    # ------------------------------------------------------------------

    def _hydrate_study(self, row: sqlite3.Row, include_payload: bool = True) -> dict[str, Any]:
        d = dict(row)
        d["critical_alert"] = bool(d.get("critical_alert"))
        if include_payload:
            d["predictions"] = _safe_json_loads(d.pop("predictions_json", None), [])
            d["anatomy"] = _safe_json_loads(d.pop("anatomy_json", None), None)
            d["decision_summary"] = _safe_json_loads(d.pop("decision_summary_json", None), None)
            d["report"] = _safe_json_loads(d.pop("report_json", None), {})
            d["ensemble_info"] = _safe_json_loads(d.pop("ensemble_info_json", None), {})
            d["timings"] = _safe_json_loads(d.pop("timings_json", None), {})
        else:
            for k in (
                "predictions_json",
                "anatomy_json",
                "decision_summary_json",
                "report_json",
                "ensemble_info_json",
                "timings_json",
            ):
                d.pop(k, None)
        return d

    def list_user_studies(self, user_id: str, include_payload: bool = False) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
                (user_id,),
            ).fetchall()
            return [self._hydrate_study(r, include_payload=include_payload) for r in rows]

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
            if not row:
                return None
            study = self._hydrate_study(row, include_payload=True)
            study["copilot_messages"] = self.list_copilot_messages(study_id)
            return study

    def save_study(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now()
        study_id = payload.get("id") or f"std-{uuid.uuid4().hex[:8]}"

        # Ensure target user exists; fallback to first seeded user if unknown ID supplied
        with self._lock, self._connect() as conn:
            exists = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not exists:
                first = conn.execute("SELECT id FROM users ORDER BY created_at ASC LIMIT 1").fetchone()
                if first:
                    user_id = first["id"]
                else:
                    self._seed_default_users()
                    user_id = SEED_USERS[0]["id"]

            predictions = payload.get("predictions") or []
            top_pred = predictions[0] if predictions else {}
            anatomy = payload.get("anatomy") or {}
            # Avoid storing bulky base64 overlay PNG inside SQLite to keep DB fast & lightweight
            anatomy_compact = {k: v for k, v in anatomy.items() if k != "overlay_base64"} if isinstance(anatomy, dict) else {}
            report = payload.get("report") or {}
            decision_summary = payload.get("decision_summary") or {}
            ensemble_info = payload.get("ensemble_info") or {}
            timings = payload.get("timings") or {}

            dictation = (
                payload.get("dictation_transcript")
                or report.get("dictation_transcript")
                or report.get("impression")
                or ""
            )

            conn.execute(
                """
                INSERT INTO studies (
                    id, user_id, study_name, filename, sample_id,
                    triage_level, critical_alert, top_pathology, top_score,
                    ctr, ctr_status, report_source, dictation_transcript,
                    predictions_json, anatomy_json, decision_summary_json,
                    report_json, ensemble_info_json, timings_json,
                    physician_name, physician_creds, physician_notes, signed_at,
                    vision_latency_ms, total_latency_ms, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    user_id,
                    payload.get("study_name") or "Chest Radiograph Study",
                    payload.get("filename") or "",
                    payload.get("sample_id"),
                    report.get("triage_level") or payload.get("triage_level") or "ROUTINE",
                    1 if (report.get("critical_alert") or payload.get("critical_alert")) else 0,
                    top_pred.get("name"),
                    float(top_pred["score"]) if top_pred.get("score") is not None else None,
                    float(anatomy["ctr"]) if anatomy.get("ctr") is not None else None,
                    anatomy.get("ctr_status"),
                    report.get("source") or "template",
                    dictation,
                    json.dumps(predictions, default=str),
                    json.dumps(anatomy_compact, default=str),
                    json.dumps(decision_summary, default=str),
                    json.dumps(report, default=str),
                    json.dumps(ensemble_info, default=str),
                    json.dumps(timings, default=str),
                    payload.get("physician_name") or "Dr. S. Patel, MD",
                    payload.get("physician_creds") or "Board-Certified Thoracic Radiologist",
                    payload.get("physician_notes") or "",
                    payload.get("signed_at"),
                    float(payload.get("vision_latency_ms") or timings.get("core_total_ms") or 0.0),
                    float(payload.get("total_latency_ms") or payload.get("inference_time_ms") or 0.0),
                    now,
                    now,
                ),
            )
            conn.execute("UPDATE users SET updated_at = ? WHERE id = ?", (now, user_id))

        return self.get_study(study_id)  # type: ignore[return-value]

    def update_study(self, study_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        direct_fields = (
            "study_name",
            "triage_level",
            "report_source",
            "dictation_transcript",
            "physician_name",
            "physician_creds",
            "physician_notes",
            "signed_at",
        )
        for f in direct_fields:
            if f in patch and patch[f] is not None:
                updates[f] = patch[f]

        if "report" in patch and isinstance(patch["report"], dict):
            rep = patch["report"]
            updates["report_json"] = json.dumps(rep, default=str)
            if rep.get("source"):
                updates["report_source"] = rep["source"]
            if rep.get("triage_level"):
                updates["triage_level"] = rep["triage_level"]
            if rep.get("dictation_transcript"):
                updates["dictation_transcript"] = rep["dictation_transcript"]
            updates["critical_alert"] = 1 if rep.get("critical_alert") else 0

        if not updates:
            return self.get_study(study_id)

        updates["updated_at"] = _utc_now()
        cols = ", ".join(f"{k} = ?" for k in updates.keys())
        vals = list(updates.values()) + [study_id]

        with self._lock, self._connect() as conn:
            cur = conn.execute(f"UPDATE studies SET {cols} WHERE id = ?", vals)
            if cur.rowcount == 0:
                return None
        return self.get_study(study_id)

    def delete_study(self, study_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM studies WHERE id = ?", (study_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Persistent Grounded Copilot Transcripts
    # ------------------------------------------------------------------

    def list_copilot_messages(self, study_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM copilot_messages WHERE study_id = ? ORDER BY created_at ASC",
                (study_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_copilot_message(
        self,
        study_id: str,
        user_id: str,
        role: str,
        content: str,
        provider: str | None = None,
        interaction_id: str | None = None,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        msg_id = f"msg-{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO copilot_messages (
                    id, study_id, user_id, role, content, provider, interaction_id, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, study_id, user_id, role, content, provider, interaction_id, latency_ms, now),
            )
        return {
            "id": msg_id,
            "study_id": study_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "provider": provider,
            "interaction_id": interaction_id,
            "latency_ms": latency_ms,
            "created_at": now,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            u_cnt = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            s_cnt = conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
            m_cnt = conn.execute("SELECT COUNT(*) FROM copilot_messages").fetchone()[0]
            return {
                "db_path": str(self.db_path),
                "total_users": int(u_cnt),
                "total_studies": int(s_cnt),
                "total_copilot_messages": int(m_cnt),
            }


CLINICAL_DB = ClinicalStore()
