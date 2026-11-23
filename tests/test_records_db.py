"""Unit & API integration tests for the persistent SQLite user/study database & evaluations endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from cxr.db.store import ClinicalStore


def test_sqlite_clinical_store_user_study_and_copilot_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_clinical.db"
        store = ClinicalStore(db_path=db_file)

        # 1. Seeded users exist automatically
        seeded = store.list_users()
        assert len(seeded) >= 4

        # 2. Create a brand-new user record
        created = store.create_user(
            {
                "full_name": "Jordan Rivera",
                "mrn": "MRN-99001",
                "age": 54,
                "sex": "Male",
                "setting": "Emergency Department",
                "indication": "Acute chest pain and dyspnea",
                "comorbidities": "Hypertension",
            }
        )
        assert created["full_name"] == "Jordan Rivera"
        assert created["mrn"] == "MRN-99001"
        user_id = created["id"]

        # 3. Update the user record
        updated = store.update_user(
            user_id,
            {"age": 55, "comorbidities": "Hypertension, Mild Asthma", "notes": "Updated profile"},
        )
        assert updated is not None
        assert updated["age"] == 55
        assert "Asthma" in updated["comorbidities"]

        # 4. Archive a diagnostic study with a real AI transcript
        study = store.save_study(
            user_id,
            {
                "study_name": "NIH Cardiomegaly",
                "filename": "nih_cardiomegaly_00000001.png",
                "predictions": [{"name": "Cardiomegaly", "score": 0.829, "severity": "severe"}],
                "anatomy": {"ctr": 0.529, "ctr_status": "Borderline (0.50-0.53)"},
                "report": {
                    "triage_level": "SEMI_URGENT",
                    "critical_alert": False,
                    "source": "gemini:gemini-3.5-flash-lite",
                    "impression": "Moderate cardiomegaly with CTR 0.529.",
                    "dictation_transcript": "Frontal chest radiograph demonstrates moderate cardiomegaly with CTR 0.529.",
                },
                "vision_latency_ms": 620.0,
                "total_latency_ms": 4200.0,
            },
        )
        assert study["user_id"] == user_id
        assert study["ctr"] == 0.529
        assert "moderate cardiomegaly" in study["dictation_transcript"].lower()
        study_id = study["id"]

        # 5. Update study transcript & physician attestation
        attested = store.update_study(
            study_id,
            {
                "dictation_transcript": "Updated attending dictation confirming cardiomegaly.",
                "physician_name": "Dr. A. Kim, MD",
                "signed_at": "2026-11-24T02:30:00Z",
            },
        )
        assert attested is not None
        assert attested["physician_name"] == "Dr. A. Kim, MD"
        assert attested["signed_at"] == "2026-11-24T02:30:00Z"
        assert "Updated attending dictation" in attested["dictation_transcript"]

        # 6. Persist Copilot consultation transcript
        store.add_copilot_message(study_id, user_id, "user", "Explain the CTR?")
        store.add_copilot_message(
            study_id,
            user_id,
            "assistant",
            "CTR is 0.529, mildly enlarged.",
            provider="gemini:gemini-3.5-flash-lite",
            latency_ms=1800.0,
        )
        msgs = store.list_copilot_messages(study_id)
        assert len(msgs) == 2
        assert msgs[1]["provider"] == "gemini:gemini-3.5-flash-lite"

        # 7. Verify CTR history appears on user detail
        detail = store.get_user(user_id)
        assert detail is not None
        assert len(detail["ctr_history"]) == 1
        assert detail["ctr_history"][0]["ctr"] == 0.529

        # 8. Cascade delete removes user and studies cleanly
        assert store.delete_user(user_id) is True
        assert store.get_user(user_id) is None
        assert store.get_study(study_id) is None


def test_records_and_evaluations_api_endpoints():
    with TestClient(app) as client:
        # List users
        r_users = client.get("/api/v1/users")
        assert r_users.status_code == 200
        assert isinstance(r_users.json(), list)
        assert len(r_users.json()) >= 1

        # Create, update, fetch, and clean up a test user via HTTP API
        r_create = client.post(
            "/api/v1/users",
            json={
                "full_name": "API Test Patient",
                "mrn": "MRN-TEST-7741",
                "age": 61,
                "sex": "Female",
                "setting": "Outpatient",
                "indication": "Routine follow-up",
            },
        )
        assert r_create.status_code == 201
        uid = r_create.json()["id"]

        r_patch = client.patch(
            f"/api/v1/users/{uid}",
            json={"age": 62, "notes": "Verified via API PATCH"},
        )
        assert r_patch.status_code == 200
        assert r_patch.json()["age"] == 62
        assert r_patch.json()["notes"] == "Verified via API PATCH"

        r_del = client.delete(f"/api/v1/users/{uid}")
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] is True

        # Verify DB stats & evaluations endpoints respond cleanly
        r_stats = client.get("/api/v1/db/stats")
        assert r_stats.status_code == 200
        assert "total_users" in r_stats.json()

        r_eval = client.get("/api/v1/evaluations")
        assert r_eval.status_code == 200
