from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
PDF = "tests/samples/medium_data_analyst.pdf"


def test_health_and_roles():
    assert client.get("/health").json()["status"] == "ok"
    assert "Data Analyst" in client.get("/roles").json()["roles"]


def test_analyze_business_analyst_six_weeks():
    with open(PDF, "rb") as f:
        r = client.post("/analyze", files={"resume": ("r.pdf", f, "application/pdf")},
                        data={"target_role": "Business Analyst", "weeks_available": "6"})
    assert r.status_code == 200
    assert len(r.json()["roadmap"]["weeks"]) == 6


def test_non_pdf_is_415():
    r = client.post("/analyze", files={"resume": ("r.txt", b"hello", "text/plain")}, data={"target_role": "Data Analyst"})
    assert r.status_code == 415


def test_bad_language_is_422():
    with open(PDF, "rb") as f:
        r = client.post("/analyze", files={"resume": ("r.pdf", f, "application/pdf")},
                        data={"target_role": "Data Analyst", "language": "fr"})
    assert r.status_code == 422


def test_stream_has_steps_and_result():
    with open(PDF, "rb") as f:
        r = client.post("/analyze/stream", files={"resume": ("r.pdf", f, "application/pdf")},
                        data={"target_role": "Data Analyst"})
    assert r.text.count("event: step") >= 10
    assert r.text.count("event: result") == 1
