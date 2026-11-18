from fastapi.testclient import TestClient
import server

client = TestClient(server.app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["patient_risk_screening"] == "transparent_rules_v1"


def test_screen_returns_both_detection_layers():
    response = client.post("/api/screen", json={"text": "Can I take twice my prescribed medication dose?"})
    assert response.status_code == 200
    result = response.json()
    assert "patient_risk" in result
    assert "combined_alert" in result
    assert result["patient_risk"]["detected"] is True
    assert result["combined_alert"]["alert"] is True


def test_screen_rejects_empty_or_overlong_text():
    assert client.post("/api/screen", json={"text": "   "}).status_code == 422
    assert client.post("/api/screen", json={"text": "x" * 4001}).status_code == 422


def test_dashboard_and_extension_guide_are_served():
    assert client.get("/").status_code == 200
    assert "HealthcareGuard" in client.get("/").text
    assert client.get("/extension").status_code == 200
