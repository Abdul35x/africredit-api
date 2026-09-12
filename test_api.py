from fastapi.testclient import TestClient
from app.main import app # Adjust import path if your app instance is elsewhere

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_docs_available():
    # Proves FastAPI Swagger UI is working
    response = client.get("/docs")
    assert response.status_code == 200
