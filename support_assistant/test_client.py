from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_calls():
    # Call 1: Policy question (triggers retrieval)
    req1 = {"query": "What is Zepto's delivery fee and delivery policy?"}
    res1 = client.post("/ask", json=req1)
    print("=== CALL 1: POLICY QUERY (RETRIEVAL TRIGGERED) ===")
    print(f"Status: {res1.status_code}")
    print(res1.json())

    # Call 2: General question (triggers direct answer)
    req2 = {"query": "What is the capital of France?"}
    res2 = client.post("/ask", json=req2)
    print("\n=== CALL 2: GENERAL QUERY (DIRECT ANSWER) ===")
    print(f"Status: {res2.status_code}")
    print(res2.json())

if __name__ == "__main__":
    test_calls()