import time
import json
import urllib.request
import urllib.error

url = "http://localhost:8000"

def get_health():
    try:
        with urllib.request.urlopen(f"{url}/health") as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return data
    except Exception:
        pass
    return None

def post_query(question, session_id):
    data = json.dumps({"question": question, "session_id": session_id}).encode('utf-8')
    req = urllib.request.Request(
        f"{url}/query",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

# Wait for server to load models
print("Waiting for server to become healthy (loading LLM)...")
for _ in range(45):
    health = get_health()
    if health and health.get("document_loaded"):
        print("Server is ready!")
        print(health)
        break
    time.sleep(2)
else:
    print("Server took too long to load or failed to start.")
    exit(1)

# Query 1: Ask about the candidate name
print("\n--- Sending Query 1 ---")
q1 = "What is the candidate's name?"
res1 = post_query(q1, "test_session")
print("Q:", q1)
print("A:", res1["answer"])

# Query 2: Ask a follow-up that requires memory
print("\n--- Sending Query 2 (Conversational Memory Test) ---")
q2 = "What did I just ask you?"
res2 = post_query(q2, "test_session")
print("Q:", q2)
print("A:", res2["answer"])
