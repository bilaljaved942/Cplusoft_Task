# Local RAG Chatbot API

A fully containerized REST API that implements a Retrieval-Augmented Generation (RAG) system using an open-source, lightweight LLM (`Qwen2.5-0.5B-Instruct`) to answer questions about a candidate's CV/resume with session-based conversational memory.

---

## 🏗️ Project Structure
The project is structured into distinct, modular Python files:
* **`chatbot.py`**: Handles local LLM loading, device auto-selection (supporting Apple Silicon MPS, CUDA, and CPU fallback), and response generation using chat templates.
* **`rag.py`**: Manages document loading (PDF/TXT), custom text cleanup (resolving spacing/layout issues), token chunking, semantic embedding generation (`all-MiniLM-L6-v2`), and cosine-similarity retrieval.
* **`app.py`**: Implements the FastAPI web server exposing the RAG pipeline as a REST API with conversational history memory.
* **`requirements.txt`**: Lists all project dependencies.
* **`Dockerfile`**: Builds a lightweight container to run the REST API.
* **`resume.pdf`**: The target resume analyzed by the RAG system.

---

## 🛠️ Step-by-Step Implementation Details

### Step 1: Open-Source Chatbot
* **LLM**: `Qwen/Qwen2.5-0.5B-Instruct` (~950MB weights, fits easily in memory, extremely fast CPU/MPS inference).
* **Device Mapping**: Checks for Apple Metal (`mps`), CUDA, or defaults to CPU.
* **Prompt Engineering**: Formats messages using Hugging Face's official model chat template.

### Step 2: RAG Pipeline
* **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` (~90MB model).
* **Text Extraction**: Uses `pypdf` to extract text from `resume.pdf`. Includes a regex-based healer to correct text where spacing got added between letters.
* **Retrieval**: Uses NumPy for fast, zero-dependency in-memory cosine-similarity checking.
* **Reliability Check**: Automatically prepends the first chunk (Header/Metadata) to every context retrieval, guaranteeing the model always knows name and contact details.

### Step 3: FastAPI REST API
* **Conversational Memory**: Maintains the last 10 exchanges (20 total user-assistant messages) using in-memory session mapping.
* **Endpoints**:
  * `POST /query`: Submits a question with a `session_id` to query the resume while maintaining session memory.
  * `DELETE /session/{session_id}`: Resets the memory context for a specific session.
  * `GET /health`: Checks server status and document indexing stats.

### Step 4: Dockerization
* **Base Image**: `python:3.10-slim`.
* **Execution**: Runs the entire API server inside a container.

---

## 🚀 How to Run the Project

### Option A: Local Development Run
If you want to run the project directly on your machine:

1. **Activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Uvicorn Server**:
   ```bash
   uvicorn app:app --port 8000
   ```

3. **Interact with Swagger UI**:
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to test endpoints directly.

---

### Option B: Docker Execution (Step 4)

1. **Build the Docker Image**:
   ```bash
   docker build -t rag-chatbot .
   ```

2. **Run the Container (Recommended with Host Cache Mounting)**:
   By default, the container downloads the model weights on startup. To mount your local Hugging Face cache so the container starts **instantly** without redownloading:
   ```bash
   docker run -d -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface --name rag-chatbot-container rag-chatbot
   ```

3. **Verify the container logs**:
   ```bash
   docker logs -f rag-chatbot-container
   ```

---

## 🧪 Example API Usage

### 1. Query the API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the candidate name?",
    "session_id": "demo_session"
  }'
```

### 2. Follow-up Query (Conversational Memory Test)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What did I just ask you?",
    "session_id": "demo_session"
  }'
```

### 3. Clear Session History
```bash
curl -X DELETE http://localhost:8000/session/demo_session
```
