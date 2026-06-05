import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAGSystem

app = FastAPI(
    title="RAG Chatbot API",
    description="A local RAG API that maintains chat context for at least 10 exchanges.",
    version="1.0.0"
)

# Global variables for RAG pipeline
rag = None

# Session history store: maps session_id -> list of message dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
session_memory = {}

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    session_id: str

@app.on_event("startup")
def startup_event():
    global rag
    # Initialize RAG System
    rag = RAGSystem()
    
    # Check for resume.pdf or fallback to sample_cv.txt
    cv_path = "resume.pdf"
    if not os.path.exists(cv_path):
        cv_path = "sample_cv.txt"
        
    if os.path.exists(cv_path):
        try:
            rag.load_document(cv_path)
        except Exception as e:
            print(f"Error loading initial document {cv_path}: {e}")
    else:
        print("Warning: No initial document found. RAG queries will return empty context.")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    global rag
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    session_id = req.session_id
    
    # Retrieve existing history
    history = session_memory.get(session_id, [])
    
    try:
        # Run query through RAG (will automatically include history in prompt context)
        answer, _ = rag.query(req.question, k=2, history=history)
        
        # Update session memory
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})
        
        # Limit context to last 10 exchanges (20 messages total)
        if len(history) > 20:
            history = history[-20:]
            
        session_memory[session_id] = history
        
        return QueryResponse(answer=answer, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in session_memory:
        del session_memory[session_id]
        return {"status": "success", "message": f"Session history for '{session_id}' cleared."}
    return {"status": "ignored", "message": f"No active history found for session '{session_id}'."}

@app.get("/health")
async def health_check():
    global rag
    return {
        "status": "healthy",
        "document_loaded": len(rag.chunks) > 0 if rag else False,
        "total_chunks": len(rag.chunks) if rag else 0
    }
