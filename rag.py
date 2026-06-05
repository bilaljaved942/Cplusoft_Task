import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from chatbot import LocalChatbot

class RAGSystem:
    def __init__(self, chatbot=None, embedding_model_name="all-MiniLM-L6-v2"):
        # Set up chatbot
        self.chatbot = chatbot if chatbot is not None else LocalChatbot()
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        self.chunks = []
        self.embeddings = None

    def load_document(self, file_path, chunk_size=500, chunk_overlap=100):
        """
        Loads a document (txt or pdf), chunks it, and generates embeddings.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")
            
        print(f"Loading document from {file_path}...")
        text = ""
        
        # Parse based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            # Default to text file parsing
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
        # Clean spaced-out text layout if present
        text = self._clean_spaced_text(text)
        
        # Simple chunking
        self.chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        print(f"Split document into {len(self.chunks)} chunks.")
        
        # Compute embeddings
        print("Generating embeddings for document chunks...")
        self.embeddings = self.embedder.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
        print("Embeddings generated successfully.")

    def _clean_spaced_text(self, text):
        """
        Heuristic to detect and fix text extractions that have spaces between every letter.
        """
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            # Count non-spaces and spaces
            non_spaces = [c for c in line if c != ' ']
            spaces = [c for c in line if c == ' ']
            if len(non_spaces) > 0 and len(spaces) / len(non_spaces) > 0.6:
                # Heavily spaced line: replace double spaces with a temporary token,
                # strip all single spaces, and replace temporary token back with a single space.
                temp = line.replace("  ", " _WS_ ")
                temp = temp.replace(" ", "")
                fixed_line = temp.replace("_WS_", " ").strip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def _chunk_text(self, text, chunk_size, chunk_overlap):
        """
        Splits text into chunks of roughly chunk_size characters, with overlap.
        Preserves paragraph structure where possible.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single paragraph is larger than chunk_size, split it by sentences or characters
                if len(para) > chunk_size:
                    start = 0
                    while start < len(para):
                        end = start + chunk_size
                        chunks.append(para[start:end].strip())
                        start += (chunk_size - chunk_overlap)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Fallback if no chunks generated
        if not chunks:
            chunks = [text.strip()]
            
        return chunks

    def retrieve(self, query, k=3):
        """
        Retrieves the top k chunks semantically closest to the query.
        """
        if not self.chunks or self.embeddings is None:
            return []
            
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarities
        # Cosine similarity = dot_product(A, B) / (norm(A) * norm(B))
        similarities = []
        for emb in self.embeddings:
            dot_prod = np.dot(query_embedding, emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            similarity = dot_prod / (norm_q * norm_e) if norm_q > 0 and norm_e > 0 else 0.0
            similarities.append(similarity)
            
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved = []
        for idx in top_k_indices:
            retrieved.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return retrieved

    def query(self, user_query, k=3, history=None):
        """
        Retrieves relevant context from the document and answers the question.
        """
        retrieved_items = self.retrieve(user_query, k=k)
        
        if not retrieved_items:
            context = "No document loaded or no relevant context found."
        else:
            context = "\n---\n".join([item["chunk"] for item in retrieved_items])
            
        system_prompt = (
            "You are a helpful, professional AI assistant. You answer questions using the provided context from the user's CV.\n"
            "If the answer cannot be found in the context, state that you do not know based on the provided CV, but do not make up facts.\n\n"
            f"Provided CV Context:\n{context}"
        )
        
        # Generate answer using local chatbot
        response = self.chatbot.generate_response(user_query, system_prompt=system_prompt, history=history)
        return response, retrieved_items

def main():
    print("=" * 60)
    print("Starting Local RAG System Interactive Session")
    print("=" * 60)
    
    rag = RAGSystem()
    
    # Load default CV
    cv_path = "resume.pdf"
    if os.path.exists(cv_path):
        rag.load_document(cv_path)
    else:
        print(f"Warning: Default CV '{cv_path}' not found. Please place your CV in the directory.")
        
    print("\nRAG System is ready! Type 'exit' or 'quit' to end the session.\n")
    
    history = []
    
    while True:
        try:
            user_input = input("\033[94mAsk about the CV:\033[0m ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("\nSearching context...")
            response, sources = rag.query(user_input, k=2, history=history)
            
            print("\n\033[92mAnswer:\033[0m")
            print(response)
            
            print("\n\033[90m[Retrieved Context Sources]:\033[0m")
            for i, src in enumerate(sources, 1):
                print(f"Source {i} (Similarity: {src['score']:.4f}):\n{src['chunk']}")
                print("-" * 40)
            print()
            
            # Keep history updated (limit history length to last 10 exchanges)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            if len(history) > 20:
                history = history[-20:]
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
