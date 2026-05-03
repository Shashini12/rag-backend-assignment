"""
RAG Backend System - Complete with FAISS, Ollama Embeddings, Chat, and DELETE
"""

import os
import uuid
import shutil
import pickle
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import numpy as np
import ollama
import faiss

print("=" * 50)
print("🚀 RAG Backend System with FAISS & Ollama")
print("=" * 50)

app = FastAPI(title="RAG Backend System", version="1.0.0")

# Create folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

# Storage
documents_db = {}
index = None
chunks_storage = []

# FAISS index file paths
FAISS_INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/metadata.pkl"

def load_index():
    global index, chunks_storage
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            chunks_storage = pickle.load(f)
        print(f"✅ Loaded existing index with {len(chunks_storage)} chunks")
    else:
        dimension = 384  # all-minilm dimension
        index = faiss.IndexFlatIP(dimension)
        chunks_storage = []
        print("✅ Created new FAISS index")

load_index()

# ============ Helper Functions ============

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list:
    if not text or len(text.strip()) == 0:
        return []
    
    text = text.strip()
    text_length = len(text)
    
    if text_length <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        break_point = end
        for i in range(end, max(start, end - 100), -1):
            if i < text_length and text[i] in ' .!?\n':
                break_point = i + 1 if text[i] == ' ' else i + 1
                break
        
        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)
        
        start = break_point - overlap
    
    return chunks

def get_embedding(text: str) -> list:
    """Generate embedding using Ollama"""
    try:
        response = ollama.embeddings(
            model="all-minilm",
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(500, f"Ollama error: {str(e)}")

def save_index():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(chunks_storage, f)

# ============ API Endpoints ============

@app.get("/")
async def root():
    return {
        "message": "RAG Backend System is running!",
        "status": "healthy",
        "embedding_model": "all-minilm (Ollama)",
        "total_chunks": len(chunks_storage)
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    global index, chunks_storage
    
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(400, "Only PDF and TXT files are supported")
    
    document_id = f"doc_{uuid.uuid4().hex[:8]}"
    file_path = f"uploads/{document_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_txt(file_path)
    
    if not text or not text.strip():
        os.remove(file_path)
        raise HTTPException(400, "No text could be extracted")
    
    chunks = chunk_text(text)
    
    if not chunks:
        os.remove(file_path)
        raise HTTPException(400, "Text could not be chunked")
    
    print(f"📊 Generating {len(chunks)} embeddings with Ollama...")
    
    for i, chunk in enumerate(chunks):
        print(f"  🔄 Chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        
        embedding_np = np.array([embedding]).astype('float32')
        index.add(embedding_np)
        
        chunks_storage.append({
            "id": f"{document_id}_chunk_{i:04d}",
            "documentId": document_id,
            "fileName": file.filename,
            "text": chunk,
            "chunkIndex": i,
            "uploadedAt": datetime.now().isoformat()
        })
    
    save_index()
    
    documents_db[document_id] = {
        "documentId": document_id,
        "fileName": file.filename,
        "uploadedAt": datetime.now().isoformat(),
        "filePath": file_path,
        "chunkCount": len(chunks),
        "status": "indexed"
    }
    
    print(f"✅ Stored {len(chunks)} chunks in FAISS index")
    
    return {
        "documentId": document_id,
        "fileName": file.filename,
        "chunkCount": len(chunks),
        "total_chunks_in_system": len(chunks_storage),
        "message": f"Stored in FAISS index with {len(chunks)} chunks"
    }

@app.get("/documents")
async def list_documents():
    return {"documents": list(documents_db.values()), "total": len(documents_db)}

@app.get("/documents/stats")
async def get_stats():
    return {
        "total_chunks_in_index": len(chunks_storage),
        "documents_uploaded": len(documents_db),
        "index_size": index.ntotal if index else 0
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the system
    """
    global index, chunks_storage
    
    # Check if document exists
    if document_id not in documents_db:
        raise HTTPException(404, f"Document {document_id} not found")
    
    # Delete the physical file
    file_path = documents_db[document_id].get("filePath")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Deleted file: {file_path}")
    
    # Remove from metadata
    doc_info = documents_db.pop(document_id)
    print(f"✅ Removed document: {document_id}")
    
    # Remove chunks from storage
    original_count = len(chunks_storage)
    chunks_storage = [c for c in chunks_storage if c["documentId"] != document_id]
    removed_count = original_count - len(chunks_storage)
    print(f"✅ Removed {removed_count} chunks from storage")
    
    # Rebuild FAISS index with remaining chunks
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    
    if chunks_storage:
        embeddings_list = []
        for chunk in chunks_storage:
            embedding = get_embedding(chunk["text"])
            embeddings_list.append(embedding)
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            index.add(embeddings_array)
            print(f"✅ Rebuilt index with {len(embeddings_list)} chunks")
    
    # Save the updated index
    save_index()
    
    return {
        "message": f"Document {document_id} deleted successfully",
        "chunks_removed": removed_count,
        "remaining_chunks": len(chunks_storage),
        "remaining_documents": len(documents_db)
    }

@app.post("/chat/query")
async def chat_query(question: str, top_k: int = 5):
    """Ask a question and get an answer based on uploaded documents"""
    import time
    start_time = time.time()
    
    if len(chunks_storage) == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": [],
            "debug": {"retrievedChunks": 0, "latencyMs": 0}
        }
    
    print(f"🔍 Processing question: {question}")
    
    # Generate embedding for the question
    question_embedding = get_embedding(question)
    question_embedding_np = np.array([question_embedding]).astype('float32')
    
    # Search in FAISS
    k = min(top_k, len(chunks_storage))
    distances, indices = index.search(question_embedding_np, k)
    
    # Get retrieved chunks
    retrieved_chunks = []
    sources = []
    
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(chunks_storage):
            chunk_info = chunks_storage[idx]
            retrieved_chunks.append(chunk_info["text"])
            sources.append({
                "fileName": chunk_info["fileName"],
                "chunkId": chunk_info["id"],
                "score": float(1 - distances[0][i])
            })
    
    # Prepare context
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I could not find enough information in the uploaded documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer using Ollama
    try:
        response = ollama.generate(
            model="llama3.2",
            prompt=prompt,
            options={"temperature": 0.3}
        )
        answer = response["response"]
    except:
        answer = f"Found {len(retrieved_chunks)} relevant chunks. To enable AI answers, please run: ollama pull llama3.2"
        sources = []
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "retrievedChunks": len(retrieved_chunks),
            "latencyMs": latency_ms
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)