# Stop the server first (Ctrl+C), then replace main.py
"""
RAG Backend System - Part 3: Text Extraction and Chunking (Fixed)
"""

import os
import uuid
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader

app = FastAPI(title="RAG Backend System", version="1.0.0")

# Create uploads folder
os.makedirs("uploads", exist_ok=True)

# Store document metadata
documents_db = {}

# ============ Helper Functions ============

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(500, f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list:
    """
    Split text into overlapping chunks.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Clean the text
    text = text.strip()
    text_length = len(text)
    
    # If text is smaller than chunk size, return as single chunk
    if text_length <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        if end >= text_length:
            # Last chunk
            chunks.append(text[start:].strip())
            break
        
        # Find a good breaking point (sentence end or space)
        # Look backwards from end to find a space or punctuation
        break_point = end
        for i in range(end, max(start, end - 100), -1):
            if i < text_length and text[i] in ' .!?\n':
                break_point = i + 1 if text[i] == ' ' else i + 1
                break
        
        # Extract chunk
        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = break_point - overlap
        
        # Prevent infinite loop
        if start <= 0 or start >= text_length:
            break
    
    return chunks

# ============ API Endpoints ============

@app.get("/")
async def root():
    return {
        "message": "RAG Backend System is running!",
        "status": "healthy",
        "chunking_config": {
            "chunk_size": 900,
            "overlap": 180
        }
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF or TXT) - Extracts text and creates chunks"""
    
    # Check file type
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(400, "Only PDF and TXT files are supported")
    
    # Generate unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    # Save file
    file_path = f"uploads/{document_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text based on file type
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_txt(file_path)
    
    if not text or not text.strip():
        os.remove(file_path)
        raise HTTPException(400, "No text could be extracted from the file")
    
    # Split into chunks
    chunks = chunk_text(text)
    
    if not chunks:
        os.remove(file_path)
        raise HTTPException(400, "Text could not be chunked properly")
    
    # Store metadata with chunk info
    documents_db[document_id] = {
        "documentId": document_id,
        "fileName": file.filename,
        "uploadedAt": datetime.now().isoformat(),
        "filePath": file_path,
        "chunkCount": len(chunks),
        "totalCharacters": len(text),
        "chunkSize": 900,
        "overlap": 180,
        "status": "chunked",
        "chunks": chunks[:5]  # Store first 5 chunks preview only to save memory
    }
    
    return {
        "documentId": document_id,
        "fileName": file.filename,
        "chunkCount": len(chunks),
        "totalCharacters": len(text),
        "chunkSize": 900,
        "overlap": 180,
        "message": f"Document chunked into {len(chunks)} pieces"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": [
            {
                "documentId": d["documentId"],
                "fileName": d["fileName"],
                "uploadedAt": d["uploadedAt"],
                "chunkCount": d.get("chunkCount", 0),
                "status": d.get("status", "uploaded")
            }
            for d in documents_db.values()
        ],
        "total": len(documents_db)
    }

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details"""
    if document_id not in documents_db:
        raise HTTPException(404, "Document not found")
    
    doc = documents_db[document_id]
    return {
        "documentId": document_id,
        "fileName": doc["fileName"],
        "chunkCount": doc.get("chunkCount", 0),
        "totalCharacters": doc.get("totalCharacters", 0),
        "status": doc.get("status", "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
