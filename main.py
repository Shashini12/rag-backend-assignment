"""
RAG Backend System - Part 2: Document Upload
"""

import os
import uuid
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException

# Create FastAPI app
app = FastAPI(
    title="RAG Backend System",
    description="Document QA system using RAG",
    version="1.0.0"
)

# Create uploads folder if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Store document metadata (temporary storage)
documents_db = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Backend System is running!",
        "status": "healthy",
        "endpoints": [
            "POST /documents/upload - Upload PDF/TXT files",
            "GET /documents - List all documents",
            "POST /chat/query - Coming soon"
        ]
    }

@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "alive"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or TXT)
    """
    # Check file type
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(400, "Only PDF and TXT files are supported")
    
    # Generate unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    # Save file
    file_path = f"uploads/{document_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store metadata
    documents_db[document_id] = {
        "documentId": document_id,
        "fileName": file.filename,
        "uploadedAt": datetime.now().isoformat(),
        "filePath": file_path,
        "status": "uploaded"
    }
    
    return {
        "documentId": document_id,
        "fileName": file.filename,
        "message": "Document uploaded successfully"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": list(documents_db.values()),
        "total": len(documents_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)