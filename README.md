# RAG Backend System - Document QA Assistant

## 🎯 Project Overview

A complete Retrieval-Augmented Generation (RAG) system backend that allows document upload, vector search, and intelligent question answering. Built as a take-home assignment.

**Tech Stack:** FastAPI, FAISS, Ollama (all-minilm), Python 3.11

---

## 📦 Build Progress (7 Commits)

| Commit | Description |
|--------|-------------|
| 1 | Basic FastAPI server with health checks |
| 2 | Document upload endpoint for PDF/TXT |
| 3 | Text extraction + intelligent chunking (900 chars, 180 overlap) |
| 4 | FAISS vector database for semantic search |
| 5 | Requirements.txt with all dependencies |
| 6 | 8 test questions suite |
| 7 | Complete documentation |

---

## ✨ Features

- ✅ **Document Upload** - Support for PDF and TXT files
- ✅ **Smart Chunking** - 900 character chunks with 180 character overlap (20%)
- ✅ **Vector Database** - FAISS for efficient similarity search
- ✅ **Embeddings** - Ollama all-minilm (free, local, no API key needed)
- ✅ **Semantic Search** - Find relevant document chunks
- ✅ **Source Citations** - Track which documents provide answers with similarity scores
- ✅ **Not-Found Handling** - Graceful responses when answer doesn't exist in documents

---

## ✂️ Chunking Strategy

| Parameter | Value | Requirement |
|-----------|-------|-------------|
| **Chunk Size** | 900 characters | Within 800-1000 ✓ |
| **Overlap** | 180 characters | Within 150-200 ✓ |
| **Overlap Percentage** | 20% | Recommended |

**Why this strategy:**
- Preserves context between chunk boundaries
- Prevents cutting in the middle of sentences when possible
- 20% overlap ensures no information loss at cut points
- Sentence boundary preservation improves chunk quality

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Ollama installed (for embeddings) - https://ollama.com

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/Shashini12/rag-backend-assignment.git
cd rag-backend-assignment

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # On Windows
# source venv/bin/activate     # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the embedding model (first time only)
ollama pull all-minilm

# 5. Run Ollama (in a separate terminal window)
ollama serve

# 6. Start the server
python main.py

Server runs at: http://localhost:3000

API Docs (Swagger UI): http://localhost:3000/docs
📡 API Endpoints
Method	Endpoint	Description	Parameters
POST	/documents/upload	Upload PDF/TXT file	file (multipart form-data)
GET	/documents	List all documents	None
GET	/documents/stats	Vector database statistics	None
DELETE	/documents/{document_id}	Delete a document	document_id (path parameter)
POST	/chat/query	Ask a question	question, top_k (optional, default 5)

