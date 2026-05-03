@'
# RAG Backend System - Document QA Assistant

## 🎯 Project Overview

A Retrieval-Augmented Generation (RAG) system backend that allows document upload, vector search, and intelligent question answering.

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

## ✨ Features

- ✅ **Document Upload** - Support for PDF and TXT files
- ✅ **Smart Chunking** - 900 character chunks with 180 character overlap (20%)
- ✅ **Vector Database** - FAISS for efficient similarity search
- ✅ **Embeddings** - Ollama all-minilm (free, local)
- ✅ **Semantic Search** - Find relevant document chunks
- ✅ **Source Citations** - Track which documents provide answers
- ✅ **Not-Found Handling** - Graceful responses when answer doesn't exist

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Ollama installed (for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/Shashini12/rag-backend-assignment.git
cd rag-backend-assignment

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Ollama (in a separate terminal)
ollama serve

# Start the server
python main.py
