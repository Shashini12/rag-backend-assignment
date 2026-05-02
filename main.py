"""
RAG Backend System - Part 1: Basic Server
"""

from fastapi import FastAPI

# Create FastAPI app
app = FastAPI(
    title="RAG Backend System",
    description="Document QA system using RAG",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Backend System is running!",
        "status": "healthy",
        "endpoints": [
            "POST /documents/upload - Coming soon",
            "POST /chat/query - Coming soon"
        ]
    }

@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "alive"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)