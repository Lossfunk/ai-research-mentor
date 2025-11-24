from __future__ import annotations

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from academic_research_mentor.runtime.context import prepare_agent
from academic_research_mentor.session import load_env_file

app = FastAPI(title="Academic Research Mentor API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent reference
agent_instance = None

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    global agent_instance
    
    # Load environment variables
    load_env_file()
    
    # Check API keys
    if not os.environ.get("OPENROUTER_API_KEY") and \
       not os.environ.get("OPENAI_API_KEY") and \
       not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: No API keys found. Agent may fail to initialize.")
    
    # Initialize agent
    print("Initializing Research Mentor Agent...")
    result = prepare_agent(prompt_arg="mentor")
    
    if result.agent:
        agent_instance = result.agent
        print("Agent initialized successfully.")
    else:
        print(f"Agent initialization failed: {result.offline_reason}")

@app.get("/health")
async def health_check():
    global agent_instance
    status = "healthy" if agent_instance else "degraded"
    return {"status": status, "agent_loaded": agent_instance is not None}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global agent_instance
    
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
        
    try:
        # Run the agent synchronously (since the wrapper is sync)
        # In a production app, we might want to run this in a threadpool
        result = agent_instance.run(request.prompt)
        
        # Extract text content from the result object
        content = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
        
        return ChatResponse(response=content)
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
