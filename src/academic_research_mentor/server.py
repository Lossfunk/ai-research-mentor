from __future__ import annotations

import os
import json
import base64
import tempfile
from typing import Optional, AsyncIterator
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from supermemory import Supermemory

from academic_research_mentor.runtime.context import prepare_agent
from academic_research_mentor.cli.session import load_env_file

app = FastAPI(title="Academic Research Mentor API")

# In-memory document store (for context)
document_store: dict[str, dict] = {}

# Direct OpenAI client for streaming with reasoning support
openai_client: Optional[AsyncOpenAI] = None

# Supermemory client for persistent memory
supermemory_client: Optional[Supermemory] = None

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
    document_context: Optional[str] = None  # Selected document content

class ChatResponse(BaseModel):
    response: str
    reasoning: Optional[str] = None

class UploadResponse(BaseModel):
    id: str
    filename: str
    content: str
    pages: Optional[int] = None


def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # pymupdf
    
    text_parts = []
    doc = fitz.open(file_path)
    page_count = len(doc)
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    
    doc.close()
    return "\n\n".join(text_parts), page_count


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx if available, otherwise return placeholder."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        return "[DOCX parsing requires python-docx. Please install it or upload a different format.]"


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and parse a document (PDF, DOCX, TXT, MD)."""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename.lower()
    file_ext = filename.split('.')[-1] if '.' in filename else ''
    
    allowed_extensions = {'pdf', 'docx', 'txt', 'md'}
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    content_bytes = await file.read()
    
    extracted_text = ""
    pages = None
    
    try:
        if file_ext == 'pdf':
            # Save to temp file for PDF processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            
            try:
                extracted_text, pages = extract_text_from_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
                
        elif file_ext == 'docx':
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            
            try:
                extracted_text = extract_text_from_docx(tmp_path)
            finally:
                os.unlink(tmp_path)
                
        elif file_ext in ('txt', 'md'):
            # Try UTF-8 first, then fallback to latin-1
            try:
                extracted_text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                extracted_text = content_bytes.decode('latin-1')
    
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")
    
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")
    
    # Generate document ID
    doc_id = f"doc-{len(document_store) + 1}"
    
    # Store in memory
    document_store[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "content": extracted_text,
        "pages": pages,
    }
    
    # Store in Supermemory for persistent context
    store_in_supermemory(doc_id, file.filename, extracted_text)
    
    return UploadResponse(
        id=doc_id,
        filename=file.filename,
        content=extracted_text,
        pages=pages,
    )


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    return {
        "documents": [
            {"id": d["id"], "filename": d["filename"], "pages": d.get("pages")}
            for d in document_store.values()
        ]
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    del document_store[doc_id]
    return {"status": "deleted", "id": doc_id}


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/api/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search Supermemory for relevant context."""
    results = search_supermemory(request.query, request.limit)
    return {"results": results, "count": len(results)}


@app.get("/api/memory/status")
async def memory_status():
    """Get Supermemory connection status."""
    global supermemory_client
    return {
        "connected": supermemory_client is not None,
        "provider": "supermemory" if supermemory_client else None
    }

def store_in_supermemory(doc_id: str, filename: str, content: str) -> bool:
    """Store a document in Supermemory for persistent memory."""
    global supermemory_client
    if not supermemory_client:
        return False
    
    try:
        # Add document to Supermemory
        supermemory_client.memory.add(
            content=content,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "type": "research_document",
            }
        )
        print(f"Stored document {filename} in Supermemory")
        return True
    except Exception as e:
        print(f"Failed to store in Supermemory: {e}")
        return False


def search_supermemory(query: str, limit: int = 5) -> list[dict]:
    """Search Supermemory for relevant context."""
    global supermemory_client
    if not supermemory_client:
        return []
    
    try:
        response = supermemory_client.search.execute(q=query, limit=limit)
        results = []
        for result in response.results or []:
            results.append({
                "content": getattr(result, "content", str(result)),
                "metadata": getattr(result, "metadata", {}),
                "score": getattr(result, "score", 0),
            })
        return results
    except Exception as e:
        print(f"Supermemory search failed: {e}")
        return []


@app.on_event("startup")
async def startup_event():
    global agent_instance, openai_client, supermemory_client
    
    # Load environment variables
    load_env_file()
    
    # Check API keys
    if not os.environ.get("OPENROUTER_API_KEY") and \
       not os.environ.get("OPENAI_API_KEY") and \
       not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: No API keys found. Agent may fail to initialize.")
    
    # Initialize Supermemory client
    supermemory_api_key = os.environ.get("SUPERMEMORY_API_KEY")
    if supermemory_api_key:
        try:
            supermemory_client = Supermemory(
                api_key=supermemory_api_key,
                base_url="https://api.supermemory.ai/"
            )
            print("Supermemory client initialized")
        except Exception as e:
            print(f"Failed to initialize Supermemory: {e}")
            supermemory_client = None
    else:
        print("SUPERMEMORY_API_KEY not set - memory features disabled")
    
    # Initialize direct OpenAI client for streaming (bypasses LangChain for reasoning support)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
        openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        print(f"OpenAI client initialized (base_url: {base_url or 'default'})")
    
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
        reasoning = getattr(result, "reasoning", None) or getattr(result, "scratchpad", None)

        # If no explicit reasoning field, attempt to pull from structured message if present
        if reasoning is None and hasattr(result, "response"):
            try:
                msg = getattr(result, "response")
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_details", None)
            except Exception:
                reasoning = None

        # Fallback: wrap internal <thinking> tag if present in content
        if reasoning is None and isinstance(content, str) and "<thinking>" in content:
            import re
            match = re.search(r"<thinking>([\s\S]*?)</thinking>", content)
            if match:
                reasoning = match.group(1).strip()

        return ChatResponse(response=content, reasoning=reasoning)
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """SSE streaming endpoint with reasoning token support.
    
    Sends events in format:
    - data: {"type": "reasoning", "content": "..."} for reasoning tokens
    - data: {"type": "content", "content": "..."} for response tokens
    - data: {"type": "done"} when complete
    """
    global openai_client, agent_instance
    
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")

    async def sse_generator() -> AsyncIterator[str]:
        try:
            model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
            
            # Get system prompt from agent if available
            system_prompt = "You are a helpful research mentor."
            if agent_instance and hasattr(agent_instance, "_system_instructions"):
                system_prompt = agent_instance._system_instructions
            
            # Build messages array
            messages = [{"role": "system", "content": system_prompt}]
            
            # Combine context from selected documents and Supermemory search
            context_parts = []
            
            # 1. Add selected document context if provided
            if request.document_context and request.document_context.strip():
                context_parts.append(f"<selected_documents>\n{request.document_context[:40000]}\n</selected_documents>")
            
            # 2. Search Supermemory for relevant past context
            memory_results = search_supermemory(request.prompt, limit=3)
            if memory_results:
                memory_context = "\n\n".join([
                    f"[Memory - {r.get('metadata', {}).get('filename', 'unknown')}]\n{r['content'][:5000]}"
                    for r in memory_results
                ])
                context_parts.append(f"<memory_context>\n{memory_context}\n</memory_context>")
            
            # Add combined context if any
            if context_parts:
                combined_context = "\n\n".join(context_parts)
                context_message = f"""The following context is available from documents and memory. Use it to inform your responses when relevant:

{combined_context}

Now respond to the user's question, citing relevant parts when applicable."""
                messages.append({"role": "user", "content": context_message})
                messages.append({"role": "assistant", "content": "I've reviewed the available context. How can I help you?"})
            
            # Add the actual user message
            messages.append({"role": "user", "content": request.prompt})
            
            # Build extra parameters for OpenRouter reasoning
            extra_body = {}
            if os.environ.get("OPENROUTER_API_KEY"):
                extra_body["include_reasoning"] = True
                
                if os.environ.get("OPENROUTER_USE_RESPONSES_API") == "1":
                    reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT", "medium")
                    extra_body["reasoning"] = {"effort": reasoning_effort}
            
            # Create streaming completion
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body=extra_body if extra_body else None,
            )
            
            first_chunk = True
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Debug: Log first chunk structure to understand format
                if first_chunk:
                    print(f"DEBUG: First chunk type: {type(chunk)}")
                    print(f"DEBUG: First chunk attrs: {[a for a in dir(chunk) if not a.startswith('_')]}")
                    print(f"DEBUG: First choice type: {type(choice)}")
                    print(f"DEBUG: First choice attrs: {[a for a in dir(choice) if not a.startswith('_')]}")
                    print(f"DEBUG: First delta type: {type(delta)}")
                    print(f"DEBUG: First delta attrs: {[a for a in dir(delta) if not a.startswith('_')]}")
                    if hasattr(chunk, "model_extra"):
                        print(f"DEBUG: chunk.model_extra: {chunk.model_extra}")
                    if hasattr(choice, "model_extra"):
                        print(f"DEBUG: choice.model_extra: {choice.model_extra}")
                    if hasattr(delta, "model_extra"):
                        print(f"DEBUG: delta.model_extra: {delta.model_extra}")
                    first_chunk = False
                
                # Check for reasoning_content in multiple possible locations
                reasoning_content = None
                
                # Method 1: Direct attribute on delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content = delta.reasoning_content
                
                # Method 2: delta.model_extra dict
                if not reasoning_content and hasattr(delta, "model_extra") and delta.model_extra:
                    reasoning_content = delta.model_extra.get("reasoning_content") or delta.model_extra.get("reasoning")
                
                # Method 3: choice level
                if not reasoning_content and hasattr(choice, "model_extra") and choice.model_extra:
                    reasoning_content = choice.model_extra.get("reasoning_content") or choice.model_extra.get("reasoning")
                
                # Method 4: chunk level  
                if not reasoning_content and hasattr(chunk, "model_extra") and chunk.model_extra:
                    reasoning_content = chunk.model_extra.get("reasoning_content") or chunk.model_extra.get("reasoning")
                
                # Method 5: Check for 'reasoning' attribute directly
                if not reasoning_content and hasattr(choice, "reasoning") and choice.reasoning:
                    reasoning_content = choice.reasoning
                
                if reasoning_content:
                    event = {"type": "reasoning", "content": reasoning_content}
                    yield f"data: {json.dumps(event)}\n\n"
                
                # Regular content
                if delta.content:
                    # Handle structured content (list of blocks)
                    content = delta.content
                    if isinstance(content, list):
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                texts.append(block)
                        content = "".join(texts)
                    
                    if content:
                        event = {"type": "content", "content": content}
                        yield f"data: {json.dumps(event)}\n\n"
            
            # Send done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"Streaming error: {e}")
            error_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# Legacy plain text streaming (fallback)
@app.post("/api/chat/stream/legacy")
async def chat_stream_legacy_endpoint(request: ChatRequest):
    global agent_instance
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def token_generator():
        async for chunk in agent_instance.stream_tokens(request.prompt):  # type: ignore[attr-defined]
            yield chunk

    return StreamingResponse(token_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
