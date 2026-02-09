"""FastAPI server for Research Mentor - uses direct OpenAI SDK."""

from __future__ import annotations

import os
import json
import tempfile
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supermemory import Supermemory

from academic_research_mentor.agent import MentorAgent, ToolRegistry, create_default_tools
from academic_research_mentor.llm import create_client
from academic_research_mentor.llm.types import Message

app = FastAPI(title="Academic Research Mentor API")

# Global instances
mentor_agent: Optional[MentorAgent] = None
supermemory_client: Optional[Supermemory] = None
document_store: dict[str, dict] = {}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class ChatRequest(BaseModel):
    prompt: str
    document_context: Optional[str] = None
    content_parts: Optional[list] = None  # structured content for vision (text + image_url parts)

class ChatResponse(BaseModel):
    response: str
    reasoning: Optional[str] = None

class UploadResponse(BaseModel):
    id: str
    filename: str
    content: str
    pages: Optional[int] = None

class TitleRequest(BaseModel):
    text: str

class TitleResponse(BaseModel):
    title: str

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 5


# --- Document Parsing ---

def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """Extract text from PDF using PyMuPDF."""
    import fitz
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
    """Extract text from DOCX."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return "[DOCX parsing requires python-docx]"


# --- Supermemory Helpers ---

def store_in_supermemory(doc_id: str, filename: str, content: str) -> bool:
    """Store document in Supermemory."""
    if not supermemory_client:
        return False
    try:
        supermemory_client.memory.add(
            content=content,
            metadata={"doc_id": doc_id, "filename": filename, "type": "research_document"}
        )
        return True
    except Exception as e:
        print(f"Supermemory store failed: {e}")
        return False


def search_supermemory(query: str, limit: int = 5) -> list[dict]:
    """Search Supermemory for context."""
    if not supermemory_client:
        return []
    try:
        response = supermemory_client.search.execute(q=query, limit=limit)
        return [
            {"content": getattr(r, "content", str(r)), "metadata": getattr(r, "metadata", {})}
            for r in (response.results or [])
        ]
    except Exception as e:
        print(f"Supermemory search failed: {e}")
        return []


async def search_supermemory_async(query: str, limit: int = 5) -> list[dict]:
    return await run_in_threadpool(search_supermemory, query, limit)


async def store_in_supermemory_async(doc_id: str, filename: str, content: str) -> bool:
    return await run_in_threadpool(store_in_supermemory, doc_id, filename, content)


async def build_context_with_memory(prompt: str, document_context: Optional[str]) -> Optional[str]:
    context = document_context or ""
    memory_results = await search_supermemory_async(prompt, limit=3)
    if memory_results:
        memory_ctx = "\n".join(f"[Memory] {r['content'][:2000]}" for r in memory_results)
        context = f"{context}\n\n{memory_ctx}" if context else memory_ctx
    return context if context else None


def _clean_title(raw: str) -> str:
    """Normalize model-produced titles."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\n", " ").strip()
    # Collapse multiple spaces
    cleaned = " ".join(cleaned.split())
    # Trim to 60 chars to fit UI
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rstrip() + "…"
    return cleaned or ""


async def generate_title_from_text(text: str) -> str:
    """Use the mentor agent's client to generate a short semantic title."""
    prompt = (
        "Generate a concise, descriptive chat title (5-8 words). "
        "No quotes, no prefix, no markdown, just the title text. "
        f"User message:\n{text}"
    )
    try:
        if mentor_agent and mentor_agent.client:
            messages = [
                Message.system("You create short, descriptive chat titles. Reply with title only."),
                Message.user(prompt)
            ]
            resp_msg, _ = await mentor_agent.client.chat_async(
                messages,
                max_tokens=16,
                temperature=0.3,
            )
            if isinstance(resp_msg.content, str):
                title = _clean_title(resp_msg.content)
                if title:
                    return title
    except Exception as e:
        print(f"Title generation failed: {e}")
    # Fallback: heuristic clip
    fallback = text.strip()
    return (fallback[:60] + ("…" if len(fallback) > 60 else "")) or "Untitled chat"


# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    """Initialize agent and clients."""
    global mentor_agent, supermemory_client
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize Supermemory
    sm_key = os.environ.get("SUPERMEMORY_API_KEY")
    if sm_key:
        try:
            supermemory_client = Supermemory(api_key=sm_key, base_url="https://api.supermemory.ai/")
            print("Supermemory initialized")
        except Exception as e:
            print(f"Supermemory init failed: {e}")
    
    # Load system prompt
    system_prompt = "You are a helpful research mentor."
    try:
        from academic_research_mentor.prompts_loader import load_instructions_from_prompt_md
        instructions, _ = load_instructions_from_prompt_md("mentor", ascii_normalize=False)
        if instructions:
            system_prompt = instructions
    except Exception as e:
        print(f"Failed to load prompt: {e}")
    
    # Initialize tools
    tool_registry = ToolRegistry()
    tools_initialized = []
    try:
        for tool in create_default_tools():
            tool_registry.register(tool)
            tools_initialized.append(tool.name)
        print(f"Tools initialized: {', '.join(tools_initialized)}")
    except Exception as e:
        print(f"Tool init warning: {e}")
    
    # Create agent with tools
    try:
        client = create_client(provider="openrouter")
        mentor_agent = MentorAgent(
            system_prompt=system_prompt,
            client=client,
            tools=tool_registry if len(tool_registry) > 0 else None
        )
        print(f"Mentor agent initialized with {len(tool_registry)} tools")
    except Exception as e:
        print(f"Agent init failed: {e}")


@app.get("/health")
async def health():
    tools_count = len(mentor_agent.tools) if mentor_agent and mentor_agent.tools else 0
    tool_names = [t.name for t in mentor_agent.tools.tools] if mentor_agent and mentor_agent.tools else []
    return {
        "status": "healthy" if mentor_agent else "degraded",
        "agent_loaded": mentor_agent is not None,
        "tools_count": tools_count,
        "tools": tool_names
    }


@app.get("/api/tools")
async def list_tools():
    """List available tools."""
    if not mentor_agent or not mentor_agent.tools:
        return {"tools": [], "count": 0}
    
    tools_info = []
    for tool in mentor_agent.tools.tools:
        tools_info.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        })
    
    return {"tools": tools_info, "count": len(tools_info)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    if not mentor_agent:
        raise HTTPException(503, "Agent not initialized")
    
    try:
        context = await build_context_with_memory(request.prompt, request.document_context)
        user_payload = request.content_parts if request.content_parts else request.prompt
        response = await mentor_agent.chat_async(user_payload, context=context)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming endpoint."""
    if not mentor_agent:
        raise HTTPException(503, "Agent not initialized")
    
    async def sse_generator() -> AsyncIterator[str]:
        context = await build_context_with_memory(request.prompt, request.document_context)
        try:
            user_payload = request.content_parts if request.content_parts else request.prompt
            async for chunk in mentor_agent.stream_async(
                user_payload,
                context=context,
                include_reasoning=True
            ):
                # Handle tool status events
                if chunk.tool_status:
                    yield f"data: {json.dumps({'type': 'tool', 'status': chunk.tool_status, 'name': chunk.tool_name, 'result': chunk.tool_result})}\n\n"
                # Handle reasoning
                if chunk.reasoning:
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk.reasoning})}\n\n"
                # Handle content
                if chunk.content and not chunk.tool_status:  # Don't emit content for tool status messages
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload and parse a document."""
    if not file.filename:
        raise HTTPException(400, "No filename")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in {'pdf', 'docx', 'txt', 'md'}:
        raise HTTPException(400, f"Unsupported type: {ext}")
    
    content_bytes = await file.read()
    text, pages = "", None
    
    try:
        if ext == 'pdf':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            try:
                text, pages = await run_in_threadpool(extract_text_from_pdf, tmp_path)
            finally:
                os.unlink(tmp_path)
        elif ext == 'docx':
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name
            try:
                text = await run_in_threadpool(extract_text_from_docx, tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            text = content_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        raise HTTPException(500, f"Parse failed: {e}")
    
    if not text.strip():
        raise HTTPException(400, "No text extracted")
    
    doc_id = f"doc-{len(document_store) + 1}"
    document_store[doc_id] = {"id": doc_id, "filename": file.filename, "content": text, "pages": pages}
    await store_in_supermemory_async(doc_id, file.filename, text)
    
    return UploadResponse(id=doc_id, filename=file.filename, content=text, pages=pages)


@app.get("/api/documents")
async def list_documents():
    return {"documents": [{"id": d["id"], "filename": d["filename"], "pages": d.get("pages")} for d in document_store.values()]}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in document_store:
        raise HTTPException(404, "Not found")
    del document_store[doc_id]
    return {"status": "deleted", "id": doc_id}


@app.post("/api/memory/search")
async def search_memory(request: MemorySearchRequest):
    results = await search_supermemory_async(request.query, request.limit)
    return {"results": results, "count": len(results)}


@app.get("/api/memory/status")
async def memory_status():
    return {"connected": supermemory_client is not None, "provider": "supermemory" if supermemory_client else None}


@app.post("/api/chat/title", response_model=TitleResponse)
async def chat_title(req: TitleRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text required")
    title = await generate_title_from_text(req.text)
    return TitleResponse(title=title)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
