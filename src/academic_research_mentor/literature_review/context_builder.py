"""Research Context Builder

Orchestrates the complete literature review process: intent extraction,
literature search, and synthesis using O3 reasoning.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import time
import os
import json
from datetime import datetime

from .intent_extractor import extract_research_intent
from .synthesis import synthesize_literature
from ..mentor_tools import arxiv_search, openreview_fetch
from .o3_client import get_o3_client


def build_research_context(user_input: str) -> Dict[str, Any]:
    """
    Build comprehensive research context for any user input.
    
    This is the main entry point that:
    1. Extracts research intent using O3
    2. Performs literature searches if research intent detected
    3. Synthesizes findings using O3
    4. Returns structured context for the mentor agent
    
    Args:
        user_input: Raw user input in natural language
        
    Returns:
        Dictionary containing:
        - has_research_context: bool
        - intent: Dict - original intent analysis
        - literature_summary: str - synthesized research overview
        - key_papers: List[Dict] - important papers found
        - recommendations: List[str] - actionable next steps
        - search_performed: bool - whether searches were actually performed
        - context_for_agent: str - formatted context for the main agent
    """
    start_time = time.time()
    
    # Initialize debug logging if enabled
    debug_log = _init_debug_logging(user_input) if _should_debug_log() else None
    
    # Step 1: Extract research intent using O3
    intent = extract_research_intent(user_input)
    
    if debug_log:
        debug_log["step1_intent_extraction"] = {
            "timestamp": datetime.now().isoformat(),
            "intent_result": intent
        }
    
    if not intent.get("has_research_intent", False):
        if debug_log:
            _save_debug_log(debug_log, "no_intent")
        return _no_research_context(intent, user_input)
    
    # Step 2: Perform literature searches based on extracted topics
    topics = intent.get("topics", [])
    if not topics:
        if debug_log:
            _save_debug_log(debug_log, "no_topics")
        return _no_research_context(intent, user_input)
    
    print(f"🔍 Research topics detected: {', '.join(topics)}")
    print("📚 Searching literature...")
    
    # Perform parallel searches
    search_results = _perform_literature_searches(topics, relax=False)
    
    if debug_log:
        debug_log["step2_literature_search"] = {
            "timestamp": datetime.now().isoformat(),
            "topics": topics,
            "search_query": _topics_to_search_query(topics),
            "arxiv_results_count": len(search_results.get("arxiv", {}).get("papers", [])),
            "openreview_results_count": len(search_results.get("openreview", {}).get("threads", [])),
            "arxiv_results": search_results.get("arxiv", {}),
            "openreview_results": search_results.get("openreview", {})
        }
    
    if not _has_meaningful_results(search_results):
        # Retry once with a relaxed strategy (broader query/older years)
        if debug_log:
            debug_log["retry1"] = {
                "timestamp": datetime.now().isoformat(),
                "reason": "No meaningful results on first attempt"
            }
            _save_debug_log(debug_log, "retry1")
        retry_results = _perform_literature_searches(topics, relax=True)
        if _has_meaningful_results(retry_results):
            search_results = retry_results
        else:
            # Fallback: LLM-only overview (no web retrieval)
            if debug_log:
                debug_log["llm_only_fallback"] = {
                    "timestamp": datetime.now().isoformat(),
                    "reason": "No meaningful results after retry; using O3-only overview"
                }
                _save_debug_log(debug_log, "llm_only")
            llm_only = _llm_only_overview(user_input=user_input, topics=topics, research_type=intent.get("research_type", "other"))
            agent_context = _build_agent_context(intent, llm_only, topics)
            duration = time.time() - start_time
            if debug_log:
                debug_log["step3_llm_only_synthesis"] = {
                    "timestamp": datetime.now().isoformat(),
                    "synthesis_result": llm_only
                }
                debug_log["step4_final"] = {
                    "timestamp": datetime.now().isoformat(),
                    "agent_context_length": len(agent_context),
                    "processing_time": duration
                }
                _save_debug_log(debug_log, "complete")
            return {
                "has_research_context": True,
                "has_literature": False,
                "is_llm_only": True,
                "grounding": "llm_only",
                "intent": intent,
                "literature_summary": llm_only.get("summary", ""),
                "key_papers": llm_only.get("key_papers", []),
                "research_gaps": llm_only.get("research_gaps", []),
                "trending_topics": llm_only.get("trending_topics", []),
                "recommendations": llm_only.get("recommendations", []),
                "search_performed": True,
                "context_for_agent": agent_context,
                "processing_time": duration
            }
    
    print("🧠 Synthesizing research insights with O3...")
    
    # Step 3: Synthesize using O3
    synthesis = synthesize_literature(
        topics=topics,
        arxiv_results=search_results["arxiv"],
        openreview_results=search_results["openreview"],
        research_type=intent.get("research_type", "other")
    )
    
    if debug_log:
        debug_log["step3_synthesis"] = {
            "timestamp": datetime.now().isoformat(),
            "synthesis_result": synthesis
        }
    
    # Step 4: Build context for the main agent
    agent_context = _build_agent_context(intent, synthesis, topics)
    
    duration = time.time() - start_time
    print(f"✅ Research context built in {duration:.1f}s")
    
    if debug_log:
        debug_log["step4_final"] = {
            "timestamp": datetime.now().isoformat(),
            "agent_context_length": len(agent_context),
            "processing_time": duration
        }
        _save_debug_log(debug_log, "complete")
    
    return {
        "has_research_context": True,
        "has_literature": True,
        "is_llm_only": False,
        "grounding": "retrieved",
        "intent": intent,
        "literature_summary": synthesis.get("summary", ""),
        "key_papers": synthesis.get("key_papers", []),
        "research_gaps": synthesis.get("research_gaps", []),
        "trending_topics": synthesis.get("trending_topics", []),
        "recommendations": synthesis.get("recommendations", []),
        "search_performed": True,
        "context_for_agent": agent_context,
        "processing_time": duration
    }


def _perform_literature_searches(topics: List[str], relax: bool = False) -> Dict[str, Any]:
    """Perform literature searches across multiple sources.

    When relax=True, broaden the search: drop year filter and increase limits.
    Uses the orchestrator for tool selection when available, falls back to legacy.
    """
    # Create search query from topics (sanitized)
    query = _topics_to_search_query(topics)
    
    # Try using orchestrator-based tool selection if available
    use_orchestrator = os.getenv("FF_REGISTRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    
    if use_orchestrator:
        try:
            from ..core.orchestrator import Orchestrator
            from ..tools import auto_discover
            
            # Ensure tools are discovered
            auto_discover()
            
            # Set up search parameters
            from_year = None if relax else 2020
            limit = 15 if relax else 10
            or_limit = 10 if relax else 8
            
            orch = Orchestrator()
            result = orch.execute_task(
                task="literature_search",
                inputs={
                    "query": query,
                    "from_year": from_year,
                    "limit": limit,
                    "or_limit": or_limit
                },
                context={"goal": f"find papers about {' '.join(topics)}"}
            )
            
            if result["execution"]["executed"] and result["results"]:
                # Convert orchestrator result back to expected format
                tool_result = result["results"]
                
                # Extract papers by source
                arxiv_papers = [p for p in tool_result.get("results", []) if p.get("source") == "arxiv"]
                openreview_papers = [p for p in tool_result.get("results", []) if p.get("source") == "openreview"]
                
                return {
                    "arxiv": {"papers": arxiv_papers},
                    "openreview": {"threads": openreview_papers},
                    "orchestrator_used": True,
                    "tool_used": result["execution"]["tool_used"]
                }
            else:
                print(f"Orchestrator execution failed: {result['execution'].get('reason', 'Unknown')}")
                # Fall through to legacy implementation
                
        except Exception as e:
            print(f"Orchestrator search failed, falling back to legacy: {e}")
            # Fall through to legacy implementation
    
    # Legacy implementation (fallback or when orchestrator disabled)
    search_results = {
        "arxiv": {},
        "openreview": {}
    }
    
    try:
        # arXiv search
        from_year = None if relax else 2020
        arxiv_limit = 15 if relax else 10
        search_results["arxiv"] = arxiv_search(query=query, from_year=from_year, limit=arxiv_limit)
    except Exception as e:
        print(f"arXiv search failed: {e}")
        search_results["arxiv"] = {"papers": [], "note": f"Search failed: {e}"}
    
    try:
        # OpenReview search
        or_limit = 10 if relax else 8
        search_results["openreview"] = openreview_fetch(query=query, limit=or_limit)
    except Exception as e:
        print(f"OpenReview search failed: {e}")
        search_results["openreview"] = {"threads": [], "note": f"Search failed: {e}"}
    
    search_results["orchestrator_used"] = False
    return search_results


def _llm_only_overview(user_input: str, topics: List[str], research_type: str) -> Dict[str, Any]:
    """Fallback: generate a literature-style overview using O3 alone (no retrieval)."""
    o3 = get_o3_client()
    if not o3.is_available():
        # Minimal heuristic fallback
        return {
            "summary": f"High-level overview for topics: {', '.join(topics)}.",
            "key_papers": [],
            "research_gaps": ["Lack of grounded references due to offline retrieval"],
            "trending_topics": topics[:5],
            "recommendations": ["Consider re-running with network access to fetch citations."]
        }

    system_message = (
        "You are an expert research mentor. No web search is available. "
        "Produce a concise literature-style overview grounded in general knowledge only. "
        "Avoid fabricating specific citation metadata or URLs."
    )
    prompt = (
        f"User research request: {user_input}\n\n"
        f"Topics (may be partial): {', '.join(topics)}\n"
        f"Research Type: {research_type}\n\n"
        "Please provide: (1) Field summary (2-3 sentences), (2) Potential research gaps, "
        "(3) Trending sub-areas, (4) 3 concrete next steps. Do not invent paper titles or links."
    )
    try:
        content = o3.reason(prompt, system_message) or ""
    except Exception:
        content = ""

    # Simple parsing: split into segments if possible
    summary = content.strip()[:800] if content else "General high-level overview produced."
    return {
        "summary": summary,
        "key_papers": [],
        "research_gaps": [],
        "trending_topics": topics[:5],
        "recommendations": [
            "Refine topics and try again",
            "Broaden/adjust keywords",
            "Add venue constraints (e.g., ICLR, NeurIPS)"
        ],
    }


def _topics_to_search_query(topics: List[str]) -> str:
    """Sanitize and compress extracted topics into a compact search query.

    - Splits composite phrases, removes parentheticals, de-noises stopwords
    - Normalizes common variants (e.g., LMMs->lmm, datasets->dataset)
    - Prioritizes domain-bearing tokens for arXiv relevance
    """
    import re

    joined = " ".join(topics or [])
    # Remove parenthetical segments
    no_paren = re.sub(r"\([^)]*\)", " ", joined)
    # Replace slashes and multiple spaces
    norm = no_paren.replace("/", " ")
    # Tokenize words and hyphenated terms
    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", norm.lower())

    # Normalize variants
    variant_map = {
        "datasets": "dataset",
        "lmms": "lmm",
        "llms": "llm",
        "preprints": "arxiv",
        "pdfs": "pdf",
        "open-source": "open source",
    }
    tokens: List[str] = []
    seen: set[str] = set()
    for t in raw_tokens:
        t = variant_map.get(t, t)
        if t not in seen:
            seen.add(t)
            tokens.append(t)

    # Remove very generic words
    stop = {
        "the","and","for","are","but","not","you","all","can","has","have","had",
        "one","two","new","now","old","see","use","using","with","via","from","into",
        "scale","scaling","build","building","project","source","open","large","large-scale",
        "strategy","strategies","resources","models","model","data","collection","sourcing",
        "curation","strategies","best","practices","mix","available","currently",
    }
    filtered = [t for t in tokens if t not in stop and len(t) >= 3]

    # Priority order for research relevance
    priority = [
        "multimodal","dataset","lmm","llm","vision-language","vlm","vision","image","text",
        "arxiv","pdf","html","pretraining","pretrain","benchmark","survey",
    ]

    def sort_key(tok: str) -> tuple[int, int]:
        try:
            idx = priority.index(tok)
        except ValueError:
            idx = len(priority)
        return (idx, -len(tok))

    ordered = sorted(filtered, key=sort_key)
    # Compose a compact query from top tokens
    core = ordered[:5] if ordered else (tokens[:5] if tokens else [])
    return " ".join(core) or " ".join((topics or [])[:3])


def _has_meaningful_results(search_results: Dict[str, Any]) -> bool:
    """Check if search results contain meaningful literature."""
    arxiv_papers = search_results.get("arxiv", {}).get("papers", [])
    openreview_threads = search_results.get("openreview", {}).get("threads", [])
    
    return len(arxiv_papers) > 0 or len(openreview_threads) > 0


def _build_agent_context(intent: Dict[str, Any], synthesis: Dict[str, Any], topics: List[str]) -> str:
    """Build formatted context string for the main agent."""
    context_parts = []
    
    # Research context header
    context_parts.append("=== RESEARCH CONTEXT ===")
    context_parts.append(f"Topics: {', '.join(topics)}")
    context_parts.append(f"Research Type: {intent.get('research_type', 'general')}")
    context_parts.append("")
    
    # Literature summary
    summary = synthesis.get("summary", "")
    if summary:
        context_parts.append("FIELD OVERVIEW:")
        context_parts.append(summary)
        context_parts.append("")
    
    # Key papers
    key_papers = synthesis.get("key_papers", [])
    if key_papers:
        context_parts.append("KEY PAPERS FOUND:")
        for i, paper in enumerate(key_papers[:5], 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "")
            venue = paper.get("venue", "")
            source = paper.get("source", "")
            
            paper_line = f"{i}. {title}"
            if year:
                paper_line += f" ({year})"
            if venue:
                paper_line += f" [{venue}]"
            elif source:
                paper_line += f" [{source}]"
            
            context_parts.append(paper_line)
        context_parts.append("")
    
    # Research gaps and opportunities
    gaps = synthesis.get("research_gaps", [])
    if gaps:
        context_parts.append("RESEARCH GAPS IDENTIFIED:")
        for gap in gaps[:3]:
            context_parts.append(f"- {gap}")
        context_parts.append("")
    
    # Trending topics
    trending = synthesis.get("trending_topics", [])
    if trending:
        context_parts.append(f"TRENDING AREAS: {', '.join(trending[:5])}")
        context_parts.append("")
    
    # Recommendations
    recommendations = synthesis.get("recommendations", [])
    if recommendations:
        context_parts.append("RESEARCH RECOMMENDATIONS:")
        for rec in recommendations[:3]:
            context_parts.append(f"- {rec}")
        context_parts.append("")
    
    context_parts.append("=== END RESEARCH CONTEXT ===")
    context_parts.append("")
    context_parts.append("Use this research context to provide informed mentoring. Ask probing questions based on the field knowledge above.")
    
    return "\n".join(context_parts)


def _no_research_context(intent: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """Return structure when no research intent is detected."""
    return {
        "has_research_context": False,
        "has_literature": False,
        "is_llm_only": False,
        "grounding": "none",
        "intent": intent,
        "literature_summary": "",
        "key_papers": [],
        "research_gaps": [],
        "trending_topics": [],
        "recommendations": [],
        "search_performed": False,
        "context_for_agent": f"No research intent detected in: '{user_input}'. Proceed with general conversation.",
        "processing_time": 0.0
    }


def _minimal_research_context(intent: Dict[str, Any], topics: List[str], search_results: Dict[str, Any]) -> Dict[str, Any]:
    """Return minimal context when searches don't yield results."""
    arxiv_note = search_results.get("arxiv", {}).get("note", "No results")
    openreview_note = search_results.get("openreview", {}).get("note", "No results")
    
    context = f"""=== RESEARCH CONTEXT ===
Topics: {', '.join(topics)}
Research Type: {intent.get('research_type', 'general')}

LITERATURE SEARCH RESULTS:
- arXiv: {arxiv_note}
- OpenReview: {openreview_note}

This appears to be a very new, niche, or interdisciplinary area.
Consider broader search terms or related fields.
=== END RESEARCH CONTEXT ===

Guide the user toward refining their research focus or exploring related areas."""
    
    return {
        "has_research_context": True,
        "has_literature": False,
        "is_llm_only": False,
        "grounding": "none",
        "intent": intent,
        "literature_summary": "Limited literature found for this specific topic.",
        "key_papers": [],
        "research_gaps": ["Potential opportunity for novel research"],
        "trending_topics": topics,
        "recommendations": ["Try broader search terms", "Explore related fields"],
        "search_performed": True,
        "context_for_agent": context,
        "processing_time": 0.0
    }


def _should_debug_log() -> bool:
    """Check if debug logging should be enabled."""
    return os.getenv("ARM_DEBUG_LITERATURE", "false").lower() in ("true", "1", "yes")


def _init_debug_logging(user_input: str) -> Dict[str, Any]:
    """Initialize debug logging structure."""
    return {
        "session_start": datetime.now().isoformat(),
        "user_input": user_input,
        "process_id": os.getpid(),
        "steps": {}
    }


def _save_debug_log(debug_log: Dict[str, Any], stage: str) -> None:
    """Save debug log to timestamped file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"literature_debug_{timestamp}_{stage}.json"
        
        # Save in current working directory for easy access
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(debug_log, f, indent=2, ensure_ascii=False)
        
        print(f"🔍 Debug log saved: {filename}")
        
    except Exception as e:
        print(f"Warning: Failed to save debug log: {e}")