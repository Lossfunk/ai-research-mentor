from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from .intent_extractor import extract_research_intent
from .synthesis import synthesize_literature
from .search import perform_literature_searches, has_meaningful_results
from .fallback import llm_only_overview
from .context_format import build_agent_context
from .debug import should_debug_log, init_debug_logging, save_debug_log


def build_research_context(user_input: str) -> Dict[str, Any]:
    start_time = time.time()

    debug_log = init_debug_logging(user_input) if should_debug_log() else None

    intent = extract_research_intent(user_input)

    if debug_log is not None:
        debug_log.setdefault("steps", {})["step1_intent_extraction"] = {
            "timestamp": datetime.now().isoformat(),
            "intent_result": intent,
        }

    if not intent.get("has_research_intent", False):
        if debug_log is not None:
            save_debug_log(debug_log, "no_intent")
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
            "processing_time": 0.0,
        }

    topics = intent.get("topics", [])
    if not topics:
        if debug_log is not None:
            save_debug_log(debug_log, "no_topics")
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
            "processing_time": 0.0,
        }

    print(f"🔍 Research topics detected: {', '.join(topics)}")
    print("📚 Searching literature...")

    search_results = perform_literature_searches(topics, relax=False)

    if debug_log is not None:
        debug_log.setdefault("steps", {})["step2_literature_search"] = {
            "timestamp": datetime.now().isoformat(),
            "topics": topics,
            "search_query": " ".join(topics),
            "arxiv_results_count": len(search_results.get("arxiv", {}).get("papers", [])),
            "openreview_results_count": len(search_results.get("openreview", {}).get("threads", [])),
            "arxiv_results": search_results.get("arxiv", {}),
            "openreview_results": search_results.get("openreview", {}),
        }

    if not has_meaningful_results(search_results):
        if debug_log is not None:
            debug_log.setdefault("steps", {})["retry1"] = {
                "timestamp": datetime.now().isoformat(),
                "reason": "No meaningful results on first attempt",
            }
            save_debug_log(debug_log, "retry1")
        retry_results = perform_literature_searches(topics, relax=True)
        if has_meaningful_results(retry_results):
            search_results = retry_results
        else:
            if debug_log is not None:
                debug_log.setdefault("steps", {})["llm_only_fallback"] = {
                    "timestamp": datetime.now().isoformat(),
                    "reason": "No meaningful results after retry; using O3-only overview",
                }
                save_debug_log(debug_log, "llm_only")
            llm_only = llm_only_overview(user_input=user_input, topics=topics, research_type=intent.get("research_type", "other"))
            agent_context = build_agent_context(intent, llm_only, topics)
            duration = time.time() - start_time
            if debug_log is not None:
                debug_log.setdefault("steps", {})["step3_llm_only_synthesis"] = {
                    "timestamp": datetime.now().isoformat(),
                    "synthesis_result": llm_only,
                }
                debug_log["step4_final"] = {
                    "timestamp": datetime.now().isoformat(),
                    "agent_context_length": len(agent_context),
                    "processing_time": duration,
                }
                save_debug_log(debug_log, "complete")
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
                "processing_time": duration,
            }

    print("🧠 Synthesizing research insights with O3...")

    synthesis = synthesize_literature(
        topics=topics,
        arxiv_results=search_results.get("arxiv", {}),
        openreview_results=search_results.get("openreview", {}),
        research_type=intent.get("research_type", "other"),
    )

    if debug_log is not None:
        debug_log.setdefault("steps", {})["step3_synthesis"] = {
            "timestamp": datetime.now().isoformat(),
            "synthesis_result": synthesis,
        }

    agent_context = build_agent_context(intent, synthesis, topics)

    duration = time.time() - start_time
    print(f"✅ Research context built in {duration:.1f}s")

    if debug_log is not None:
        debug_log["step4_final"] = {
            "timestamp": datetime.now().isoformat(),
            "agent_context_length": len(agent_context),
            "processing_time": duration,
        }
        save_debug_log(debug_log, "complete")

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
        "processing_time": duration,
    }
