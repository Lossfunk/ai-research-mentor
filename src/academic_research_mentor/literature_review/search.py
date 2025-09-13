from __future__ import annotations

import os
from typing import Any, Dict, List

from ..mentor_tools import arxiv_search


def topics_to_search_query(topics: List[str]) -> str:
    import re

    joined = " ".join(topics or [])
    no_paren = re.sub(r"\([^)]*\)", " ", joined)
    norm = no_paren.replace("/", " ")
    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", norm.lower())

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

    stop = {
        "the","and","for","are","but","not","you","all","can","has","have","had",
        "one","two","new","now","old","see","use","using","with","via","from","into",
        "scale","scaling","build","building","project","source","open","large","large-scale",
        "strategy","strategies","resources","models","model","data","collection","sourcing",
        "curation","strategies","best","practices","mix","available","currently",
    }
    filtered = [t for t in tokens if t not in stop and len(t) >= 3]

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
    core = ordered[:5] if ordered else (tokens[:5] if tokens else [])
    return " ".join(core) or " ".join((topics or [])[:3])


def perform_literature_searches(topics: List[str], relax: bool = False) -> Dict[str, Any]:
    query = topics_to_search_query(topics)

    use_orchestrator = os.getenv("FF_REGISTRY_ENABLED", "true").lower() in ("1", "true", "yes", "on")

    if use_orchestrator:
        try:
            from ..core.orchestrator import Orchestrator
            from ..tools import auto_discover

            auto_discover()

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
                    "or_limit": or_limit,
                },
                context={"goal": f"find papers about {' '.join(topics)}"},
            )

            if result["execution"]["executed"] and result["results"]:
                tool_result = result["results"]
                arxiv_papers = [p for p in tool_result.get("results", []) if p.get("source") == "arxiv"]
                openreview_papers = [p for p in tool_result.get("results", []) if p.get("source") == "openreview"]

                return {
                    "arxiv": {"papers": arxiv_papers},
                    "openreview": {"threads": openreview_papers},
                    "orchestrator_used": True,
                    "tool_used": result["execution"]["tool_used"],
                }
            else:
                print(f"Orchestrator execution failed: {result['execution'].get('reason', 'Unknown')}")
        except Exception as e:
            print(f"Orchestrator search failed, falling back to legacy: {e}")

    search_results = {"arxiv": {}, "openreview": {}}

    try:
        from_year = None if relax else 2020
        arxiv_limit = 15 if relax else 10
        search_results["arxiv"] = arxiv_search(query=query, from_year=from_year, limit=arxiv_limit)
    except Exception as e:
        print(f"arXiv search failed: {e}")
        search_results["arxiv"] = {"papers": [], "note": f"Search failed: {e}"}

    search_results["orchestrator_used"] = False
    return search_results


def has_meaningful_results(search_results: Dict[str, Any]) -> bool:
    arxiv_papers = search_results.get("arxiv", {}).get("papers", [])
    openreview_threads = search_results.get("openreview", {}).get("threads", [])
    return len(arxiv_papers) > 0 or len(openreview_threads) > 0
