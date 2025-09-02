"""Intent Extraction using O3 Deep Reasoning

Extracts research intent and topics from natural language using O3's
advanced reasoning capabilities.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import json
from .o3_client import get_o3_client


def extract_research_intent(user_input: str) -> Dict[str, Any]:
    """
    Extract research intent from user input using O3 deep reasoning.
    
    Args:
        user_input: Raw user input in natural language
        
    Returns:
        Dictionary containing:
        - has_research_intent: bool
        - topics: List[str] - extracted research topics/keywords
        - research_type: str - type of research need (survey, specific_paper, methodology, etc.)
        - urgency: str - low/medium/high
        - context: str - additional context about the research need
    """
    o3_client = get_o3_client()
    
    if not o3_client.is_available():
        # Fallback to simple keyword detection
        return _fallback_intent_extraction(user_input)
    
    system_message = """You are an expert research intent analyzer. Your job is to deeply understand what research-related information or guidance the user needs, even from casual or indirect queries.

Extract research intent from user messages with the following considerations:
1. Academic research can be mentioned directly or implied
2. Users might ask about concepts, methods, tools, or fields without explicitly mentioning "research"
3. Even general questions about technical topics often have research implications
4. Learning goals, project planning, and methodology questions are research-related

Respond with a JSON object containing:
{
    "has_research_intent": boolean,
    "topics": ["list", "of", "research", "keywords"],
    "research_type": "survey|specific_paper|methodology|conceptual|tools|venue_info|other",
    "urgency": "low|medium|high",
    "context": "brief description of what the user likely needs"
}

Be generous in detecting research intent - err on the side of "yes" unless clearly unrelated."""

    prompt = f"""Analyze this user input for research intent:

"{user_input}"

Extract research topics, determine the type of research need, and assess urgency. Consider both explicit and implicit research needs."""

    try:
        response = o3_client.reason(prompt, system_message)
        if response:
            # Try to parse JSON response
            try:
                result = json.loads(response.strip())
                return _validate_intent_result(result)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information from text
                return _parse_text_response(response, user_input)
    except Exception as e:
        print(f"Intent extraction failed: {e}")
    
    # Fallback to simple detection
    return _fallback_intent_extraction(user_input)


def _validate_intent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean the intent extraction result."""
    validated = {
        "has_research_intent": bool(result.get("has_research_intent", False)),
        "topics": result.get("topics", []) if isinstance(result.get("topics"), list) else [],
        "research_type": result.get("research_type", "other"),
        "urgency": result.get("urgency", "medium"),
        "context": str(result.get("context", "")).strip()
    }
    
    # Ensure research_type is valid
    valid_types = {"survey", "specific_paper", "methodology", "conceptual", "tools", "venue_info", "other"}
    if validated["research_type"] not in valid_types:
        validated["research_type"] = "other"
    
    # Ensure urgency is valid
    valid_urgency = {"low", "medium", "high"}
    if validated["urgency"] not in valid_urgency:
        validated["urgency"] = "medium"
    
    return validated


def _parse_text_response(response: str, user_input: str) -> Dict[str, Any]:
    """Parse O3 text response when JSON parsing fails."""
    # Simple text parsing as fallback
    has_intent = any(word in response.lower() for word in 
                    ["research", "paper", "literature", "study", "academic", "topic"])
    
    # Extract potential topics from the response
    topics = []
    if has_intent:
        # Try to find quoted terms or technical keywords
        import re
        quoted_terms = re.findall(r'"([^"]+)"', response)
        topics = [term.strip() for term in quoted_terms if len(term.strip()) > 2]
        
        # If no quoted terms, use input for topic extraction
        if not topics:
            topics = _extract_topics_from_text(user_input)
    
    return {
        "has_research_intent": has_intent,
        "topics": topics[:5],  # Limit to 5 topics
        "research_type": "other",
        "urgency": "medium",
        "context": "Research intent detected but details unclear"
    }


def _fallback_intent_extraction(user_input: str) -> Dict[str, Any]:
    """Simple fallback intent extraction when O3 is unavailable."""
    text = user_input.lower().strip()
    
    # Keywords that suggest research intent
    research_keywords = [
        "research", "paper", "papers", "study", "literature", "review",
        "arxiv", "publication", "journal", "conference", "academic",
        "methodology", "experiment", "analysis", "survey", "thesis",
        "dissertation", "cite", "citation", "reference", "algorithm",
        "model", "framework", "approach", "technique", "method"
    ]
    
    has_intent = any(keyword in text for keyword in research_keywords)
    
    # If no obvious research keywords, check for technical/scientific terms
    if not has_intent:
        technical_patterns = [
            r'\b\w+ing\b',  # Methods/techniques ending in -ing
            r'\b\w+tion\b',  # Concepts ending in -tion
            r'\b\w+ism\b',   # Theories/concepts ending in -ism
        ]
        import re
        has_intent = any(re.search(pattern, text) for pattern in technical_patterns)
    
    topics = _extract_topics_from_text(user_input) if has_intent else []
    
    return {
        "has_research_intent": has_intent,
        "topics": topics,
        "research_type": "other",
        "urgency": "medium",
        "context": "Basic keyword-based detection"
    }


def _extract_topics_from_text(text: str) -> List[str]:
    """Extract potential research topics from text."""
    import re
    
    # Normalize and pre-tokenize
    raw = text.strip()
    lowered = raw.lower()

    # Extract word tokens (letters and hyphens) and common abbreviations/numerics
    word_tokens = re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", lowered)
    abbrev_tokens = re.findall(r"\b(?:llms?|lmms?|pdfs?|pdf|html|arxiv)\b", lowered)

    # Merge and de-duplicate while preserving order
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in word_tokens + abbrev_tokens:
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    # Domain-aware stopwords (expanded to avoid generic noise in queries)
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
        "did", "she", "use", "way", "say", "each", "which", "what", "about",
        "would", "there", "could", "other", "after", "first", "well", "many",
        "some", "time", "very", "when", "much", "before", "right", "too", "any",
        "same", "tell", "does", "most", "also", "back", "good", "with", "into",
        "from", "over", "than", "then", "this", "that", "these", "those", "your",
        # Domain-generic words we don't want as topics
        "open", "source", "huge", "build", "building", "project", "aiming", "mix",
        "scale", "scaling", "available", "current", "work", "works", "token",
        "trillion", "billion", "images", "image", "data", "sources",
    }

    # Keep domain-relevant candidates
    candidates = [t for t in tokens if t not in stop_words]

    # Promote key domain terms by priority and collapse variants
    priority = [
        "multimodal", "dataset", "datasets", "lmm", "lmms", "llm", "llms",
        "preprint", "preprints", "arxiv", "pdf", "pdfs", "html", "web",
        "extraction", "parsing", "ocr", "crawl", "crawling", "pipeline",
    ]

    # Map some variants
    variant_map = {
        "lmms": "lmm",
        "llms": "llm",
        "pdfs": "pdf",
    }
    normalized: list[str] = []
    for t in candidates:
        normalized.append(variant_map.get(t, t))

    # Ensure phrase 'open-source' is considered if both parts exist
    if "open" in lowered and "source" in lowered and "open-source" not in normalized:
        normalized.append("open-source")

    # Order by priority first, then by token length (desc), then original order
    def sort_key(tok: str) -> tuple[int, int]:
        try:
            p_idx = priority.index(tok)
        except ValueError:
            p_idx = len(priority)
        return (p_idx, -len(tok))

    unique_ordered: list[str] = []
    seen2: set[str] = set()
    for tok in sorted(normalized, key=sort_key):
        if tok not in seen2:
            seen2.add(tok)
            unique_ordered.append(tok)

    # Return up to top 5 topics
    return unique_ordered[:5]