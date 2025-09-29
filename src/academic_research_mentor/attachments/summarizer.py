from __future__ import annotations

"""LLM-based document summarization for attachments."""

from typing import Any
import os


def generate_document_summary(docs: list[Any]) -> str:
    """Generate LLM-based summary of document content to extract key info.
    
    Extracts:
    - Experiments already conducted
    - Key findings and results
    - Research methods and datasets used
    - Research questions being investigated
    """
    try:
        # Sample first N pages for summary (avoid token overload)
        sample_pages = docs[:min(20, len(docs))]
        combined_text = "\n\n".join([d.page_content[:1500] for d in sample_pages if d.page_content])
        
        if len(combined_text) < 100:
            return "Insufficient content for summary generation."
        
        # Build prompt for LLM to extract key information
        prompt = f"""Analyze this research document and extract:

1. EXPERIMENTS CONDUCTED: List specific experiments, evaluations, or validations already performed (with brief descriptions)
2. KEY FINDINGS: Main results, conclusions, or discoveries
3. RESEARCH METHODS: Methodologies, datasets, models, or techniques used
4. RESEARCH QUESTIONS: Questions or hypotheses being investigated

Be specific and concise. If a section is not present, write "Not found".

Document excerpt:
{combined_text[:8000]}

Provide structured output in this format:

EXPERIMENTS CONDUCTED:
- [experiment 1]
- [experiment 2]
...

KEY FINDINGS:
- [finding 1]
- [finding 2]
...

RESEARCH METHODS:
- [method 1]
- [method 2]
...

RESEARCH QUESTIONS:
- [question 1]
- [question 2]
..."""

        # Try to get LLM for summarization
        llm = None
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            if os.environ.get("OPENAI_API_KEY"):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
        
        if llm is None:
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore
                if os.environ.get("ANTHROPIC_API_KEY"):
                    llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)
            except Exception:
                pass
        
        if llm is None:
            return "LLM unavailable for summary generation. Ensure OPENAI_API_KEY or ANTHROPIC_API_KEY is set."
        
        response = llm.invoke(prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        return summary_text.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
