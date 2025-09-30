from __future__ import annotations

"""LLM-based document summarization for attachments."""

from typing import Any
import os


def _get_llm() -> Any | None:
    """Get LLM instance for summarization."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        if os.environ.get("OPENROUTER_API_KEY"):
            return ChatOpenAI(
                model="google/gemini-2.5-flash-preview-09-2025",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0
            )
    except Exception:
        pass
    return None


def _extract_experiment_structure(text: str, llm: Any) -> str:
    """Pass 1: Identify experiment boundaries and structure."""
    prompt = f"""Analyze this research document and identify ALL experiments/studies conducted.

TASK: List each experiment with:
1. Experiment number and title
2. Where it appears in the document (section/paragraph markers)
3. Brief objective (1 sentence)

Be careful NOT to conflate multiple experiments into one. If you see sub-variants (e.g., score-based vs ranking-based), list them as separate items.

Document excerpt:
{text[:15000]}

Output format:
- Experiment 1: [Title] - [Location] - [Objective]
- Experiment 2: [Title] - [Location] - [Objective]
..."""
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def _extract_experiment_details(text: str, exp_summary: str, llm: Any) -> str:
    """Pass 2: Extract detailed info for each identified experiment."""
    prompt = f"""Based on the identified experiments below, extract DETAILED information for each one.

IDENTIFIED EXPERIMENTS:
{exp_summary}

For EACH experiment, find and extract:
- Experiment N: [Title from above]
  * Setup: What was tested? How many variants/conditions? What evaluation metrics?
  * Key finding: What specific verdict or conclusion was reached? Use exact phrasing from document.
  * Details: Numerical results, language-specific findings, comparisons between conditions.
  * Caveats: Any limitations or warnings mentioned.

CRITICAL: Do NOT mix details from different experiments. Quote specific findings verbatim where possible.

Full document excerpt:
{text[:15000]}

Output format:
- Experiment 1: [Title]
  * Setup: [Detailed setup]
  * Key finding: [Explicit verdict from document]
  * Details: [Specific results with numbers]
  * Caveats: [Limitations if mentioned]

- Experiment 2: ...
"""
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def generate_document_summary(docs: list[Any]) -> str:
    """Generate LLM-based summary using multi-pass extraction.
    
    Pass 1: Identify experiment boundaries and structure
    Pass 2: Extract detailed info for each experiment
    Pass 3: Extract methods, findings, and questions
    
    This prevents experiment conflation and misattribution.
    """
    try:
        # Sample pages strategically for better coverage
        sample_pages = docs[:min(30, len(docs))]
        combined_text = "\n\n".join([d.page_content[:2000] for d in sample_pages if d.page_content])
        
        if len(combined_text) < 100:
            return "Insufficient content for summary generation."
        
        llm = _get_llm()
        if llm is None:
            return "LLM unavailable for summary generation. Ensure OPENROUTER_API_KEY is set."
        
        # Pass 1: Identify experiment structure
        exp_structure = _extract_experiment_structure(combined_text, llm)
        
        # Pass 2: Extract detailed info for each experiment
        exp_details = _extract_experiment_details(combined_text, exp_structure, llm)
        
        # Pass 3: Extract methods, key findings, research questions
        methods_prompt = f"""Based on this research document, extract:

1. RESEARCH METHODS:
   - Datasets: Names, sizes, languages
   - Models: Names, versions, configurations
   - Evaluation metrics: Specific metrics used
   - Analysis techniques: Statistical tests, visualizations

2. KEY FINDINGS (cross-experiment):
   - Overall conclusions
   - Correlations found or NOT found
   - Important caveats and limitations

3. RESEARCH QUESTIONS:
   - Main questions or hypotheses tested

Document excerpt:
{combined_text[:15000]}

Previous experiment extraction:
{exp_details[:2000]}"""

        methods_response = llm.invoke(methods_prompt)
        methods_text = methods_response.content if hasattr(methods_response, 'content') else str(methods_response)
        
        # Combine into structured output
        final_summary = f"""EXPERIMENTS CONDUCTED:
{exp_details}

{methods_text}"""
        
        return final_summary.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
