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
        # Sample pages strategically for better coverage
        sample_pages = docs[:min(30, len(docs))]
        combined_text = "\n\n".join([d.page_content[:2000] for d in sample_pages if d.page_content])
        
        if len(combined_text) < 100:
            return "Insufficient content for summary generation."
        
        # Build detailed prompt for comprehensive extraction
        prompt = f"""Analyze this research document and extract detailed, structured information. Be SPECIFIC and preserve important details.

CRITICAL: For each experiment, include:
- Sub-experiments or variants (e.g., "Experiment 1a: score-based", "Experiment 1b: ranking-based")
- Evaluation dimensions/metrics used
- Explicit findings and verdicts (what was learned?)
- Numerical results when mentioned
- Important caveats or limitations

Document excerpt:
{combined_text[:12000]}

Provide output in this EXACT format:

EXPERIMENTS CONDUCTED:
For each experiment, include:
- Experiment N: [Title/objective]
  * Setup: [What was tested, how many variants, evaluation metrics]
  * Key finding: [Explicit verdict - what was learned?]
  * Details: [Specific results, numbers, language-specific findings if any]
  
Example format:
- Experiment 1: Multilingual judge configuration
  * Setup: Tested score-based (Cohere, translated, English query with 2/4/8 examples) vs ranking-based (Judge score 4 and 8)
  * Key finding: More examples improve results; translation does not help evaluation
  * Details: Score-based showed X pattern, ranking-based showed Y pattern

KEY FINDINGS:
List explicit verdicts, conclusions, and numerical results:
- [Specific finding with numbers/languages if mentioned]
- [Correlations found or NOT found]
- [Caveats and limitations explicitly stated]

RESEARCH METHODS:
- Datasets: [Names, sizes, languages]
- Models: [Names, versions, configurations]
- Evaluation metrics: [Specific metrics used]
- Analysis techniques: [Statistical tests, visualizations]

RESEARCH QUESTIONS:
- [Specific questions or hypotheses tested]"""

        # Try to get LLM for summarization (prefer OpenRouter for consistency)
        llm = None
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            if os.environ.get("OPENROUTER_API_KEY"):
                llm = ChatOpenAI(
                    model="google/gemini-2.5-flash-preview-09-2025",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0
                )
        except Exception:
            pass
        
        if llm is None:
            return "LLM unavailable for summary generation. Ensure OPENROUTER_API_KEY is set."
        
        response = llm.invoke(prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        return summary_text.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
