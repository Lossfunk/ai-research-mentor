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


def generate_document_summary(docs: list[Any]) -> str:
    """Generate grounded summary by extracting document content as-written.
    
    Uses single-pass extraction with explicit grounding requirements:
    - Extract experiments as they appear in document (don't reorganize)
    - Include page numbers for all claims
    - Quote key findings verbatim
    - Preserve document structure and experiment labels
    """
    try:
        # Use maximum available content for grounding
        sample_pages = docs[:min(30, len(docs))]
        # Include page metadata for grounding
        page_texts = []
        for page in sample_pages:
            page_num = page.metadata.get("page", "?")
            content = page.page_content[:2500] if page.page_content else ""
            page_texts.append(f"[Page {page_num}]\n{content}")
        
        combined_text = "\n\n".join(page_texts)
        
        if len(combined_text) < 100:
            return "Insufficient content for summary generation."
        
        llm = _get_llm()
        if llm is None:
            return "LLM unavailable for summary generation. Ensure OPENROUTER_API_KEY is set."
        
        # Grounded single-pass extraction
        prompt = f"""Extract information from this research document EXACTLY as written. Do NOT reorganize, infer, or combine information.

CRITICAL INSTRUCTIONS:
1. If the document labels experiments (e.g., "Experiment 1:", "Study 2:"), use those EXACT labels
2. For each experiment, include the PAGE NUMBER where it's described
3. Quote key findings VERBATIM from the document - use quotation marks
4. Preserve the document's organization - extract in the order things appear
5. Do NOT combine separate experiments into one
6. Do NOT split one experiment into multiple entries
7. If evaluation dimensions are listed (e.g., "Detailedness, Language Quality"), list them ALL

Document with page markers:
{combined_text[:20000]}

Output format:

EXPERIMENTS CONDUCTED:
- Experiment N: [Exact title/objective from document] (Page X)
  * Setup: [What was tested - languages, configurations, metrics AS LISTED in doc]
  * Key finding: "[Direct quote of verdict/conclusion from document]"
  * Details: [Specific results with numbers, language-specific findings]
  * Caveats: [Any limitations mentioned]

RESEARCH METHODS:
- Datasets: [Names, sizes, languages AS STATED]
- Models: [Names, versions AS STATED]
- Evaluation metrics: [ALL metrics listed in document]

KEY FINDINGS:
- [Direct quotes or paraphrases with page numbers]
- [Correlations found/not found with page numbers]

RESEARCH QUESTIONS:
- [Questions as stated in document]

Remember: EXTRACT, don't reorganize. Quote verbatim when possible. Include page numbers."""

        response = llm.invoke(prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        return summary_text.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
