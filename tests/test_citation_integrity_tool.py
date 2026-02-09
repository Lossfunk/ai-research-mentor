from __future__ import annotations

from academic_research_mentor.agent.tool_adapters import CitationIntegrityToolAdapter
from academic_research_mentor.citations.integrity import audit_reference_text


def test_audit_reference_text_extracts_reference_signals_without_network_checks() -> None:
    text = """
    References:
    Smith et al. (2024). Great Paper. https://example.com/paper
    DOI: 10.1000/xyz123
    arXiv:2401.12345
    """
    report = audit_reference_text(
        text,
        check_urls=False,
        verify_doi=False,
        verify_arxiv=False,
    )
    totals = report["summary"]["totals"]
    assert totals["urls"] == 1
    assert totals["dois"] == 1
    assert totals["arxiv_ids"] == 1
    assert report["summary"]["score"] >= 70


def test_citation_integrity_tool_adapter_formats_report() -> None:
    tool = CitationIntegrityToolAdapter()
    output = tool.execute(
        reference_text="Ref: https://example.com DOI: 10.1000/xyz123",
        check_urls=False,
        verify_doi=False,
        verify_arxiv=False,
    )
    assert "Citation Integrity Report:" in output
    assert "Detected:" in output
