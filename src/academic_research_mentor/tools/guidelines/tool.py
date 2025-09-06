"""
Guidelines tool for research mentoring advice.

Searches curated research guidance sources to provide evidence-based
academic mentoring advice on methodology, problem selection, and research taste.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..base_tool import BaseTool
from .config import GuidelinesConfig


class GuidelinesTool(BaseTool):
    """Tool for searching research guidelines and providing mentoring advice."""
    
    name = "research_guidelines"
    version = "1.0"
    
    def __init__(self) -> None:
        self.config = GuidelinesConfig()
        self._search_tool = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the guidelines tool with optional configuration."""
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self._search_tool = DuckDuckGoSearchRun()
        except ImportError:
            # Graceful fallback if DuckDuckGo search not available
            self._search_tool = None
    
    def can_handle(self, task_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this tool can handle research guidelines queries."""
        if not task_context:
            return False
            
        # Check for task goal or query text
        text = ""
        goal = task_context.get("goal", "")
        query = task_context.get("query", "")
        text = f"{goal} {query}".strip().lower()
        
        if not text:
            return False
            
        # Detect research guidance patterns
        guidelines_patterns = [
            r'\b(research\s+methodology|problem\s+selection|research\s+taste)\b',
            r'\b(academic\s+advice|phd\s+guidance|research\s+strategy)\b',
            r'\b(how\s+to\s+choose|develop\s+taste|research\s+skills)\b',
            r'\b(research\s+best\s+practices|methodology\s+advice)\b',
            r'\b(academic\s+career|research\s+planning|project\s+selection)\b',
            r'\b(hamming|effective\s+research|research\s+principles)\b',
            r'\bphd\b|\bcareer\s+guidance\b|\bmentoring\b|\bacademic\s+guidance\b',
            r'\bresearch\s+advice\b|\bgraduate\s+school\b|\bacademic\s+career\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in guidelines_patterns)
    
    def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search research guidelines and return raw content for agent reasoning (RAG-style)."""
        query = str(inputs.get("query", "")).strip()
        topic = str(inputs.get("topic", query)).strip()
        
        if not topic:
            return {
                "retrieved_guidelines": [],
                "formatted_content": "No topic provided for guidelines search.",
                "total_guidelines": 0,
                "note": "Empty query provided"
            }
        
        if not self._search_tool:
            return {
                "retrieved_guidelines": [],
                "formatted_content": f"Guidelines search unavailable for topic: {topic}",
                "total_guidelines": 0,
                "note": "Search tool not available"
            }
        
        try:
            # Generate targeted search queries based on topic
            search_queries = self._get_prioritized_queries(topic)[:self.config.MAX_SEARCH_QUERIES]
            
            retrieved_guidelines = []
            
            for query_str in search_queries:
                try:
                    results = self._search_tool.run(query_str)
                    if results and len(results) > 20:  # Basic quality check
                        source_type = self._identify_source_type(query_str)
                        guide_id = f"guide_{hash(results) & 0xfffff:05x}"
                        source_domain = self._extract_domain(query_str)
                        
                        retrieved_guidelines.append({
                            "guide_id": guide_id,
                            "source_type": source_type,
                            "source_domain": source_domain,
                            "content": results[:500],  # Truncate for token efficiency
                            "search_query": query_str
                        })
                        
                except Exception:
                    # Continue with other queries if one fails
                    continue
            
            if not retrieved_guidelines:
                return {
                    "retrieved_guidelines": [],
                    "formatted_content": f"No guidelines found for '{topic}' in curated sources. Try rephrasing your query.",
                    "total_guidelines": 0,
                    "note": "No guidelines retrieved from search"
                }
            
            # Format content for agent consumption (RAG-style)
            formatted_content = self._format_guidelines_for_agent(topic, retrieved_guidelines)
            
            return {
                "retrieved_guidelines": retrieved_guidelines,
                "formatted_content": formatted_content,
                "total_guidelines": len(retrieved_guidelines),
                "topic": topic,
                "note": f"Retrieved {len(retrieved_guidelines)} relevant guidelines for agent reasoning"
            }
            
        except Exception as e:
            return {
                "retrieved_guidelines": [],
                "formatted_content": f"Error searching guidelines: {e}",
                "total_guidelines": 0,
                "error": str(e),
                "note": "Error in guidelines search"
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata for selection and usage."""
        meta = super().get_metadata()
        meta["identity"]["owner"] = "guidelines"
        meta["capabilities"] = {
            "task_types": ["research_advice", "methodology_guidance", "academic_mentoring"],
            "domains": ["research_methodology", "academic_career", "problem_selection", "research_taste"]
        }
        meta["io"] = {
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research question or topic"},
                    "topic": {"type": "string", "description": "Specific area for guidance"}
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "retrieved_guidelines": {"type": "array"},
                    "formatted_content": {"type": "string"},
                    "total_guidelines": {"type": "integer"}
                }
            }
        }
        meta["operational"] = {
            "cost_estimate": "low-medium",
            "latency_profile": "5-10 seconds",
            "rate_limits": "3 searches per query"
        }
        meta["usage"] = {
            "ideal_inputs": ["research methodology questions", "academic career advice", "problem selection guidance"],
            "anti_patterns": ["very broad questions", "non-research topics"],
            "prerequisites": ["internet connection for search"]
        }
        return meta
    
    def _identify_source_type(self, query: str) -> str:
        """Identify the source type based on the search query."""
        if "gwern.net" in query:
            return "Hamming's research methodology"
        elif "lesswrong.com" in query:
            return "Research project selection"
        elif "colah.github.io" in query:
            return "Research taste and judgment"
        elif "michaelnielsen.org" in query:
            return "Research methodology principles"
        elif "letters.lossfunk.com" in query:
            return "Research methodology and good science"
        elif "alignmentforum.org" in query:
            return "Research process and ML guidance"
        elif "neelnanda.io" in query:
            return "Mechanistic interpretability methodology"
        elif "joschu.net" in query:
            return "ML research methodology"
        elif "arxiv.org" in query:
            return "Academic research papers"
        else:
            return "Research guidance"
    
    def _get_prioritized_queries(self, topic: str) -> List[str]:
        """Generate prioritized search queries based on topic keywords."""
        topic_lower = topic.lower()
        base_queries = self.config.get_search_queries(topic)
        
        # Prioritize queries based on topic content
        if any(keyword in topic_lower for keyword in ['problem', 'choose', 'select', 'pick']):
            # Prioritize problem selection sources
            return [
                f"site:lesswrong.com {topic} research project",
                f"site:gwern.net {topic} research methodology",
                f"site:letters.lossfunk.com {topic} research methodology",
                f"site:alignmentforum.org {topic} research process",
                f"site:michaelnielsen.org {topic} research principles"
            ]
        elif any(keyword in topic_lower for keyword in ['taste', 'judgment', 'quality', 'good']):
            # Prioritize research taste sources  
            return [
                f"site:colah.github.io {topic} research taste",
                f"site:01.me {topic} research taste",
                f"site:cuhk.edu.hk {topic} research taste",
                f"site:letters.lossfunk.com {topic} research methodology",
                f"site:thoughtforms.life {topic} research advice"
            ]
        elif any(keyword in topic_lower for keyword in ['method', 'process', 'approach', 'how']):
            # Prioritize methodology sources
            return [
                f"site:michaelnielsen.org {topic} research principles",
                f"site:gwern.net {topic} research methodology",
                f"site:letters.lossfunk.com {topic} research methodology",
                f"site:alignmentforum.org {topic} research process",
                f"site:neelnanda.io {topic} research methodology"
            ]
        else:
            # Default to diverse mix
            return base_queries[:6]
    
    def _extract_domain(self, query_str: str) -> str:
        """Extract domain from site: query."""
        match = re.search(r'site:(\S+)', query_str)
        return match.group(1) if match else "unknown"
    
    def _format_guidelines_for_agent(self, topic: str, guidelines: List[Dict[str, Any]]) -> str:
        """Format retrieved guidelines for agent reasoning (RAG-style)."""
        content_parts = [
            f"Retrieved {len(guidelines)} research guidelines for topic: '{topic}'\n"
        ]
        
        for guideline in guidelines:
            guide_id = guideline["guide_id"]
            source_type = guideline["source_type"] 
            content = guideline["content"]
            
            content_parts.extend([
                f"GUIDELINE [{guide_id}]:",
                f"Source: {source_type}",
                f"Content: {content}",
                "---"
            ])
        
        content_parts.append(
            "\nINSTRUCTION: Use the above guidelines to provide evidence-based research advice. "
            "When referencing guidelines in your response, cite them as '[guide_id]' so users "
            "can see which specific sources influenced your advice."
        )
        
        return "\n".join(content_parts)
