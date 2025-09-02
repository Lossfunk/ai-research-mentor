"""Guidelines formatter for converting guidelines into prompt-ready text."""

from typing import Dict, List, Any, Optional


class GuidelinesFormatter:
    """Formats research mentorship guidelines for injection into prompts."""
    
    def __init__(self, max_guidelines: Optional[int] = None):
        """Initialize formatter.
        
        Args:
            max_guidelines: Maximum number of guidelines to include in output.
                          If None, includes all guidelines.
        """
        self.max_guidelines = max_guidelines
    
    def format_guidelines_for_prompt(
        self, 
        guidelines: List[Dict[str, Any]], 
        format_style: str = "comprehensive"
    ) -> str:
        """Format guidelines into text suitable for prompt injection.
        
        Args:
            guidelines: List of guideline dictionaries
            format_style: Formatting style - "comprehensive", "compact", or "minimal"
        
        Returns:
            Formatted guidelines text ready for prompt injection
        """
        if not guidelines:
            return ""
        
        # Limit guidelines if max_guidelines is set
        if self.max_guidelines and len(guidelines) > self.max_guidelines:
            guidelines = guidelines[:self.max_guidelines]
        
        if format_style == "comprehensive":
            return self._format_comprehensive(guidelines)
        elif format_style == "compact":
            return self._format_compact(guidelines)
        elif format_style == "minimal":
            return self._format_minimal(guidelines)
        else:
            raise ValueError(f"Unknown format_style: {format_style}")
    
    def _format_comprehensive(self, guidelines: List[Dict[str, Any]]) -> str:
        """Format guidelines with full details including metadata."""
        sections = ["# Research Mentorship Guidelines", ""]
        
        # Group by category
        categories = {}
        for guideline in guidelines:
            category = guideline.get('category', 'uncategorized')
            if category not in categories:
                categories[category] = []
            categories[category].append(guideline)
        
        for category, cat_guidelines in categories.items():
            sections.append(f"## {category.replace('_', ' ').title()}")
            sections.append("")
            
            for guideline in cat_guidelines:
                sections.append(f"**{guideline.get('title', 'Untitled')}** ({guideline.get('id', 'no-id')})")
                sections.append(f"{guideline.get('content', 'No content available')}")
                
                # Add metadata
                metadata_parts = []
                if guideline.get('type'):
                    metadata_parts.append(f"Type: {guideline['type']}")
                if guideline.get('source'):
                    metadata_parts.append(f"Source: {guideline['source']}")
                if guideline.get('year'):
                    metadata_parts.append(f"Year: {guideline['year']}")
                
                if metadata_parts:
                    sections.append(f"*{' | '.join(metadata_parts)}*")
                
                sections.append("")
        
        return "\n".join(sections)
    
    def _format_compact(self, guidelines: List[Dict[str, Any]]) -> str:
        """Format guidelines in a more concise style."""
        sections = ["# Research Mentorship Guidelines", ""]
        
        for i, guideline in enumerate(guidelines, 1):
            title = guideline.get('title', 'Untitled')
            content = guideline.get('content', 'No content available')
            category = guideline.get('category', 'general')
            
            sections.append(f"{i}. **{title}** [{category}]")
            sections.append(f"   {content}")
            sections.append("")
        
        return "\n".join(sections)
    
    def _format_minimal(self, guidelines: List[Dict[str, Any]]) -> str:
        """Format guidelines with just essential content."""
        sections = ["# Research Mentorship Guidelines", ""]
        
        for guideline in guidelines:
            content = guideline.get('content', 'No content available')
            sections.append(f"• {content}")
        
        sections.append("")
        return "\n".join(sections)
    
    def format_guidelines_by_category(
        self, 
        guidelines: List[Dict[str, Any]], 
        categories: List[str],
        format_style: str = "comprehensive"
    ) -> str:
        """Format only guidelines from specified categories.
        
        Args:
            guidelines: List of all guidelines
            categories: List of categories to include
            format_style: Formatting style
        
        Returns:
            Formatted guidelines text for specified categories only
        """
        filtered_guidelines = [
            g for g in guidelines 
            if g.get('category') in categories
        ]
        return self.format_guidelines_for_prompt(filtered_guidelines, format_style)
    
    def format_guidelines_by_tags(
        self,
        guidelines: List[Dict[str, Any]],
        tags: List[str],
        format_style: str = "comprehensive"
    ) -> str:
        """Format guidelines that contain any of the specified tags.
        
        Args:
            guidelines: List of all guidelines
            tags: List of tags to filter by
            format_style: Formatting style
        
        Returns:
            Formatted guidelines text for guidelines with matching tags
        """
        filtered_guidelines = []
        for guideline in guidelines:
            guideline_tags = guideline.get('tags', [])
            if any(tag in guideline_tags for tag in tags):
                filtered_guidelines.append(guideline)
        
        return self.format_guidelines_for_prompt(filtered_guidelines, format_style)
    
    def create_guidelines_section(
        self,
        guidelines: List[Dict[str, Any]],
        section_title: str = "Research Mentorship Guidelines",
        format_style: str = "comprehensive",
        include_stats: bool = False
    ) -> str:
        """Create a complete guidelines section for prompt injection.
        
        Args:
            guidelines: List of guidelines to format
            section_title: Title for the guidelines section
            format_style: Formatting style
            include_stats: Whether to include statistics about guidelines
        
        Returns:
            Complete formatted section ready for prompt injection
        """
        sections = []
        
        # Add section header
        sections.append(f"# {section_title}")
        sections.append("")
        
        # Add stats if requested
        if include_stats:
            categories = set(g.get('category', 'uncategorized') for g in guidelines)
            sections.append(f"*{len(guidelines)} guidelines across {len(categories)} categories*")
            sections.append("")
        
        # Add formatted guidelines
        formatted_guidelines = self.format_guidelines_for_prompt(guidelines, format_style)
        # Remove the header since we added our own
        if formatted_guidelines.startswith("# Research Mentorship Guidelines"):
            lines = formatted_guidelines.split('\n')
            # Skip the first two lines (header and empty line)
            formatted_guidelines = '\n'.join(lines[2:])
        
        sections.append(formatted_guidelines)
        
        # Add closing note
        sections.append("")
        sections.append("*Use these guidelines to inform your mentorship approach and responses.*")
        
        return "\n".join(sections)
    
    def get_token_estimate(self, text: str) -> int:
        """Rough estimate of token count for the formatted text.
        
        Args:
            text: Formatted guidelines text
        
        Returns:
            Estimated token count (rough approximation)
        """
        # Rough approximation: 4 characters per token on average
        return len(text) // 4