"""Guidelines injector for integrating guidelines into agent prompts."""

from typing import Optional, List, Dict, Any
from .loader import GuidelinesLoader
from .formatter import GuidelinesFormatter
from .config import GuidelinesConfig


class GuidelinesInjector:
    """Handles injection of research mentorship guidelines into agent prompts."""
    
    def __init__(self, config: Optional[GuidelinesConfig] = None):
        """Initialize injector with configuration.
        
        Args:
            config: Guidelines configuration. If None, creates default config.
        """
        self.config = config or GuidelinesConfig()
        self.loader = GuidelinesLoader(self.config.guidelines_path)
        self.formatter = GuidelinesFormatter(self.config.max_guidelines)
        self._guidelines_cache: Optional[str] = None
    
    def inject_guidelines(self, base_prompt: str) -> str:
        """Inject guidelines into a base prompt.
        
        Args:
            base_prompt: The original prompt to enhance with guidelines
        
        Returns:
            Enhanced prompt with guidelines injected
        """
        if not self.config.is_enabled:
            return base_prompt
        
        guidelines_section = self.get_guidelines_section()
        if not guidelines_section:
            return base_prompt
        
        # Inject guidelines between prompt sections
        # Look for a good injection point or append at the end
        injection_point = self._find_injection_point(base_prompt)
        
        if injection_point == -1:
            # No good injection point found, append at the end
            return f"{base_prompt}\n\n{guidelines_section}"
        else:
            # Insert at the found injection point
            before = base_prompt[:injection_point]
            after = base_prompt[injection_point:]
            return f"{before}\n\n{guidelines_section}\n\n{after}"
    
    def get_guidelines_section(self) -> str:
        """Get the formatted guidelines section.
        
        Returns:
            Formatted guidelines text ready for injection
        """
        if not self.config.is_enabled:
            return ""
        
        # Use cached version if available
        if self._guidelines_cache is not None:
            return self._guidelines_cache
        
        try:
            # Load guidelines
            if self.config.categories:
                # Filter by categories if specified
                all_guidelines = self.loader.load_guidelines()
                guidelines = [
                    g for g in all_guidelines 
                    if g.get('category') in self.config.categories
                ]
            else:
                # Load all guidelines
                guidelines = self.loader.load_guidelines()
            
            if not guidelines:
                return ""
            
            # Format guidelines
            self._guidelines_cache = self.formatter.create_guidelines_section(
                guidelines=guidelines,
                section_title="Research Mentorship Guidelines",
                format_style=self.config.format_style,
                include_stats=self.config.include_stats
            )
            
            return self._guidelines_cache
            
        except Exception as e:
            # Log error but don't break the application
            print(f"Warning: Failed to load guidelines: {e}")
            return ""
    
    def _find_injection_point(self, prompt: str) -> int:
        """Find the best point to inject guidelines in the prompt.
        
        Args:
            prompt: The base prompt text
        
        Returns:
            Character index for injection, or -1 if no good point found
        """
        # Look for common section markers where guidelines could be injected
        injection_markers = [
            "\n## Guidelines",
            "\n## Instructions",
            "\n## Context",
            "\n## Additional Information",
            "\n## Background",
            "\n# Guidelines",
            "\n# Instructions", 
            "\n# Context",
            "\n# Additional Information",
            "\n# Background"
        ]
        
        for marker in injection_markers:
            index = prompt.find(marker)
            if index != -1:
                return index
        
        # Look for end of main content before examples
        example_markers = [
            "\n## Examples",
            "\n# Examples",
            "\n## Example",
            "\n# Example"
        ]
        
        for marker in example_markers:
            index = prompt.find(marker)
            if index != -1:
                return index
        
        # No good injection point found
        return -1
    
    def get_token_estimate(self) -> int:
        """Get estimated token count for the guidelines section.
        
        Returns:
            Estimated token count
        """
        guidelines_section = self.get_guidelines_section()
        return self.formatter.get_token_estimate(guidelines_section)
    
    def clear_cache(self):
        """Clear the cached guidelines section."""
        self._guidelines_cache = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded guidelines.
        
        Returns:
            Dictionary with guidelines statistics and config info
        """
        stats = {
            'config': self.config.to_dict(),
            'guidelines_stats': {},
            'token_estimate': 0
        }
        
        if self.config.is_enabled:
            try:
                guidelines = self.loader.load_guidelines()
                stats['guidelines_stats'] = self.loader.get_stats()
                stats['token_estimate'] = self.get_token_estimate()
            except Exception as e:
                stats['error'] = str(e)
        
        return stats


def create_guidelines_injector() -> GuidelinesInjector:
    """Factory function to create a configured guidelines injector.
    
    Returns:
        Configured GuidelinesInjector instance
    """
    config = GuidelinesConfig()
    return GuidelinesInjector(config)