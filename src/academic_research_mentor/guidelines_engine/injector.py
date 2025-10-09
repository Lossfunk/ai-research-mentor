"""Guidelines injector for integrating guidelines into agent prompts."""

from typing import Optional, Dict, Any

from .config import GuidelinesConfig


class GuidelinesInjector:
    """Handles injection of research mentorship guidelines into agent prompts."""

    def __init__(self, config: Optional[GuidelinesConfig] = None):
        """Initialize injector with configuration.

        Args:
            config: Guidelines configuration. If None, creates default config.
        """
        self.config = config or GuidelinesConfig()

    def inject_guidelines(self, base_prompt: str) -> str:
        """Inject guidelines into a base prompt.

        Args:
            base_prompt: The original prompt to enhance with guidelines

        Returns:
            Enhanced prompt with guidelines injected
        """
        if not self.config.is_enabled:
            return base_prompt

        hint = (
            "When the user asks for research mentorship, methodology, problem selection, "
            "or research taste guidance, prefer calling the research_guidelines tool to gather "
            "evidence-based guidance before answering."
        )
        return f"{base_prompt}\n\n{hint}"

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

        if self.config.is_enabled and self.config.is_dynamic_mode:
            sources_count = None
            approx_tokens = 0
            try:
                from ..tools.guidelines.config import GuidelinesConfig as ToolsGuidelinesConfig  # type: ignore
                sources_count = len(getattr(ToolsGuidelinesConfig, 'GUIDELINE_SOURCES', {}).keys())
                if not sources_count:
                    sources_count = len(getattr(ToolsGuidelinesConfig, 'GUIDELINE_URLS', []))
                per_source = getattr(ToolsGuidelinesConfig, 'DEFAULT_MAX_PER_SOURCE', 3)
                global_cap = getattr(ToolsGuidelinesConfig, 'RESULT_CAP', 30)
                approx_items = min(global_cap, sources_count * max(1, per_source))
                approx_tokens = int(approx_items * 75)
            except Exception:
                sources_count = 0
                approx_tokens = 0

            stats['guidelines_stats'] = {
                "tool_backed": True,
                "sources": sources_count,
                "total_guidelines": sources_count,
            }
            stats['token_estimate'] = approx_tokens

        return stats


def create_guidelines_injector() -> GuidelinesInjector:
    """Factory function to create a configured guidelines injector.
    
    Returns:
        Configured GuidelinesInjector instance
    """
    config = GuidelinesConfig()
    return GuidelinesInjector(config)