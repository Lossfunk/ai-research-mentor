"""Configuration for guidelines engine."""

import os
from typing import List, Optional
from enum import Enum


class GuidelinesMode(Enum):
    """Guidelines integration modes."""
    OFF = "off"
    STATIC = "static"
    DYNAMIC = "dynamic"  # Future implementation


class GuidelinesConfig:
    """Configuration class for guidelines engine."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.mode = self._get_guidelines_mode()
        self.max_guidelines = self._get_max_guidelines()
        self.categories = self._get_categories_filter()
        self.format_style = self._get_format_style()
        self.include_stats = self._get_include_stats()
        self.guidelines_path = self._get_guidelines_path()
    
    def _get_guidelines_mode(self) -> GuidelinesMode:
        """Get guidelines mode from environment."""
        mode_str = os.getenv('ARM_GUIDELINES_MODE', 'static').lower()
        try:
            return GuidelinesMode(mode_str)
        except ValueError:
            # Default to static if invalid mode specified
            return GuidelinesMode.STATIC
    
    def _get_max_guidelines(self) -> Optional[int]:
        """Get maximum guidelines count from environment."""
        max_str = os.getenv('ARM_GUIDELINES_MAX')
        if max_str:
            try:
                return int(max_str)
            except ValueError:
                pass
        return None
    
    def _get_categories_filter(self) -> Optional[List[str]]:
        """Get categories filter from environment."""
        categories_str = os.getenv('ARM_GUIDELINES_CATEGORIES')
        if categories_str:
            # Split by comma and clean up whitespace
            return [cat.strip() for cat in categories_str.split(',') if cat.strip()]
        return None
    
    def _get_format_style(self) -> str:
        """Get format style from environment."""
        return os.getenv('ARM_GUIDELINES_FORMAT', 'comprehensive')
    
    def _get_include_stats(self) -> bool:
        """Get whether to include stats in guidelines."""
        return os.getenv('ARM_GUIDELINES_INCLUDE_STATS', 'false').lower() in ('true', '1', 'yes')
    
    def _get_guidelines_path(self) -> Optional[str]:
        """Get custom guidelines file path from environment."""
        return os.getenv('ARM_GUIDELINES_PATH')
    
    @property
    def is_enabled(self) -> bool:
        """Check if guidelines are enabled."""
        return self.mode != GuidelinesMode.OFF
    
    @property
    def is_static_mode(self) -> bool:
        """Check if in static mode."""
        return self.mode == GuidelinesMode.STATIC
    
    @property
    def is_dynamic_mode(self) -> bool:
        """Check if in dynamic mode."""
        return self.mode == GuidelinesMode.DYNAMIC
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for debugging."""
        return {
            'mode': self.mode.value,
            'max_guidelines': self.max_guidelines,
            'categories': self.categories,
            'format_style': self.format_style,
            'include_stats': self.include_stats,
            'guidelines_path': self.guidelines_path,
            'is_enabled': self.is_enabled
        }