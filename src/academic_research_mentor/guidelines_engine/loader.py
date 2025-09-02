"""Guidelines loader for reading and parsing unified guidelines JSON."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class GuidelinesLoader:
    """Loads and parses unified research mentorship guidelines."""
    
    def __init__(self, guidelines_path: Optional[str] = None):
        """Initialize loader with path to guidelines JSON file.
        
        Args:
            guidelines_path: Path to unified_guidelines.json. If None, looks for it
                           relative to the package or in common locations.
        """
        self.guidelines_path = guidelines_path or self._find_guidelines_file()
        self._guidelines_cache: Optional[List[Dict[str, Any]]] = None
    
    def _find_guidelines_file(self) -> str:
        """Find the unified_guidelines.json file in expected locations."""
        # Get the directory containing this file
        current_dir = Path(__file__).parent
        
        # Look for guidelines relative to the package
        search_paths = [
            # New location: academic-research-mentor/unified_guidelines.json
            current_dir.parent.parent.parent / "unified_guidelines.json",
            # Fallback: mentor-guidelines/ directory (from guidelines_engine module)
            current_dir.parent.parent.parent.parent / "mentor-guidelines" / "unified_guidelines.json",
            # Current directory
            current_dir / "unified_guidelines.json",
            # Parent directories
            current_dir.parent / "unified_guidelines.json",
            current_dir.parent.parent / "unified_guidelines.json",
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            f"Could not find unified_guidelines.json in expected locations: {search_paths}"
        )
    
    def load_guidelines(self) -> List[Dict[str, Any]]:
        """Load all guidelines from the JSON file.
        
        Returns:
            List of guideline dictionaries with keys: id, title, content, 
            category, type, source, tags, year
        """
        if self._guidelines_cache is not None:
            return self._guidelines_cache
        
        try:
            with open(self.guidelines_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both direct array and wrapper object formats
            if isinstance(data, list):
                self._guidelines_cache = data
            elif isinstance(data, dict) and 'guidelines' in data:
                self._guidelines_cache = data['guidelines']
            else:
                raise ValueError("Invalid guidelines JSON format: expected array or object with 'guidelines' key")
            
            return self._guidelines_cache
        except FileNotFoundError:
            raise FileNotFoundError(f"Guidelines file not found: {self.guidelines_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in guidelines file: {e}")
    
    def get_guidelines_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all guidelines for a specific category.
        
        Args:
            category: Category to filter by (e.g., 'mentorship', 'research_methods')
        
        Returns:
            List of guidelines matching the category
        """
        guidelines = self.load_guidelines()
        return [g for g in guidelines if g.get('category') == category]
    
    def get_guidelines_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Get guidelines that contain any of the specified tags.
        
        Args:
            tags: List of tags to search for
        
        Returns:
            List of guidelines containing at least one of the tags
        """
        guidelines = self.load_guidelines()
        result = []
        
        for guideline in guidelines:
            guideline_tags = guideline.get('tags', [])
            if any(tag in guideline_tags for tag in tags):
                result.append(guideline)
        
        return result
    
    def get_guidelines_by_type(self, guideline_type: str) -> List[Dict[str, Any]]:
        """Get guidelines of a specific type.
        
        Args:
            guideline_type: Type to filter by (e.g., 'framework', 'technique')
        
        Returns:
            List of guidelines matching the type
        """
        guidelines = self.load_guidelines()
        return [g for g in guidelines if g.get('type') == guideline_type]
    
    def get_guideline_by_id(self, guideline_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific guideline by its ID.
        
        Args:
            guideline_id: The ID of the guideline to retrieve
        
        Returns:
            The guideline dictionary or None if not found
        """
        guidelines = self.load_guidelines()
        for guideline in guidelines:
            if guideline.get('id') == guideline_id:
                return guideline
        return None
    
    def get_categories(self) -> List[str]:
        """Get all unique categories in the guidelines.
        
        Returns:
            List of unique category names
        """
        guidelines = self.load_guidelines()
        categories = set()
        for guideline in guidelines:
            if 'category' in guideline:
                categories.add(guideline['category'])
        return sorted(list(categories))
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all guidelines.
        
        Returns:
            List of unique tags
        """
        guidelines = self.load_guidelines()
        all_tags = set()
        for guideline in guidelines:
            tags = guideline.get('tags', [])
            all_tags.update(tags)
        return sorted(list(all_tags))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded guidelines.
        
        Returns:
            Dictionary with counts and statistics
        """
        guidelines = self.load_guidelines()
        categories = self.get_categories()
        
        stats = {
            'total_guidelines': len(guidelines),
            'categories': len(categories),
            'category_breakdown': {},
            'total_tags': len(self.get_all_tags()),
            'types': []
        }
        
        # Count guidelines per category
        for category in categories:
            stats['category_breakdown'][category] = len(
                self.get_guidelines_by_category(category)
            )
        
        # Get unique types
        types = set()
        for guideline in guidelines:
            if 'type' in guideline:
                types.add(guideline['type'])
        stats['types'] = sorted(list(types))
        
        return stats