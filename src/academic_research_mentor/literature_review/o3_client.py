"""O3 Client for OpenRouter Integration

Provides interface to O3 model via OpenRouter for deep reasoning tasks.
"""

from __future__ import annotations

import os
from typing import Optional, Any

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    ChatOpenAI = None


class O3Client:
    """Client for accessing O3 model via OpenRouter."""
    
    def __init__(self) -> None:
        """Initialize O3 client with OpenRouter configuration."""
        self._client: Optional[Any] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the O3 client if API key is available."""
        if not ChatOpenAI:
            return
            
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return
            
        try:
            self._client = ChatOpenAI(
                model="openai/o3-mini",  # Use O3 mini for better performance
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=4000,  # Sufficient for detailed analysis
            )
        except Exception:
            self._client = None
    
    def is_available(self) -> bool:
        """Check if O3 client is available and ready to use."""
        return self._client is not None
    
    def reason(self, prompt: str, system_message: Optional[str] = None) -> Optional[str]:
        """
        Use O3 for deep reasoning on the given prompt.
        
        Args:
            prompt: The reasoning task/question
            system_message: Optional system context
            
        Returns:
            O3's reasoning response, or None if unavailable
        """
        if not self.is_available():
            return None
            
        try:
            messages = []
            if system_message:
                from langchain_core.messages import SystemMessage  # type: ignore
                messages.append(SystemMessage(content=system_message))
            
            from langchain_core.messages import HumanMessage  # type: ignore
            messages.append(HumanMessage(content=prompt))
            
            result = self._client.invoke(messages)
            return getattr(result, "content", None) or str(result)
            
        except Exception as e:
            print(f"O3 reasoning failed: {e}")
            return None


# Global O3 client instance
_o3_client: Optional[O3Client] = None


def get_o3_client() -> O3Client:
    """Get the global O3 client instance."""
    global _o3_client
    if _o3_client is None:
        _o3_client = O3Client()
    return _o3_client