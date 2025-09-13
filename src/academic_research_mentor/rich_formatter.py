"""Rich formatting utilities for enhanced console output."""

from __future__ import annotations

import re
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule


class RichFormatter:
    """Enhanced formatter using Rich library for better console output."""
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the Rich formatter.
        
        Args:
            console: Optional Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()
    
    def print_response(self, content: str, title: Optional[str] = None) -> None:
        """Print a formatted response using Rich.
        
        Args:
            content: The content to display
            title: Optional title for the response panel
        """
        if not content.strip():
            return
            
        # Detect if content looks like markdown or contains code blocks
        if self._has_markdown_elements(content):
            self._print_markdown_response(content, title)
        else:
            self._print_text_response(content, title)
    
    def print_streaming_chunk(self, chunk: str) -> None:
        """Print a streaming chunk with basic formatting.
        
        Args:
            chunk: Text chunk to print
        """
        self.console.print(chunk, end="", highlight=False)
    
    def start_streaming_response(self, title: str = "Mentor") -> None:
        """Start a streaming response with proper header.
        
        Args:
            title: Title for the response section
        """
        self.console.print("")  # Add spacing
        self.console.print(f"[bold green]{title}:[/bold green]")
    
    def end_streaming_response(self) -> None:
        """End a streaming response with proper footer."""
        self.console.print("")  # Add spacing after response
    
    def print_error(self, message: str) -> None:
        """Print an error message with Rich formatting.
        
        Args:
            message: Error message to display
        """
        self.console.print(f"[bold red]Error:[/bold red] {message}")
    
    def print_info(self, message: str) -> None:
        """Print an info message with Rich formatting.
        
        Args:
            message: Info message to display
        """
        self.console.print(f"[bold blue]Info:[/bold blue] {message}")
    
    def print_success(self, message: str) -> None:
        """Print a success message with Rich formatting.
        
        Args:
            message: Success message to display
        """
        self.console.print(f"[bold green]Success:[/bold green] {message}")
    
    def print_rule(self, title: Optional[str] = None) -> None:
        """Print a horizontal rule with optional title.
        
        Args:
            title: Optional title for the rule
        """
        self.console.print(Rule(title=title, style="blue"))
    
    def _has_markdown_elements(self, content: str) -> bool:
        """Check if content contains markdown elements.
        
        Args:
            content: Content to check
            
        Returns:
            True if content appears to contain markdown elements
        """
        # Check for common markdown patterns
        patterns = [
            r'^#{1,6}\s',  # Headers
            r'```',        # Code blocks
            r'`[^`]+`',    # Inline code
            r'\*\*[^*]+\*\*',  # Bold text
            r'\*[^*]+\*',      # Italic text
            r'^\s*[-*+]\s',    # Lists
            r'^\s*\d+\.\s',    # Numbered lists
            r'\[.+\]\(.+\)',   # Links
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    def _print_markdown_response(self, content: str, title: Optional[str] = None) -> None:
        """Print response as markdown with Rich formatting.
        
        Args:
            content: Markdown content to display
            title: Optional title for the panel
        """
        try:
            # Process the content to handle special elements
            processed_content = self._process_markdown_content(content)
            markdown = Markdown(processed_content)
            
            if title:
                panel = Panel(markdown, title=f"[bold blue]{title}[/bold blue]", border_style="blue")
                self.console.print(panel)
            else:
                self.console.print(markdown)
        except Exception:
            # Fallback to text rendering if markdown parsing fails
            self._print_text_response(content, title)
    
    def _print_text_response(self, content: str, title: Optional[str] = None) -> None:
        """Print response as plain text with basic Rich formatting.
        
        Args:
            content: Text content to display
            title: Optional title for the panel
        """
        # Apply basic highlighting for URLs, email addresses, etc.
        text = Text(content)
        
        # Highlight URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, content):
            start, end = match.span()
            text.stylize("blue underline", start, end)
        
        if title:
            panel = Panel(text, title=f"[bold green]{title}[/bold green]", border_style="green")
            self.console.print(panel)
        else:
            self.console.print(text)
    
    def print_section(self, content: str, title: str, border_style: str = "blue") -> None:
        """Print a titled panel section with consistent styling.
        
        Args:
            content: Content to display
            title: Section title
            border_style: Rich style for panel border
        """
        if not content.strip():
            return
        try:
            if self._has_markdown_elements(content):
                body = Markdown(self._process_markdown_content(content))
            else:
                # Apply basic URL highlighting even in sections
                text = Text(content)
                url_pattern = r'https?://[^\s]+'
                for match in re.finditer(url_pattern, content):
                    start, end = match.span()
                    text.stylize("blue underline", start, end)
                body = text
            panel = Panel(body, title=f"[bold]{title}[/bold]", border_style=border_style)
            self.console.print(panel)
        except Exception:
            # Graceful fallback
            self.console.print(f"[bold]{title}[/bold]\n{content}")

    def _process_markdown_content(self, content: str) -> str:
        """Process markdown content to enhance certain elements.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Processed markdown content
        """
        # Handle code blocks with syntax highlighting hints
        # This is already handled by Rich's Markdown renderer
        
        # Ensure proper spacing around elements
        content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize multiple newlines
        
        return content


# Global formatter instance
_global_formatter: Optional[RichFormatter] = None


def get_formatter() -> RichFormatter:
    """Get the global Rich formatter instance.
    
    Returns:
        Global RichFormatter instance
    """
    global _global_formatter
    if _global_formatter is None:
        _global_formatter = RichFormatter()
    return _global_formatter


def print_formatted_response(content: str, title: Optional[str] = None) -> None:
    """Convenience function to print a formatted response.
    
    Args:
        content: Content to display
        title: Optional title for the response
    """
    get_formatter().print_response(content, title)


def print_streaming_chunk(chunk: str) -> None:
    """Convenience function to print a streaming chunk.
    
    Args:
        chunk: Text chunk to print
    """
    get_formatter().print_streaming_chunk(chunk)


def start_streaming_response(title: str = "Mentor") -> None:
    """Convenience function to start a streaming response.
    
    Args:
        title: Title for the response section
    """
    get_formatter().start_streaming_response(title)


def end_streaming_response() -> None:
    """Convenience function to end a streaming response."""
    get_formatter().end_streaming_response()


def print_error(message: str) -> None:
    """Convenience function to print an error message.
    
    Args:
        message: Error message to display
    """
    get_formatter().print_error(message)


def print_info(message: str) -> None:
    """Convenience function to print an info message.
    
    Args:
        message: Info message to display
    """
    get_formatter().print_info(message)


def print_success(message: str) -> None:
    """Convenience function to print a success message.
    
    Args:
        message: Success message to display
    """
    get_formatter().print_success(message)


def print_agent_reasoning(content: str) -> None:
    """Print content as an 'Agent's reasoning' section.
    
    Args:
        content: Reasoning or tool traces to show as internal context.
    """
    get_formatter().print_section(content, "Agent's reasoning", border_style="magenta")


def print_user_input(content: str) -> None:
    """Print the user's prompt in a consistent panel.
    
    Args:
        content: The user's input message.
    """
    get_formatter().print_section(content, "You", border_style="cyan")
