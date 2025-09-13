from __future__ import annotations

import re
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule


class RichFormatter:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def print_response(self, content: str, title: Optional[str] = None) -> None:
        if not content.strip():
            return
        if self._has_markdown_elements(content):
            self._print_markdown_response(content, title)
        else:
            self._print_text_response(content, title)

    def print_streaming_chunk(self, chunk: str) -> None:
        self.console.print(chunk, end="", highlight=False)

    def start_streaming_response(self, title: str = "Mentor") -> None:
        self.console.print("")
        self.console.print(f"[bold green]{title}:[/bold green]")

    def end_streaming_response(self) -> None:
        self.console.print("")

    def print_error(self, message: str) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_info(self, message: str) -> None:
        self.console.print(f"[bold blue]Info:[/bold blue] {message}")

    def print_success(self, message: str) -> None:
        self.console.print(f"[bold green]Success:[/bold green] {message}")

    def print_rule(self, title: Optional[str] = None) -> None:
        self.console.print(Rule(title=title, style="blue"))

    def print_section(self, content: str, title: str, border_style: str = "blue") -> None:
        if not content.strip():
            return
        try:
            if self._has_markdown_elements(content):
                body = Markdown(self._process_markdown_content(content))
            else:
                text = Text(content)
                url_pattern = r'https?://[^\s]+'
                for match in re.finditer(url_pattern, content):
                    start, end = match.span()
                    text.stylize("blue underline", start, end)
                body = text
            panel = Panel(body, title=f"[bold]{title}[/bold]", border_style=border_style)
            self.console.print(panel)
        except Exception:
            self.console.print(f"[bold]{title}[/bold]\n{content}")

    def _has_markdown_elements(self, content: str) -> bool:
        patterns = [
            r'^#{1,6}\s',
            r'```',
            r'`[^`]+`',
            r'\*\*[^*]+\*\*',
            r'\*[^*]+\*',
            r'^\s*[-*+]\s',
            r'^\s*\d+\.\s',
            r'\[.+\]\(.+\)',
        ]
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False

    def _print_markdown_response(self, content: str, title: Optional[str] = None) -> None:
        try:
            processed_content = self._process_markdown_content(content)
            markdown = Markdown(processed_content)
            if title:
                panel = Panel(markdown, title=f"[bold blue]{title}[/bold blue]", border_style="blue")
                self.console.print(panel)
            else:
                self.console.print(markdown)
        except Exception:
            self._print_text_response(content, title)

    def _print_text_response(self, content: str, title: Optional[str] = None) -> None:
        text = Text(content)
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, content):
            start, end = match.span()
            text.stylize("blue underline", start, end)
        if title:
            panel = Panel(text, title=f"[bold green]{title}[/bold green]", border_style="green")
            self.console.print(panel)
        else:
            self.console.print(text)

    def _process_markdown_content(self, content: str) -> str:
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content


_global_formatter: Optional[RichFormatter] = None


def get_formatter() -> RichFormatter:
    global _global_formatter
    if _global_formatter is None:
        _global_formatter = RichFormatter()
    return _global_formatter
