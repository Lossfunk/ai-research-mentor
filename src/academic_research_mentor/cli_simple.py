"""Simplified CLI using direct OpenAI SDK agent."""

from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from academic_research_mentor.agent import MentorAgent, ToolRegistry, create_default_tools
from academic_research_mentor.llm import create_client


console = Console()


def load_system_prompt() -> str:
    """Load system prompt from prompt.md."""
    try:
        from academic_research_mentor.prompts_loader import load_instructions_from_prompt_md
        instructions, _ = load_instructions_from_prompt_md("mentor", ascii_normalize=False)
        return instructions or "You are a helpful research mentor."
    except Exception:
        return "You are a helpful research mentor."


def print_response(content: str) -> None:
    """Print formatted response."""
    # Parse thinking blocks
    thinking = None
    main_content = content
    
    if "<thinking>" in content and "</thinking>" in content:
        import re
        match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            main_content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
    
    # Print thinking block if present
    if thinking:
        console.print(Panel(
            thinking,
            title="[yellow]Thinking[/yellow]",
            border_style="yellow",
            expand=False
        ))
    
    # Print main response
    console.print(Markdown(main_content))


def repl(agent: MentorAgent) -> None:
    """Simple REPL loop."""
    console.print(Panel(
        "[bold]Academic Research Mentor[/bold]\n"
        "Type your questions below. Commands: 'exit', 'clear', 'help'",
        border_style="blue"
    ))
    
    while True:
        try:
            console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        lower_input = user_input.lower()
        if lower_input in ('exit', 'quit', 'q'):
            console.print("[dim]Goodbye![/dim]")
            break
        elif lower_input == 'clear':
            agent.clear_history()
            console.print("[dim]History cleared.[/dim]")
            continue
        elif lower_input == 'help':
            console.print(Panel(
                "Commands:\n"
                "  exit/quit/q - Exit the mentor\n"
                "  clear - Clear conversation history\n"
                "  help - Show this help\n\n"
                "Just type your research questions to get started!",
                title="Help",
                border_style="green"
            ))
            continue
        
        # Get response
        console.print("\n[bold green]Mentor:[/bold green]")
        
        try:
            response = agent.chat(user_input)
            print_response(response)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main() -> None:
    """Main entry point."""
    load_dotenv()
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: No API key found.[/red]")
        console.print("Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    # Initialize tools
    tool_registry = ToolRegistry()
    tools_loaded = []
    try:
        for tool in create_default_tools():
            tool_registry.register(tool)
            tools_loaded.append(tool.name)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load some tools: {e}[/yellow]")
    
    # Create agent
    try:
        system_prompt = load_system_prompt()
        provider = "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai"
        client = create_client(provider=provider)
        agent = MentorAgent(
            system_prompt=system_prompt,
            client=client,
            tools=tool_registry if len(tool_registry) > 0 else None
        )
        console.print(f"[dim]Using provider: {provider} | Tools: {', '.join(tools_loaded) or 'none'}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Start REPL
    repl(agent)


if __name__ == "__main__":
    main()
