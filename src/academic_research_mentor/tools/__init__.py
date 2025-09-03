from __future__ import annotations

"""
Tool registry scaffolding.

WS1 goal: provide a minimal API for registering and fetching tools without
altering existing code paths. Auto-discovery will be added later.
"""

from typing import Dict, Optional, Type
import importlib
import pkgutil
import inspect

from .base_tool import BaseTool


class ToolBase(BaseTool):
    """Backward-compatible alias for BaseTool."""


_registry: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    _registry[tool.name] = tool


def get_tool(name: str) -> Optional[BaseTool]:
    return _registry.get(name)


def list_tools() -> Dict[str, BaseTool]:
    return dict(_registry)


def auto_discover(package: str = __name__) -> None:
    """Discover and register tools under this package.

    Convention: any subpackage containing a module named `tool` with at least
    one subclass of BaseTool will be imported and an instance registered.
    """
    # Walk submodules of the tools package recursively
    pkg = importlib.import_module(package)
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        modname = modinfo.name
        # Only consider leaf named 'tool' or modules ending with '.tool'
        if not (modname.endswith(".tool") or modname.split(".")[-1] == "tool"):
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        # Find BaseTool subclasses
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseTool) and obj is not BaseTool:
                try:
                    instance: BaseTool = obj()
                    instance.initialize({})
                    if validate_tool_instance(instance):
                        register_tool(instance)
                except Exception:
                    # Skip tools that fail to initialize
                    continue


def validate_tool_instance(tool: BaseTool) -> bool:
    """Basic sanity checks for a tool instance and its metadata."""
    try:
        if not isinstance(tool.name, str) or not tool.name:
            return False
        meta = tool.get_metadata() or {}
        ident = meta.get("identity", {}) if isinstance(meta, dict) else {}
        if ident.get("name") != tool.name:
            return False
        # Optional minimal IO schema presence
        io = meta.get("io", {})
        if not isinstance(io, dict):
            return False
        return True
    except Exception:
        return False
