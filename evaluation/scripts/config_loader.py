from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


METRICS_CONFIG_PATH = Path("evaluation/config/metrics.yaml")
CITATION_DOMAINS_PATH = Path("evaluation/config/citation_domains.yaml")


@lru_cache(maxsize=1)
def load_metrics_config() -> Dict[str, Any]:
    if not METRICS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing metrics config: {METRICS_CONFIG_PATH}")
    with METRICS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


@lru_cache(maxsize=1)
def metrics_config_digest() -> str:
    if not METRICS_CONFIG_PATH.exists():
        return "missing"
    raw = METRICS_CONFIG_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def compute_file_digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_citation_domains() -> Dict[str, list[str]]:
    if not CITATION_DOMAINS_PATH.exists():
        raise FileNotFoundError(f"Missing citation domains config: {CITATION_DOMAINS_PATH}")
    with CITATION_DOMAINS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    normalized: Dict[str, list[str]] = {}
    for kind, domains in data.items():
        if not isinstance(domains, list):
            continue
        normalized[kind] = [str(domain).strip().lower() for domain in domains if domain]
    return normalized


@lru_cache(maxsize=1)
def citation_domains_digest() -> str:
    if not CITATION_DOMAINS_PATH.exists():
        return "missing"
    raw = CITATION_DOMAINS_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()
