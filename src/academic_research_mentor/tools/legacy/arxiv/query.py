from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def extract_phrases_and_tokens(raw_query: str) -> Tuple[List[str], List[str]]:
    if not raw_query:
        return [], []

    text = raw_query.strip()

    # If the query already contains arXiv field operators, do not tokenize aggressively
    if re.search(r"\b(?:ti|abs|au|cat|all):", text, flags=re.IGNORECASE):
        return [text], []

    # Pull out quoted phrases first
    phrases = [m.group(1).strip().lower() for m in re.finditer(r'"([^"]+)"', text)]
    text_wo_quotes = re.sub(r'"[^"]+"', ' ', text)

    # Tokenize remaining text on non-alphanumeric boundaries
    raw_tokens = re.split(r"[^A-Za-z0-9_-]+", text_wo_quotes)
    raw_tokens = [tok.lower() for tok in raw_tokens if tok]

    stopwords = {
        "a", "an", "the", "and", "or", "for", "to", "of", "on", "in", "with", "by", "from",
        "at", "as", "is", "are", "be", "being", "into", "via", "using", "use", "based",
        "towards", "toward", "new", "novel",
    }
    tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= 2]

    return phrases, tokens


def detect_ml_domain(query: str) -> Optional[str]:
    query_lower = query.lower()

    domain_keywords: Dict[str, List[str]] = {
        "cs.LG": [
            "machine learning", "neural network", "deep learning", "diffusion", "transformer",
            "gan", "vae", "reinforcement learning", "supervised learning", "unsupervised learning",
            "training", "optimization", "gradient", "backprop", "lstm", "cnn", "rnn",
        ],
        "cs.CV": [
            "computer vision", "image", "video", "visual", "detection", "segmentation",
            "classification", "recognition", "object detection", "face", "ocr", "opencv",
            "multimodal", "vision-language", "vlm", "image-text", "cross-modal", "grounding", "clip",
        ],
        "cs.CL": [
            "natural language", "nlp", "text", "language model", "bert", "gpt", "llm",
            "translation", "sentiment", "tokenization", "parsing", "dialogue",
        ],
        "cs.AI": [
            "artificial intelligence", "planning", "reasoning", "knowledge", "expert system",
            "agent", "multi-agent", "search algorithm", "heuristic",
        ],
        "cs.RO": [
            "robot", "robotics", "manipulation", "navigation", "control", "autonomous",
        ],
        "stat.ML": [
            "statistical learning", "bayesian", "mcmc", "inference", "probability", "statistics",
        ],
    }

    domain_scores: Dict[str, int] = {}
    for category, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            domain_scores[category] = score

    if not domain_scores:
        return None

    return max(domain_scores, key=lambda k: domain_scores[k])


def build_arxiv_query(raw_query: str, from_year: Optional[int]) -> str:
    phrases, tokens = extract_phrases_and_tokens(raw_query)

    clauses: List[str] = []

    if phrases == [raw_query] and not tokens:
        clauses.append(raw_query.strip())
    else:
        detected_domain = detect_ml_domain(raw_query)
        if detected_domain:
            clauses.append(f"cat:{detected_domain}")

        phrase_clauses: List[str] = []
        for phr in phrases:
            safe = phr.replace('"', '')
            hyphen_variant = safe.replace('-', ' ')
            if hyphen_variant != safe and len(hyphen_variant) >= 3:
                phrase_clauses.append(f'(ti:"{safe}" OR abs:"{safe}" OR ti:"{hyphen_variant}" OR abs:"{hyphen_variant}")')
            else:
                phrase_clauses.append(f'(ti:"{safe}" OR abs:"{safe}")')

        token_terms: List[str] = []
        sorted_tokens = sorted(tokens, key=lambda t: (-len(t), t))[:5]
        for tok in sorted_tokens:
            variants = {tok}
            if '-' in tok:
                variants.add(tok.replace('-', ' '))
            variant_clauses: List[str] = []
            for v in variants:
                if len(v) >= 4:
                    variant_clauses.append(f'(ti:{v} OR abs:{v})')
                else:
                    variant_clauses.append(f'(ti:{v} OR abs:{v} OR all:{v})')
            token_terms.append('(' + ' OR '.join(variant_clauses) + ')')

        if phrase_clauses:
            clauses.extend(phrase_clauses)
        if token_terms:
            clauses.append('(' + ' OR '.join(token_terms) + ')')

    if from_year is not None:
        clauses.append(f"submittedDate:[{from_year}01010000+TO+300001010000]")

    return " AND ".join(clauses) if clauses else raw_query.strip()


def relevance_score(title: str, summary: str, phrases: List[str], tokens: List[str]) -> float:
    t = (title or "").lower()
    s = (summary or "").lower()
    score = 0.0

    for p in phrases:
        if p in t:
            score += 5.0
        elif p in s:
            score += 2.0

    for tok in tokens:
        token_pattern = rf"\b{re.escape(tok)}\b"
        if re.search(token_pattern, t):
            weight = 1.5 if len(tok) >= 4 else 1.0
            score += weight
        elif re.search(token_pattern, s):
            weight = 0.8 if len(tok) >= 4 else 0.5
            score += weight

    title_token_matches = sum(1 for tok in tokens if re.search(rf"\b{re.escape(tok)}\b", t))
    if title_token_matches >= 2:
        score += title_token_matches * 0.5

    if len(title) > 100:
        score *= 0.95

    return score
