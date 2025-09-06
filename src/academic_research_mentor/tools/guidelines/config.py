"""
Configuration for guidelines tool.

Contains curated research guidance sources from leading researchers
and institutions for evidence-based academic mentoring.
"""

from __future__ import annotations

from typing import Dict, List


class GuidelinesConfig:
    """Configuration for research guidelines sources and search parameters."""
    
    # Cost Optimization Settings
    ENABLE_CACHING = True
    CACHE_TTL_HOURS = 24  # Cache responses for 24 hours
    MAX_SEARCH_QUERIES = 3  # Limit to 3 queries to control costs
    ENABLE_COST_MONITORING = True
    
    # Guidelines Sources - Domain mapping for filtering
    GUIDELINE_SOURCES: Dict[str, str] = {
        "gwern.net": "Hamming on research methodology and important problems",
        "lesswrong.com": "Research project selection and evaluation", 
        "colah.github.io": "Research taste and judgment",
        "01.me": "Research taste development",
        "arxiv.org": "Academic papers on research methodology",
        "lifescied.org": "Research process and methodology",
        "trendspider.com": "ML research framing",
        "news.ycombinator.com": "Community discussion on research",
        "cuhk.edu.hk": "Research taste academic perspectives",
        "michaelnielsen.org": "Research methodology principles and effective research",
        "febs.onlinelibrary.wiley.com": "Academic research practices and methodology",
        "researchgate.net": "Research methodology guides and best practices",
        "gigazine.net": "AI research impact and methodology",
        "academic.oup.com": "Academic research practices and methodology",
        "thoughtforms.life": "Student advice and research guidance",
        "letters.lossfunk.com": "Research methodology and good science manifesto",
        "alignmentforum.org": "Research process and ML paper writing guidance",
        "neelnanda.io": "Mechanistic interpretability and research methodology",
        "joschu.net": "ML research methodology and best practices"
    }
    
    # Specific URLs for direct fetching if needed
    GUIDELINE_URLS: List[str] = [
        "https://gwern.net/doc/science/1986-hamming",
        "https://www.lesswrong.com/posts/kDsywodAKgQAAAxE8/how-not-to-choose-a-research-project",
        "https://news.ycombinator.com/item?id=35776480",
        "https://trendspider.com/learning-center/framing-machine-learning-research/",
        "https://arxiv.org/abs/2412.05683",
        "https://www.lifescied.org/doi/10.1187/cbe.20-12-0276",
        "https://arxiv.org/abs/2304.05585",
        "https://colah.github.io/notes/taste/",
        "https://01.me/en/2024/04/research-taste/",
        "https://home.ie.cuhk.edu.hk/~dmchiu/research_taste.pdf",
        "http://michaelnielsen.org/blog/principles-of-effective-research/",
        "https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.15755",
        "https://www.researchgate.net/publication/31052323_Best_Practices_Research_A_Methodological_Guide_for_the_Perplexed",
        "https://gigazine.net/gsc_news/en/20240926-how-to-impactful-ai-research/",
        "https://academic.oup.com/icesjms/article/82/6/fsae121/7754918",
        "https://thoughtforms.life/what-advice-do-i-give-to-my-students/",
        "https://letters.lossfunk.com/p/what-is-research-and-how-to-do-it",
        "https://letters.lossfunk.com/p/manifesto-for-doing-good-science",
        "https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand",
        "https://www.alignmentforum.org/posts/Xt8tMtwfsLo2jRCEj/highly-opinionated-advice-on-how-to-write-ml-papers",
        "https://www.neelnanda.io/mechanistic-interpretability/getting-started",
        "http://joschu.net/blog/opinionated-guide-ml-research.html"
    ]
    
    @classmethod
    def get_search_queries(cls, topic: str) -> List[str]:
        """Generate targeted search queries for different guideline sources."""
        return [
            f"site:gwern.net {topic} research methodology",
            f"site:lesswrong.com {topic} research project", 
            f"site:colah.github.io {topic} research taste",
            f"site:01.me {topic} research taste",
            f"{topic} research methodology site:arxiv.org",
            f"site:lifescied.org {topic} research process",
            f"site:cuhk.edu.hk {topic} research taste",
            f"site:michaelnielsen.org {topic} research",
            f"site:letters.lossfunk.com {topic} research",
            f"site:alignmentforum.org {topic} research",
            f"site:neelnanda.io {topic} research",
            f"site:joschu.net {topic} research"
        ]
