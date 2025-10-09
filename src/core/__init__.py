"""
Core processing components for catalog auto-tagging
"""

from .catalog_processor import CatalogProcessor
from .tag_generator import TagGenerator
from .web_scraper import WebScraper

# Optional ML components (requires torch)
try:
    from .tag_classifier import TagClassifier
    __all__ = ["CatalogProcessor", "TagClassifier", "TagGenerator", "WebScraper"]
except ImportError:
    # torch not installed - ML features unavailable
    __all__ = ["CatalogProcessor", "TagGenerator", "WebScraper"]
