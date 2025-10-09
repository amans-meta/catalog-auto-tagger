"""
External service integrations for web search and scraping
"""

from .google_search import GoogleSearchAPI
from .scrapers import AmazonScraper, BaseScraper, ScraperRegistry, ZillowScraper

__all__ = [
    "GoogleSearchAPI",
    "BaseScraper",
    "AmazonScraper",
    "ZillowScraper",
    "ScraperRegistry",
]
