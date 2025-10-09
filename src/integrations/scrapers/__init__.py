"""
Site-specific scrapers for different product types
"""

from .base_scraper import AmazonScraper, BaseScraper, ScraperRegistry, ZillowScraper

__all__ = ["BaseScraper", "AmazonScraper", "ZillowScraper", "ScraperRegistry"]
