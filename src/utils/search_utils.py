import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .config import ConfigManager

logger = logging.getLogger(__name__)


class SearchUtils:
    """Utility functions for web search operations"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.get(
                    "web_search.user_agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
            }
        )

        # Rate limiting
        self.last_request_time = 0
        self.request_delay = self.config.get("web_search.rate_limit_delay", 1.0)

    def build_search_query(
        self,
        product_title: str,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        additional_terms: Optional[List[str]] = None,
    ) -> str:
        """Build optimized search query for a product"""
        query_parts = [f'"{product_title}"']

        if brand:
            query_parts.append(brand)

        if category:
            query_parts.append(category)

        if additional_terms:
            query_parts.extend(additional_terms)

        return " ".join(query_parts)

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc
        except:
            return ""

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except:
            return False

    def extract_price_from_text(self, text: str) -> List[float]:
        """Extract price values from text"""
        import re

        prices = []

        # Common price patterns
        price_patterns = [
            r"\$[\d,]+\.?\d*",
            r"USD\s*[\d,]+\.?\d*",
            r"[\d,]+\.?\d*\s*dollars?",
            r"Price:\s*\$?[\d,]+\.?\d*",
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert to float
                    clean_price = re.sub(r"[^\d.]", "", match)
                    if clean_price:
                        prices.append(float(clean_price))
                except (ValueError, TypeError):
                    continue

        return prices

    def extract_specifications_from_text(self, text: str) -> Dict[str, str]:
        """Extract product specifications from text"""
        import re

        specs = {}

        # Common specification patterns
        spec_patterns = [
            r"([A-Za-z\s]+):\s*([^\n\r]+)",
            r"([A-Za-z\s]+)\s*-\s*([^\n\r]+)",
            r"(\w+(?:\s+\w+)*)\s*:\s*([^\n\r]+)",
        ]

        for pattern in spec_patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()

                # Filter for relevant specs
                if (
                    len(key) > 2
                    and len(value) > 1
                    and any(
                        keyword in key
                        for keyword in [
                            "size",
                            "weight",
                            "color",
                            "material",
                            "brand",
                            "model",
                            "type",
                            "dimension",
                            "capacity",
                        ]
                    )
                ):
                    specs[key] = value

        return specs

    def categorize_website(self, url: str) -> str:
        """Categorize website type based on URL"""
        domain = self.extract_domain(url).lower()

        # E-commerce sites
        if any(
            site in domain for site in ["amazon", "ebay", "etsy", "shopify", "store"]
        ):
            return "ecommerce"

        # Real estate sites
        if any(
            site in domain
            for site in ["zillow", "realtor", "trulia", "redfin", "homes"]
        ):
            return "real_estate"

        # Review sites
        if any(site in domain for site in ["yelp", "tripadvisor", "reviews", "rating"]):
            return "reviews"

        # News/blog sites
        if any(site in domain for site in ["blog", "news", "article", "post"]):
            return "content"

        # Social media
        if any(
            site in domain for site in ["facebook", "twitter", "instagram", "linkedin"]
        ):
            return "social"

        return "general"

    def rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time

        if time_diff < self.request_delay:
            time.sleep(self.request_delay - time_diff)

        self.last_request_time = time.time()

    def clean_search_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean and filter search results"""
        cleaned_results = []
        seen_urls = set()

        for result in results:
            url = result.get("url", "")

            # Skip duplicates
            if url in seen_urls:
                continue

            # Skip invalid URLs
            if not url or not url.startswith(("http://", "https://")):
                continue

            # Skip common non-useful domains
            domain = self.extract_domain(url).lower()
            if any(
                skip_domain in domain
                for skip_domain in [
                    "pinterest",
                    "youtube",
                    "facebook",
                    "twitter",
                    "instagram",
                ]
            ):
                continue

            seen_urls.add(url)

            # Add website category
            result["website_category"] = self.categorize_website(url)

            cleaned_results.append(result)

        return cleaned_results

    def generate_search_variations(self, base_query: str) -> List[str]:
        """Generate search query variations"""
        variations = [base_query]

        # Add common search modifiers
        modifiers = ["specifications", "reviews", "price", "buy", "features", "details"]

        for modifier in modifiers:
            variations.append(f"{base_query} {modifier}")

        return variations[:5]  # Limit to 5 variations

    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """Extract contact information from text"""
        import re

        contact_info = {"phones": [], "emails": [], "addresses": []}

        # Phone numbers
        phone_patterns = [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\(\d{3}\)\s*\d{3}[-.]?\d{4}",
            r"\b\d{3}\s\d{3}\s\d{4}\b",
        ]

        for pattern in phone_patterns:
            contact_info["phones"].extend(re.findall(pattern, text))

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        contact_info["emails"] = re.findall(email_pattern, text)

        # Simple address patterns (can be enhanced)
        address_pattern = r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl)"
        contact_info["addresses"] = re.findall(address_pattern, text, re.IGNORECASE)

        return contact_info
