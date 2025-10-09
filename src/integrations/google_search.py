import logging
from typing import Any, Dict, List, Optional

import requests

from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class GoogleSearchAPI:
    """Google Custom Search API integration"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.api_key = self.config.get_api_key("google")
        self.search_engine_id = self.config.get("web_search.google_cse_id")

        if not self.api_key:
            logger.warning("Google Search API key not configured")
        if not self.search_engine_id:
            logger.warning("Google Custom Search Engine ID not configured")

    def is_available(self) -> bool:
        """Check if Google Search API is available"""
        return bool(self.api_key and self.search_engine_id)

    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: Optional[str] = None,
        site_restrict: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform Google Custom Search

        Args:
            query: Search query
            num_results: Number of results to return (max 10 per request)
            search_type: Type of search ('image' for images)
            site_restrict: Restrict search to specific site

        Returns:
            List of search results
        """
        if not self.is_available():
            logger.error("Google Search API not properly configured")
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(num_results, 10),
            }

            if search_type:
                params["searchType"] = search_type

            if site_restrict:
                params["siteSearch"] = site_restrict

            response = requests.get(
                url,
                params=params,
                timeout=self.config.get("web_search.timeout_seconds", 30),
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("items", []):
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_url": item.get("displayLink", ""),
                    "source": "google_custom_search",
                }

                # Add image info if available
                if "pagemap" in item:
                    pagemap = item["pagemap"]
                    if "cse_image" in pagemap:
                        result["image"] = pagemap["cse_image"][0].get("src", "")

                    # Extract metadata
                    if "metatags" in pagemap:
                        metatags = pagemap["metatags"][0] if pagemap["metatags"] else {}
                        result["description"] = metatags.get(
                            "og:description", result["snippet"]
                        )
                        result["site_name"] = metatags.get("og:site_name", "")

                results.append(result)

            # Log search info
            search_info = data.get("searchInformation", {})
            total_results = search_info.get("totalResults", 0)
            search_time = search_info.get("searchTime", 0)

            logger.debug(
                f"Google search: '{query}' returned {len(results)} results "
                f"({total_results} total in {search_time}s)"
            )

            return results

        except requests.RequestException as e:
            logger.error(f"Google Search API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Google Search API error: {e}")
            return []

    def search_images(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search for images related to query"""
        return self.search(query, num_results, search_type="image")

    def search_site(
        self, query: str, site: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search within a specific site"""
        return self.search(query, num_results, site_restrict=site)

    def get_usage_info(self) -> Dict[str, Any]:
        """Get API usage information (if available)"""
        # This would require additional API calls to get quota info
        # For now, return basic info
        return {
            "api_configured": self.is_available(),
            "api_key_length": len(self.api_key) if self.api_key else 0,
            "search_engine_id": self.search_engine_id or "not_configured",
        }
