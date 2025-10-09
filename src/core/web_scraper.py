import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from models.product import ProductInfo, WebEnhancedInfo
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from utils.config import ConfigManager
from utils.text_processing import TextProcessor
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class WebScraper:
    """Handles web scraping and information retrieval for products"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.text_processor = TextProcessor()

        # Configure session with headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.get(
                    "scraping.user_agent",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        # Selenium driver (initialized lazily)
        self._driver = None

        # Rate limiting
        self.last_request_time = 0
        self.request_delay = self.config.get("scraping.delay_between_requests", 1.0)
        self.timeout = self.config.get("scraping.timeout_seconds", 30)

    def enhance_product_with_web_data(self, product: ProductInfo) -> WebEnhancedInfo:
        """
        Enhance product information by searching and scraping web data

        Args:
            product: ProductInfo object to enhance

        Returns:
            WebEnhancedInfo with scraped data
        """
        web_info = WebEnhancedInfo()

        try:
            # Generate search queries
            search_queries = self._generate_search_queries(product)

            # Search for information
            for query in search_queries[:3]:  # Limit to 3 searches
                search_results = self._web_search(query)
                web_info.search_results.extend(search_results)

                # Scrape top results
                for result in search_results[:2]:  # Top 2 results per query
                    if "url" in result:
                        scraped_content = self._scrape_url(result["url"])
                        if scraped_content:
                            web_info.scraped_content[result["url"]] = scraped_content

                # Rate limiting
                time.sleep(self.request_delay)

            # Extract structured information from scraped content
            web_info.specifications = self._extract_specifications(
                web_info.scraped_content
            )
            web_info.reviews = self._extract_reviews(web_info.scraped_content)

        except Exception as e:
            logger.error(f"Error enhancing product {product.id} with web data: {e}")

        return web_info

    def _generate_search_queries(self, product: ProductInfo) -> List[str]:
        """Generate search queries for a product"""
        queries = []

        # Basic product search
        base_query = f'"{product.title}"'
        if product.brand:
            base_query = f"{product.brand} {base_query}"
        queries.append(base_query)

        # Product with category
        if product.category:
            queries.append(f"{product.title} {product.category}")

        # Product specs/reviews
        queries.append(f"{product.title} specifications reviews")

        # Price comparison
        queries.append(f"{product.title} price buy")

        return queries

    def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using multiple search engines

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        results = []

        # Try Google search first
        google_results = self._google_search(query, max_results)
        results.extend(google_results)

        # If not enough results, try DuckDuckGo
        if len(results) < max_results:
            ddg_results = self._duckduckgo_search(query, max_results - len(results))
            results.extend(ddg_results)

        return results[:max_results]

    def _google_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Google (requires API key)"""
        api_key = self.config.get_api_key("google")
        cse_id = self.config.get("web_search.google_cse_id")

        if not api_key or not cse_id:
            logger.debug("Google search API key or CSE ID not configured")
            return []

        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": min(max_results, 10),
            }

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("items", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "google",
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

    def _duckduckgo_search(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            # Use DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            self._rate_limit()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            results = []

            # Get related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "FirstURL" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "").split(" - ")[0],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                            "source": "duckduckgo",
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _scrape_url(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """
        Scrape content from a URL

        Args:
            url: URL to scrape
            use_selenium: Whether to use Selenium for JavaScript-heavy sites

        Returns:
            Scraped text content
        """
        try:
            if use_selenium:
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None

    def _scrape_with_requests(self, url: str) -> Optional[str]:
        """Scrape URL using requests and BeautifulSoup"""
        try:
            self._rate_limit()

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Check content size
            max_size = self.config.get("scraping.max_page_size_mb", 5) * 1024 * 1024
            if len(response.content) > max_size:
                logger.warning(f"Page too large: {url}")
                return None

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script, style, and other non-content elements
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                element.decompose()

            # Extract text content
            text = soup.get_text(separator=" ", strip=True)

            # Clean and limit text
            text = self.text_processor.clean_text(text)

            # Limit text length
            max_length = 10000  # 10k characters
            if len(text) > max_length:
                text = text[:max_length]

            return text

        except Exception as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None

    def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape URL using Selenium for JavaScript-heavy sites"""
        try:
            driver = self._get_selenium_driver()
            if not driver:
                return None

            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Get page text
            text = driver.find_element(By.TAG_NAME, "body").text

            # Clean text
            text = self.text_processor.clean_text(text)

            # Limit text length
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length]

            return text

        except Exception as e:
            logger.error(f"Selenium scraping failed for {url}: {e}")
            return None

    def _get_selenium_driver(self):
        """Get or create Selenium WebDriver"""
        if self._driver is None:
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")

                self._driver = webdriver.Chrome(
                    service=webdriver.chrome.service.Service(
                        ChromeDriverManager().install()
                    ),
                    options=chrome_options,
                )

            except Exception as e:
                logger.error(f"Failed to initialize Selenium driver: {e}")
                return None

        return self._driver

    def _extract_specifications(
        self, scraped_content: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract product specifications from scraped content"""
        specifications = {}

        for url, content in scraped_content.items():
            if not content:
                continue

            # Look for common specification patterns
            specs = {}

            # Extract key-value pairs that look like specifications
            import re

            # Pattern for "Key: Value" format
            spec_patterns = [
                r"([A-Za-z\s]+):\s*([^\n\r]+)",
                r"([A-Za-z\s]+)\s*-\s*([^\n\r]+)",
                r"(\w+(?:\s+\w+)*)\s*:\s*([^\n\r]+)",
            ]

            for pattern in spec_patterns:
                matches = re.findall(pattern, content)
                for key, value in matches:
                    key = key.strip().lower()
                    value = value.strip()

                    # Filter relevant specifications
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
                            ]
                        )
                    ):
                        specs[key] = value

            if specs:
                specifications[url] = specs

        return specifications

    def _extract_reviews(self, scraped_content: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract product reviews from scraped content"""
        reviews = []

        for url, content in scraped_content.items():
            if not content:
                continue

            # Look for review-like content
            sentences = content.split(".")

            for sentence in sentences:
                sentence = sentence.strip()

                # Identify potential reviews (sentences with sentiment indicators)
                if len(sentence) > 20 and any(
                    word in sentence.lower()
                    for word in [
                        "love",
                        "hate",
                        "great",
                        "terrible",
                        "recommend",
                        "avoid",
                        "excellent",
                        "poor",
                    ]
                ):

                    sentiment = self.text_processor.calculate_sentiment(sentence)

                    if (
                        abs(sentiment["polarity"]) > 0.1
                    ):  # Only keep sentences with clear sentiment
                        reviews.append(
                            {
                                "text": sentence,
                                "sentiment": sentiment,
                                "source_url": url,
                            }
                        )

            # Limit reviews per URL
            if len(reviews) >= 5:
                break

        return reviews[:10]  # Return top 10 reviews

    def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time

        if time_diff < self.request_delay:
            time.sleep(self.request_delay - time_diff)

        self.last_request_time = time.time()

    def search_product_info(
        self, product_name: str, category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for general product information

        Args:
            product_name: Name of the product to search for
            category: Optional category to refine search

        Returns:
            List of search results with product information
        """
        query = product_name
        if category:
            query += f" {category}"

        return self._web_search(query, max_results=10)

    def get_product_reviews(
        self, product_name: str, brand: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for product reviews

        Args:
            product_name: Name of the product
            brand: Optional brand name

        Returns:
            List of reviews found
        """
        query = f"{product_name} reviews"
        if brand:
            query = f"{brand} {query}"

        search_results = self._web_search(query, max_results=5)

        all_reviews = []
        for result in search_results:
            if "url" in result:
                content = self._scrape_url(result["url"])
                if content:
                    reviews = self._extract_reviews({result["url"]: content})
                    all_reviews.extend(reviews)

        return all_reviews

    def __del__(self):
        """Cleanup Selenium driver on deletion"""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
