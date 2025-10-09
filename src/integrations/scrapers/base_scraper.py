import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ...utils.config import ConfigManager
from ...utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for site-specific scrapers"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.text_processor = TextProcessor()

        # Configure session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.get(
                    "scraping.user_agent",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                )
            }
        )

        self.timeout = self.config.get("scraping.timeout_seconds", 30)

    @abstractmethod
    def can_scrape(self, url: str) -> bool:
        """Check if this scraper can handle the given URL"""
        pass

    @abstractmethod
    def scrape_product_info(self, url: str) -> Dict[str, Any]:
        """Extract product information from URL"""
        pass

    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Get page content as BeautifulSoup object"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return BeautifulSoup(response.content, "html.parser")

        except Exception as e:
            logger.error(f"Failed to get page content for {url}: {e}")
            return None

    def extract_text_safe(self, element, default: str = "") -> str:
        """Safely extract text from BeautifulSoup element"""
        if element:
            text = element.get_text(strip=True)
            return self.text_processor.clean_text(text) if text else default
        return default

    def extract_price_from_element(self, element) -> Optional[float]:
        """Extract price from BeautifulSoup element"""
        if not element:
            return None

        price_text = self.extract_text_safe(element)
        prices = self.text_processor.extract_entities(price_text).get("prices", [])

        if prices:
            # Try to convert first price found
            price_str = prices[0].replace("$", "").replace(",", "")
            try:
                return float(price_str)
            except (ValueError, TypeError):
                pass

        return None

    def extract_images(self, soup: BeautifulSoup, selectors: List[str]) -> List[str]:
        """Extract image URLs using multiple selectors"""
        images = []

        for selector in selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    # Try different attributes
                    for attr in ["src", "data-src", "data-lazy-src"]:
                        img_url = element.get(attr)
                        if (
                            img_url
                            and isinstance(img_url, str)
                            and img_url.startswith(("http", "//"))
                        ):
                            if img_url.startswith("//"):
                                img_url = "https:" + img_url
                            images.append(img_url)
                            break
            except Exception as e:
                logger.debug(f"Error extracting images with selector '{selector}': {e}")
                continue

        return list(set(images))  # Remove duplicates

    def extract_specifications(
        self, soup: BeautifulSoup, spec_selectors: Dict[str, str]
    ) -> Dict[str, str]:
        """Extract specifications using CSS selectors"""
        specs = {}

        for spec_name, selector in spec_selectors.items():
            try:
                element = soup.select_one(selector)
                if element:
                    value = self.extract_text_safe(element)
                    if value:
                        specs[spec_name] = value
            except Exception as e:
                logger.debug(f"Error extracting spec '{spec_name}': {e}")
                continue

        return specs


class AmazonScraper(BaseScraper):
    """Amazon product page scraper"""

    def can_scrape(self, url: str) -> bool:
        """Check if URL is an Amazon product page"""
        return "amazon.com" in url and "/dp/" in url

    def scrape_product_info(self, url: str) -> Dict[str, Any]:
        """Scrape Amazon product information"""
        soup = self.get_page_content(url)
        if not soup:
            return {}

        product_info = {}

        try:
            # Product title
            title_selectors = ["#productTitle", ".product-title", "h1.a-size-large"]

            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    product_info["title"] = self.extract_text_safe(title_element)
                    break

            # Price
            price_selectors = [
                ".a-price-whole",
                ".a-offscreen",
                ".a-price .a-offscreen",
            ]

            for selector in price_selectors:
                price_element = soup.select_one(selector)
                price = self.extract_price_from_element(price_element)
                if price:
                    product_info["price"] = price
                    break

            # Description/Features
            desc_selectors = [
                "#feature-bullets ul",
                ".a-unordered-list.a-vertical",
                "#productDescription p",
            ]

            features = []
            for selector in desc_selectors:
                elements = soup.select(f"{selector} li, {selector} p")
                for element in elements:
                    text = self.extract_text_safe(element)
                    if text and len(text) > 10:
                        features.append(text)
                if features:
                    break

            if features:
                product_info["description"] = " ".join(features[:5])  # Top 5 features

            # Images
            image_selectors = ["#landingImage", ".a-dynamic-image", "#altImages img"]
            product_info["images"] = self.extract_images(soup, image_selectors)

            # Brand
            brand_element = soup.select_one("#bylineInfo, .a-color-secondary")
            if brand_element:
                brand_text = self.extract_text_safe(brand_element)
                if "by " in brand_text.lower():
                    product_info["brand"] = (
                        brand_text.lower().replace("by ", "").strip()
                    )

            # Specifications
            spec_table = soup.select_one(
                "#productDetails_techSpec_section_1 tr, .a-keyvalue tr"
            )
            if spec_table:
                specs = {}
                for row in soup.select(
                    "#productDetails_techSpec_section_1 tr, .a-keyvalue tr"
                ):
                    cells = row.select("td, th")
                    if len(cells) >= 2:
                        key = self.extract_text_safe(cells[0])
                        value = self.extract_text_safe(cells[1])
                        if key and value:
                            specs[key.lower()] = value
                product_info["specifications"] = specs

            product_info["source"] = "amazon_scraper"

        except Exception as e:
            logger.error(f"Error scraping Amazon product {url}: {e}")

        return product_info


class ZillowScraper(BaseScraper):
    """Zillow real estate listing scraper"""

    def can_scrape(self, url: str) -> bool:
        """Check if URL is a Zillow listing"""
        return "zillow.com" in url and "/homedetails/" in url

    def scrape_product_info(self, url: str) -> Dict[str, Any]:
        """Scrape Zillow listing information"""
        soup = self.get_page_content(url)
        if not soup:
            return {}

        listing_info = {}

        try:
            # Property title/address
            address_selectors = [
                'h1[data-testid="property-details-address"]',
                ".ds-address-container h1",
                ".zsg-photo-card-address",
            ]

            for selector in address_selectors:
                address_element = soup.select_one(selector)
                if address_element:
                    listing_info["title"] = self.extract_text_safe(address_element)
                    break

            # Price
            price_selectors = [
                '[data-testid="property-details-price"]',
                ".ds-estimate-value",
                ".zsg-photo-card-price",
            ]

            for selector in price_selectors:
                price_element = soup.select_one(selector)
                price = self.extract_price_from_element(price_element)
                if price:
                    listing_info["price"] = price
                    break

            # Property details
            details_selectors = {
                "bedrooms": '[data-testid="property-details-beds"]',
                "bathrooms": '[data-testid="property-details-baths"]',
                "square_feet": '[data-testid="property-details-sqft"]',
                "lot_size": '[data-testid="property-details-lot-size"]',
            }

            for detail_name, selector in details_selectors.items():
                element = soup.select_one(selector)
                if element:
                    value = self.extract_text_safe(element)
                    if value:
                        listing_info[detail_name] = value

            # Description
            description_element = soup.select_one(
                ".ds-overview-section, .zsg-lg-1-3 .text-base"
            )
            if description_element:
                listing_info["description"] = self.extract_text_safe(
                    description_element
                )

            # Images
            image_selectors = [
                ".media-stream-photo img",
                ".ds-media-col img",
                "picture img",
            ]
            listing_info["images"] = self.extract_images(soup, image_selectors)

            # Property type
            type_element = soup.select_one(
                ".ds-property-facts-container .Text-c11n-8-65-2__sc-aiai24-0"
            )
            if type_element:
                listing_info["category"] = self.extract_text_safe(type_element)

            listing_info["source"] = "zillow_scraper"

        except Exception as e:
            logger.error(f"Error scraping Zillow listing {url}: {e}")

        return listing_info


class ScraperRegistry:
    """Registry for managing different scrapers"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.scrapers = [AmazonScraper(config_manager), ZillowScraper(config_manager)]

    def get_scraper(self, url: str) -> Optional[BaseScraper]:
        """Get appropriate scraper for URL"""
        for scraper in self.scrapers:
            if scraper.can_scrape(url):
                return scraper
        return None

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL using appropriate scraper"""
        scraper = self.get_scraper(url)
        if scraper:
            return scraper.scrape_product_info(url)

        # Fallback to generic scraping
        logger.debug(f"No specific scraper found for {url}, using generic approach")
        return {}

    def add_scraper(self, scraper: BaseScraper):
        """Add custom scraper to registry"""
        self.scrapers.append(scraper)

    def get_supported_domains(self) -> List[str]:
        """Get list of supported domains"""
        domains = []
        for scraper in self.scrapers:
            # This is a simplified approach - in practice, scrapers might need
            # a method to return their supported domains
            if isinstance(scraper, AmazonScraper):
                domains.append("amazon.com")
            elif isinstance(scraper, ZillowScraper):
                domains.append("zillow.com")
        return domains
