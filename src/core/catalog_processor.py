import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from models.product import ProductInfo
from utils.config import ConfigManager
from utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class CatalogProcessor:
    """Processes catalog data from various formats and converts to ProductInfo objects"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.text_processor = TextProcessor()

        # Supported file formats
        self.supported_formats = {".csv", ".xlsx", ".xls", ".json", ".jsonl"}

    def process_catalog_file(
        self, file_path: str, catalog_type: str = "generic"
    ) -> List[ProductInfo]:
        """
        Process a catalog file and return list of ProductInfo objects

        Args:
            file_path: Path to catalog file
            catalog_type: Type of catalog (real_estate, ecommerce, etc.)

        Returns:
            List of ProductInfo objects
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"Catalog file not found: {file_path}")

        if file_path_obj.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")

        logger.info(f"Processing catalog file: {file_path}")

        try:
            if file_path_obj.suffix.lower() == ".json":
                return self._process_json(file_path_obj, catalog_type)
            elif file_path_obj.suffix.lower() == ".jsonl":
                return self._process_jsonl(file_path_obj, catalog_type)
            elif file_path_obj.suffix.lower() == ".csv":
                return self._process_csv(file_path_obj, catalog_type)
            elif file_path_obj.suffix.lower() in {".xlsx", ".xls"}:
                return self._process_excel(file_path_obj, catalog_type)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
        except Exception as e:
            logger.error(f"Error processing catalog file {file_path}: {e}")
            raise

    def _process_csv(self, file_path: Path, catalog_type: str) -> List[ProductInfo]:
        """Process CSV catalog file"""
        products = []

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to read CSV file with any supported encoding")

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        for idx, row in df.iterrows():
            try:
                product = self._row_to_product(row.to_dict(), catalog_type, int(idx))
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(products)} products from CSV")
        return products

    def _process_excel(self, file_path: Path, catalog_type: str) -> List[ProductInfo]:
        """Process Excel catalog file"""
        products = []

        try:
            df = pd.read_excel(file_path)
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            for idx, row in df.iterrows():
                try:
                    product = self._row_to_product(
                        row.to_dict(), catalog_type, int(idx)
                    )
                    if product:
                        products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

        logger.info(f"Successfully processed {len(products)} products from Excel")
        return products

    def _process_json(self, file_path: Path, catalog_type: str) -> List[ProductInfo]:
        """Process JSON catalog file"""
        products = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Look for common keys that might contain product list
            possible_keys = ["products", "items", "listings", "data", "results"]
            items = None
            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break

            if items is None:
                # Treat the dict as a single product
                items = [data]
        else:
            raise ValueError("JSON file must contain an object or array")

        for idx, item in enumerate(items):
            try:
                product = self._dict_to_product(item, catalog_type, idx)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Failed to process item {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(products)} products from JSON")
        return products

    def _process_jsonl(self, file_path: Path, catalog_type: str) -> List[ProductInfo]:
        """Process JSONL (JSON Lines) catalog file"""
        products = []

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    product = self._dict_to_product(item, catalog_type, idx)
                    if product:
                        products.append(product)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {idx + 1}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to process line {idx + 1}: {e}")
                    continue

        logger.info(f"Successfully processed {len(products)} products from JSONL")
        return products

    def _row_to_product(
        self, row: Dict[str, Any], catalog_type: str, row_idx: int
    ) -> Optional[ProductInfo]:
        """Convert a row dictionary to ProductInfo"""
        # Handle NaN values
        row = {k: (v if pd.notna(v) else None) for k, v in row.items()}
        return self._dict_to_product(row, catalog_type, row_idx)

    def _dict_to_product(
        self, data: Dict[str, Any], catalog_type: str, item_idx: int
    ) -> Optional[ProductInfo]:
        """Convert a dictionary to ProductInfo"""
        try:
            # Extract ID - try multiple possible field names
            product_id = self._extract_field(
                data, ["id", "product_id", "listing_id", "item_id", "uuid"]
            )
            if not product_id:
                product_id = f"{catalog_type}_{item_idx}"

            # Extract title/name - required field
            title = self._extract_field(
                data,
                ["title", "name", "product_name", "listing_title", "property_name"],
            )
            if not title:
                logger.warning(f"No title found for item {item_idx}, skipping")
                return None

            # Extract description
            description = self._extract_field(
                data, ["description", "desc", "details", "summary", "about"]
            )

            # Extract price
            price = self._extract_price(data)

            # Extract currency
            currency = (
                self._extract_field(data, ["currency", "curr", "price_currency"])
                or "USD"
            )

            # Extract category
            category = self._extract_field(
                data, ["category", "type", "property_type", "product_category"]
            )

            # Extract brand
            brand = self._extract_field(
                data, ["brand", "manufacturer", "make", "builder"]
            )

            # Extract URL
            url = self._extract_field(data, ["url", "link", "website", "listing_url"])

            # Extract images
            images = self._extract_images(data)

            # Extract attributes (remaining fields)
            attributes = self._extract_attributes(data, catalog_type)

            return ProductInfo(
                id=str(product_id),
                title=str(title),
                description=description,
                price=price,
                currency=currency,
                category=category,
                brand=brand,
                url=url,
                images=images,
                attributes=attributes,
            )

        except Exception as e:
            logger.error(f"Error creating ProductInfo for item {item_idx}: {e}")
            return None

    def _get_field_mappings(self, catalog_type: str) -> Dict[str, List[str]]:
        """Get field mappings for the catalog type"""
        # Get catalog configuration from settings
        catalog_config = (
            self.config.get(f"catalog_types.{catalog_type}", {}) if self.config else {}
        )
        field_mappings = catalog_config.get("field_mappings", {})

        # Load Meta real estate field mappings
        meta_fields = self._load_meta_field_mappings()

        # Default field mappings if not configured
        default_mappings = {
            "id": [
                "id",
                "property_id",
                "listing_id",
                "home_listing_id",
                "mls_id",
                "unique_id",
                "item_id",
            ],
            "title": ["name", "title", "property_name", "listing_title", "heading"],
            "description": [
                "description",
                "desc",
                "details",
                "property_description",
                "summary",
                "about",
                "overview",
            ],
            "price": [
                "price",
                "sale_price",
                "listing_price",
                "cost",
                "amount",
                "rent",
                "rental_price",
            ],
            "category": [
                "property_type",
                "type",
                "building_type",
                "home_type",
                "listing_category",
            ],
            "brand": [
                "brand",
                "agent_company",
                "builder",
                "developer",
                "construction_company",
            ],
            "url": ["url", "link", "website", "listing_url"],
            # Meta-specific real estate fields
            "address": [
                "address",
                "full_address",
                "street_address",
                "property_address",
                "location",
            ],
            "city": ["city", "town", "municipality", "locality"],
            "region": ["region", "state", "province", "territory"],
            "postal_code": ["postal_code", "zip_code", "zip", "pincode"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lng", "long"],
            "neighborhood": [
                "neighborhood",
                "area",
                "district",
                "sector",
                "locality",
                "suburb",
            ],
            # Property details
            "num_beds": [
                "num_beds",
                "bedrooms",
                "beds",
                "bedroom_count",
                "bed_count",
                "bhk",
            ],
            "num_baths": [
                "num_baths",
                "bathrooms",
                "baths",
                "bathroom_count",
                "bath_count",
            ],
            "size_sqft": [
                "size_sqft",
                "sqft",
                "square_feet",
                "area",
                "size",
                "built_up_area",
                "carpet_area",
                "super_area",
            ],
            "year_built": ["year_built", "construction_year", "built_year", "age"],
            "furnish_type": [
                "furnish_type",
                "furnished",
                "furnishing",
                "furnish_status",
            ],
            "parking_type": [
                "parking_type",
                "parking",
                "parking_spaces",
                "garage",
                "covered_parking",
            ],
            "availability": [
                "availability",
                "status",
                "listing_status",
                "property_status",
            ],
            "listing_type": [
                "listing_type",
                "transaction_type",
                "deal_type",
                "purpose",
            ],
            # Images and media
            "image_url": [
                "image_url",
                "primary_image",
                "main_image",
                "photo",
                "picture",
            ],
            "additional_image_url": [
                "additional_image_url",
                "images",
                "photos",
                "pictures",
                "gallery",
                "image_urls",
            ],
            # Agent information
            "agent_name": [
                "agent_name",
                "realtor_name",
                "broker_name",
                "contact_person",
            ],
            "agent_phone": [
                "agent_phone",
                "contact_phone",
                "phone",
                "mobile",
                "telephone",
            ],
            "agent_email": ["agent_email", "contact_email", "email"],
            # Features and amenities
            "features": [
                "features",
                "amenities",
                "facilities",
                "highlights",
                "property_features",
            ],
        }

        # Add Meta field aliases if available
        if meta_fields:
            for field, aliases in meta_fields.items():
                if isinstance(aliases, list):
                    if field in default_mappings:
                        default_mappings[field].extend(aliases)
                    else:
                        default_mappings[field] = aliases

        # Merge with configured mappings
        for key, values in field_mappings.items():
            if key in default_mappings:
                # Extend default with configured values
                default_mappings[key] = list(set(default_mappings[key] + values))
            else:
                # Add new mapping
                default_mappings[key] = values

        return default_mappings

    def _load_meta_field_mappings(self) -> Optional[Dict[str, List[str]]]:
        """Load Meta real estate field mappings from YAML file"""
        try:
            import yaml

            meta_fields_path = (
                Path(__file__).parent.parent.parent
                / "config"
                / "meta_real_estate_fields.yaml"
            )

            if meta_fields_path.exists():
                with open(meta_fields_path, "r") as f:
                    meta_config = yaml.safe_load(f)
                return meta_config.get("field_aliases", {})
        except Exception as e:
            logger.debug(f"Could not load Meta field mappings: {e}")

        return None

    def _extract_field(
        self, data: Dict[str, Any], field_names: List[str]
    ) -> Optional[str]:
        """Extract field value trying multiple possible field names"""
        for field_name in field_names:
            # Try exact match
            if field_name in data and data[field_name] is not None:
                value = data[field_name]
                return str(value).strip() if value else None

            # Try case variations
            for key in data.keys():
                if key.lower() == field_name.lower() and data[key] is not None:
                    value = data[key]
                    return str(value).strip() if value else None

        return None

    def _extract_price(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract and parse price from various formats including INR"""
        price_fields = [
            "price",
            "cost",
            "amount",
            "value",
            "listing_price",
            "sale_price",
        ]

        for field_name in price_fields:
            price_value = self._extract_field(data, [field_name])
            if price_value:
                try:
                    price_str = str(price_value).lower().strip()

                    # Remove currency symbols and commas
                    clean_price = (
                        price_str.replace("$", "")
                        .replace("₹", "")  # Indian Rupee symbol
                        .replace("rs.", "")  # Rs. notation
                        .replace("rs", "")  # Rs notation
                        .replace("inr", "")  # INR notation
                        .replace(",", "")
                        .replace("€", "")
                        .replace("£", "")
                        .replace(" ", "")
                    )

                    # Handle Indian number notations
                    # Handle 'lakh' notation (e.g., "50 lakh" = 5000000)
                    if "lakh" in clean_price or "lac" in clean_price:
                        number_part = (
                            clean_price.replace("lakh", "").replace("lac", "").strip()
                        )
                        return float(number_part) * 100000

                    # Handle 'crore' notation (e.g., "2.5 crore" = 25000000)
                    if "crore" in clean_price or "cr" in clean_price:
                        number_part = (
                            clean_price.replace("crore", "").replace("cr", "").strip()
                        )
                        return float(number_part) * 10000000

                    # Handle 'k' notation (e.g., "250k" = 250000)
                    if clean_price.endswith("k"):
                        return float(clean_price[:-1]) * 1000

                    # Handle 'm' notation (e.g., "1.5m" = 1500000)
                    if clean_price.endswith("m"):
                        return float(clean_price[:-1]) * 1000000

                    # Handle plain numbers
                    return float(clean_price)
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_images(self, data: Dict[str, Any]) -> List[str]:
        """Extract image URLs from data"""
        image_fields = [
            "images",
            "image",
            "photos",
            "photo",
            "picture",
            "pictures",
            "image_url",
            "photo_url",
        ]
        images = []

        for field_name in image_fields:
            value = self._extract_field(data, [field_name])
            if value:
                # Handle comma-separated URLs
                if "," in value:
                    urls = [url.strip() for url in value.split(",") if url.strip()]
                    images.extend(urls)
                else:
                    images.append(value)

        # Also check for numbered image fields (image1, image2, etc.)
        for key, value in data.items():
            if (
                key.lower().startswith("image") or key.lower().startswith("photo")
            ) and value:
                if str(value).startswith("http"):
                    images.append(str(value))

        return list(set(images))  # Remove duplicates

    def _extract_attributes(
        self, data: Dict[str, Any], catalog_type: str
    ) -> Dict[str, Any]:
        """Extract remaining attributes based on catalog type"""
        # Standard fields that are handled separately
        standard_fields = {
            "id",
            "product_id",
            "listing_id",
            "item_id",
            "uuid",
            "title",
            "name",
            "product_name",
            "listing_title",
            "property_name",
            "description",
            "desc",
            "details",
            "summary",
            "about",
            "price",
            "cost",
            "amount",
            "value",
            "listing_price",
            "sale_price",
            "currency",
            "curr",
            "price_currency",
            "category",
            "type",
            "property_type",
            "product_category",
            "brand",
            "manufacturer",
            "make",
            "builder",
            "url",
            "link",
            "website",
            "listing_url",
            "images",
            "image",
            "photos",
            "photo",
            "picture",
            "pictures",
            "image_url",
            "photo_url",
        }

        attributes = {}

        for key, value in data.items():
            if key.lower() not in standard_fields and not key.lower().startswith(
                ("image", "photo")
            ):
                if value is not None and str(value).strip():
                    attributes[key] = value

        return attributes

    def validate_products(self, products: List[ProductInfo]) -> List[ProductInfo]:
        """Validate and clean product data"""
        valid_products = []

        for product in products:
            # Basic validation
            if not product.title or len(product.title.strip()) < 2:
                logger.warning(f"Product {product.id} has invalid title, skipping")
                continue

            # Clean and validate price
            if product.price is not None and (product.price < 0 or product.price > 1e9):
                logger.warning(
                    f"Product {product.id} has suspicious price {product.price}"
                )
                product.price = None

            # Clean description
            if product.description:
                product.description = self.text_processor.clean_text(
                    product.description
                )

            valid_products.append(product)

        logger.info(f"Validated {len(valid_products)} out of {len(products)} products")
        return valid_products

    def get_catalog_stats(self, products: List[ProductInfo]) -> Dict[str, Any]:
        """Generate statistics about the catalog"""
        if not products:
            return {}

        stats = {
            "total_products": len(products),
            "has_description": sum(1 for p in products if p.description),
            "has_price": sum(1 for p in products if p.price is not None),
            "has_category": sum(1 for p in products if p.category),
            "has_brand": sum(1 for p in products if p.brand),
            "has_images": sum(1 for p in products if p.images),
            "avg_title_length": sum(len(p.title) for p in products) / len(products),
            "categories": list(set(p.category for p in products if p.category)),
            "brands": list(set(p.brand for p in products if p.brand)),
        }

        # Price statistics
        prices = [p.price for p in products if p.price is not None]
        if prices:
            stats["price_stats"] = {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
                "count": len(prices),
            }

        return stats
