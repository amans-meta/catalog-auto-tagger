import json
import logging
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.product import (
    CatalogProcessingResult,
    EnhancedProduct,
    GeneratedTag,
    ProductInfo,
    WebEnhancedInfo,
)
from ..utils.config import ConfigManager
from .catalog_processor import CatalogProcessor
from .tag_classifier import TagClassifier
from .web_scraper import WebScraper

logger = logging.getLogger(__name__)


class TagGenerator:
    """Main orchestrator for the catalog auto-tagging system"""

    def __init__(
        self,
        catalog_type: str = "generic",
        config_manager: Optional[ConfigManager] = None,
    ):
        self.catalog_type = catalog_type
        self.config = config_manager or ConfigManager()

        # Initialize components
        self.catalog_processor = CatalogProcessor(self.config)
        self.web_scraper = WebScraper(self.config)
        self.tag_classifier = TagClassifier(self.config)

        # Processing settings
        self.max_concurrent_products = self.config.get(
            "processing.max_concurrent_products", 3
        )
        self.batch_size = self.config.get("processing.batch_size", 10)
        self.retry_attempts = self.config.get("processing.retry_attempts", 3)

        # Setup logging
        self._setup_logging()

        logger.info(f"TagGenerator initialized for catalog type: {catalog_type}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get("logging.level", "INFO")
        log_file = self.config.get("logging.file", "./logs/catalog_tagger.log")

        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def process_catalog(
        self,
        catalog_path: str,
        enable_web_enhancement: bool = True,
        output_path: Optional[str] = None,
    ) -> CatalogProcessingResult:
        """
        Process an entire catalog file

        Args:
            catalog_path: Path to catalog file
            enable_web_enhancement: Whether to enhance with web data
            output_path: Optional path to save results

        Returns:
            CatalogProcessingResult with processed products
        """
        start_time = time.time()
        logger.info(f"Starting catalog processing: {catalog_path}")

        try:
            # Load and process catalog
            products = self.catalog_processor.process_catalog_file(
                catalog_path, self.catalog_type
            )

            if not products:
                logger.warning("No products found in catalog")
                return CatalogProcessingResult(
                    total_products=0,
                    processed_products=0,
                    failed_products=0,
                    processing_time_seconds=time.time() - start_time,
                    enhanced_products=[],
                )

            logger.info(f"Loaded {len(products)} products from catalog")

            # Validate products
            valid_products = self.catalog_processor.validate_products(products)

            # Process products in batches
            enhanced_products = []
            failed_count = 0

            for i in range(0, len(valid_products), self.batch_size):
                batch = valid_products[i : i + self.batch_size]
                batch_results = self._process_batch(batch, enable_web_enhancement)

                for result in batch_results:
                    if result.processing_status == "complete":
                        enhanced_products.append(result)
                    else:
                        failed_count += 1

                logger.info(
                    f"Processed batch {i//self.batch_size + 1}/{(len(valid_products) + self.batch_size - 1)//self.batch_size}"
                )

            # Create result summary
            processing_time = time.time() - start_time
            result = CatalogProcessingResult(
                total_products=len(products),
                processed_products=len(enhanced_products),
                failed_products=failed_count,
                processing_time_seconds=processing_time,
                enhanced_products=enhanced_products,
                summary_stats=self._generate_summary_stats(enhanced_products),
            )

            # Save results if output path provided
            if output_path:
                self._save_results(result, output_path)

            logger.info(
                f"Catalog processing completed in {processing_time:.2f}s. "
                f"Processed: {len(enhanced_products)}, Failed: {failed_count}"
            )

            return result

        except Exception as e:
            logger.error(f"Catalog processing failed: {e}")
            raise

    def _process_batch(
        self, products: List[ProductInfo], enable_web_enhancement: bool
    ) -> List[EnhancedProduct]:
        """Process a batch of products concurrently"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent_products) as executor:
            # Submit tasks
            future_to_product = {
                executor.submit(
                    self._process_single_product, product, enable_web_enhancement
                ): product
                for product in products
            }

            # Collect results
            for future in as_completed(future_to_product):
                product = future_to_product[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process product {product.id}: {e}")
                    # Create failed result
                    enhanced_product = EnhancedProduct(
                        product_info=product,
                        web_info=WebEnhancedInfo(),
                        processing_status="error",
                        error_message=str(e),
                    )
                    results.append(enhanced_product)

        return results

    def _process_single_product(
        self, product: ProductInfo, enable_web_enhancement: bool
    ) -> EnhancedProduct:
        """Process a single product with retries"""
        for attempt in range(self.retry_attempts):
            try:
                return self._process_product_once(product, enable_web_enhancement)
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for product {product.id}: {e}"
                )
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(1)  # Brief delay before retry

    def _process_product_once(
        self, product: ProductInfo, enable_web_enhancement: bool
    ) -> EnhancedProduct:
        """Process a single product once"""
        start_time = time.time()

        try:
            # Initialize enhanced product
            enhanced_product = EnhancedProduct(
                product_info=product,
                web_info=WebEnhancedInfo(),
                processing_status="processing",
            )

            # Web enhancement (if enabled)
            if enable_web_enhancement and self.config.get("web_search.enabled", True):
                try:
                    enhanced_product.web_info = (
                        self.web_scraper.enhance_product_with_web_data(product)
                    )
                    logger.debug(f"Web enhancement completed for product {product.id}")
                except Exception as e:
                    logger.warning(
                        f"Web enhancement failed for product {product.id}: {e}"
                    )
                    # Continue without web enhancement

            # Tag generation
            try:
                tags = self.tag_classifier.classify_product_tags(
                    product, enhanced_product.web_info, self.catalog_type
                )
                enhanced_product.generated_tags = tags
                logger.debug(f"Generated {len(tags)} tags for product {product.id}")
            except Exception as e:
                logger.error(f"Tag generation failed for product {product.id}: {e}")
                enhanced_product.generated_tags = []

            # Update status
            enhanced_product.processing_status = "complete"

            processing_time = time.time() - start_time
            logger.debug(f"Product {product.id} processed in {processing_time:.2f}s")

            return enhanced_product

        except Exception as e:
            logger.error(f"Product processing failed for {product.id}: {e}")
            raise

    def generate_tags(
        self, product_data: Dict[str, Any], enable_web_enhancement: bool = True
    ) -> List[GeneratedTag]:
        """
        Generate tags for a single product from raw data

        Args:
            product_data: Dictionary containing product information
            enable_web_enhancement: Whether to enhance with web data

        Returns:
            List of GeneratedTag objects
        """
        try:
            # Convert to ProductInfo
            products = self.catalog_processor._dict_to_product(
                product_data, self.catalog_type, 0
            )
            if not products:
                logger.error("Failed to create ProductInfo from data")
                return []

            # Validate
            valid_products = self.catalog_processor.validate_products([products])
            if not valid_products:
                logger.error("Product validation failed")
                return []

            product = valid_products[0]

            # Process product
            enhanced_product = self._process_single_product(
                product, enable_web_enhancement
            )

            return enhanced_product.generated_tags

        except Exception as e:
            logger.error(f"Tag generation failed: {e}")
            return []

    def _generate_summary_stats(
        self, enhanced_products: List[EnhancedProduct]
    ) -> Dict[str, Any]:
        """Generate summary statistics for processed products"""
        if not enhanced_products:
            return {}

        # Tag statistics
        all_tags = []
        tag_sources = {}
        confidence_scores = []

        for product in enhanced_products:
            for tag in product.generated_tags:
                all_tags.append(tag.name)
                confidence_scores.append(tag.confidence)

                if tag.source not in tag_sources:
                    tag_sources[tag.source] = 0
                tag_sources[tag.source] += 1

        # Calculate tag frequency
        tag_frequency = {}
        for tag in all_tags:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        # Most common tags
        most_common_tags = sorted(
            tag_frequency.items(), key=lambda x: x[1], reverse=True
        )[:10]

        stats = {
            "total_tags_generated": len(all_tags),
            "unique_tags": len(set(all_tags)),
            "avg_tags_per_product": (
                len(all_tags) / len(enhanced_products) if enhanced_products else 0
            ),
            "avg_confidence": (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0
            ),
            "tag_sources": tag_sources,
            "most_common_tags": dict(most_common_tags),
            "products_with_web_data": sum(
                1
                for p in enhanced_products
                if p.web_info.search_results or p.web_info.scraped_content
            ),
            "processing_success_rate": len(
                [p for p in enhanced_products if p.processing_status == "complete"]
            )
            / len(enhanced_products),
        }

        return stats

    def _save_results(self, result: CatalogProcessingResult, output_path: str):
        """Save processing results to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary for JSON serialization
            result_dict = {
                "total_products": result.total_products,
                "processed_products": result.processed_products,
                "failed_products": result.failed_products,
                "processing_time_seconds": result.processing_time_seconds,
                "summary_stats": result.summary_stats,
                "products": [],
            }

            for enhanced_product in result.enhanced_products:
                product_dict = {
                    "id": enhanced_product.product_info.id,
                    "title": enhanced_product.product_info.title,
                    "description": enhanced_product.product_info.description,
                    "price": enhanced_product.product_info.price,
                    "currency": enhanced_product.product_info.currency,
                    "category": enhanced_product.product_info.category,
                    "brand": enhanced_product.product_info.brand,
                    "url": enhanced_product.product_info.url,
                    "images": enhanced_product.product_info.images,
                    "attributes": enhanced_product.product_info.attributes,
                    "generated_tags": [
                        {
                            "name": tag.name,
                            "confidence": tag.confidence,
                            "source": tag.source,
                            "reasoning": tag.reasoning,
                        }
                        for tag in enhanced_product.generated_tags
                    ],
                    "processing_status": enhanced_product.processing_status,
                    "processing_timestamp": enhanced_product.processing_timestamp.isoformat(),
                    "web_search_results_count": len(
                        enhanced_product.web_info.search_results
                    ),
                    "scraped_pages_count": len(
                        enhanced_product.web_info.scraped_content
                    ),
                }
                result_dict["products"].append(product_dict)

            # Save to JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_catalog_stats(self, catalog_path: str) -> Dict[str, Any]:
        """Get statistics about a catalog without processing"""
        try:
            products = self.catalog_processor.process_catalog_file(
                catalog_path, self.catalog_type
            )
            return self.catalog_processor.get_catalog_stats(products)
        except Exception as e:
            logger.error(f"Failed to get catalog stats: {e}")
            return {}

    def get_available_tags(self) -> List[str]:
        """Get list of available tags for the current catalog type"""
        return self.tag_classifier.get_available_tags(self.catalog_type)

    def explain_tag(self, tag_name: str) -> Optional[str]:
        """Get explanation for a specific tag"""
        self.tag_classifier.load_catalog_type_tags(self.catalog_type)
        return self.tag_classifier.get_tag_explanation(tag_name)

    def test_single_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test tag generation on a single product (for debugging/testing)

        Args:
            product_data: Dictionary containing product information

        Returns:
            Dictionary with test results
        """
        try:
            start_time = time.time()

            # Generate tags
            tags = self.generate_tags(product_data, enable_web_enhancement=True)

            processing_time = time.time() - start_time

            result = {
                "success": True,
                "processing_time_seconds": processing_time,
                "generated_tags": [
                    {
                        "name": tag.name,
                        "confidence": tag.confidence,
                        "source": tag.source,
                        "reasoning": tag.reasoning,
                    }
                    for tag in tags
                ],
                "tag_count": len(tags),
                "high_confidence_tags": len([t for t in tags if t.confidence > 0.7]),
            }

            return result

        except Exception as e:
            logger.error(f"Single product test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generated_tags": [],
                "tag_count": 0,
                "high_confidence_tags": 0,
            }

    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration settings"""
        for key, value in config_updates.items():
            self.config.update(key, value)

        # Save updated config
        self.config.save_config()
        logger.info("Configuration updated")

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and capabilities"""
        return {
            "catalog_type": self.catalog_type,
            "web_search_enabled": self.config.get("web_search.enabled", True),
            "ml_models_enabled": self.config.get(
                "ml_models.text_classification.enabled", True
            ),
            "available_tags": len(self.get_available_tags()),
            "max_concurrent_products": self.max_concurrent_products,
            "batch_size": self.batch_size,
            "supported_formats": list(self.catalog_processor.supported_formats),
        }
