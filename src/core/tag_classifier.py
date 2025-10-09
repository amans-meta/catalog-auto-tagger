import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.product import GeneratedTag, ProductInfo, WebEnhancedInfo
from models.tag_config import (
    EcommerceTagConfig,
    RealEstateTagConfig,
    TagDefinition,
    TagType,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoTokenizer
from utils.config import ConfigManager
from utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class TagClassifier:
    """ML-based tag classification system"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.text_processor = TextProcessor()

        # Initialize models
        self.tfidf_vectorizer = None
        self.ml_model = None
        self.transformer_model = None
        self.transformer_tokenizer = None

        # Tag configurations
        self.tag_definitions = {}
        self.loaded_catalog_types = set()

        # Model cache directory
        self.cache_dir = Path(
            self.config.get("ml_models.text_classification.cache_dir", "./models/cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_catalog_type_tags(self, catalog_type: str) -> None:
        """Load predefined tags for a catalog type"""
        if catalog_type in self.loaded_catalog_types:
            return

        if catalog_type == "real_estate":
            tags = RealEstateTagConfig.get_tags()
        elif catalog_type == "ecommerce":
            tags = EcommerceTagConfig.get_tags()
        else:
            # Generic tags - you can expand this
            tags = self._get_generic_tags()

        for tag in tags:
            self.tag_definitions[tag.name] = tag

        self.loaded_catalog_types.add(catalog_type)
        logger.info(f"Loaded {len(tags)} tag definitions for {catalog_type}")

    def _get_generic_tags(self) -> List[TagDefinition]:
        """Get generic tag definitions"""
        return [
            TagDefinition(
                name="popular",
                tag_type=TagType.POPULARITY,
                keywords=["popular", "trending", "bestseller", "top-rated"],
                description="Popular or trending item",
                priority=3,
            ),
            TagDefinition(
                name="affordable",
                tag_type=TagType.PRICE_RANGE,
                keywords=["affordable", "cheap", "budget", "low-cost"],
                patterns=[r"\$[1-9][0-9]?(?!\d)", r"under.*100"],
                description="Budget-friendly option",
                priority=4,
            ),
            TagDefinition(
                name="premium",
                tag_type=TagType.PRICE_RANGE,
                keywords=["premium", "luxury", "high-end", "expensive"],
                patterns=[r"\$[5-9][0-9]{2,}", r"luxury"],
                description="Premium/luxury option",
                priority=3,
            ),
        ]

    def classify_product_tags(
        self,
        product: ProductInfo,
        web_info: Optional[WebEnhancedInfo] = None,
        catalog_type: str = "generic",
    ) -> List[GeneratedTag]:
        """
        Classify and generate tags for a product

        Args:
            product: ProductInfo object
            web_info: Optional web-enhanced information
            catalog_type: Type of catalog for tag selection

        Returns:
            List of GeneratedTag objects
        """
        # Load appropriate tag definitions
        self.load_catalog_type_tags(catalog_type)

        all_tags = []

        # Rule-based tagging
        rule_based_tags = self._rule_based_classification(product, web_info)
        all_tags.extend(rule_based_tags)

        # ML-based tagging (if enabled)
        if self.config.get("ml_models.text_classification.enabled", True):
            ml_tags = self._ml_based_classification(product, web_info)
            all_tags.extend(ml_tags)

        # Transformer-based tagging (if available)
        if self._is_transformer_available():
            transformer_tags = self._transformer_based_classification(product, web_info)
            all_tags.extend(transformer_tags)

        # Consolidate and filter tags
        final_tags = self._consolidate_tags(all_tags)

        # Apply confidence threshold
        min_confidence = self.config.get("output.min_confidence_threshold", 0.3)
        final_tags = [tag for tag in final_tags if tag.confidence >= min_confidence]

        # Limit number of tags
        max_tags = self.config.get("output.max_tags_per_product", 20)
        final_tags = sorted(final_tags, key=lambda x: x.confidence, reverse=True)[
            :max_tags
        ]

        return final_tags

    def _rule_based_classification(
        self, product: ProductInfo, web_info: Optional[WebEnhancedInfo] = None
    ) -> List[GeneratedTag]:
        """Apply rule-based tag classification"""
        tags = []

        # Combine all text content
        text_content = self._get_combined_text(product, web_info)

        for tag_name, tag_def in self.tag_definitions.items():
            confidence = self._calculate_rule_based_confidence(
                text_content, product, tag_def
            )

            if confidence >= tag_def.min_confidence:
                tags.append(
                    GeneratedTag(
                        name=tag_name,
                        confidence=confidence,
                        source="rule_based",
                        reasoning=f"Matched keywords/patterns for {tag_def.tag_type.value}",
                    )
                )

        return tags

    def _calculate_rule_based_confidence(
        self, text_content: str, product: ProductInfo, tag_def: TagDefinition
    ) -> float:
        """Calculate confidence score for rule-based matching"""
        confidence = 0.0

        # Keyword matching
        if tag_def.keywords:
            keyword_matches = sum(
                1
                for keyword in tag_def.keywords
                if self.text_processor.contains_keywords(text_content, [keyword])
            )
            keyword_confidence = min(keyword_matches / len(tag_def.keywords), 1.0) * 0.6
            confidence += keyword_confidence

        # Pattern matching
        if tag_def.patterns:
            pattern_matches = len(
                self.text_processor.extract_patterns(text_content, tag_def.patterns)
            )
            pattern_confidence = min(pattern_matches / len(tag_def.patterns), 1.0) * 0.4
            confidence += pattern_confidence

        # Synonym matching
        if tag_def.synonyms:
            synonym_matches = sum(
                1
                for synonym in tag_def.synonyms
                if self.text_processor.contains_keywords(text_content, [synonym])
            )
            synonym_confidence = min(synonym_matches / len(tag_def.synonyms), 1.0) * 0.3
            confidence += synonym_confidence

        # Category-specific rules
        confidence += self._apply_category_specific_rules(product, tag_def)

        # Price-based rules
        if product.price and tag_def.tag_type == TagType.PRICE_RANGE:
            confidence += self._apply_price_rules(product.price, tag_def)

        return min(confidence, 1.0)

    def _apply_category_specific_rules(
        self, product: ProductInfo, tag_def: TagDefinition
    ) -> float:
        """Apply category-specific classification rules"""
        confidence_boost = 0.0

        # Brand matching
        if tag_def.tag_type == TagType.BRAND and product.brand:
            if any(
                keyword.lower() in product.brand.lower() for keyword in tag_def.keywords
            ):
                confidence_boost += 0.4

        # Category matching
        if product.category:
            category_lower = product.category.lower()
            if any(keyword.lower() in category_lower for keyword in tag_def.keywords):
                confidence_boost += 0.3

        return confidence_boost

    def _apply_price_rules(self, price: float, tag_def: TagDefinition) -> float:
        """Apply price-based classification rules"""
        confidence_boost = 0.0

        # Simple price range rules
        if tag_def.name == "budget_friendly" and price < 100:
            confidence_boost += 0.5
        elif tag_def.name == "mid_range" and 100 <= price <= 1000:
            confidence_boost += 0.5
        elif tag_def.name == "premium" and price > 1000:
            confidence_boost += 0.5

        # Real estate specific price rules
        elif tag_def.name == "budget_friendly" and price < 300000:
            confidence_boost += 0.4
        elif tag_def.name == "high_end" and price > 700000:
            confidence_boost += 0.4

        return confidence_boost

    def _ml_based_classification(
        self, product: ProductInfo, web_info: Optional[WebEnhancedInfo] = None
    ) -> List[GeneratedTag]:
        """Apply ML-based tag classification"""
        tags = []

        try:
            # Get text features
            text_content = self._get_combined_text(product, web_info)
            features = self._extract_ml_features(text_content, product)

            # Use TF-IDF for text similarity
            if not self.tfidf_vectorizer:
                self._initialize_tfidf()

            # Calculate similarity with tag definitions
            for tag_name, tag_def in self.tag_definitions.items():
                similarity = self._calculate_text_similarity(text_content, tag_def)

                if similarity > 0.3:  # Minimum similarity threshold
                    confidence = min(similarity * 1.2, 1.0)  # Boost similarity score
                    tags.append(
                        GeneratedTag(
                            name=tag_name,
                            confidence=confidence,
                            source="ml_tfidf",
                            reasoning=f"Text similarity score: {similarity:.3f}",
                        )
                    )

        except Exception as e:
            logger.error(f"ML classification failed: {e}")

        return tags

    def _transformer_based_classification(
        self, product: ProductInfo, web_info: Optional[WebEnhancedInfo] = None
    ) -> List[GeneratedTag]:
        """Apply transformer-based tag classification"""
        tags = []

        try:
            if not self._initialize_transformer():
                return tags

            text_content = self._get_combined_text(product, web_info)

            # Get embeddings for product text
            product_embedding = self._get_text_embedding(text_content)

            # Compare with tag definition embeddings
            for tag_name, tag_def in self.tag_definitions.items():
                tag_text = " ".join(tag_def.keywords + tag_def.synonyms)
                if tag_def.description:
                    tag_text += " " + tag_def.description

                tag_embedding = self._get_text_embedding(tag_text)

                # Calculate cosine similarity
                similarity = cosine_similarity([product_embedding], [tag_embedding])[0][
                    0
                ]

                if similarity > 0.4:  # Minimum similarity threshold
                    confidence = min(similarity * 1.1, 1.0)
                    tags.append(
                        GeneratedTag(
                            name=tag_name,
                            confidence=confidence,
                            source="transformer",
                            reasoning=f"Semantic similarity: {similarity:.3f}",
                        )
                    )

        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")

        return tags

    def _get_combined_text(
        self, product: ProductInfo, web_info: Optional[WebEnhancedInfo] = None
    ) -> str:
        """Combine all text content for analysis"""
        text_parts = [product.title]

        if product.description:
            text_parts.append(product.description)

        if product.category:
            text_parts.append(product.category)

        if product.brand:
            text_parts.append(product.brand)

        # Add web-scraped content
        if web_info and web_info.scraped_content:
            for content in web_info.scraped_content.values():
                if content:
                    # Limit web content length
                    content_excerpt = content[:1000]
                    text_parts.append(content_excerpt)

        # Add search results snippets
        if web_info and web_info.search_results:
            for result in web_info.search_results:
                if "snippet" in result:
                    text_parts.append(result["snippet"])

        return " ".join(text_parts)

    def _extract_ml_features(
        self, text_content: str, product: ProductInfo
    ) -> Dict[str, Any]:
        """Extract ML features from product data"""
        features = {}

        # Text features
        text_features = self.text_processor.extract_features(text_content)
        features.update(text_features)

        # Product-specific features
        features["has_price"] = product.price is not None
        features["has_brand"] = product.brand is not None
        features["has_category"] = product.category is not None
        features["has_images"] = len(product.images) > 0
        features["num_images"] = len(product.images)

        if product.price:
            features["log_price"] = np.log10(max(product.price, 1))
            features["price_range"] = self._get_price_range(product.price)

        return features

    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges"""
        if price < 100:
            return "very_low"
        elif price < 500:
            return "low"
        elif price < 2000:
            return "medium"
        elif price < 10000:
            return "high"
        else:
            return "very_high"

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        try:
            # Create training corpus from tag definitions
            corpus = []
            for tag_def in self.tag_definitions.values():
                tag_text = " ".join(tag_def.keywords + tag_def.synonyms)
                if tag_def.description:
                    tag_text += " " + tag_def.description
                corpus.append(tag_text)

            if corpus:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000, stop_words="english", ngram_range=(1, 2)
                )
                self.tfidf_vectorizer.fit(corpus)

        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF: {e}")

    def _calculate_text_similarity(self, text: str, tag_def: TagDefinition) -> float:
        """Calculate text similarity using TF-IDF"""
        try:
            if not self.tfidf_vectorizer:
                return 0.0

            tag_text = " ".join(tag_def.keywords + tag_def.synonyms)
            if tag_def.description:
                tag_text += " " + tag_def.description

            # Vectorize texts
            vectors = self.tfidf_vectorizer.transform([text, tag_text])

            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity

        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return 0.0

    def _is_transformer_available(self) -> bool:
        """Check if transformer model is available"""
        return self.config.get("ml_models.text_classification.enabled", True)

    def _initialize_transformer(self) -> bool:
        """Initialize transformer model"""
        try:
            if self.transformer_model is None:
                model_name = self.config.get(
                    "ml_models.text_classification.model_name",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )

                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_model = AutoModel.from_pretrained(model_name)

                # Set device
                device = self.config.get("ml_models.text_classification.device", "auto")
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                self.transformer_model.to(device)

            return True

        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            return False

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using transformer model"""
        try:
            # Tokenize text
            inputs = self.transformer_tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy()[0]

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return np.zeros(384)  # Default embedding size

    def _consolidate_tags(self, tags: List[GeneratedTag]) -> List[GeneratedTag]:
        """Consolidate tags from multiple sources"""
        # Group tags by name
        tag_groups = {}
        for tag in tags:
            if tag.name not in tag_groups:
                tag_groups[tag.name] = []
            tag_groups[tag.name].append(tag)

        consolidated_tags = []

        for tag_name, tag_list in tag_groups.items():
            if len(tag_list) == 1:
                consolidated_tags.append(tag_list[0])
            else:
                # Combine multiple sources for the same tag
                max_confidence = max(tag.confidence for tag in tag_list)
                sources = list(set(tag.source for tag in tag_list))

                # Average confidence with boost for multiple sources
                avg_confidence = sum(tag.confidence for tag in tag_list) / len(tag_list)
                final_confidence = min((avg_confidence + max_confidence) / 2 * 1.1, 1.0)

                consolidated_tags.append(
                    GeneratedTag(
                        name=tag_name,
                        confidence=final_confidence,
                        source="+".join(sources),
                        reasoning=f"Combined from {len(tag_list)} sources",
                    )
                )

        return consolidated_tags

    def get_tag_explanation(self, tag_name: str) -> Optional[str]:
        """Get explanation for a specific tag"""
        if tag_name in self.tag_definitions:
            tag_def = self.tag_definitions[tag_name]
            return tag_def.description
        return None

    def get_available_tags(self, catalog_type: str = None) -> List[str]:
        """Get list of available tags for a catalog type"""
        if catalog_type:
            self.load_catalog_type_tags(catalog_type)

        return list(self.tag_definitions.keys())
