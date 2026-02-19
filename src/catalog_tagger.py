#!/usr/bin/env python3
"""
Catalog-Only Tagger - Production Ready
Uses ONLY catalog data for tagging (no web enrichment)
Optimized for speed and large-scale processing
"""

import concurrent.futures
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils import load_catalog_config, load_settings, load_tags

# Load settings early for logging config
_settings = load_settings()
_log_cfg = _settings.get("logging", {})
logging.basicConfig(
    level=getattr(logging, _log_cfg.get("level", "INFO")),
    format=_log_cfg.get("format", "%(levelname)s: %(message)s"),
)
logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    """Result from tagging"""

    tag_name: str
    confidence: float
    category: str
    evidence: List[str]
    source: str


@dataclass
class CatalogEntry:
    """Represents a catalog entry"""

    id: str
    title: str
    description: str
    price: float
    category: str
    location: str
    full_text: str


class CatalogOnlyTagger:
    """Fast tagger that uses ONLY catalog data (no web scraping)"""

    def __init__(self, catalog_type: str):
        if not catalog_type:
            raise ValueError(
                "catalog_type must be specified (e.g., 'real_estate', 'automotive', 'electronics')"
            )
        self.catalog_type = catalog_type
        self.settings = _settings
        self.embedding_model = None
        self._initialize_embeddings()

        # Load catalog configuration and tags - FAIL if not available
        try:
            self.catalog_config = load_catalog_config(catalog_type)
            self.tags = load_tags(catalog_type)
            if not self.tags:
                raise ValueError(f"No tags found for catalog_type '{catalog_type}'")

            logger.info(
                f"Loaded {len(self.tags)} tags for '{catalog_type}' from configuration"
            )
        except Exception as e:
            logger.error(f"Failed to load tags for '{catalog_type}': {e}")
            logger.error(
                f"Please ensure config/tags/{catalog_type}_tags.yaml exists"
            )
            logger.error(
                f"Available examples: config/tags/real_estate_tags.yaml, config/tags/custom_example.yaml"
            )
            raise ValueError(
                f"Cannot initialize tagger without valid tag configuration for '{catalog_type}'"
            )

        self._prepare_tag_embeddings()

    def _initialize_embeddings(self):
        """Initialize sentence transformers"""
        try:
            import logging
            import os

            from sentence_transformers import SentenceTransformer

            # Disable verbose output
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)

            ml_cfg = self.settings.get("ml_models", {}).get("sentence_transformers", {})
            model_name = ml_cfg.get("model_name", "all-MiniLM-L6-v2")
            show_progress = ml_cfg.get("show_progress_bar", False)

            # Try with show_progress_bar first, fall back if not supported
            try:
                self.embedding_model = SentenceTransformer(
                    model_name, show_progress_bar=show_progress
                )
            except TypeError:
                # Older versions don't support show_progress_bar parameter
                self.embedding_model = SentenceTransformer(model_name)
            logger.info("Loaded sentence-transformers for semantic similarity")
        except ImportError:
            logger.info(
                "sentence-transformers not available, using keyword matching only"
            )
            self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            self.embedding_model = None

    def _prepare_tag_embeddings(self):
        """Prepare embeddings for tags"""
        if not self.embedding_model:
            return

        embeddings_created = 0
        for tag_name, tag_def in self.tags.items():
            tag_text = f"{tag_name} {' '.join(tag_def.get('keywords', []))}"
            try:
                embedding = self.embedding_model.encode([tag_text])[0]
                tag_def["embedding"] = embedding
                embeddings_created += 1
            except Exception as e:
                logger.warning(f"Failed to create embedding for {tag_name}: {e}")

        logger.info(f"Created embeddings for {embeddings_created} tags")

    def _compute_semantic_similarity(self, text: str, tag_def: dict) -> float:
        """Compute semantic similarity using embeddings"""
        if not self.embedding_model or "embedding" not in tag_def:
            return 0.0

        try:
            text_embedding = self.embedding_model.encode([text])[0]
            tag_embedding = tag_def["embedding"]

            # Cosine similarity
            text_norm = text_embedding / np.linalg.norm(text_embedding)
            tag_norm = tag_embedding / np.linalg.norm(tag_embedding)

            similarity = np.dot(text_norm, tag_norm)
            return float(similarity)
        except Exception as e:
            logger.debug(f"Semantic similarity computation failed: {e}")
            return 0.0

    def match_tags_catalog_only(self, entry: CatalogEntry) -> List[TagResult]:
        """Match tags using ONLY catalog data (no web enrichment)"""
        results = []
        text_lower = entry.full_text.lower()

        thresholds = self.settings.get("processing", {}).get("confidence_thresholds", {})
        default_threshold = thresholds.get("default", 0.15)
        semantic_threshold = thresholds.get("semantic_only", default_threshold)
        confidence_overrides = self.catalog_config.get("confidence_overrides", {})

        for tag_name, tag_def in self.tags.items():
            confidence = 0.0
            evidence = []
            source = "textual"

            # Check matching mode
            matching_mode = tag_def.get("matching_mode", "semantic")  # default to semantic

            # Skip semantic/text matching for price_only tags
            if matching_mode == "price_only":
                # Only use price-based matching below, skip all text matching
                pass
            else:
                # SEMANTIC SIMILARITY on catalog text only (skip if text_only mode)
                semantic_score = 0.0
                if matching_mode != "text_only":
                    semantic_score = self._compute_semantic_similarity(entry.full_text, tag_def)
                    if semantic_score > 0:
                        confidence += semantic_score * 0.6  # 60% weight for semantic
                        evidence.append(f"semantic_similarity: {semantic_score:.3f}")
                        source = "semantic"

                # Keyword matching
                keywords = tag_def.get("keywords", [])
                keyword_matches = []

                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        keyword_matches.append(keyword)
                        confidence += 0.3 * tag_def["weight"]

                if keyword_matches:
                    evidence.append(f"keywords: {keyword_matches}")

                # Pattern matching
                patterns = tag_def.get("patterns", [])
                pattern_matches = []
                for pattern in patterns:
                    try:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            pattern_matches.append(pattern)
                            confidence += 0.4 * tag_def["weight"]
                    except re.error:
                        continue

                if pattern_matches:
                    evidence.append(f"patterns: {pattern_matches}")

            # Price-based matching
            price_range = tag_def.get("price_range")
            if price_range and entry.price > 0:
                min_price, max_price = price_range
                if min_price <= entry.price < max_price:
                    confidence += 0.8 * tag_def["weight"]
                    evidence.append(f"price: {entry.price:,.0f} in range")

            # Category-specific boosts and verification
            if tag_def["category"] == "property_type":
                # Verify property_type tags against catalog data (if available)
                if entry.category and entry.category.strip():
                    if tag_name in entry.category.lower():
                        confidence += 0.5
                        evidence.append("category match")
                    else:
                        confidence *= 0.2  # Reduce confidence by 80%
                        evidence.append("category_mismatch_penalty")
            elif entry.category and tag_name in entry.category.lower():
                confidence += 0.5
                evidence.append("category match")

            # Location specific boosts
            if tag_def["category"] == "location" and entry.location:
                location_lower = entry.location.lower()
                for keyword in tag_def.get("keywords", []):
                    if keyword.lower() in location_lower:
                        confidence += 0.3
                        evidence.append(f"location match: {keyword}")

            # Apply confidence threshold
            category = tag_def["category"]
            if category in confidence_overrides:
                threshold = confidence_overrides[category]
            else:
                threshold = semantic_threshold if semantic_score > 0 else default_threshold

            if confidence >= threshold:
                results.append(
                    TagResult(
                        tag_name=tag_name,
                        confidence=min(confidence, 1.0),
                        category=tag_def["category"],
                        evidence=evidence,
                        source=source,
                    )
                )

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results


class MultithreadedCatalogTagger:
    """Multithreaded catalog-only tagger for large-scale processing"""

    def __init__(self, max_workers: int = 8, catalog_type: str = "real_estate"):
        self.max_workers = max_workers
        self.matcher = CatalogOnlyTagger(catalog_type)
        self.catalog_config = self.matcher.catalog_config
        self.settings = self.matcher.settings

    def _resolve_field(self, row: Dict[str, Any], config_key: str, settings_key: str) -> str:
        """Resolve a field value from row data using catalog config then settings fallback.

        1. If catalog_config has an explicit field name, use that
        2. Otherwise, try each alias from settings field_mappings
        3. Return the first matching column value, or empty string
        """
        # Try catalog config first
        field_name = self.catalog_config.get(config_key)
        if field_name:
            val = row.get(field_name)
            if val is not None and pd.notna(val):
                return str(val)

        # Fall back to settings field_mappings aliases
        aliases = self.settings.get("field_mappings", {}).get(settings_key, [])
        for alias in aliases:
            val = row.get(alias)
            if val is not None and pd.notna(val):
                return str(val)

        return ""

    def process_chunk(self, chunk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of catalog entries (catalog-only)"""
        results = []
        basic_categories = self.catalog_config.get("basic_tag_categories", [])
        output_fields = self.catalog_config.get("output_fields", {})

        for row_data in chunk_data:
            try:
                # Create catalog entry
                entry = self._create_catalog_entry(row_data)

                # Generate tags using catalog data only
                tags = self.matcher.match_tags_catalog_only(entry)

                # Categorize tags
                basic_tags = []
                advanced_tags = []

                for tag in tags:
                    if basic_categories and tag.category in basic_categories:
                        basic_tags.append(tag.tag_name)
                    else:
                        advanced_tags.append(tag.tag_name)

                # Location extraction from catalog data
                city_tag = entry.location.lower() if entry.location else ""
                state_tag = self._extract_state_from_location(city_tag)

                # Create result
                result = {
                    "Entry_ID": entry.id,
                    "Title": entry.title,
                    "Price": entry.price,
                    "Location": entry.location,
                    "Category": entry.category,
                    "Total_Tags": len(tags),
                    "basic_tags": ", ".join(basic_tags),
                    "advanced_tags": ", ".join(advanced_tags),
                    "city_tag": city_tag,
                    "state_tag": state_tag,
                    "Top_5_Tags": ", ".join([tag.tag_name for tag in tags[:5]]),
                    "Tag_Details": " | ".join(
                        [f"{tag.tag_name}({tag.confidence:.2f})" for tag in tags]
                    ),
                    "Categories": ", ".join(list(set([tag.category for tag in tags]))),
                    "Web_Enhanced": "No",  # Always No for catalog-only
                }

                # Add domain-specific output columns from config
                for output_col, input_field in output_fields.items():
                    if output_col not in result:
                        val = row_data.get(input_field)
                        result[output_col] = str(val) if val is not None and pd.notna(val) else ""

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process entry: {e}")
                continue

        return results

    def _extract_state_from_location(self, location: str) -> str:
        """Extract state from location using config-driven mapping"""
        if not location:
            return ""

        city_state_mapping = self.catalog_config.get("city_state_mapping", {})
        return city_state_mapping.get(location.lower(), "")

    def _create_catalog_entry(self, row_data: Dict[str, Any]) -> CatalogEntry:
        """Create catalog entry from row data"""
        # Resolve fields using config with settings fallback
        entry_id = self._resolve_field(row_data, "id_field", "id")
        if not entry_id:
            entry_id = f"entry_{hash(str(row_data))}"

        title = self._resolve_field(row_data, "title_field", "title")

        # Description: try catalog_config description_fields, then settings fallback
        description = ""
        desc_fields = self.catalog_config.get("description_fields")
        if not desc_fields:
            desc_fields = self.settings.get("field_mappings", {}).get("description", [])
        for field in desc_fields:
            if field in row_data and pd.notna(row_data[field]):
                description = str(row_data[field])
                break

        # Parse price
        price = 0.0
        price_field = self.catalog_config.get("price_field")
        if not price_field:
            # Try settings fallback
            for alias in self.settings.get("field_mappings", {}).get("price", []):
                if alias in row_data:
                    price_field = alias
                    break
        if price_field and price_field in row_data and pd.notna(row_data[price_field]):
            price_str = (
                str(row_data[price_field]).replace("INR", "").replace(",", "").strip()
            )
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                price = 0.0

        category = self._resolve_field(row_data, "category_field", "category")
        location = self._resolve_field(row_data, "location_field", "location")

        # Create full text
        full_text_parts = [title, description, category, location]

        # Build excluded fields dynamically from config
        excluded_fields = set()
        for config_key in ["id_field", "title_field", "price_field", "category_field", "location_field", "cache_key_field"]:
            field_val = self.catalog_config.get(config_key)
            if field_val:
                excluded_fields.add(field_val)
        # Also exclude description fields
        if desc_fields:
            excluded_fields.update(desc_fields)

        for key, value in row_data.items():
            # Skip custom labels and excluded fields
            if "custom_label" in key.lower() or "customlabel" in key.lower():
                continue

            if pd.notna(value) and isinstance(value, str) and len(value) < 200:
                if key not in excluded_fields:
                    full_text_parts.append(value)

        full_text = " ".join([str(part) for part in full_text_parts if part])

        return CatalogEntry(
            id=entry_id,
            title=title,
            description=description,
            price=price,
            category=category,
            location=location,
            full_text=full_text,
        )

    def process_large_file(
        self, input_file: str, output_file: str, chunk_size: int = 200
    ):
        """Process large files with multithreading (catalog-only, faster)"""
        print(f"MULTITHREADED CATALOG-ONLY TAGGER")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Web enrichment: DISABLED (catalog-only mode)")
        print(f"Workers: {self.max_workers}")
        print(f"Chunk size: {chunk_size}")

        # Load data
        try:
            if input_file.endswith(".xlsx"):
                df = pd.read_excel(input_file)
            else:
                df = pd.read_csv(input_file)

            print(f"Loaded {len(df)} entries from {input_file}")
        except Exception as e:
            print(f"Failed to load input file: {e}")
            return False

        # Split into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]
            chunk_data = chunk_df.to_dict("records")
            chunks.append(chunk_data)

        print(f"Split into {len(chunks)} chunks of max {chunk_size} entries each")

        # Process chunks in parallel
        all_results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk_data): i
                for i, chunk_data in enumerate(chunks)
            }

            # Collect results
            completed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result(
                        timeout=120
                    )  # 2 minute timeout per chunk
                    all_results.extend(chunk_results)
                    completed_chunks += 1

                    # Progress update
                    progress = (completed_chunks / len(chunks)) * 100
                    print(
                        f"Progress: {completed_chunks}/{len(chunks)} chunks ({progress:.1f}%) - {len(all_results)} entries processed"
                    )

                except concurrent.futures.TimeoutError:
                    logger.error(f"Chunk {chunk_idx} timed out")
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")

        # Save results
        try:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)

            processing_time = time.time() - start_time

            print(f"\nPROCESSING COMPLETED!")
            print(f"Results saved to: {output_file}")
            print(f"Processed: {len(all_results)} entries")
            print(f"Total time: {processing_time:.1f} seconds")
            print(f"Rate: {len(all_results)/processing_time:.1f} entries/second")

            # Statistics
            total_tags = sum(row["Total_Tags"] for row in all_results)

            print(f"Total tags generated: {total_tags}")
            print(f"Average tags per entry: {total_tags/len(all_results):.1f}")
            print(f"Web enrichment: Disabled (faster processing)")

            return True

        except Exception as e:
            print(f"Failed to save results: {e}")
            return False


def main():
    """Main function for production catalog-only tagger"""
    import argparse

    settings = _settings
    file_io = settings.get("file_io", {})
    processing = settings.get("processing", {})

    print("PRODUCTION MULTITHREADED CATALOG-ONLY TAGGER")
    print("Fast Processing with Semantic Similarity (No Web)")
    print("=" * 70)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Catalog Auto-Tagger - Fast Processing (Catalog-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python catalog_tagger.py real_estate                # Process using real estate tags
  python catalog_tagger.py automotive                 # Process using automotive tags
  python catalog_tagger.py electronics                # Process using electronics tags

Available catalog types depend on config files in: config/tags/
- real_estate_tags.yaml (pre-built)
- custom_example.yaml (example template)
- [create your own]_tags.yaml (custom domains)
        """,
    )

    parser.add_argument(
        "catalog_type",
        help="Catalog type to use for tagging (e.g., 'real_estate', 'automotive', 'electronics')",
    )

    parser.add_argument(
        "--input",
        "-i",
        default=file_io.get("default_input_file", "input/Book3.xlsx"),
        help=f"Input file path (default: {file_io.get('default_input_file', 'input/Book3.xlsx')})",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=file_io.get("default_output_files", {}).get("catalog_only", "output/catalog_only_results.csv"),
        help=f"Output file path (default: {file_io.get('default_output_files', {}).get('catalog_only', 'output/catalog_only_results.csv')})",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=processing.get("max_workers", 8),
        help=f"Number of worker threads (default: {processing.get('max_workers', 8)})",
    )

    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=processing.get("chunk_size", 200),
        help=f"Entries per processing chunk (default: {processing.get('chunk_size', 200)})",
    )

    args = parser.parse_args()

    # Settings from arguments
    catalog_type = args.catalog_type
    input_file = args.input
    output_file = args.output
    max_workers = args.workers
    chunk_size = args.chunk_size

    print(f"Catalog Type: {catalog_type}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Workers: {max_workers}")
    print(f"Chunk Size: {chunk_size}")

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        print(f"Please place your catalog file in the input/ directory")
        return False

    # Create output directory if needed
    Path("output").mkdir(exist_ok=True)

    # Initialize tagger
    try:
        print(f"Initializing catalog-only tagger for '{catalog_type}'...")
        tagger = MultithreadedCatalogTagger(max_workers, catalog_type)
    except ValueError as e:
        print(f"{e}")
        print(f"Available catalog types depend on config files in: config/tags/")
        print(
            f"   Create {catalog_type}_tags.yaml or use existing ones like 'real_estate'"
        )
        return False

    # Process the file
    success = tagger.process_large_file(
        input_file=input_file, output_file=output_file, chunk_size=chunk_size
    )

    if success:
        print(f"\nSUCCESS! Check the results in {output_file}")
        print(f"Catalog-only advantages:")
        print(f"   - 5-10x faster than web-enhanced version")
        print(f"   - No API quota limitations")
        print(f"   - Suitable for very large catalogs (1M+ entries)")
        print(f"   - Still uses semantic similarity for quality")
    else:
        print(f"\nPROCESSING FAILED")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
