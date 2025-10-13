#!/usr/bin/env python3
"""
Multithreaded Hybrid Tagger - Production Ready
Combines semantic similarity with Google Search web enrichment
Supports large-scale processing with multithreading
"""

import concurrent.futures
import json
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not installed. Install with: pip install python-dotenv"
    )

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
    property_type: str
    city: str
    full_text: str


class ThreadSafeWebEnricher:
    """Thread-safe Google Search based web enricher with title-based caching"""

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.rate_limit_lock = Lock()
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )

        # Initialize cache
        self.cache_file = Path("cache/web_search_cache.pkl")
        self.cache_lock = Lock()
        self._load_cache()

    def _load_cache(self):
        """Load existing cache from disk"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.web_cache = pickle.load(f)
                logger.info(f"📂 Loaded {len(self.web_cache)} cached web searches")
            else:
                self.web_cache = {}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.web_cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Don't acquire lock here if already held - potential deadlock
            logger.info(f"💾 Saving cache with {len(self.web_cache)} entries...")
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.web_cache, f)
            logger.info(f"✅ Cache saved successfully to {self.cache_file}")
        except Exception as e:
            logger.warning(f"❌ Failed to save cache: {e}")

    def _get_cache_key(self, title: str, city: str) -> str:
        """Generate cache key from title and city"""
        # Normalize title for consistent caching
        normalized_title = re.sub(r"\s+", " ", title.lower().strip())
        normalized_city = city.lower().strip()
        return f"{normalized_title}|{normalized_city}"

    def enrich_entry(self, entry: CatalogEntry) -> str:
        """Thread-safe web enrichment with caching"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(entry.title, entry.city)

            with self.cache_lock:
                if cache_key in self.web_cache:
                    logger.debug(f"🎯 Cache hit for: {entry.title}")
                    return self.web_cache[cache_key]

            query = f'"{entry.title}" {entry.city} specifications features details'

            # Thread-safe rate limiting
            with self.rate_limit_lock:
                current_time = time.time()
                if current_time - self.last_request_time < 1:
                    time.sleep(1 - (current_time - self.last_request_time))
                self.last_request_time = time.time()

            logger.debug(f"🌐 Web search for: {entry.title}")

            # Google Custom Search API
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": self.api_key, "cx": self.cse_id, "q": query, "num": 5}

            response = self.session.get(search_url, params=params, timeout=15)

            enriched_text = ""
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])

                enriched_parts = []
                for item in items:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    if snippet and len(snippet) > 30:
                        # Generic filtering - can work across industries
                        relevant_indicators = [
                            "features",
                            "specifications",
                            "details",
                            "price",
                            "amenities",
                            "facilities",
                            "quality",
                            "premium",
                            "luxury",
                            "advanced",
                            "model",
                            "brand",
                            "review",
                        ]

                        if any(
                            keyword in snippet.lower()
                            for keyword in relevant_indicators
                        ):
                            enriched_parts.append(f"{title} | {snippet}")

                if enriched_parts:
                    enriched_text = " | ".join(enriched_parts)

            # Cache the result (even empty results to avoid repeated API calls)
            with self.cache_lock:
                self.web_cache[cache_key] = enriched_text

                # Save cache periodically (every 10 entries)
                if len(self.web_cache) % 10 == 0:
                    try:
                        with open(self.cache_file, "wb") as f:
                            pickle.dump(self.web_cache, f)
                        logger.info(
                            f"💾 Cache saved with {len(self.web_cache)} entries"
                        )
                    except Exception as save_error:
                        logger.warning(f"Cache save failed: {save_error}")

            return enriched_text

        except Exception as e:
            logger.warning(f"Web enrichment failed for {entry.id}: {e}")
            # Cache failure as empty result
            try:
                with self.cache_lock:
                    self.web_cache[cache_key] = ""
            except Exception:
                pass
            return ""


class SemanticMatcher:
    """Thread-safe semantic matching with embeddings"""

    def __init__(self, catalog_type: str):
        if not catalog_type:
            raise ValueError(
                "catalog_type must be specified (e.g., 'real_estate', 'automotive', 'electronics')"
            )
        self.catalog_type = catalog_type
        self.embedding_model = None
        self._initialize_embeddings()

        # Load tags from configuration - FAIL if not available
        try:
            from utils import load_tags

            self.tags = load_tags(catalog_type)
            if not self.tags:
                raise ValueError(f"No tags found for catalog_type '{catalog_type}'")

            logger.info(
                f"✅ Loaded {len(self.tags)} tags for '{catalog_type}' from configuration"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load tags for '{catalog_type}': {e}")
            logger.error(
                f"💡 Please ensure config/tags/{catalog_type}_tags.yaml exists"
            )
            logger.error(
                f"📋 Available examples: config/tags/real_estate_tags.yaml, config/tags/custom_example.yaml"
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
            # Try with show_progress_bar first, fall back if not supported
            try:
                self.embedding_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", show_progress_bar=False
                )
            except TypeError:
                # Older versions don't support show_progress_bar parameter
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Loaded sentence-transformers for semantic similarity")
        except ImportError:
            logger.info(
                "⚠️  sentence-transformers not available, using keyword matching only"
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

        logger.info(f"✅ Created embeddings for {embeddings_created} tags")

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

    def match_tags_with_web_boost(
        self, entry: CatalogEntry, web_text: str = ""
    ) -> List[TagResult]:
        """Match tags with web content boost"""
        full_text = f"{entry.full_text} {web_text}".strip()
        web_lower = web_text.lower() if web_text else ""

        results = []

        for tag_name, tag_def in self.tags.items():
            confidence = 0.0
            evidence = []
            source = "textual"

            # SEMANTIC SIMILARITY on enriched text
            semantic_score = self._compute_semantic_similarity(full_text, tag_def)
            if semantic_score > 0:
                confidence += (
                    semantic_score * 0.6
                )  # Higher weight for enriched semantic
                evidence.append(f"semantic_similarity: {semantic_score:.3f}")
                source = "semantic"

            # Web-specific keyword matching
            keywords = tag_def.get("keywords", [])
            keyword_matches = []
            web_keyword_matches = []

            for keyword in keywords:
                if keyword.lower() in full_text.lower():
                    keyword_matches.append(keyword)
                    confidence += 0.25 * tag_def["weight"]

                # Additional boost if keyword found in web content
                if web_text and keyword.lower() in web_lower:
                    web_keyword_matches.append(keyword)
                    confidence += 0.15 * tag_def["weight"]  # Web boost

            if keyword_matches:
                evidence.append(f"keywords: {keyword_matches}")
            if web_keyword_matches:
                evidence.append(f"web_keywords: {web_keyword_matches}")
                source = "hybrid"

            # Pattern matching
            patterns = tag_def.get("patterns", [])
            pattern_matches = []
            for pattern in patterns:
                try:
                    if re.search(pattern, full_text.lower(), re.IGNORECASE):
                        pattern_matches.append(pattern)
                        confidence += 0.4 * tag_def["weight"]
                    elif web_text and re.search(pattern, web_lower, re.IGNORECASE):
                        pattern_matches.append(f"{pattern}(web)")
                        confidence += 0.3 * tag_def["weight"]
                        source = "hybrid"
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
                    evidence.append(f"price: ₹{entry.price:,.0f} in range")

            # Web content analysis for specific categories
            if tag_def["category"] == "amenity" and web_text:
                amenity_indicators = ["amenities", "facilities", "features"]
                web_amenity_score = (
                    sum(1 for ind in amenity_indicators if ind in web_lower) * 0.1
                )
                if web_amenity_score > 0:
                    confidence += web_amenity_score
                    evidence.append(f"web_amenity_context: {web_amenity_score:.2f}")
                    source = "hybrid"

            # Property type specific boosts
            if tag_name in entry.property_type.lower():
                confidence += 0.5
                evidence.append("property_type match")

            # Location specific boosts
            if tag_def["category"] == "location" and entry.city:
                city_lower = entry.city.lower()
                for keyword in keywords:
                    if keyword.lower() in city_lower or (
                        web_text and keyword.lower() in web_lower
                    ):
                        confidence += 0.3
                        evidence.append(f"location match: {keyword}")
                        if web_text and keyword.lower() in web_lower:
                            source = "hybrid"

            # Apply confidence threshold
            threshold = 0.12 if semantic_score > 0 else 0.18
            if confidence >= threshold:
                if web_text and any("web" in str(ev) for ev in evidence):
                    evidence.append("web_boost")

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

    def _extract_location_from_web(self, web_text: str) -> Tuple[str, str]:
        """Extract city and state from web content"""
        if not web_text:
            return ("", "")

        web_lower = web_text.lower()
        city = ""
        state = ""

        # Indian cities pattern matching
        city_patterns = {
            "mumbai": ["mumbai", "bombay"],
            "delhi": ["delhi", "new delhi"],
            "bangalore": ["bangalore", "bengaluru"],
            "gurgaon": ["gurgaon", "gurugram"],
            "noida": ["noida", "greater noida"],
            "pune": ["pune", "puna"],
            "hyderabad": ["hyderabad"],
            "chennai": ["chennai", "madras"],
            "kolkata": ["kolkata", "calcutta"],
            "ahmedabad": ["ahmedabad"],
            "faridabad": ["faridabad"],
            "ghaziabad": ["ghaziabad"],
        }

        # Find city in web content
        for standard_city, variations in city_patterns.items():
            for variation in variations:
                if variation in web_lower:
                    city = standard_city
                    break
            if city:
                break

        # State patterns
        state_patterns = {
            "haryana": ["haryana", "gurgaon", "gurugram", "faridabad"],
            "delhi": ["delhi", "new delhi"],
            "uttar pradesh": ["uttar pradesh", "noida", "ghaziabad", "greater noida"],
            "karnataka": ["karnataka", "bangalore", "bengaluru"],
            "maharashtra": ["maharashtra", "mumbai", "pune"],
            "telangana": ["telangana", "hyderabad"],
            "tamil nadu": ["tamil nadu", "chennai"],
            "west bengal": ["west bengal", "kolkata"],
            "gujarat": ["gujarat", "ahmedabad"],
        }

        # Find state in web content
        for standard_state, indicators in state_patterns.items():
            for indicator in indicators:
                if indicator in web_lower:
                    state = standard_state
                    break
            if state:
                break

        return (city, state)


class MultithreadedHybridTagger:
    """Multithreaded hybrid tagger for large-scale processing"""

    def __init__(
        self, api_key: str, cse_id: str, max_workers: int = 4, catalog_type: str = None
    ):
        if not catalog_type:
            raise ValueError(
                "catalog_type must be specified (e.g., 'real_estate', 'automotive', 'electronics')"
            )
        self.api_key = api_key
        self.cse_id = cse_id
        self.max_workers = max_workers
        self.matcher = SemanticMatcher(catalog_type)
        self.web_enricher = (
            ThreadSafeWebEnricher(api_key, cse_id) if api_key and cse_id else None
        )

    def process_chunk(
        self, chunk_data: List[Dict[str, Any]], enable_web: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a chunk of catalog entries"""
        results = []

        for row_data in chunk_data:
            try:
                # Create catalog entry
                entry = self._create_catalog_entry(row_data)

                # Get web content if enabled
                web_content = ""
                web_content_retrieved = False
                if enable_web and self.web_enricher:
                    try:
                        web_content = self.web_enricher.enrich_entry(entry)
                        # Consider web content retrieved if it has actual content
                        web_content_retrieved = len(web_content.strip()) > 0
                    except Exception as e:
                        logger.warning(f"Web enrichment failed for {entry.id}: {e}")

                # Generate tags
                tags = self.matcher.match_tags_with_web_boost(entry, web_content)

                # Categorize tags
                basic_tags = []
                advanced_tags = []

                for tag in tags:
                    if tag.category in ["bedrooms", "amenity", "price_range", "status"]:
                        basic_tags.append(tag.tag_name)
                    else:
                        advanced_tags.append(tag.tag_name)

                # Extract location from web content
                city_tag, state_tag = (
                    self.matcher._extract_location_from_web(web_content)
                    if web_content
                    else ("", "")
                )

                # Create result - Web_Enhanced only "Yes" if web content was actually retrieved and used
                result = {
                    "Entry_ID": entry.id,
                    "Title": entry.title,
                    "Price_INR": entry.price,
                    "City": entry.city,
                    "Property_Type": entry.property_type,
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
                    "Web_Enhanced": "Yes" if web_content_retrieved else "No",
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process entry: {e}")
                continue

        return results

    def _create_catalog_entry(self, row_data: Dict[str, Any]) -> CatalogEntry:
        """Create catalog entry from row data"""
        # Extract fields
        entry_id = str(
            row_data.get(
                "home_listing_id", row_data.get("id", f"prop_{hash(str(row_data))}")
            )
        )
        title = str(row_data.get("name", ""))

        # Description
        description = ""
        desc_fields = ["description", "desc", "details", "summary"]
        for field in desc_fields:
            if field in row_data and pd.notna(row_data[field]):
                description = str(row_data[field])
                break

        # Parse price
        price = 0.0
        if "Price" in row_data and pd.notna(row_data["Price"]):
            price_str = (
                str(row_data["Price"]).replace("INR", "").replace(",", "").strip()
            )
            try:
                price = float(price_str)
            except:
                price = 0.0

        property_type = str(row_data.get("Property_Type", ""))
        city = str(row_data.get("Address.city", ""))

        # Create full text
        full_text_parts = [title, description, property_type, city]
        for key, value in row_data.items():
            if pd.notna(value) and isinstance(value, str) and len(value) < 200:
                if key not in [
                    "home_listing_id",
                    "name",
                    "Price",
                    "Property_Type",
                    "Address.city",
                ]:
                    full_text_parts.append(value)

        full_text = " ".join([str(part) for part in full_text_parts if part])

        return CatalogEntry(
            id=entry_id,
            title=title,
            description=description,
            price=price,
            property_type=property_type,
            city=city,
            full_text=full_text,
        )

    def process_large_file(
        self,
        input_file: str,
        output_file: str,
        enable_web: bool = True,
        chunk_size: int = 100,
    ):
        """Process large files with multithreading"""
        print(f"🚀 MULTITHREADED HYBRID TAGGER")
        print(f"📁 Input: {input_file}")
        print(f"📁 Output: {output_file}")
        print(f"🌐 Web enrichment: {'Enabled' if enable_web else 'Disabled'}")
        print(f"⚙️  Workers: {self.max_workers}")
        print(f"📦 Chunk size: {chunk_size}")

        # Load data
        try:
            if input_file.endswith(".xlsx"):
                df = pd.read_excel(input_file)
            else:
                df = pd.read_csv(input_file)

            print(f"✅ Loaded {len(df)} entries from {input_file}")
        except Exception as e:
            print(f"❌ Failed to load input file: {e}")
            return False

        # Split into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]
            chunk_data = chunk_df.to_dict("records")
            chunks.append(chunk_data)

        print(f"📦 Split into {len(chunks)} chunks of max {chunk_size} entries each")

        # Process chunks in parallel
        all_results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk_data, enable_web): i
                for i, chunk_data in enumerate(chunks)
            }

            # Collect results
            completed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result(
                        timeout=300
                    )  # 5 minute timeout per chunk
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

            print(f"\n✅ PROCESSING COMPLETED!")
            print(f"📄 Results saved to: {output_file}")
            print(f"📊 Processed: {len(all_results)} entries")
            print(f"⏱️  Total time: {processing_time:.1f} seconds")
            print(f"⚡ Rate: {len(all_results)/processing_time:.1f} entries/second")

            # Statistics
            total_tags = sum(row["Total_Tags"] for row in all_results)
            web_enhanced = len(
                [row for row in all_results if row["Web_Enhanced"] == "Yes"]
            )

            print(f"🏷️  Total tags generated: {total_tags}")
            print(
                f"🌐 Web-enhanced entries: {web_enhanced}/{len(all_results)} ({web_enhanced/len(all_results)*100:.1f}%)"
            )
            print(f"📈 Average tags per entry: {total_tags/len(all_results):.1f}")

            return True

        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False


def main():
    """Main function for production hybrid tagger"""
    import argparse

    print("🚀 PRODUCTION MULTITHREADED HYBRID TAGGER")
    print("🎯 Semantic + Web Enrichment with Parallel Processing")
    print("=" * 70)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Catalog Auto-Tagger - Web Enhanced Processing (Hybrid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_tagger.py real_estate                 # Process using real estate tags
  python hybrid_tagger.py automotive                  # Process using automotive tags
  python hybrid_tagger.py electronics                 # Process using electronics tags

Available catalog types depend on config files in: config/tags/
- real_estate_tags.yaml (pre-built)
- custom_example.yaml (example template)
- [create your own]_tags.yaml (custom domains)

Requirements:
- Google Custom Search API key (GOOGLE_API_KEY in .env)
- Google Custom Search Engine ID (GOOGLE_CSE_ID in .env)
        """,
    )

    parser.add_argument(
        "catalog_type",
        help="Catalog type to use for tagging (e.g., 'real_estate', 'automotive', 'electronics')",
    )

    parser.add_argument(
        "--input",
        "-i",
        default="input/Book3.xlsx",
        help="Input file path (default: input/Book3.xlsx)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output/hybrid_tagged_results.csv",
        help="Output file path (default: output/hybrid_tagged_results.csv)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )

    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=50,
        help="Entries per processing chunk (default: 50)",
    )

    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web enrichment (faster processing)",
    )

    args = parser.parse_args()

    # Configuration from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not args.no_web and (not api_key or not cse_id):
        print("❌ Missing Google API credentials!")
        print("💡 Please ensure you have set the following environment variables:")
        print("   • GOOGLE_API_KEY - Your Google Custom Search API key")
        print("   • GOOGLE_CSE_ID - Your Google Custom Search Engine ID")
        print("📄 You can set these in the .env file (see .env.example for template)")
        print("🔧 Or use --no-web flag to disable web enrichment")
        return False

    # Settings from arguments
    catalog_type = args.catalog_type
    input_file = args.input
    output_file = args.output
    enable_web = not args.no_web
    max_workers = args.workers
    chunk_size = args.chunk_size

    print(f"📋 Catalog Type: {catalog_type}")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"🌐 Web enrichment: {'Enabled' if enable_web else 'Disabled'}")
    print(f"⚙️  Workers: {max_workers}")
    print(f"📦 Chunk size: {chunk_size}")

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"📁 Please place your catalog file in the input/ directory")
        return False

    # Create output directory if needed
    Path("output").mkdir(exist_ok=True)

    # Initialize tagger
    try:
        print(f"⚙️  Initializing hybrid tagger for '{catalog_type}'...")
        tagger = MultithreadedHybridTagger(api_key, cse_id, max_workers, catalog_type)
    except ValueError as e:
        print(f"❌ {e}")
        print(f"💡 Available catalog types depend on config files in: config/tags/")
        print(
            f"   Create {catalog_type}_tags.yaml or use existing ones like 'real_estate'"
        )
        return False

    # Process the file
    success = tagger.process_large_file(
        input_file=input_file,
        output_file=output_file,
        enable_web=enable_web,
        chunk_size=chunk_size,
    )

    if success:
        print(f"\n🎉 SUCCESS! Check the results in {output_file}")
        print(f"💡 For large catalogs (100K+ entries), consider:")
        print(f"   • Increasing max_workers (up to 8-10)")
        print(f"   • Splitting input files into smaller batches")
        print(f"   • Running during off-peak hours for API quota")
    else:
        print(f"\n❌ PROCESSING FAILED")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
