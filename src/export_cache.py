#!/usr/bin/env python3
"""
Export web search cache for use in internal Meta version.

This script exports the web_search_cache.pkl file to a specified location
so it can be imported by the internal version running on Meta infrastructure.
"""

import argparse
import logging
import pickle
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_cache(output_dir: str) -> None:
    """
    Export web search cache to specified directory.

    Args:
        output_dir: Directory to export cache to
    """
    # Paths
    project_root = Path(__file__).parent.parent
    cache_file = project_root / "cache" / "web_search_cache.pkl"
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if cache exists
    if not cache_file.exists():
        logger.error(f"❌ Cache file not found: {cache_file}")
        logger.info("Run hybrid_tagger first to build the cache.")
        return

    # Load cache to get stats
    logger.info(f"📂 Loading cache from: {cache_file}")
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    logger.info(f"✅ Cache loaded: {len(cache)} entries")

    # Copy cache to output directory
    output_file = output_path / "web_search_cache.pkl"
    logger.info(f"📤 Exporting cache to: {output_file}")

    shutil.copy2(cache_file, output_file)

    logger.info(f"✅ Cache exported successfully!")
    logger.info(f"   Entries: {len(cache)}")
    logger.info(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")

    # Show sample keys
    logger.info(f"\n📋 Sample cache keys (first 5):")
    for i, key in enumerate(list(cache.keys())[:5]):
        content_preview = cache[key][:100] if cache[key] else "(empty)"
        logger.info(f"   {i+1}. {key}")
        logger.info(f"      Content: {content_preview}...")


def main():
    parser = argparse.ArgumentParser(
        description="Export web search cache for internal Meta version"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for exported cache"
    )

    args = parser.parse_args()

    logger.info("🚀 Starting cache export...")
    export_cache(args.output)
    logger.info("✨ Done!")


if __name__ == "__main__":
    main()
