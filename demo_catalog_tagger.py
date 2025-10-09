#!/usr/bin/env python3
"""
Complete Demo of the Catalog Auto-Tagger System

This script demonstrates how to use the catalog auto-tagger system
to process sample data and generate tags.
"""

import json
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def demo_real_estate_tagging():
    """Demonstrate real estate catalog tagging"""
    print("🏠 REAL ESTATE CATALOG TAGGING DEMO")
    print("=" * 60)

    # Import required modules
    from models.product import GeneratedTag, ProductInfo
    from models.tag_config import RealEstateTagConfig
    from utils.config import ConfigManager
    from utils.text_processing import TextProcessor

    # Initialize components
    config = ConfigManager()
    text_processor = TextProcessor()
    tag_definitions = RealEstateTagConfig.get_tags()

    print(f"📋 Loaded {len(tag_definitions)} tag definitions")

    # Sample real estate products
    sample_properties = [
        {
            "id": "prop_001",
            "title": "Beautiful 3BR Luxury Home with Pool",
            "description": "Stunning newly renovated single family home with hardwood floors, granite countertops, swimming pool, and 2-car garage. Located in excellent school district.",
            "price": 650000,
            "category": "Single Family Home",
            "attributes": {
                "bedrooms": 3,
                "bathrooms": 2,
                "square_feet": 2200,
                "year_built": 2020,
            },
        },
        {
            "id": "prop_002",
            "title": "Waterfront Luxury Condo Downtown",
            "description": "High-end condominium with ocean views, marble floors, and resort-style amenities. Premium location in the heart of downtown.",
            "price": 850000,
            "category": "Condominium",
            "attributes": {
                "bedrooms": 2,
                "bathrooms": 2,
                "square_feet": 1800,
                "floor": 15,
            },
        },
        {
            "id": "prop_003",
            "title": "Fixer Upper with Great Potential",
            "description": "Property needs some TLC but has great bones and is perfect for investors or DIY enthusiasts. Cash offers preferred.",
            "price": 125000,
            "category": "Single Family Home",
            "attributes": {
                "bedrooms": 2,
                "bathrooms": 1,
                "square_feet": 1200,
                "condition": "needs_work",
            },
        },
    ]

    # Process each property
    for i, prop_data in enumerate(sample_properties, 1):
        print(f"\n🏘️  Processing Property {i}: {prop_data['title']}")
        print("-" * 50)

        # Create ProductInfo object
        product = ProductInfo(**prop_data)

        # Generate full text for analysis
        full_text = f"{product.title} {product.description or ''}"

        # Extract text features
        features = text_processor.extract_features(full_text)
        keywords = features["keywords"]
        sentiment = features["sentiment"]
        entities = features["entities"]

        print(f"💰 Price: ${product.price:,}")
        print(f"📝 Keywords: {keywords[:8]}")
        print(
            f"😊 Sentiment: {sentiment['polarity']:.2f} (polarity), {sentiment['subjectivity']:.2f} (subjectivity)"
        )

        # Generate tags based on rules
        generated_tags = []

        for tag_def in tag_definitions:
            confidence = 0.0
            reasoning = []

            # Check keyword matches
            keyword_matches = sum(
                1
                for keyword in tag_def.keywords
                if keyword.lower() in full_text.lower()
            )
            if keyword_matches > 0:
                confidence += keyword_matches * 0.3
                reasoning.append(f"keyword matches: {keyword_matches}")

            # Check pattern matches
            if hasattr(tag_def, "patterns") and tag_def.patterns:
                import re

                pattern_matches = 0
                for pattern in tag_def.patterns:
                    try:
                        if re.search(pattern, full_text, re.IGNORECASE):
                            pattern_matches += 1
                    except re.error:
                        continue
                if pattern_matches > 0:
                    confidence += pattern_matches * 0.4
                    reasoning.append(f"pattern matches: {pattern_matches}")

            # Check price-based tags
            if product.price and "price" in tag_def.name.lower():
                if "budget" in tag_def.name and product.price < 300000:
                    confidence += 0.8
                    reasoning.append(f"price ${product.price:,} < $300k")
                elif "mid_range" in tag_def.name and 300000 <= product.price <= 700000:
                    confidence += 0.8
                    reasoning.append(f"price ${product.price:,} in mid-range")
                elif "high_end" in tag_def.name and product.price > 700000:
                    confidence += 0.8
                    reasoning.append(f"price ${product.price:,} > $700k")

            # Check category matches
            if (
                product.category
                and tag_def.name.lower().replace("_", " ") in product.category.lower()
            ):
                confidence += 0.6
                reasoning.append(f"category match: {product.category}")

            # Apply minimum confidence threshold
            if confidence >= tag_def.min_confidence:
                generated_tags.append(
                    GeneratedTag(
                        name=tag_def.name,
                        confidence=min(confidence, 1.0),
                        source="rule_based",
                        reasoning="; ".join(reasoning) if reasoning else None,
                    )
                )

        # Sort tags by confidence
        generated_tags.sort(key=lambda x: x.confidence, reverse=True)

        # Display results
        print(f"\n🏷️  Generated {len(generated_tags)} tags:")
        for tag in generated_tags[:10]:  # Show top 10
            confidence_icon = (
                "🟢" if tag.confidence > 0.7 else "🟡" if tag.confidence > 0.5 else "🔴"
            )
            print(f"   {confidence_icon} {tag.name} ({tag.confidence:.2f})")
            if tag.reasoning:
                print(f"      ↳ {tag.reasoning}")

    print(f"\n✅ Demo completed successfully!")


def demo_csv_processing():
    """Demonstrate processing a CSV file"""
    print(f"\n📊 CSV FILE PROCESSING DEMO")
    print("=" * 60)

    # Check if sample data exists
    sample_file = Path("data/samples/real_estate_sample.csv")
    if not sample_file.exists():
        print("❌ Sample data file not found. Run setup.py first.")
        return

    # Import catalog processor
    from core.catalog_processor import CatalogProcessor
    from utils.config import ConfigManager

    # Initialize processor
    config = ConfigManager()
    processor = CatalogProcessor(config)

    print(f"📁 Processing file: {sample_file}")

    try:
        # Process the CSV file
        products = processor.process_catalog_file(str(sample_file), "real_estate")

        print(f"✅ Successfully processed {len(products)} products")

        # Show statistics
        stats = processor.get_catalog_stats(products)
        print(f"\n📈 Catalog Statistics:")
        print(f"   • Total products: {stats['total_products']}")
        print(f"   • Has descriptions: {stats['has_description']}")
        print(f"   • Has prices: {stats['has_price']}")
        print(f"   • Has categories: {stats['has_category']}")
        print(f"   • Average title length: {stats['avg_title_length']:.1f} chars")

        if "price_stats" in stats:
            price_stats = stats["price_stats"]
            print(
                f"   • Price range: ${price_stats['min']:,} - ${price_stats['max']:,}"
            )
            print(f"   • Average price: ${price_stats['avg']:,.0f}")

        # Show first few products
        print(f"\n📦 First 3 products:")
        for i, product in enumerate(products[:3], 1):
            print(f"   {i}. {product.title} - ${product.price:,}")

    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main demo function"""
    print("🚀 CATALOG AUTO-TAGGER COMPLETE DEMO")
    print("🤖 Intelligent Product Tagging System")
    print("=" * 70)

    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")

    try:
        # Run real estate tagging demo
        demo_real_estate_tagging()

        # Run CSV processing demo
        demo_csv_processing()

        print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print(f"\n📖 System Features Demonstrated:")
        print(f"   ✅ Text processing and feature extraction")
        print(f"   ✅ Rule-based tag generation")
        print(f"   ✅ Confidence scoring and reasoning")
        print(f"   ✅ CSV file processing")
        print(f"   ✅ Product data modeling")
        print(f"   ✅ Statistical analysis")

        print(f"\n🔧 To extend the system:")
        print(f"   • Add web scraping by implementing WebScraper")
        print(f"   • Add ML models by implementing TagClassifier")
        print(f"   • Add new tag definitions in config/tags/")
        print(f"   • Add support for new file formats")
        print(f"   • Add API integrations for search engines")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
