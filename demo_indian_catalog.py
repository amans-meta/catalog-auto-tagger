#!/usr/bin/env python3
"""
Indian Catalog Auto-Tagger Demo

This script demonstrates the catalog auto-tagger system
customized for Indian market with INR pricing and local websites.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def create_sample_indian_catalog():
    """Create a sample Indian catalog CSV file"""
    sample_data = [
        {
            "property_id": "IND001",
            "title": "Spacious 3BHK Apartment in Gurgaon",
            "description": "Ready to move 3BHK flat with modern amenities including gym, swimming pool, 24x7 security, power backup, and covered parking. Located in prime location with metro connectivity.",
            "price": "85 lakh",
            "property_type": "Apartment",
            "city": "Gurgaon",
            "bhk": "3",
            "area": "1200 sqft",
            "furnishing": "Semi-furnished",
            "floor": "5",
            "total_floors": "12",
            "parking": "Covered",
            "amenities": "Gym, Pool, Security, Power Backup",
        },
        {
            "property_id": "IND002",
            "title": "Luxury Villa with Swimming Pool in Bangalore",
            "description": "Independent villa in premium gated community with swimming pool, landscaped garden, club house, and 24x7 security. Near IT parks and excellent connectivity.",
            "price": "2.5 crore",
            "property_type": "Villa",
            "city": "Bangalore",
            "bhk": "4",
            "area": "2800 sqft",
            "furnishing": "Unfurnished",
            "parking": "2 car garage",
            "amenities": "Swimming Pool, Club House, Garden, Security",
        },
        {
            "property_id": "IND003",
            "title": "Budget 2BHK Flat in Mumbai Suburbs",
            "description": "Affordable 2BHK apartment in upcoming area with good investment potential. Basic amenities available with lift and parking facility.",
            "price": "45 lakh",
            "property_type": "Apartment",
            "city": "Mumbai",
            "bhk": "2",
            "area": "850 sqft",
            "furnishing": "Unfurnished",
            "floor": "3",
            "total_floors": "8",
            "parking": "Open",
            "amenities": "Lift, Basic Security",
        },
    ]

    # Create CSV
    csv_path = Path("data/samples/indian_sample.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)

    print(f"✅ Created sample Indian catalog: {csv_path}")
    return str(csv_path)


def demo_indian_price_parsing():
    """Demonstrate Indian price parsing (Lakh/Crore notation)"""
    print("🏷️ INDIAN PRICE PARSING DEMO")
    print("=" * 50)

    from core.catalog_processor import CatalogProcessor
    from utils.config import ConfigManager

    config = ConfigManager()
    processor = CatalogProcessor(config)

    # Test Indian price formats
    test_prices = [
        {"price": "85 lakh"},
        {"price": "₹85 lakh"},
        {"price": "Rs. 2.5 crore"},
        {"price": "45,00,000"},
        {"price": "1.2 cr"},
        {"price": "50 lac"},
    ]

    print("Testing Indian price formats:")
    for i, test_data in enumerate(test_prices, 1):
        parsed_price = processor._extract_price(test_data)
        original = test_data["price"]
        formatted_price = f"₹{parsed_price:,.0f}" if parsed_price else "Could not parse"
        print(f"  {i}. '{original}' → {formatted_price}")


def demo_indian_tagging():
    """Demonstrate Indian real estate tagging"""
    print("\n🏠 INDIAN REAL ESTATE TAGGING DEMO")
    print("=" * 60)

    import yaml
    from models.product import GeneratedTag, ProductInfo
    from utils.config import ConfigManager
    from utils.text_processing import TextProcessor

    config = ConfigManager()
    text_processor = TextProcessor()

    # Load Indian tags configuration manually for demo
    try:
        with open("config/tags/indian_real_estate.yaml", "r") as f:
            tags_yaml = yaml.safe_load(f)

        # Create simple tag objects for demo
        class SimpleTag:
            def __init__(self, name, keywords=None, patterns=None, min_confidence=0.3):
                self.name = name
                self.keywords = keywords or []
                self.patterns = patterns or []
                self.min_confidence = min_confidence

        tag_definitions = []
        if tags_yaml:
            for tags_list in tags_yaml.values():
                if isinstance(tags_list, list):
                    for tag_info in tags_list:
                        tag_definitions.append(
                            SimpleTag(
                                name=tag_info.get("name", ""),
                                keywords=tag_info.get("keywords", []),
                                patterns=tag_info.get("patterns", []),
                                min_confidence=tag_info.get("min_confidence", 0.3),
                            )
                        )

        print(f"📋 Loaded {len(tag_definitions)} Indian tag definitions")
    except Exception as e:
        print(f"⚠️  Could not load tag definitions: {e}")
        tag_definitions = []

    # Sample Indian properties
    sample_properties = [
        {
            "id": "IND001",
            "title": "Spacious 3BHK Apartment in Gurgaon",
            "description": "Ready to move 3BHK flat with modern amenities including gym, swimming pool, 24x7 security, power backup, and covered parking. Located in prime location with metro connectivity.",
            "price": 8500000,  # 85 lakh
            "category": "Apartment",
            "attributes": {
                "city": "Gurgaon",
                "bhk": "3",
                "area": "1200 sqft",
                "furnishing": "Semi-furnished",
                "parking": "Covered",
            },
        },
        {
            "id": "IND002",
            "title": "Luxury Villa with Swimming Pool in Bangalore",
            "description": "Independent villa in premium gated community with swimming pool, landscaped garden, club house, and 24x7 security. Near IT parks and excellent connectivity.",
            "price": 25000000,  # 2.5 crore
            "category": "Villa",
            "attributes": {
                "city": "Bangalore",
                "bhk": "4",
                "area": "2800 sqft",
                "parking": "2 car garage",
            },
        },
        {
            "id": "IND003",
            "title": "Budget 2BHK Flat in Mumbai Suburbs",
            "description": "Affordable 2BHK apartment in upcoming area with good investment potential. Basic amenities available.",
            "price": 4500000,  # 45 lakh
            "category": "Apartment",
            "attributes": {"city": "Mumbai", "bhk": "2", "area": "850 sqft"},
        },
    ]

    # Process each property
    for i, prop_data in enumerate(sample_properties, 1):
        print(f"\n🏘️  Processing Property {i}: {prop_data['title']}")
        print("-" * 50)

        product = ProductInfo(**prop_data)
        full_text = f"{product.title} {product.description or ''}"

        # Get Indian price ranges from config
        price_ranges = config.get("currency.inr_price_ranges", {})

        # Determine price category
        price_category = "unknown"
        if product.price:
            for range_name, range_info in price_ranges.items():
                min_price = range_info.get("min", 0)
                max_price = range_info.get("max")

                if min_price <= product.price and (
                    max_price is None or product.price <= max_price
                ):
                    price_category = range_name
                    break

        print(f"💰 Price: ₹{product.price:,} ({price_category})")

        # Extract features
        features = text_processor.extract_features(full_text)
        print(f"📝 Keywords: {features['keywords'][:6]}")

        # Generate tags
        generated_tags = []

        # Rule-based tagging using Indian tag definitions
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

            # Price-based tags (Indian ranges)
            if product.price and "price" in tag_def.name.lower():
                if "budget" in tag_def.name and product.price < 2000000:  # < 20 lakh
                    confidence += 0.8
                    reasoning.append(f"price ₹{product.price:,} < ₹20 lakh")
                elif (
                    "mid_range" in tag_def.name and 2000000 <= product.price <= 10000000
                ):  # 20L - 1Cr
                    confidence += 0.8
                    reasoning.append(f"price ₹{product.price:,} in mid-range")
                elif (
                    "high_end" in tag_def.name and 10000000 < product.price <= 50000000
                ):  # 1Cr - 5Cr
                    confidence += 0.8
                    reasoning.append(f"price ₹{product.price:,} in high-end")
                elif "luxury" in tag_def.name and product.price > 50000000:  # > 5 Cr
                    confidence += 0.8
                    reasoning.append(f"price ₹{product.price:,} > ₹5 crore")

            # Apply minimum confidence threshold
            min_confidence = getattr(tag_def, "min_confidence", 0.3)
            if confidence >= min_confidence:
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
        for tag in generated_tags[:8]:  # Show top 8
            confidence_icon = (
                "🟢" if tag.confidence > 0.7 else "🟡" if tag.confidence > 0.5 else "🔴"
            )
            print(f"   {confidence_icon} {tag.name} ({tag.confidence:.2f})")
            if tag.reasoning:
                print(f"      ↳ {tag.reasoning}")


def demo_custom_catalog_processing():
    """Demonstrate processing a custom catalog format"""
    print(f"\n📊 CUSTOM CATALOG PROCESSING DEMO")
    print("=" * 60)

    # Create sample file if it doesn't exist
    sample_file = create_sample_indian_catalog()

    from core.catalog_processor import CatalogProcessor
    from utils.config import ConfigManager

    config = ConfigManager()
    processor = CatalogProcessor(config)

    print(f"📁 Processing file: {sample_file}")

    try:
        # Process with Indian configuration
        products = processor.process_catalog_file(sample_file, "indian_real_estate")

        print(f"✅ Successfully processed {len(products)} products")

        # Show field mappings used
        field_mappings = processor._get_field_mappings("indian_real_estate")
        print(f"\n🗺️  Field mappings used:")
        for field, variations in field_mappings.items():
            print(f"   • {field}: {variations[:3]}...")  # Show first 3 variations

        # Show processed products
        print(f"\n📦 Processed products:")
        for i, product in enumerate(products, 1):
            print(f"   {i}. {product.title}")
            if product.price:
                print(f"      💰 ₹{product.price:,.0f}")
            if product.attributes:
                attrs = list(product.attributes.items())[:3]  # Show first 3 attributes
                attr_str = ", ".join([f"{k}: {v}" for k, v in attrs])
                print(f"      📋 {attr_str}")

    except Exception as e:
        print(f"❌ Error processing catalog: {e}")
        import traceback

        traceback.print_exc()


def demo_web_search_configuration():
    """Demonstrate web search with Indian website prioritization"""
    print(f"\n🔍 INDIAN WEB SEARCH CONFIGURATION")
    print("=" * 60)

    from utils.config import ConfigManager

    config = ConfigManager()

    # Show Indian real estate search configuration
    indian_config = config.get("catalog_types.indian_real_estate", {})

    print("🏠 Indian Real Estate Search Configuration:")

    search_templates = indian_config.get("web_search_templates", [])
    print(f"\n📝 Search Templates:")
    for i, template in enumerate(search_templates, 1):
        print(f"   {i}. {template}")

    scraping_rules = indian_config.get("scraping_rules", {})
    priority_sites = scraping_rules.get("priority_sites", [])

    print(f"\n🎯 Priority Websites (Indian Market):")
    for i, site in enumerate(priority_sites, 1):
        print(f"   {i}. {site}")

    skip_sites = scraping_rules.get("skip_sites", [])
    print(f"\n🚫 Skipped Websites:")
    for i, site in enumerate(skip_sites, 1):
        print(f"   {i}. {site}")

    # Show e-commerce configuration too
    ecom_config = config.get("catalog_types.indian_ecommerce", {})
    ecom_priority = ecom_config.get("scraping_rules", {}).get("priority_sites", [])

    print(f"\n🛒 Indian E-commerce Priority Sites:")
    for i, site in enumerate(ecom_priority, 1):
        print(f"   {i}. {site}")


def main():
    """Main demo function"""
    print("🇮🇳 INDIAN CATALOG AUTO-TAGGER DEMO")
    print("🚀 Customized for Indian Market & Currency")
    print("=" * 70)

    print(f"📍 Working directory: {os.getcwd()}")

    try:
        # Demo Indian price parsing
        demo_indian_price_parsing()

        # Demo Indian tagging
        demo_indian_tagging()

        # Demo custom catalog processing
        demo_custom_catalog_processing()

        # Demo web search configuration
        demo_web_search_configuration()

        print(f"\n🎉 ALL INDIAN MARKET DEMOS COMPLETED!")
        print(f"\n📖 Indian Market Features Demonstrated:")
        print(f"   ✅ INR price parsing (Lakh/Crore notation)")
        print(f"   ✅ Indian real estate terminology (BHK, etc.)")
        print(f"   ✅ Indian website prioritization")
        print(f"   ✅ Custom field mappings")
        print(f"   ✅ Indian price range categorization")
        print(f"   ✅ Local amenity and feature detection")

        print(f"\n🔧 To use with your catalog:")
        print(f"   1. Update field mappings in config/settings.yaml")
        print(f"   2. Add your catalog type configuration")
        print(f"   3. Modify tag definitions in config/tags/")
        print(f"   4. Process with: catalog_type='your_type'")

        print(f"\n💡 Next Steps:")
        print(f"   • Add your specific catalog format")
        print(f"   • Configure web scraping for your domain")
        print(f"   • Train ML models with your data")
        print(f"   • Set up API integrations")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
