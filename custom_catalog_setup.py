#!/usr/bin/env python3
"""
Custom Catalog Setup Utility

This script helps you configure the system for your specific catalog format.
Run this to set up field mappings and configurations for your data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def analyze_catalog_sample(file_path: str) -> Dict[str, Any]:
    """Analyze a sample catalog file to understand its structure"""
    print(f"🔍 Analyzing catalog structure: {file_path}")

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    analysis = {
        "columns": [],
        "sample_data": {},
        "data_types": {},
        "recommendations": {},
    }

    if file_path_obj.suffix.lower() == ".csv":
        import pandas as pd

        try:
            # Read first few rows to analyze structure
            df = pd.read_csv(file_path, nrows=5)
            analysis["columns"] = list(df.columns)

            # Sample data from first row
            if len(df) > 0:
                analysis["sample_data"] = df.iloc[0].to_dict()

            # Data types
            analysis["data_types"] = {
                col: str(dtype) for col, dtype in df.dtypes.items()
            }

            print(f"   ✅ Found {len(analysis['columns'])} columns")
            for col in analysis["columns"][:10]:  # Show first 10
                print(f"      • {col}")

        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            return analysis

    elif file_path_obj.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            sample_item = data[0]
        elif isinstance(data, dict):
            # Look for common keys containing arrays
            for key in ["products", "items", "listings", "data"]:
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    sample_item = data[key][0]
                    break
            else:
                sample_item = data
        else:
            sample_item = {}

        analysis["columns"] = (
            list(sample_item.keys()) if isinstance(sample_item, dict) else []
        )
        analysis["sample_data"] = sample_item

        print(f"   ✅ Found {len(analysis['columns'])} fields")
        for col in analysis["columns"][:10]:
            print(f"      • {col}")

    # Generate recommendations
    analysis["recommendations"] = _generate_field_recommendations(analysis["columns"])

    return analysis


def _generate_field_recommendations(columns: List[str]) -> Dict[str, List[str]]:
    """Generate field mapping recommendations based on column names"""
    recommendations = {
        "id": [],
        "title": [],
        "description": [],
        "price": [],
        "category": [],
        "brand": [],
        "url": [],
    }

    # Common patterns for each field type
    patterns = {
        "id": ["id", "product_id", "listing_id", "item_id", "uuid", "sku"],
        "title": [
            "title",
            "name",
            "product_name",
            "listing_title",
            "property_name",
            "item_name",
        ],
        "description": [
            "description",
            "desc",
            "details",
            "summary",
            "about",
            "overview",
        ],
        "price": [
            "price",
            "cost",
            "amount",
            "value",
            "listing_price",
            "sale_price",
            "rent",
        ],
        "category": ["category", "type", "property_type", "product_category", "class"],
        "brand": ["brand", "manufacturer", "make", "builder", "company"],
        "url": ["url", "link", "website", "listing_url", "product_url"],
    }

    for field_type, keywords in patterns.items():
        for col in columns:
            col_lower = col.lower().replace("_", "").replace(" ", "")
            for keyword in keywords:
                keyword_clean = keyword.lower().replace("_", "")
                if keyword_clean in col_lower or col_lower in keyword_clean:
                    recommendations[field_type].append(col)
                    break

    return recommendations


def create_custom_configuration():
    """Interactive setup for custom catalog configuration"""
    print("\n🛠️  CUSTOM CATALOG CONFIGURATION SETUP")
    print("=" * 60)

    # Get catalog type name
    catalog_type = input(
        "📝 Enter your catalog type name (e.g., 'my_real_estate', 'my_products'): "
    ).strip()
    if not catalog_type:
        print("❌ Catalog type name is required")
        return

    # Get sample file for analysis
    sample_file = input(
        "📁 Enter path to your sample catalog file (CSV/JSON): "
    ).strip()

    try:
        analysis = analyze_catalog_sample(sample_file)
    except Exception as e:
        print(f"❌ Could not analyze sample file: {e}")
        return

    print(f"\n🔍 Analysis Results:")
    print(f"   📊 Columns found: {len(analysis['columns'])}")

    # Show recommendations
    print(f"\n💡 Field mapping recommendations:")
    config_mappings = {}

    for field_type, recommended_cols in analysis["recommendations"].items():
        if recommended_cols:
            print(f"\n   {field_type.upper()}:")
            for i, col in enumerate(recommended_cols, 1):
                print(f"      {i}. {col}")

            # Ask user to confirm or modify
            user_input = (
                input(f"   ✅ Use these for '{field_type}'? (y/n/custom): ")
                .strip()
                .lower()
            )

            if user_input == "y":
                config_mappings[field_type] = recommended_cols
            elif user_input == "custom":
                custom_cols = input(
                    f"   📝 Enter column names for '{field_type}' (comma-separated): "
                ).strip()
                if custom_cols:
                    config_mappings[field_type] = [
                        col.strip() for col in custom_cols.split(",")
                    ]

    # Ask for additional custom fields
    print(f"\n📋 Additional custom fields in your data:")
    remaining_cols = [
        col
        for col in analysis["columns"]
        if not any(col in mapping for mapping in config_mappings.values())
    ]

    if remaining_cols:
        print("   Unused columns:")
        for i, col in enumerate(remaining_cols, 1):
            print(f"      {i}. {col}")

        custom_fields = input(
            "   📝 Enter any additional field mappings (format: field_name:col1,col2): "
        ).strip()
        if custom_fields:
            for mapping in custom_fields.split():
                if ":" in mapping:
                    field_name, cols = mapping.split(":", 1)
                    config_mappings[field_name.strip()] = [
                        col.strip() for col in cols.split(",")
                    ]

    # Ask for websites to prioritize
    print(f"\n🌐 Website prioritization for web search:")
    websites = input(
        "   📝 Enter priority websites (comma-separated, e.g., 'site1.com,site2.com'): "
    ).strip()
    priority_sites = (
        [site.strip() for site in websites.split(",") if site.strip()]
        if websites
        else []
    )

    # Ask for search templates
    print(f"\n🔍 Search query templates:")
    print("   Example: '{title} {city} property details'")
    search_template = input(
        "   📝 Enter search template (or press Enter for default): "
    ).strip()
    search_templates = (
        [search_template]
        if search_template
        else ["{title} details", "{title} price information", "{title} specifications"]
    )

    # Create configuration
    catalog_config = {
        "web_search_templates": search_templates,
        "scraping_rules": {
            "priority_sites": priority_sites,
            "skip_sites": [
                "pinterest.com",
                "youtube.com",
                "facebook.com",
                "instagram.com",
            ],
        },
        "ml_model_config": {"focus_keywords": ["product", "item", "listing"]},
        "field_mappings": config_mappings,
    }

    # Save configuration
    settings_path = Path("config/settings.yaml")

    try:
        # Load existing settings
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
        else:
            settings = {}

        # Add new catalog type
        if "catalog_types" not in settings:
            settings["catalog_types"] = {}

        settings["catalog_types"][catalog_type] = catalog_config

        # Save updated settings
        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)

        print(f"\n✅ Configuration saved for catalog type: '{catalog_type}'")
        print(f"📁 Updated: {settings_path}")

    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return

    # Show usage example
    print(f"\n🚀 Usage Example:")
    print(f"```python")
    print(f"from src.core.catalog_processor import CatalogProcessor")
    print(f"from src.utils.config import ConfigManager")
    print(f"")
    print(f"config = ConfigManager()")
    print(f"processor = CatalogProcessor(config)")
    print(
        f"products = processor.process_catalog_file('your_file.csv', '{catalog_type}')"
    )
    print(f"```")

    return catalog_type


def test_configuration(catalog_type: str, sample_file: str):
    """Test the configuration with a sample file"""
    print(f"\n🧪 TESTING CONFIGURATION")
    print("=" * 50)

    try:
        from core.catalog_processor import CatalogProcessor
        from utils.config import ConfigManager

        config = ConfigManager()
        processor = CatalogProcessor(config)

        print(f"📁 Processing sample file with '{catalog_type}' configuration...")

        products = processor.process_catalog_file(sample_file, catalog_type)

        print(f"✅ Successfully processed {len(products)} products")

        if products:
            print(f"\n📦 Sample product:")
            sample_product = products[0]
            print(f"   ID: {sample_product.id}")
            print(f"   Title: {sample_product.title}")
            if sample_product.price:
                print(f"   Price: {sample_product.currency}{sample_product.price:,.2f}")
            if sample_product.category:
                print(f"   Category: {sample_product.category}")

            print(f"   Attributes: {list(sample_product.attributes.keys())[:5]}...")

        # Show field mappings used
        field_mappings = processor._get_field_mappings(catalog_type)
        print(f"\n🗺️  Field mappings used:")
        for field, variations in field_mappings.items():
            if variations:
                print(f"   • {field}: {variations}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main setup function"""
    print("🛠️  CUSTOM CATALOG SETUP UTILITY")
    print("🔧 Configure the system for your specific data format")
    print("=" * 70)

    try:
        # Interactive configuration
        catalog_type = create_custom_configuration()

        if catalog_type:
            # Ask if user wants to test
            test_choice = (
                input(f"\n🧪 Test the configuration now? (y/n): ").strip().lower()
            )
            if test_choice == "y":
                sample_file = input("📁 Enter path to test file: ").strip()
                if sample_file and Path(sample_file).exists():
                    test_configuration(catalog_type, sample_file)

        print(f"\n🎉 SETUP COMPLETED!")
        print(f"\n📖 What you can do next:")
        print(
            f"   1. Run: python demo_indian_catalog.py (to see Indian market example)"
        )
        print(f"   2. Process your data with the new configuration")
        print(f"   3. Create custom tag definitions in config/tags/")
        print(f"   4. Add web scraping for your domain-specific sites")

    except KeyboardInterrupt:
        print(f"\n⏹️  Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
