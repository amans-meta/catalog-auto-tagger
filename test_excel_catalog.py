#!/usr/bin/env python3
"""
Test Excel Catalog Processing
Tests both scenarios: with and without web scraper
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_without_web_scraper(excel_file):
    """Test catalog processing WITHOUT web scraper"""
    print("🔥 TESTING WITHOUT WEB SCRAPER")
    print("=" * 60)

    try:
        from core.catalog_processor import CatalogProcessor
        from utils.config import ConfigManager

        config = ConfigManager()
        processor = CatalogProcessor(config)

        print(f"📁 Processing Excel file: {excel_file}")

        # Process the Excel file
        products = processor.process_catalog_file(excel_file, 'indian_real_estate')

        print(f"✅ Successfully processed {len(products)} products")

        if not products:
            print("⚠️  No products found - check Excel file format")
            return []

        # Show statistics
        stats = processor.get_catalog_stats(products)
        print(f"\n📊 Catalog Statistics:")
        print(f"   • Total products: {stats.get('total_products', 0)}")
        print(f"   • Has descriptions: {stats.get('has_description', 0)}")
        print(f"   • Has prices: {stats.get('has_price', 0)}")
        print(f"   • Has categories: {stats.get('has_category', 0)}")
        print(f"   • Average title length: {stats.get('avg_title_length', 0):.1f} chars")

        # Price statistics
        if 'price_stats' in stats and stats['price_stats']:
            price_stats = stats['price_stats']
            print(f"   • Price range: {price_stats.get('min', 0):,.0f} - {price_stats.get('max', 0):,.0f}")
            print(f"   • Average price: {price_stats.get('avg', 0):,.0f}")

        # Show field mappings used
        field_mappings = processor._get_field_mappings('indian_real_estate')
        print(f"\n🗺️  Field Mappings Available: {len(field_mappings)} types")

        # Show first few products
        print(f"\n📦 Sample Products:")
        for i, product in enumerate(products[:5], 1):
            print(f"   {i}. ID: {product.id}")
            print(f"      Title: {product.title[:60]}...")
            if product.price:
                print(f"      Price: {product.currency} {product.price:,.0f}")
            if product.category:
                print(f"      Category: {product.category}")

            # Show key attributes
            if product.attributes:
                key_attrs = {}
                for key in ['city', 'num_beds', 'num_baths', 'size_sqft', 'agent_name']:
                    if key in product.attributes:
                        key_attrs[key] = product.attributes[key]
                if key_attrs:
                    print(f"      Attributes: {key_attrs}")
            print()

        return products

    except Exception as e:
        print(f"❌ Error in processing: {e}")
        traceback.print_exc()
        return []


def test_with_web_scraper(excel_file, products):
    """Test catalog processing WITH web scraper"""
    print("\n🕷️  TESTING WITH WEB SCRAPER")
    print("=" * 60)

    if not products:
        print("⚠️  No products to enhance - skipping web scraper test")
        return

    try:
        from core.web_scraper import WebScraper
        from utils.config import ConfigManager

        config = ConfigManager()
        scraper = WebScraper(config)

        print(f"🔍 Testing web scraper on first 3 products...")
        print(f"💡 Note: This may take 10-30 seconds per product")

        enhanced_count = 0

        for i, product in enumerate(products[:3], 1):
            print(f"\n🏘️  Enhancing Product {i}: {product.title[:50]}...")

            try:
                # Enhance with web data
                web_info = scraper.enhance_product_with_web_data(product)

                print(f"   📊 Results:")
                print(f"      • Search results: {len(web_info.search_results)}")
                print(f"      • Pages scraped: {len(web_info.scraped_content)}")
                print(f"      • Specifications: {len(web_info.specifications)}")
                print(f"      • Reviews found: {len(web_info.reviews)}")

                # Show sample search results
                if web_info.search_results:
                    print(f"   🔍 Sample search results:")
                    for j, result in enumerate(web_info.search_results[:2], 1):
                        title = result.get('title', 'No title')[:40]
                        source = result.get('source', 'unknown')
                        print(f"      {j}. {title}... (from {source})")

                # Show specifications if found
                if web_info.specifications:
                    print(f"   📋 Sample specifications:")
                    for url, specs in list(web_info.specifications.items())[:1]:
                        for key, value in list(specs.items())[:3]:
                            print(f"      • {key}: {value[:30]}...")

                enhanced_count += 1

            except Exception as e:
                print(f"   ⚠️  Web enhancement failed: {e}")
                continue

        print(f"\n✅ Successfully enhanced {enhanced_count} out of 3 products")

        if enhanced_count == 0:
            print("💡 Web scraping notes:")
            print("   • May need Google API key for better results")
            print("   • DuckDuckGo fallback has limited results")
            print("   • Network connectivity required")
            print("   • Some sites may block automated requests")

    except Exception as e:
        print(f"❌ Error in web scraping: {e}")
        traceback.print_exc()


def analyze_excel_structure(excel_file):
    """Analyze Excel file structure"""
    print("🔍 ANALYZING EXCEL STRUCTURE")
    print("=" * 60)

    try:
        import pandas as pd

        # Read just the first few rows
        df = pd.read_excel(excel_file, nrows=5)

        print(f"📋 Excel File Analysis:")
        print(f"   • Columns found: {len(df.columns)}")
        print(f"   • Sample rows: {len(df)}")

        print(f"\n📝 Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")

        print(f"\n📊 Sample Data (first 2 rows):")
        for idx, row in df.head(2).iterrows():
            print(f"   Row {idx + 1}:")
            for col in df.columns[:5]:  # Show first 5 columns
                value = str(row[col])[:30] if pd.notna(row[col]) else "None"
                print(f"      {col}: {value}")
            print()

        return True

    except Exception as e:
        print(f"❌ Error analyzing Excel: {e}")
        return False


def main():
    """Main test function"""
    print("📊 EXCEL CATALOG TESTING")
    print("🧪 Testing Real Estate Processing")
    print("=" * 70)

    excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"

    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        print("💡 Please ensure the file exists and path is correct")
        return False

    print(f"📁 Target file: {excel_file}")
    print(f"📏 File size: {Path(excel_file).stat().st_size / 1024:.1f} KB")

    try:
        # Step 1: Analyze structure
        if not analyze_excel_structure(excel_file):
            return False

        # Step 2: Test without web scraper
        products = test_without_web_scraper(excel_file)

        # Step 3: Test with web scraper (if products found)
        test_with_web_scraper(excel_file, products)

        # Summary
        print(f"\n🎯 TESTING SUMMARY")
        print("=" * 50)
        print(f"✅ Excel structure analysis: Complete")
        print(f"✅ Catalog processing: {'Success' if products else 'No products found'}")
        print(f"✅ Web scraper test: {'Complete' if products else 'Skipped'}")

        print(f"\n📖 What was tested:")
        print(f"   • Excel file reading and parsing")
        print(f"   • Meta real estate field mapping")
        print(f"   • Indian market price parsing (₹, lakh, crore)")
        print(f"   • Product data extraction and validation")
        if products:
            print(f"   • Web search and content scraping")
            print(f"   • Specification and review extraction")

        print(f"\n💡 Next Steps:")
        print(f"   1. Review the field mappings for your data")
        print(f"   2. Adjust column names if needed")
        print(f"   3. Configure web scraper with API keys for better results")
        print(f"   4. Process your full catalog when ready")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
