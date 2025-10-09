#!/usr/bin/env python3
"""
Simple Excel Catalog Test (No ML Dependencies)
Tests catalog processing without PyTorch/ML components
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def analyze_excel_structure(excel_file):
    """Analyze Excel file structure"""
    print("🔍 ANALYZING EXCEL STRUCTURE")
    print("=" * 60)

    try:
        import pandas as pd

        # Read just the first few rows
        df = pd.read_excel(excel_file, nrows=10)

        print(f"📋 Excel File Analysis:")
        print(f"   • Columns found: {len(df.columns)}")
        print(f"   • Sample rows: {len(df)}")
        print(f"   • File size: {Path(excel_file).stat().st_size / 1024:.1f} KB")

        print(f"\n📝 All Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")

        print(f"\n📊 Sample Data (first 3 rows):")
        for idx, row in df.head(3).iterrows():
            print(f"\n   Row {idx + 1}:")
            for col in df.columns:
                value = str(row[col]) if pd.notna(row[col]) else "None"
                # Truncate long values
                if len(value) > 50:
                    value = value[:50] + "..."
                print(f"      {col}: {value}")

        return df

    except Exception as e:
        print(f"❌ Error analyzing Excel: {e}")
        traceback.print_exc()
        return None


def test_basic_processing(excel_file, df):
    """Test basic catalog processing without ML"""
    print(f"\n🔥 TESTING BASIC CATALOG PROCESSING")
    print("=" * 60)

    try:
        # Import only the basic components we need
        import pandas as pd
        from utils.config import ConfigManager

        # Test config loading
        config = ConfigManager()
        print("✅ Configuration loaded successfully")

        # Test field mapping
        print(f"\n🗺️  Testing Field Mappings:")

        # Manually simulate field mapping
        columns = list(df.columns)
        mapped_fields = {}

        # Common field patterns
        field_patterns = {
            'id': ['id', 'property_id', 'listing_id', 'item_id', 'unique_id'],
            'title': ['title', 'name', 'property_name', 'listing_title', 'heading'],
            'description': ['description', 'desc', 'details', 'summary', 'about'],
            'price': ['price', 'cost', 'amount', 'value', 'sale_price', 'rent'],
            'category': ['category', 'type', 'property_type', 'building_type'],
            'city': ['city', 'town', 'location', 'area', 'locality'],
            'bedrooms': ['bedrooms', 'beds', 'num_beds', 'bhk'],
            'bathrooms': ['bathrooms', 'baths', 'num_baths'],
            'size': ['size', 'area', 'sqft', 'square_feet', 'built_up_area']
        }

        # Map fields
        for field_type, patterns in field_patterns.items():
            for col in columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern in col_lower:
                        if field_type not in mapped_fields:
                            mapped_fields[field_type] = []
                        mapped_fields[field_type].append(col)
                        break

        print(f"📋 Detected Field Mappings:")
        for field_type, cols in mapped_fields.items():
            print(f"   • {field_type}: {cols}")

        # Test price parsing
        print(f"\n💰 Testing Price Parsing:")

        # Find price column
        price_col = None
        for col in columns:
            if any(p in col.lower() for p in ['price', 'cost', 'amount', 'value']):
                price_col = col
                break

        if price_col and price_col in df.columns:
            print(f"   Price column found: {price_col}")

            # Test parsing different price formats
            sample_prices = df[price_col].dropna().head(5)

            def parse_price(price_str):
                """Simple price parsing"""
                if pd.isna(price_str):
                    return None

                price_str = str(price_str).lower().strip()

                # Remove currency symbols
                price_str = price_str.replace('₹', '').replace('rs.', '').replace('rs', '').replace(',', '').replace(' ', '')

                try:
                    # Handle lakh/crore
                    if 'lakh' in price_str or 'lac' in price_str:
                        number = float(price_str.replace('lakh', '').replace('lac', ''))
                        return number * 100000
                    elif 'crore' in price_str or 'cr' in price_str:
                        number = float(price_str.replace('crore', '').replace('cr', ''))
                        return number * 10000000
                    else:
                        return float(price_str)
                except:
                    return None

            print(f"   Sample price parsing:")
            for i, price in enumerate(sample_prices, 1):
                parsed = parse_price(price)
                formatted = f"₹{parsed:,.0f}" if parsed else "Could not parse"
                print(f"      {i}. '{price}' → {formatted}")
        else:
            print(f"   No price column detected")

        # Statistics
        print(f"\n📊 Data Statistics:")
        print(f"   • Total rows: {len(df)}")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Non-empty cells: {df.notna().sum().sum()}")
        print(f"   • Empty cells: {df.isna().sum().sum()}")

        # Column completeness
        print(f"\n📈 Column Completeness:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            percentage = (non_null / len(df)) * 100
            print(f"   • {col}: {non_null}/{len(df)} ({percentage:.1f}%)")

        return True

    except Exception as e:
        print(f"❌ Error in basic processing: {e}")
        traceback.print_exc()
        return False


def test_web_scraper_availability():
    """Test if web scraper can be loaded"""
    print(f"\n🕷️  TESTING WEB SCRAPER AVAILABILITY")
    print("=" * 60)

    try:
        from core.web_scraper import WebScraper
        from utils.config import ConfigManager

        config = ConfigManager()
        scraper = WebScraper(config)

        print("✅ Web scraper loaded successfully")
        print("💡 Web scraper is available for enhanced processing")

        # Show configured sites
        indian_config = config.get("catalog_types.indian_real_estate", {})
        if indian_config:
            scraping_rules = indian_config.get("scraping_rules", {})
            priority_sites = scraping_rules.get("priority_sites", [])

            print(f"\n🎯 Configured Indian Real Estate Sites:")
            for i, site in enumerate(priority_sites[:5], 1):
                print(f"   {i}. {site}")

            if len(priority_sites) > 5:
                print(f"   ... and {len(priority_sites) - 5} more")

        return True

    except Exception as e:
        print(f"⚠️  Web scraper not available: {e}")
        return False


def generate_processing_guide(mapped_fields):
    """Generate a guide for processing the real catalog"""
    print(f"\n📖 PROCESSING GUIDE FOR YOUR DATA")
    print("=" * 60)

    print("🔧 To process your full Excel catalog:")
    print()

    print("1️⃣  INSTALL DEPENDENCIES (if needed):")
    print("   pip install pandas openpyxl requests beautifulsoup4 pyyaml")
    print()

    print("2️⃣  BASIC PROCESSING (Fast, No Web Scraping):")
    print("```python")
    print("import pandas as pd")
    print("from src.utils.config import ConfigManager")
    print("")
    print("# Read your Excel file")
    print("df = pd.read_excel('/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx')")
    print("")
    print("# Process each row")
    print("products = []")
    print("for idx, row in df.iterrows():")
    print("    product = {")
    for field, cols in mapped_fields.items():
        if cols:
            print(f"        '{field}': row.get('{cols[0]}', ''),")
    print("    }")
    print("    products.append(product)")
    print("")
    print("print(f'Processed {len(products)} products')")
    print("```")
    print()

    print("3️⃣  ENHANCED PROCESSING (With Web Scraping):")
    print("   • Add Google Search API key (optional)")
    print("   • Enable web scraping for live data")
    print("   • Extract specifications and reviews")
    print()

    print("4️⃣  FIELD MAPPING CUSTOMIZATION:")
    print("   • Modify config/settings.yaml")
    print("   • Add your specific column names")
    print("   • Configure for your data format")
    print()

    print("💡 What your data can become:")
    print("   ✅ Standardized real estate catalog")
    print("   ✅ Price normalization (₹, lakh, crore)")
    print("   ✅ Property feature extraction")
    print("   ✅ Location and amenity tagging")
    print("   ✅ Web-enhanced property details")
    print("   ✅ Meta ads compatible format")


def main():
    """Main test function"""
    print("📊 SIMPLE EXCEL CATALOG TEST")
    print("🚀 No ML Dependencies Required")
    print("=" * 70)

    excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"

    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        print("💡 Please ensure the file exists and path is correct")
        return False

    try:
        # Step 1: Analyze Excel structure
        df = analyze_excel_structure(excel_file)
        if df is None:
            return False

        # Step 2: Test basic processing
        success = test_basic_processing(excel_file, df)
        if not success:
            return False

        # Step 3: Test web scraper availability
        web_available = test_web_scraper_availability()

        # Step 4: Generate processing guide
        # Get mapped fields for guide
        columns = list(df.columns)
        mapped_fields = {}
        field_patterns = {
            'id': ['id', 'property_id', 'listing_id'],
            'title': ['title', 'name', 'property_name'],
            'price': ['price', 'cost', 'amount'],
            'city': ['city', 'location', 'area']
        }

        for field_type, patterns in field_patterns.items():
            for col in columns:
                if any(p in col.lower() for p in patterns):
                    mapped_fields[field_type] = [col]
                    break

        generate_processing_guide(mapped_fields)

        # Final summary
        print(f"\n🎯 TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Excel file analysis: Complete")
        print(f"✅ Column detection: {len(df.columns)} columns found")
        print(f"✅ Data parsing: {len(df)} rows detected")
        print(f"✅ Field mapping: Ready")
        print(f"✅ Web scraper: {'Available' if web_available else 'Install torch for ML features'}")

        print(f"\n🚀 READY FOR FULL PROCESSING!")
        print("Your Excel file is compatible with the system.")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
