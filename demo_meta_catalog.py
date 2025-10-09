#!/usr/bin/env python3
"""
Meta Real Estate Catalog Demo

This script demonstrates the catalog auto-tagger system
configured for Meta real estate ads with comprehensive field mapping.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def create_meta_real_estate_sample():
    """Create a sample Meta real estate catalog with comprehensive fields"""
    sample_data = [
        {
            # Required Meta fields
            "id": "META_RE_001",
            "address": "123 Oak Street, Downtown Mumbai, Maharashtra 400001",
            "availability": "AVAILABLE",
            "description": "Stunning 3BHK luxury apartment in the heart of Mumbai with premium amenities including swimming pool, gym, 24x7 security, and covered parking. Perfect for families with excellent connectivity to IT hubs and metro stations.",
            "listing_type": "FOR_SALE",
            "price": "8500000",  # 85 lakh
            "property_type": "APARTMENT",
            "url": "https://example.com/property/META_RE_001",
            # Location details
            "city": "Mumbai",
            "region": "Maharashtra",
            "postal_code": "400001",
            "neighborhood": "Downtown",
            "latitude": "19.0760",
            "longitude": "72.8777",
            # Property details
            "num_beds": "3",
            "num_baths": "2",
            "size_sqft": "1200",
            "year_built": "2020",
            "furnish_type": "SEMI_FURNISHED",
            "parking_type": "COVERED",
            # Images and media
            "image_url": "https://example.com/images/META_RE_001_main.jpg",
            "additional_image_url": "https://example.com/images/META_RE_001_1.jpg,https://example.com/images/META_RE_001_2.jpg,https://example.com/images/META_RE_001_3.jpg",
            # Agent information
            "agent_name": "Rajesh Kumar",
            "agent_phone": "+91-9876543210",
            "agent_email": "rajesh@premiumrealty.in",
            "agent_company": "Premium Realty Mumbai",
            # Additional features
            "features": "Swimming Pool, Gym, 24x7 Security, Power Backup, Lift, Covered Parking, Garden, Children Play Area",
            "currency": "INR",
            "condition": "NEW",
            "days_on_market": "15",
            # Custom labels for Meta ads
            "custom_label_0": "Luxury",
            "custom_label_1": "Family Friendly",
            "custom_label_2": "IT Hub Nearby",
            "custom_label_3": "Metro Connected",
            "custom_label_4": "Investment Grade",
        },
        {
            # Villa example with different field structure
            "property_id": "META_RE_002",  # Alternative ID field
            "full_address": "Plot 45, Sector 7, Gurgaon, Haryana 122001",
            "status": "AVAILABLE",
            "details": "Independent 4BHK villa in premium gated community with private garden, swimming pool, and 2-car garage. Vastu compliant with east-facing orientation.",
            "transaction_type": "FOR_SALE",
            "sale_price": "25000000",  # 2.5 crore
            "home_type": "VILLA",
            "listing_url": "https://example.com/villa/META_RE_002",
            # Location with different field names
            "town": "Gurgaon",
            "state": "Haryana",
            "pincode": "122001",
            "locality": "Sector 7",
            "lat": "28.4595",
            "lng": "77.0266",
            # Property details with variations
            "bedrooms": "4",
            "bathrooms": "3",
            "square_feet": "2800",
            "construction_year": "2021",
            "furnished": "UNFURNISHED",
            "garage": "2 CAR GARAGE",
            # Media
            "primary_image": "https://example.com/images/META_RE_002_main.jpg",
            "photos": "https://example.com/images/META_RE_002_1.jpg,https://example.com/images/META_RE_002_2.jpg",
            # Contact details
            "realtor_name": "Priya Sharma",
            "contact_phone": "+91-9988776655",
            "contact_email": "priya@luxuryvilas.in",
            "builder": "Luxury Villas Pvt Ltd",
            # Features
            "amenities": "Private Garden, Swimming Pool, Garage, Security, Vastu Compliant",
            "curr": "INR",
            "property_condition": "EXCELLENT",
            "listing_age": "7",
            # Indian specific
            "facing": "East",
            "possession": "READY_TO_MOVE",
            "vastu_compliant": "Yes",
        },
        {
            # Budget apartment example
            "mls_id": "META_RE_003",
            "street_address": "Flat 302, Building C, Sunrise Apartments, Pune",
            "property_status": "AVAILABLE",
            "property_description": "Affordable 2BHK flat perfect for first-time buyers. Well-maintained building with basic amenities and good connectivity.",
            "deal_type": "FOR_SALE",
            "cost": "4500000",  # 45 lakh
            "building_type": "APARTMENT",
            "website": "https://example.com/budget/META_RE_003",
            # Location
            "municipality": "Pune",
            "province": "Maharashtra",
            "zip_code": "411001",
            "area": "Koregaon Park",
            "latitude": "18.5204",
            "longitude": "73.8567",
            # Details
            "bedroom_count": "2",
            "bath_count": "1",
            "area_sqft": "850",
            "built_year": "2015",
            "furnishing": "UNFURNISHED",
            "parking_spaces": "OPEN",
            # Media
            "main_image": "https://example.com/images/META_RE_003.jpg",
            # Agent
            "broker_name": "Amit Patel",
            "mobile": "+91-9123456789",
            "email": "amit@budgetflats.com",
            # Features
            "facilities": "Lift, Security Guard, Water Supply",
            "price_currency": "INR",
            "age": "8",  # Years old
            # Budget category indicators
            "custom_label_0": "Budget Friendly",
            "custom_label_1": "First Time Buyer",
            "custom_label_2": "Good Connectivity",
        },
    ]

    # Create CSV
    csv_path = Path("data/samples/meta_real_estate_sample.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)

    print(f"✅ Created Meta real estate sample: {csv_path}")
    return str(csv_path)


def demo_meta_field_mapping():
    """Demonstrate Meta real estate field mapping capabilities"""
    print("🗺️  META REAL ESTATE FIELD MAPPING DEMO")
    print("=" * 60)

    from core.catalog_processor import CatalogProcessor
    from utils.config import ConfigManager

    config = ConfigManager()
    processor = CatalogProcessor(config)

    # Show comprehensive field mappings
    field_mappings = processor._get_field_mappings("indian_real_estate")

    print("📋 Comprehensive Field Mappings (Meta + Indian + Custom):")
    print()

    # Group by category for better display
    categories = {
        "Core Identity": ["id", "title", "description", "price", "category", "url"],
        "Location": [
            "address",
            "city",
            "region",
            "postal_code",
            "latitude",
            "longitude",
            "neighborhood",
        ],
        "Property Details": [
            "num_beds",
            "num_baths",
            "size_sqft",
            "year_built",
            "furnish_type",
            "parking_type",
        ],
        "Listing Info": [
            "availability",
            "listing_type",
            "agent_name",
            "agent_phone",
            "agent_email",
        ],
        "Media & Features": ["image_url", "additional_image_url", "features"],
    }

    for category, fields in categories.items():
        print(f"📁 {category}:")
        for field in fields:
            if field in field_mappings:
                aliases = field_mappings[field][:5]  # Show first 5 aliases
                more_count = len(field_mappings[field]) - 5
                aliases_str = ", ".join(aliases)
                if more_count > 0:
                    aliases_str += f" (+{more_count} more)"
                print(f"   • {field}: {aliases_str}")
        print()


def demo_meta_catalog_processing():
    """Demonstrate processing Meta real estate catalog"""
    print("📊 META CATALOG PROCESSING DEMO")
    print("=" * 60)

    # Create sample file
    sample_file = create_meta_real_estate_sample()

    from core.catalog_processor import CatalogProcessor
    from utils.config import ConfigManager

    config = ConfigManager()
    processor = CatalogProcessor(config)

    print(f"📁 Processing Meta catalog: {sample_file}")

    try:
        # Process with Indian real estate configuration (includes Meta fields)
        products = processor.process_catalog_file(sample_file, "indian_real_estate")

        print(f"✅ Successfully processed {len(products)} products")

        # Analyze each product
        for i, product in enumerate(products, 1):
            print(f"\n🏘️  Product {i}: {product.title}")
            print("-" * 50)

            print(f"   📍 ID: {product.id}")
            if product.price:
                print(f"   💰 Price: {product.currency} {product.price:,.0f}")
            if product.category:
                print(f"   🏠 Type: {product.category}")

            # Show attributes extracted
            if product.attributes:
                print(f"   📋 Extracted Attributes:")
                # Show key attributes
                key_attrs = [
                    "city",
                    "num_beds",
                    "num_baths",
                    "size_sqft",
                    "agent_name",
                    "parking_type",
                ]
                for attr in key_attrs:
                    if attr in product.attributes:
                        print(f"      • {attr}: {product.attributes[attr]}")

                # Show count of other attributes
                other_attrs = len(
                    [k for k in product.attributes.keys() if k not in key_attrs]
                )
                if other_attrs > 0:
                    print(f"      • (+{other_attrs} more attributes)")

            if product.images:
                print(f"   🖼️  Images: {len(product.images)} image(s)")

    except Exception as e:
        print(f"❌ Error processing Meta catalog: {e}")
        import traceback

        traceback.print_exc()


def demo_meta_validation():
    """Demonstrate Meta catalog validation"""
    print("\n✅ META CATALOG VALIDATION DEMO")
    print("=" * 60)

    import yaml

    # Load Meta field specifications
    try:
        with open("config/meta_real_estate_fields.yaml", "r") as f:
            meta_config = yaml.safe_load(f)

        print("📋 Meta Real Estate Field Specifications:")

        # Show required fields
        required_fields = meta_config.get("required_fields", [])
        print(f"\n🔴 Required Fields ({len(required_fields)}):")
        for field in required_fields:
            print(f"   • {field}")

        # Show validation rules
        validation_rules = meta_config.get("validation_rules", {})

        if "allowed_values" in validation_rules:
            print(f"\n📝 Allowed Values:")
            for field, values in validation_rules["allowed_values"].items():
                print(
                    f"   • {field}: {', '.join(values[:5])}{'...' if len(values) > 5 else ''}"
                )

        if "numeric_ranges" in validation_rules:
            print(f"\n🔢 Numeric Ranges:")
            for field, range_info in validation_rules["numeric_ranges"].items():
                min_val = range_info.get("min", "N/A")
                max_val = range_info.get("max", "N/A")
                print(f"   • {field}: {min_val} - {max_val}")

        # Show field types
        field_types = meta_config.get("field_types", {})
        print(f"\n📊 Field Type Summary:")
        for field_type, fields in field_types.items():
            print(f"   • {field_type.upper()}: {len(fields)} fields")

    except Exception as e:
        print(f"⚠️  Could not load Meta specifications: {e}")


def demo_meta_integration_guide():
    """Show integration guide for Meta ads"""
    print(f"\n🔗 META ADS INTEGRATION GUIDE")
    print("=" * 60)

    print("📖 To use this system with Meta Real Estate Ads:")
    print()

    print("1️⃣  CATALOG PREPARATION")
    print("   • Ensure you have all required Meta fields:")
    print("     - id, address, availability, description")
    print("     - listing_type, price, property_type, url")
    print("   • Use standard Meta field names or supported aliases")
    print("   • Include INR prices in any format (₹, lakh, crore)")
    print()

    print("2️⃣  FIELD MAPPING")
    print("   • System auto-detects 50+ field variations")
    print("   • Handles Indian-specific fields (BHK, pincode, etc.)")
    print("   • Maps to Meta catalog format automatically")
    print()

    print("3️⃣  PROCESSING WORKFLOW")
    print("   ```python")
    print("   processor = CatalogProcessor(config)")
    print(
        "   products = processor.process_catalog_file('your_meta_catalog.csv', 'indian_real_estate')"
    )
    print("   ```")
    print()

    print("4️⃣  OUTPUT VALIDATION")
    print("   • Validates required Meta fields")
    print("   • Checks data types and ranges")
    print("   • Generates tags for ad targeting")
    print()

    print("5️⃣  SUPPORTED FORMATS")
    print("   • CSV (recommended for Meta)")
    print("   • Excel (.xlsx, .xls)")
    print("   • JSON/JSONL")
    print()

    print("💡 PRO TIPS:")
    print("   • Include agent information for better ads")
    print("   • Add multiple images for better performance")
    print("   • Use custom_label_0-4 for ad categorization")
    print("   • Ensure address format matches Meta requirements")


def main():
    """Main demo function"""
    print("🏢 META REAL ESTATE CATALOG AUTO-TAGGER")
    print("📊 Facebook Marketing API Compatible System")
    print("=" * 70)

    print(f"📍 Working directory: {os.getcwd()}")

    try:
        # Demo Meta field mapping
        demo_meta_field_mapping()

        # Demo Meta catalog processing
        demo_meta_catalog_processing()

        # Demo validation
        demo_meta_validation()

        # Show integration guide
        demo_meta_integration_guide()

        print(f"\n🎉 META INTEGRATION DEMO COMPLETED!")
        print(f"\n📈 System Capabilities Demonstrated:")
        print(f"   ✅ Meta real estate field mapping (50+ variations)")
        print(f"   ✅ Indian market customization (INR, BHK, etc.)")
        print(f"   ✅ Comprehensive validation rules")
        print(f"   ✅ Multi-format catalog processing")
        print(f"   ✅ Agent & property detail extraction")
        print(f"   ✅ Image URL handling")
        print(f"   ✅ Auto-tagging for ad targeting")

        print(f"\n🔧 Ready for Meta Ads Integration:")
        print(f"   • Upload your real estate catalog (any format)")
        print(f"   • System will map to Meta fields automatically")
        print(f"   • Generate tags for better ad targeting")
        print(f"   • Validate data meets Meta requirements")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
