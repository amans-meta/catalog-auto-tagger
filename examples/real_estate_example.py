#!/usr/bin/env python3
"""
Real Estate Catalog Auto-Tagging Example

This example demonstrates how to use the catalog auto-tagger system
to process real estate listings and generate relevant tags.
"""

import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tag_generator import TagGenerator


def create_sample_real_estate_data():
    """Create sample real estate data for testing"""

    sample_listings = [
        {
            "id": "listing_001",
            "title": "Beautiful 3BR/2BA Single Family Home in Downtown",
            "description": "Stunning newly renovated home featuring hardwood floors, granite countertops, and a spacious backyard. Located in excellent school district with easy access to shopping and restaurants. This luxury property includes a 2-car garage and has been completely updated with modern amenities.",
            "price": 450000,
            "currency": "USD",
            "category": "Single Family Home",
            "address": "123 Main Street, Downtown",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 2100,
            "lot_size": "0.25 acres",
            "year_built": 2020,
            "features": "hardwood floors, granite countertops, 2-car garage, backyard",
            "school_district": "Excellent School District",
            "images": [
                "https://example.com/listing1_1.jpg",
                "https://example.com/listing1_2.jpg",
            ],
        },
        {
            "id": "listing_002",
            "title": "Luxury Waterfront Condo with Pool and Spa",
            "description": "High-end condominium unit with stunning ocean views. Features include marble floors, stainless steel appliances, and access to resort-style amenities including pool, spa, and fitness center. This premium property is perfect for those seeking luxury living.",
            "price": 850000,
            "currency": "USD",
            "category": "Condominium",
            "address": "456 Ocean Drive, Waterfront District",
            "bedrooms": 2,
            "bathrooms": 2,
            "square_feet": 1800,
            "year_built": 2018,
            "features": "ocean views, marble floors, stainless appliances, pool, spa, fitness center",
            "amenities": "pool, spa, fitness center, concierge",
            "images": ["https://example.com/listing2_1.jpg"],
        },
        {
            "id": "listing_003",
            "title": "Fixer Upper - Great Potential in Suburban Neighborhood",
            "description": "This property needs some TLC but has great bones and potential. Located in a quiet family-friendly suburban area with good schools nearby. Perfect for investors or DIY enthusiasts looking for a project. Priced to sell quickly.",
            "price": 125000,
            "currency": "USD",
            "category": "Single Family Home",
            "address": "789 Suburban Lane, Family Neighborhood",
            "bedrooms": 2,
            "bathrooms": 1,
            "square_feet": 1200,
            "year_built": 1985,
            "condition": "needs work",
            "features": "suburban location, family neighborhood, good schools",
            "images": [],
        },
        {
            "id": "listing_004",
            "title": "Modern Townhouse - 4BR/3BA with Garage",
            "description": "Spacious modern townhouse in desirable location. Features include open floor plan, updated kitchen with island, master suite with walk-in closet, and attached 2-car garage. Great for families looking for move-in ready home.",
            "price": 320000,
            "currency": "USD",
            "category": "Townhouse",
            "address": "321 Townhouse Way, Modern Community",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2500,
            "year_built": 2019,
            "features": "open floor plan, updated kitchen, master suite, 2-car garage",
            "images": [
                "https://example.com/listing4_1.jpg",
                "https://example.com/listing4_2.jpg",
            ],
        },
    ]

    return sample_listings


def main():
    """Main example function"""

    # Initialize the tag generator for real estate
    print("🏠 Real Estate Catalog Auto-Tagger Example")
    print("=" * 50)

    tagger = TagGenerator(catalog_type="real_estate")

    # Get available tags
    available_tags = tagger.get_available_tags()
    print(f"\n📋 Available Real Estate Tags ({len(available_tags)}):")
    for tag in available_tags:
        explanation = tagger.explain_tag(tag)
        print(f"  • {tag}: {explanation}")

    # Create sample data
    sample_listings = create_sample_real_estate_data()

    print(f"\n🏘️  Processing {len(sample_listings)} Sample Listings:")
    print("-" * 50)

    # Process each listing
    for i, listing in enumerate(sample_listings, 1):
        print(f"\n{i}. Processing: {listing['title']}")
        print(f"   Price: ${listing['price']:,}")
        print(f"   Type: {listing['category']}")

        # Test single product tag generation (without web enhancement for demo)
        result = tagger.test_single_product(listing)

        if result["success"]:
            print(
                f"   ✅ Generated {result['tag_count']} tags in {result['processing_time_seconds']:.2f}s"
            )

            # Show top tags
            for tag in result["generated_tags"][:5]:  # Top 5 tags
                confidence_indicator = (
                    "🟢"
                    if tag["confidence"] > 0.7
                    else "🟡" if tag["confidence"] > 0.5 else "🔴"
                )
                print(
                    f"      {confidence_indicator} {tag['name']} ({tag['confidence']:.2f}) - {tag['source']}"
                )
                if tag["reasoning"]:
                    print(f"         Reason: {tag['reasoning']}")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

    # Demonstrate configuration
    print(f"\n⚙️  System Configuration:")
    status = tagger.get_processing_status()
    print(f"   • Catalog Type: {status['catalog_type']}")
    print(
        f"   • Web Search: {'Enabled' if status['web_search_enabled'] else 'Disabled'}"
    )
    print(f"   • ML Models: {'Enabled' if status['ml_models_enabled'] else 'Disabled'}")
    print(f"   • Available Tags: {status['available_tags']}")
    print(f"   • Supported Formats: {', '.join(status['supported_formats'])}")

    print(f"\n🎯 Example Complete!")
    print("To process a real catalog file, use:")
    print("  result = tagger.process_catalog('path/to/your/catalog.csv')")


if __name__ == "__main__":
    main()
