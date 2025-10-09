#!/usr/bin/env python3
"""
Catalog Auto-Tagger Command Line Interface

Usage:
    python cli.py process catalog.csv --type real_estate --output results.json
    python cli.py test --product '{"title": "Beautiful Home", "price": 450000}'
    python cli.py tags --type real_estate
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.tag_generator import TagGenerator


def process_catalog_command(args):
    """Process a catalog file"""
    print(f"🏭 Processing catalog: {args.catalog}")
    print(f"📋 Catalog type: {args.type}")
    print(f"🌐 Web enhancement: {'Enabled' if args.web else 'Disabled'}")

    try:
        # Initialize tag generator
        tagger = TagGenerator(catalog_type=args.type)

        # Process catalog
        result = tagger.process_catalog(
            catalog_path=args.catalog,
            enable_web_enhancement=args.web,
            output_path=args.output,
        )

        # Print summary
        print(f"\n✅ Processing Complete!")
        print(f"   📊 Total products: {result.total_products}")
        print(f"   ✅ Successfully processed: {result.processed_products}")
        print(f"   ❌ Failed: {result.failed_products}")
        print(f"   ⏱️  Processing time: {result.processing_time_seconds:.2f}s")

        if args.output:
            print(f"   💾 Results saved to: {args.output}")

        # Show summary stats
        if result.summary_stats:
            stats = result.summary_stats
            print(f"\n📈 Summary Statistics:")
            print(f"   🏷️  Total tags generated: {stats.get('total_tags_generated', 0)}")
            print(f"   🎯 Unique tags: {stats.get('unique_tags', 0)}")
            print(
                f"   📊 Avg tags per product: {stats.get('avg_tags_per_product', 0):.1f}"
            )
            print(f"   🎖️  Avg confidence: {stats.get('avg_confidence', 0):.2f}")

            # Show most common tags
            most_common = stats.get("most_common_tags", {})
            if most_common:
                print(f"\n🔥 Most Common Tags:")
                for tag, count in list(most_common.items())[:5]:
                    print(f"   • {tag}: {count} products")

    except Exception as e:
        print(f"❌ Error processing catalog: {e}")
        sys.exit(1)


def test_product_command(args):
    """Test tag generation on a single product"""
    print("🧪 Testing single product tag generation")

    try:
        # Parse product data
        if args.product.startswith("{"):
            product_data = json.loads(args.product)
        else:
            # Assume it's a simple title
            product_data = {"title": args.product}

        print(f"📦 Product: {product_data.get('title', 'Unknown')}")

        # Initialize tag generator
        tagger = TagGenerator(catalog_type=args.type)

        # Test product
        result = tagger.test_single_product(product_data)

        if result["success"]:
            print(
                f"✅ Generated {result['tag_count']} tags in {result['processing_time_seconds']:.2f}s"
            )
            print(f"🎯 High confidence tags: {result['high_confidence_tags']}")

            print(f"\n🏷️  Generated Tags:")
            for tag in result["generated_tags"]:
                confidence_icon = (
                    "🟢"
                    if tag["confidence"] > 0.7
                    else "🟡" if tag["confidence"] > 0.5 else "🔴"
                )
                print(
                    f"   {confidence_icon} {tag['name']} ({tag['confidence']:.2f}) - {tag['source']}"
                )
                if tag.get("reasoning"):
                    print(f"      ↳ {tag['reasoning']}")
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")

    except json.JSONDecodeError:
        print("❌ Invalid JSON format for product data")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error testing product: {e}")
        sys.exit(1)


def list_tags_command(args):
    """List available tags for a catalog type"""
    print(f"📋 Available tags for catalog type: {args.type}")

    try:
        tagger = TagGenerator(catalog_type=args.type)
        tags = tagger.get_available_tags()

        if not tags:
            print("❌ No tags found for this catalog type")
            return

        print(f"\n🏷️  Found {len(tags)} predefined tags:")
        for tag in sorted(tags):
            explanation = tagger.explain_tag(tag)
            print(f"   • {tag}")
            if explanation:
                print(f"     ↳ {explanation}")

    except Exception as e:
        print(f"❌ Error listing tags: {e}")
        sys.exit(1)


def status_command(args):
    """Show system status and configuration"""
    print("⚙️  System Status")

    try:
        tagger = TagGenerator(catalog_type=args.type)
        status = tagger.get_processing_status()

        print(f"\n📊 Configuration:")
        print(f"   • Catalog Type: {status['catalog_type']}")
        print(
            f"   • Web Search: {'✅ Enabled' if status['web_search_enabled'] else '❌ Disabled'}"
        )
        print(
            f"   • ML Models: {'✅ Enabled' if status['ml_models_enabled'] else '❌ Disabled'}"
        )
        print(f"   • Available Tags: {status['available_tags']}")
        print(f"   • Max Concurrent: {status['max_concurrent_products']}")
        print(f"   • Batch Size: {status['batch_size']}")
        print(f"   • Supported Formats: {', '.join(status['supported_formats'])}")

    except Exception as e:
        print(f"❌ Error getting status: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Catalog Auto-Tagger - Automatically generate tags for catalog products",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a real estate catalog
    python cli.py process listings.csv --type real_estate --output results.json

    # Test a single product
    python cli.py test --product "Luxury 3BR Home with Pool" --type real_estate

    # Test with JSON data
    python cli.py test --product '{"title": "iPhone 15", "price": 999, "brand": "Apple"}' --type ecommerce

    # List available tags
    python cli.py tags --type real_estate

    # Check system status
    python cli.py status --type real_estate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process catalog command
    process_parser = subparsers.add_parser("process", help="Process a catalog file")
    process_parser.add_argument(
        "catalog", help="Path to catalog file (CSV, JSON, Excel)"
    )
    process_parser.add_argument(
        "--type",
        default="generic",
        choices=["real_estate", "ecommerce", "generic"],
        help="Catalog type (default: generic)",
    )
    process_parser.add_argument("--output", help="Output file path (JSON format)")
    process_parser.add_argument(
        "--no-web", dest="web", action="store_false", help="Disable web enhancement"
    )

    # Test single product command
    test_parser = subparsers.add_parser(
        "test", help="Test tag generation on a single product"
    )
    test_parser.add_argument(
        "--product", required=True, help="Product data (JSON string or simple title)"
    )
    test_parser.add_argument(
        "--type",
        default="generic",
        choices=["real_estate", "ecommerce", "generic"],
        help="Catalog type (default: generic)",
    )

    # List tags command
    tags_parser = subparsers.add_parser(
        "tags", help="List available tags for catalog type"
    )
    tags_parser.add_argument(
        "--type",
        default="generic",
        choices=["real_estate", "ecommerce", "generic"],
        help="Catalog type (default: generic)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument(
        "--type",
        default="generic",
        choices=["real_estate", "ecommerce", "generic"],
        help="Catalog type (default: generic)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute appropriate command
    if args.command == "process":
        process_catalog_command(args)
    elif args.command == "test":
        test_product_command(args)
    elif args.command == "tags":
        list_tags_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()
