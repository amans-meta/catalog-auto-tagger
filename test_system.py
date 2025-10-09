#!/usr/bin/env python3
"""
Quick test script for the Catalog Auto-Tagger system
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 Testing Catalog Auto-Tagger System")
    print("=" * 50)

    try:
        # Test imports
        print("📦 Testing imports...")
        from models.product import GeneratedTag, ProductInfo
        from models.tag_config import RealEstateTagConfig
        from utils.config import ConfigManager
        from utils.text_processing import TextProcessor

        print("   ✅ All imports successful!")

        # Test configuration
        print("\n⚙️  Testing configuration...")
        config = ConfigManager()
        print(f"   ✅ Config loaded successfully")
        print(f"   📊 Web search enabled: {config.get('web_search.enabled', False)}")
        print(f"   🔧 Processing batch size: {config.get('processing.batch_size', 10)}")

        # Test text processing
        print("\n📝 Testing text processing...")
        text_processor = TextProcessor()
        sample_text = "Beautiful 3BR luxury home with pool, $450,000"
        features = text_processor.extract_features(sample_text)
        print(f"   ✅ Extracted {len(features['keywords'])} keywords")
        print(f"   🏷️  Keywords: {features['keywords'][:5]}")

        # Test tag configuration
        print("\n🏷️  Testing tag configuration...")
        real_estate_tags = RealEstateTagConfig.get_tags()
        print(f"   ✅ Loaded {len(real_estate_tags)} real estate tags")
        for tag in real_estate_tags[:3]:
            print(f"   • {tag.name} ({tag.tag_type})")

        # Test product model
        print("\n📦 Testing product model...")
        product = ProductInfo(
            id="test_001",
            title="Beautiful 3BR Home",
            description="Luxury home with modern amenities",
            price=450000,
            category="Single Family Home",
        )
        print(f"   ✅ Created product: {product.title}")
        print(f"   💰 Price: ${product.price:,}")

        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ System is ready to use!")
        print("\n📖 Next steps:")
        print("   1. Process a sample catalog:")
        print("      python test_catalog.py")
        print("   2. Try the CLI (if imports work):")
        print("      PYTHONPATH=src python cli.py --help")
        print("   3. Read the README.md for full documentation")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
