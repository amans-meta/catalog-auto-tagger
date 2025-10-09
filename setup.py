#!/usr/bin/env python3
"""
Setup script for Catalog Auto-Tagger

This script helps with initial setup and installation.
"""

import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = ["logs", "models/cache", "cache", "data/samples", "output"]

    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("   ✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False
    return True


def create_sample_data():
    """Create sample data files for testing"""
    print("\n📊 Creating sample data...")

    # Sample real estate CSV
    real_estate_csv = """id,title,description,price,category,address,bedrooms,bathrooms,square_feet
1,"Beautiful 3BR Home in Downtown","Stunning newly renovated home with hardwood floors and granite countertops. Located in excellent school district.",450000,"Single Family Home","123 Main St, Downtown",3,2,2100
2,"Luxury Waterfront Condo","High-end condominium with ocean views, marble floors, and resort amenities.",850000,"Condominium","456 Ocean Dr, Waterfront",2,2,1800
3,"Fixer Upper with Great Potential","Property needs TLC but has great bones. Perfect for investors or DIY enthusiasts.",125000,"Single Family Home","789 Suburban Ln, Family Area",2,1,1200
4,"Modern Townhouse with Garage","Spacious townhouse with open floor plan, updated kitchen, and 2-car garage.",320000,"Townhouse","321 Modern Way, New Community",4,3,2500"""

    sample_data_path = Path("data/samples/real_estate_sample.csv")
    sample_data_path.write_text(real_estate_csv)
    print(f"   ✅ Created: {sample_data_path}")

    # Sample e-commerce CSV
    ecommerce_csv = """id,title,description,price,category,brand,url
1,"iPhone 15 Pro","Latest iPhone with advanced camera system and titanium design.",999,"Electronics","Apple","https://example.com/iphone15"
2,"Nike Air Max Sneakers","Comfortable running shoes with Air Max cushioning technology.",120,"Clothing & Shoes","Nike","https://example.com/airmax"
3,"Samsung 55\" 4K TV","Smart TV with HDR and streaming capabilities.",599,"Electronics","Samsung","https://example.com/samsung-tv"
4,"Instant Pot Pressure Cooker","Multi-use electric pressure cooker for quick and easy meals.",89,"Home & Kitchen","Instant Pot","https://example.com/instant-pot"""

    ecommerce_data_path = Path("data/samples/ecommerce_sample.csv")
    ecommerce_data_path.write_text(ecommerce_csv)
    print(f"   ✅ Created: {ecommerce_data_path}")


def setup_environment():
    """Setup environment file"""
    print("\n🔧 Setting up environment...")

    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists() and env_example_path.exists():
        env_path.write_text(env_example_path.read_text())
        print("   ✅ Created .env file from .env.example")
        print("   ⚠️  Please edit .env file to add your API keys")
    else:
        print("   ℹ️  .env file already exists or .env.example not found")


def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")

    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        from core.tag_generator import TagGenerator

        # Test basic functionality
        tagger = TagGenerator(catalog_type="real_estate")
        status = tagger.get_processing_status()

        print(f"   ✅ System initialized successfully")
        print(f"   📋 Catalog type: {status['catalog_type']}")
        print(f"   🏷️  Available tags: {status['available_tags']}")

        return True

    except Exception as e:
        print(f"   ❌ Installation test failed: {e}")
        return False


def main():
    print("🚀 Catalog Auto-Tagger Setup")
    print("=" * 40)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Run setup steps
    success = True

    create_directories()

    if not install_dependencies():
        success = False

    create_sample_data()
    setup_environment()

    if success and test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Edit .env file to add your API keys (optional)")
        print("   2. Try the examples:")
        print("      python examples/real_estate_example.py")
        print("   3. Test with sample data:")
        print(
            "      python cli.py process data/samples/real_estate_sample.csv --type real_estate"
        )
        print("   4. Read the README.md for detailed usage instructions")
    else:
        print("\n❌ Setup encountered errors. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
