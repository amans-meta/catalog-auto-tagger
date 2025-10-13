# Catalog Auto-Tagger System

An intelligent system that automatically generates tags for catalog products using two complementary approaches:

## 🎯 **Two Production-Ready Taggers**

### **1. Catalog-Only Tagger** (`catalog_tagger.py`)
- **Fast & Reliable**: Uses ONLY catalog data for tagging
- **High Performance**: 5-10x faster processing
- **No API Limits**: Perfect for large-scale catalogs (1M+ entries)
- **Semantic Similarity**: Advanced ML-based matching with sentence transformers

### **2. Hybrid Tagger** (`hybrid_tagger.py`)
- **Enhanced Accuracy**: Combines catalog data with Google Search web enrichment
- **Intelligent Matching**: Semantic similarity + web-scraped context
- **Production Ready**: Multithreaded with rate limiting and error handling

## 📁 **Clean Project Structure**

```
catalog-auto-tagger/
├── input/                        # 📥 All input files go here
│   └── [your-catalog-file.xlsx] # Your catalog data
├── output/                      # 📤 All results saved here
│   ├── catalog_only_results.csv
│   └── hybrid_tagged_results.csv
├── src/                        # 💻 Production code
│   ├── catalog_tagger.py       # Fast catalog-only tagger
│   ├── hybrid_tagger.py        # Web-enhanced hybrid tagger
│   └── utils/                   # 🔧 Utility modules
│       ├── config_loader.py    # Configuration management
│       └── __init__.py
├── config/                     # ⚙️ Flexible configuration system
│   ├── settings.yaml           # System settings & parameters
│   └── tags/
│       └── real_estate_tags.yaml  # Tag definitions (customizable!)
├── .env                        # 🔒 API credentials (keep secure)
├── .env.example               # 📋 Template for API setup
└── requirements.txt
```

## 🚀 **Quick Start**

### **Option 1: Fast Catalog-Only Processing**
```bash
# Real estate processing
python3 src/catalog_tagger.py real_estate

# Other industries (requires creating config file first)
python3 src/catalog_tagger.py automotive
python3 src/catalog_tagger.py electronics
python3 src/catalog_tagger.py custom_domain
```
- ✅ **Uses**: Only catalog data
- ✅ **Speed**: Very fast (5-10x faster)
- ✅ **Best for**: Large catalogs, regular processing
- 🎯 **Industry-agnostic**: Just specify your domain!

### **Option 2: Enhanced Web-Augmented Processing**
```bash
# Real estate with web enhancement
python3 src/hybrid_tagger.py real_estate

# Other industries with web boost
python3 src/hybrid_tagger.py automotive
python3 src/hybrid_tagger.py electronics

# Custom options
python3 src/hybrid_tagger.py real_estate --workers 8 --chunk-size 100
```
- ✅ **Uses**: Catalog data + Google Search enrichment
- ✅ **Accuracy**: Higher accuracy through web context
- ✅ **Best for**: High-value catalogs, detailed analysis
- 🎯 **Industry-agnostic**: Works with any domain configuration!

## 📊 **Input/Output**

**Input**: Place your catalog files in `/input/` folder
- Supports: `.xlsx`, `.csv` files
- The system will automatically detect and process files in the input directory

**Output**: Results automatically saved to `/output/` folder
- Catalog-only: `output/catalog_only_results.csv`
- Hybrid: `output/hybrid_tagged_results.csv`

## Supported Catalog Types

- Real Estate Listings
- E-commerce Products
- Automotive Listings
- Generic Product Catalogs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Google API Setup (Required for Hybrid Tagger)

To use the hybrid tagger with web enrichment, you need:

1. **Google Custom Search API Key** - Get from [Google Cloud Console](https://developers.google.com/custom-search/v1/introduction)
2. **Google Custom Search Engine ID** - Create at [Google CSE](https://cse.google.com/cse/)

### Setup Steps:
```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env and add your API credentials
# GOOGLE_API_KEY=your_api_key_here
# GOOGLE_CSE_ID=your_cse_id_here
```

⚠️ **Security Note**: Never commit the `.env` file with real credentials to version control!

## 🔧 **Flexible Configuration System**

The system now uses a flexible YAML-based configuration that separates logic from data:

### **Tag Configuration** (`config/tags/real_estate_tags.yaml`)
- ✅ **32 pre-built tags** for real estate (villas, BHK configurations, luxury features, etc.)
- ✅ **Easy customization** - Add/modify tags without touching code
- ✅ **Semantic patterns** - Keywords + regex patterns for precise matching
- ✅ **Price range integration** - Automatic INR price-based tagging
- ✅ **Category organization** - Property types, amenities, luxury indicators, etc.

**Example Tag Definition:**
```yaml
luxury_indicators:
  concierge_service:
    keywords:
      - concierge
      - butler service
      - personal assistant
    patterns:
      - "concierge.*service"
      - "butler.*service"
    category: luxury_indicators
    weight: 1.0
```

### **System Settings** (`config/settings.yaml`)
- ⚙️ **ML model settings** - Configure sentence transformers model
- ⚙️ **Processing parameters** - Workers, chunk sizes, confidence thresholds
- ⚙️ **File I/O paths** - Input/output file configurations
- ⚙️ **Web enrichment settings** - Rate limits, timeouts, search templates

### **Benefits of Configuration-Driven System:**
- 🎯 **No code changes needed** to add new tags or adjust settings
- 🚀 **Hot configuration reloads** - Update tags without restarting
- 📈 **Easy scaling** - Adjust processing parameters for different workloads
- 🔄 **Version control friendly** - Track tag evolution over time
- 🏗️ **Multiple catalog types** - Easy to create new tag sets for different domains

### **Quick Customization Examples:**

**Adding New Tags:**
```yaml
# Add to config/tags/real_estate_tags.yaml
new_category:
  eco_friendly:
    keywords:
      - solar panels
      - rainwater harvesting
      - energy efficient
    category: sustainability
    weight: 0.8
```

**Creating Custom Catalog Types:**
```bash
# 1. Copy the example template
cp config/tags/custom_example.yaml config/tags/automotive_tags.yaml

# 2. Edit for your domain (cars, electronics, etc.)
# 3. Run with your new catalog type:
python3 src/catalog_tagger.py automotive
python3 src/hybrid_tagger.py automotive
```

### **✅ Industry-Agnostic Design:**
- 🚫 **No hardcoded industry assumptions** - The system requires explicit catalog type specification
- 🛡️ **Fail-safe design** - Won't run with missing or invalid configurations
- ⚙️ **Pure configuration-driven** - Zero code changes needed for new industries
- 🎯 **Flexible tag structure** - Support any domain: automotive, electronics, retail, etc.

**Adjusting Processing Settings:**
```yaml
# Edit config/settings.yaml
processing:
  max_workers: 8        # More parallel processing
  chunk_size: 100       # Larger batches
  confidence_thresholds:
    default: 0.20       # Higher confidence requirement
```
