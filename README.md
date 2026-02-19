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
├── src/                        # Production code
│   ├── catalog_tagger.py       # Fast catalog-only tagger
│   ├── hybrid_tagger.py        # Web-enhanced hybrid tagger
│   ├── export_cache.py         # Cache export utility
│   └── utils/                   # Utility modules
│       ├── config_loader.py    # Configuration management
│       └── __init__.py
├── config/                     # ⚙️ Flexible configuration system
│   ├── settings.yaml           # System settings & parameters
│   ├── catalog_types/          # Industry-specific field mappings
│   │   └── real_estate.yaml    # Real estate field configuration
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

The architecture supports any domain. Currently, only `real_estate` ships with a pre-built configuration (tags, field mappings, and search templates). To add a new domain:

1. Create `config/tags/<your_domain>_tags.yaml` with tag definitions
2. Create `config/catalog_types/<your_domain>.yaml` with field mappings
3. Run with `python3 src/catalog_tagger.py <your_domain>`

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
- **No code changes needed** to add new tags or adjust settings
- **Easy scaling** - Adjust processing parameters for different workloads
- **Version control friendly** - Track tag evolution over time
- **Multiple catalog types** - Easy to create new tag sets for different domains

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

### **Catalog Type Configuration** (`config/catalog_types/*.yaml`)
Industry-specific field mappings and domain configuration:

```yaml
# config/catalog_types/real_estate.yaml

# Field mappings - tells the tagger which columns to read
id_field: "home_listing_id"
title_field: "name"
description_fields: ["description", "desc", "details", "summary"]
price_field: "Price"
category_field: "Property_Type"
location_field: "Address.city"
cache_key_field: "name"

# Which tag categories count as "basic" vs "advanced" in output
basic_tag_categories: ["bedrooms", "amenity", "price_range", "status"]

# Output column mappings
output_fields:
  city_tag: "Address.city"
  state_tag: "Address.state"

# Web search templates (hybrid tagger)
search_templates:
  - '"{title}" {location} specifications features details'
  - '"{title}" real estate property India'

# Web content relevance indicators
web_relevance_indicators:
  - features
  - amenities
  - luxury
  - bedroom

# City-to-state mapping (when catalog lacks a state field)
city_state_mapping:
  mumbai: maharashtra
  bangalore: karnataka

# Confidence threshold overrides per category
confidence_overrides:
  property_type: 0.5
```

**Field Resolution Strategy:**
1. If `catalog_config` specifies a field name (e.g., `title_field: "name"`), that column is used directly
2. If not specified, the system tries each alias from `settings.yaml` `field_mappings` (e.g., `["name", "title", "product_name", ...]`)
3. First matching column wins; empty string if nothing matches

This means you can omit fields from `catalog_types/*.yaml` and the system will auto-detect columns using the generic aliases in `settings.yaml`.

**Benefits:**
- **Config-driven field resolution** - System adapts to any catalog structure
- **Industry-specific outputs** - Each domain can define custom output columns
- **Location intelligence** - Config-driven city/state extraction from catalog and web data
- **Clear field mapping** - Documents which input fields are used

### **Advanced Tag Matching Modes:**
Tags now support multiple matching strategies:

1. **`semantic`** (default): ML-based semantic similarity matching
2. **`text_only`**: Exact keyword/pattern matching only (no semantic)
3. **`price_only`**: Price-range based matching (no text analysis)

**Example:**
```yaml
# Bedroom tags use text_only for precise matching
3bhk:
  keywords: [3bhk, 3 bhk, 3 bedroom]
  matching_mode: text_only  # No semantic confusion
  category: bedrooms

# Price tags use price_only for exact bracketing
premium:
  price_field: "Price"          # Which field to check
  price_range:
    min: 10000000               # 1 Cr
    max: 40000000               # 4 Cr
  matching_mode: price_only     # Only price-based
  category: price_range
```

### **Property Type Verification:**
System now verifies property_type tags against catalog data:
- ✅ **80% penalty** for property_type mismatches (e.g., "plot" tag on "Apartment")
- ✅ **Confidence boost** for correct property_type matches
- ✅ **Works even when catalog lacks property_type field**

### **Industry-Agnostic Design:**
- **No hardcoded field names** - All field mappings come from config files
- **Fail-safe design** - Won't run with missing or invalid configurations
- **Configuration-driven** - Zero code changes needed for new industries
- **Flexible tag structure** - Support any domain: automotive, electronics, retail, etc.
- **Settings consumed at runtime** - ML model, confidence thresholds, workers, file paths, web enrichment params all read from `settings.yaml`

**Adjusting Processing Settings:**
```yaml
# Edit config/settings.yaml
processing:
  max_workers: 8        # More parallel processing
  chunk_size: 100       # Larger batches
  confidence_thresholds:
    default: 0.20       # Higher confidence requirement
```
