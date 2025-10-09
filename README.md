# Catalog Auto-Tagger System

An intelligent system that automatically generates tags for catalog products by combining catalog information with web-sourced data.

## Features

- **Multi-source Analysis**: Combines catalog data with web-scraped information
- **Multi-label Classification**: Products can have multiple relevant tags
- **Extensible Tag System**: Easy to add new predefined tags
- **Web Enhancement**: Automatically searches for additional product information online
- **Configurable**: Supports different catalog types (real estate, e-commerce, etc.)

## Architecture

```
catalog-auto-tagger/
├── src/
│   ├── core/
│   │   ├── catalog_processor.py    # Process catalog data
│   │   ├── web_scraper.py         # Web information retrieval
│   │   ├── tag_classifier.py     # ML-based tag classification
│   │   └── tag_generator.py      # Main orchestrator
│   ├── models/
│   │   ├── product.py            # Product data models
│   │   └── tag_config.py         # Tag configuration
│   ├── utils/
│   │   ├── search_utils.py       # Web search utilities
│   │   ├── text_processing.py    # Text processing helpers
│   │   └── config.py             # Configuration management
│   └── integrations/
│       ├── google_search.py      # Google Search API
│       └── scrapers/             # Site-specific scrapers
├── config/
│   ├── tags/                     # Predefined tag configurations
│   └── settings.yaml            # System settings
├── examples/
│   └── real_estate_example.py   # Home listings example
└── requirements.txt
```

## Usage

```python
from src.core.tag_generator import TagGenerator

# Initialize the system
tagger = TagGenerator(catalog_type="real_estate")

# Process a catalog
results = tagger.process_catalog("path/to/catalog.csv")

# Get enhanced tags for a single product
tags = tagger.generate_tags(product_data)
```

## Supported Catalog Types

- Real Estate Listings
- E-commerce Products
- Automotive Listings
- Generic Product Catalogs

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configure predefined tags and system settings in the `config/` directory.
