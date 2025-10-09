"""
Data models for catalog auto-tagging system
"""

from .product import (
    CatalogProcessingResult,
    EnhancedProduct,
    GeneratedTag,
    ProductInfo,
    WebEnhancedInfo,
)
from .tag_config import (
    CatalogTypeConfig,
    EcommerceTagConfig,
    RealEstateTagConfig,
    TagDefinition,
    TagType,
)

__all__ = [
    "ProductInfo",
    "WebEnhancedInfo",
    "GeneratedTag",
    "EnhancedProduct",
    "CatalogProcessingResult",
    "TagDefinition",
    "TagType",
    "CatalogTypeConfig",
    "RealEstateTagConfig",
    "EcommerceTagConfig",
]
