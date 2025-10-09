from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ProductInfo(BaseModel):
    """Core product information from catalog"""
    id: str
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "USD"
    category: Optional[str] = None
    brand: Optional[str] = None
    url: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class WebEnhancedInfo(BaseModel):
    """Additional information gathered from web sources"""
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    scraped_content: Dict[str, str] = Field(default_factory=dict)
    social_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    competitor_info: List[Dict[str, Any]] = Field(default_factory=list)


class GeneratedTag(BaseModel):
    """A generated tag with confidence score"""
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # 'catalog', 'web', 'ml_model'
    reasoning: Optional[str] = None


class EnhancedProduct(BaseModel):
    """Complete product with catalog info, web data, and generated tags"""
    product_info: ProductInfo
    web_info: WebEnhancedInfo
    generated_tags: List[GeneratedTag] = Field(default_factory=list)
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    processing_status: str = "pending"  # pending, processing, complete, error
    error_message: Optional[str] = None


class CatalogProcessingResult(BaseModel):
    """Results from processing an entire catalog"""
    total_products: int
    processed_products: int
    failed_products: int
    processing_time_seconds: float
    enhanced_products: List[EnhancedProduct]
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
