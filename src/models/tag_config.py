from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum


class TagType(str, Enum):
    """Types of tags that can be generated"""
    CATEGORY = "category"
    FEATURE = "feature"
    CONDITION = "condition"
    PRICE_RANGE = "price_range"
    BRAND = "brand"
    STYLE = "style"
    LOCATION = "location"
    SIZE = "size"
    COLOR = "color"
    MATERIAL = "material"
    SENTIMENT = "sentiment"
    POPULARITY = "popularity"


class TagDefinition(BaseModel):
    """Definition of a predefined tag"""
    name: str
    tag_type: TagType
    keywords: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)  # regex patterns
    synonyms: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)  # 1=highest, 10=lowest
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    exclusive_with: List[str] = Field(default_factory=list)  # mutually exclusive tags


class CatalogTypeConfig(BaseModel):
    """Configuration for a specific catalog type"""
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str] = Field(default_factory=list)
    predefined_tags: List[TagDefinition]
    web_search_templates: List[str] = Field(default_factory=list)
    scraping_rules: Dict[str, Any] = Field(default_factory=dict)
    ml_model_config: Dict[str, Any] = Field(default_factory=dict)


class RealEstateTagConfig:
    """Predefined tags for real estate listings"""

    @staticmethod
    def get_tags() -> List[TagDefinition]:
        return [
            # Property Type Tags
            TagDefinition(
                name="single_family_home",
                tag_type=TagType.CATEGORY,
                keywords=["single family", "detached", "house", "home"],
                synonyms=["sfh", "detached home", "standalone house"],
                description="Single family detached home",
                priority=1
            ),
            TagDefinition(
                name="condominium",
                tag_type=TagType.CATEGORY,
                keywords=["condo", "condominium", "unit"],
                synonyms=["condo unit", "residential unit"],
                description="Condominium unit",
                priority=1
            ),
            TagDefinition(
                name="townhouse",
                tag_type=TagType.CATEGORY,
                keywords=["townhouse", "townhome", "row house"],
                synonyms=["th", "town home"],
                description="Townhouse or row house",
                priority=1
            ),

            # Property Features
            TagDefinition(
                name="luxury",
                tag_type=TagType.FEATURE,
                keywords=["luxury", "premium", "high-end", "upscale", "executive"],
                patterns=[r"\$[\d,]+k", r"luxury", r"executive"],
                description="High-end luxury property",
                priority=2,
                min_confidence=0.7
            ),
            TagDefinition(
                name="waterfront",
                tag_type=TagType.FEATURE,
                keywords=["waterfront", "lakefront", "oceanfront", "beachfront", "riverfront"],
                synonyms=["water view", "lake view", "ocean view"],
                description="Property with water access or views",
                priority=2
            ),
            TagDefinition(
                name="pool",
                tag_type=TagType.FEATURE,
                keywords=["pool", "swimming pool", "in-ground pool", "above ground pool"],
                synonyms=["spa", "hot tub", "jacuzzi"],
                description="Property with swimming pool",
                priority=3
            ),
            TagDefinition(
                name="garage",
                tag_type=TagType.FEATURE,
                keywords=["garage", "car garage", "attached garage", "detached garage"],
                patterns=[r"(\d+)\s*car\s*garage", r"garage"],
                description="Property with garage",
                priority=3
            ),

            # Size Categories
            TagDefinition(
                name="compact",
                tag_type=TagType.SIZE,
                keywords=["compact", "cozy", "efficient"],
                patterns=[r"[<]?\s*1000\s*sq", r"studio", r"1\s*bed"],
                description="Smaller property under 1000 sq ft",
                priority=4
            ),
            TagDefinition(
                name="spacious",
                tag_type=TagType.SIZE,
                keywords=["spacious", "large", "expansive", "roomy"],
                patterns=[r"[>]?\s*3000\s*sq", r"4\+?\s*bed"],
                description="Large property over 3000 sq ft",
                priority=4
            ),

            # Condition Tags
            TagDefinition(
                name="newly_built",
                tag_type=TagType.CONDITION,
                keywords=["new construction", "newly built", "brand new", "never lived in"],
                patterns=[r"202[0-9]\s*built", r"new\s*construction"],
                description="Recently constructed property",
                priority=2
            ),
            TagDefinition(
                name="renovated",
                tag_type=TagType.CONDITION,
                keywords=["renovated", "updated", "remodeled", "upgraded"],
                synonyms=["refreshed", "modernized"],
                description="Recently renovated property",
                priority=3
            ),
            TagDefinition(
                name="fixer_upper",
                tag_type=TagType.CONDITION,
                keywords=["fixer upper", "needs work", "handyman special", "as-is"],
                synonyms=["tlc", "renovation opportunity"],
                description="Property needing significant work",
                priority=5
            ),

            # Price Range Tags
            TagDefinition(
                name="budget_friendly",
                tag_type=TagType.PRICE_RANGE,
                patterns=[r"\$[1-9][0-9]{4,5}(?![0-9])", r"under.*300k"],
                description="Affordable properties under $300k",
                priority=4
            ),
            TagDefinition(
                name="mid_range",
                tag_type=TagType.PRICE_RANGE,
                patterns=[r"\$[3-7][0-9]{5}", r"300k.*700k"],
                description="Mid-range properties $300k-$700k",
                priority=4
            ),
            TagDefinition(
                name="high_end",
                tag_type=TagType.PRICE_RANGE,
                patterns=[r"\$[7-9][0-9]{5}", r"700k.*million"],
                description="High-end properties $700k-$1M",
                priority=4
            ),

            # Location Features
            TagDefinition(
                name="downtown",
                tag_type=TagType.LOCATION,
                keywords=["downtown", "city center", "urban core", "cbd"],
                synonyms=["central", "metropolitan"],
                description="Located in downtown area",
                priority=3
            ),
            TagDefinition(
                name="suburban",
                tag_type=TagType.LOCATION,
                keywords=["suburban", "suburbs", "residential area", "family neighborhood"],
                description="Located in suburban area",
                priority=3
            ),
            TagDefinition(
                name="school_district",
                tag_type=TagType.FEATURE,
                keywords=["school district", "top schools", "excellent schools", "rated schools"],
                patterns=[r"school.*district", r"top.*school"],
                description="In desirable school district",
                priority=2
            )
        ]


class EcommerceTagConfig:
    """Predefined tags for e-commerce products"""

    @staticmethod
    def get_tags() -> List[TagDefinition]:
        return [
            # Product Categories
            TagDefinition(
                name="electronics",
                tag_type=TagType.CATEGORY,
                keywords=["electronics", "electronic", "tech", "technology", "gadget"],
                description="Electronic devices and gadgets",
                priority=1
            ),
            TagDefinition(
                name="clothing",
                tag_type=TagType.CATEGORY,
                keywords=["clothing", "apparel", "fashion", "wear", "garment"],
                description="Clothing and fashion items",
                priority=1
            ),
            TagDefinition(
                name="home_garden",
                tag_type=TagType.CATEGORY,
                keywords=["home", "garden", "household", "furniture", "decor"],
                description="Home and garden products",
                priority=1
            ),

            # Brand Indicators
            TagDefinition(
                name="premium_brand",
                tag_type=TagType.BRAND,
                keywords=["apple", "samsung", "sony", "nike", "gucci", "louis vuitton"],
                description="Premium/luxury brand",
                priority=2,
                min_confidence=0.8
            ),

            # Condition Tags
            TagDefinition(
                name="new",
                tag_type=TagType.CONDITION,
                keywords=["new", "brand new", "unopened", "sealed"],
                description="New product condition",
                priority=1
            ),
            TagDefinition(
                name="refurbished",
                tag_type=TagType.CONDITION,
                keywords=["refurbished", "renewed", "restored", "open box"],
                description="Refurbished product condition",
                priority=2
            ),

            # Price Categories
            TagDefinition(
                name="budget",
                tag_type=TagType.PRICE_RANGE,
                patterns=[r"\$[1-9][0-9]?(?!\d)", r"under.*100"],
                description="Budget-friendly under $100",
                priority=3
            ),
            TagDefinition(
                name="premium_price",
                tag_type=TagType.PRICE_RANGE,
                patterns=[r"\$[5-9][0-9]{2,}", r"500\+"],
                description="Premium priced $500+",
                priority=3
            )
        ]
