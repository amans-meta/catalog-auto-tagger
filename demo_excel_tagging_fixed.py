#!/usr/bin/env python3
"""
Fixed Excel Auto-Tagging Demo
Addresses the confidence threshold and price parsing issues
"""

import os
import sys
import traceback
import re
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def load_tag_definitions():
    """Load Indian real estate tag definitions with FIXED thresholds"""
    try:
        import yaml

        tag_file = Path("config/tags/indian_real_estate.yaml")
        if not tag_file.exists():
            print("⚠️  Tag definitions file not found")
            return {}

        with open(tag_file, 'r') as f:
            tags_config = yaml.safe_load(f)

        # IMPORTANT FIX: Lower all confidence thresholds
        for category, tag_list in tags_config.items():
            if isinstance(tag_list, list):
                for tag_info in tag_list:
                    if isinstance(tag_info, dict) and 'min_confidence' in tag_info:
                        # Lower thresholds significantly
                        original = tag_info['min_confidence']
                        tag_info['min_confidence'] = max(0.2, original - 0.3)  # Reduce by 0.3, minimum 0.2

        print(f"✅ Loaded tag definitions with LOWERED thresholds")
        return tags_config

    except Exception as e:
        print(f"❌ Error loading tag definitions: {e}")
        return {}


def create_product_from_row(row, row_idx):
    """Create a product object from Excel row"""
    import pandas as pd

    # Extract basic fields with multiple possible column names
    def get_field(row, possible_names, default=""):
        for name in possible_names:
            for col in row.index:
                if col.lower().strip() == name.lower().strip():
                    value = row[col]
                    return str(value) if pd.notna(value) else default
        return default

    # Extract fields
    product = {
        'id': get_field(row, ['id', 'property_id', 'listing_id', 'unique_id'], f"PROP_{row_idx}"),
        'title': get_field(row, ['title', 'name', 'property_name', 'heading', 'property title']),
        'description': get_field(row, ['description', 'desc', 'details', 'summary', 'about']),
        'price_text': get_field(row, ['price', 'cost', 'amount', 'value', 'sale_price']),
        'category': get_field(row, ['category', 'type', 'property_type', 'building_type']),
        'city': get_field(row, ['city', 'location', 'area', 'locality', 'town']),
        'bedrooms': get_field(row, ['bedrooms', 'beds', 'num_beds', 'bhk', 'bedroom', 'bed']),
        'bathrooms': get_field(row, ['bathrooms', 'baths', 'num_baths', 'bathroom', 'bath']),
        'size': get_field(row, ['size', 'area', 'sqft', 'square_feet', 'built_up_area', 'carpet_area']),
        'address': get_field(row, ['address', 'full_address', 'location', 'property_address']),
        'amenities': get_field(row, ['amenities', 'features', 'facilities', 'highlights']),
        'agent_name': get_field(row, ['agent_name', 'agent', 'contact_person', 'broker']),
        'agent_phone': get_field(row, ['agent_phone', 'phone', 'contact_phone', 'mobile']),
    }

    # Parse price with FIXED parsing
    product['price'] = parse_price_fixed(product['price_text'])

    # Create full text for analysis
    product['full_text'] = f"{product['title']} {product['description']} {product['amenities']} {product['city']} {product['address']}"

    return product


def parse_price_fixed(price_str):
    """FIXED price parsing that handles INR suffix and various formats"""
    if not price_str or str(price_str).lower() in ['nan', 'none', '']:
        return None

    try:
        price_str = str(price_str).lower().strip()

        # Remove currency symbols and suffixes
        clean_price = (
            price_str
            .replace('₹', '')
            .replace('rs.', '')
            .replace('rs', '')
            .replace('inr', '')  # This was the issue - remove INR
            .replace(',', '')
            .replace(' ', '')
        )

        # Handle Indian notations
        if 'lakh' in clean_price or 'lac' in clean_price:
            number_part = clean_price.replace('lakh', '').replace('lac', '').strip()
            return float(number_part) * 100000
        elif 'crore' in clean_price or 'cr' in clean_price:
            number_part = clean_price.replace('crore', '').replace('cr', '').strip()
            return float(number_part) * 10000000
        elif 'k' in clean_price:
            number_part = clean_price.replace('k', '').strip()
            return float(number_part) * 1000
        else:
            return float(clean_price)
    except Exception as e:
        print(f"Price parsing error for '{price_str}': {e}")
        return None


def generate_tags_for_product_enhanced(product, tags_config):
    """ENHANCED tag generation with multiple scoring methods"""
    generated_tags = []
    full_text = product['full_text'].lower()
    price = product['price']

    print(f"🔍 Analyzing: {product['title'][:50]}...")
    print(f"💰 Parsed price: ₹{price:,.0f}" if price else "💰 No price")

    # Process each tag category
    for category, tag_list in tags_config.items():
        if not isinstance(tag_list, list):
            continue

        for tag_info in tag_list:
            if not isinstance(tag_info, dict) or 'name' not in tag_info:
                continue

            tag_name = tag_info['name']
            keywords = tag_info.get('keywords', [])
            patterns = tag_info.get('patterns', [])
            min_confidence = tag_info.get('min_confidence', 0.2)  # Lower default

            confidence = 0.0
            reasons = []

            # ENHANCED: Check keyword matches with partial matching
            keyword_matches = 0
            matched_keywords = []
            for keyword in keywords:
                # Exact match
                if keyword.lower() in full_text:
                    keyword_matches += 1
                    matched_keywords.append(keyword)
                # Partial match for compound words
                elif any(word in full_text for word in keyword.lower().split()):
                    keyword_matches += 0.5  # Partial credit
                    matched_keywords.append(f"{keyword}(partial)")

            if keyword_matches > 0:
                confidence += keyword_matches * 0.4  # Increased weight
                reasons.append(f"keywords: {matched_keywords}")

            # ENHANCED: Check regex patterns
            pattern_matches = 0
            for pattern in patterns:
                try:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        pattern_matches += 1
                except:
                    continue

            if pattern_matches > 0:
                confidence += pattern_matches * 0.5
                reasons.append(f"patterns: {pattern_matches}")

            # ENHANCED: Price-based tagging with better logic
            if price:
                if 'budget' in tag_name.lower() and price < 2000000:
                    confidence += 0.9
                    reasons.append(f"budget: ₹{price:,.0f} < ₹20L")
                elif 'mid_range' in tag_name.lower() and 2000000 <= price <= 10000000:
                    confidence += 0.9
                    reasons.append(f"mid-range: ₹{price:,.0f}")
                elif 'high_end' in tag_name.lower() and 10000000 < price <= 50000000:
                    confidence += 0.9
                    reasons.append(f"high-end: ₹{price:,.0f}")
                elif 'luxury' in tag_name.lower() and price > 50000000:
                    confidence += 0.9
                    reasons.append(f"luxury: ₹{price:,.0f} > ₹5Cr")

            # ENHANCED: BHK detection from multiple fields
            if 'bhk' in tag_name.lower():
                bhk_sources = [product['bedrooms'], product['title'], product['description']]
                bhk_text = ' '.join(str(s) for s in bhk_sources if s).lower()

                # Look for BHK patterns
                bhk_patterns = [tag_name, tag_name.replace('bhk', ' bhk'), tag_name.replace('bhk', 'bed')]
                for pattern in bhk_patterns:
                    if pattern.lower() in bhk_text:
                        confidence += 0.8
                        reasons.append(f"BHK match in: {pattern}")
                        break

            # ENHANCED: Amenity detection with broader matching
            amenity_keywords = ['swimming', 'gym', 'parking', 'security', 'lift', 'garden', 'club']
            if any(amenity in tag_name.lower() for amenity in amenity_keywords):
                amenities_text = f"{product['amenities']} {product['description']} {product['title']}".lower()

                amenity_score = 0
                for keyword in keywords:
                    if keyword.lower() in amenities_text:
                        amenity_score += 0.6
                    # Check for partial matches
                    elif any(word in amenities_text for word in keyword.lower().split() if len(word) > 3):
                        amenity_score += 0.3

                if amenity_score > 0:
                    confidence += amenity_score
                    reasons.append(f"amenity: {amenity_score:.1f}")

            # ENHANCED: Location-based features
            location_keywords = ['metro', 'airport', 'it', 'tech', 'city', 'center']
            if any(loc in tag_name.lower() for loc in location_keywords):
                location_text = f"{product['city']} {product['address']} {product['description']}".lower()

                for keyword in keywords:
                    if keyword.lower() in location_text:
                        confidence += 0.5
                        reasons.append(f"location: {keyword}")

            # Log the scoring for debugging
            if confidence > 0:
                print(f"   🏷️  {tag_name}: {confidence:.2f} ({'✅' if confidence >= min_confidence else '❌'}) - {reasons}")

            # Apply confidence threshold
            if confidence >= min_confidence:
                generated_tags.append({
                    'name': tag_name,
                    'confidence': min(confidence, 1.0),
                    'category': category,
                    'reasoning': '; '.join(reasons)
                })

    # Sort by confidence
    generated_tags.sort(key=lambda x: x['confidence'], reverse=True)

    return generated_tags


def process_excel_with_enhanced_tagging(excel_file):
    """Process Excel file with enhanced tagging"""
    print("🏷️  ENHANCED EXCEL AUTO-TAGGING")
    print("=" * 60)

    try:
        import pandas as pd

        # Load Excel file
        print(f"📁 Loading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)

        print(f"   • Found {len(df)} rows")
        print(f"   • Found {len(df.columns)} columns")

        # Load tag definitions
        tags_config = load_tag_definitions()
        if not tags_config:
            print("❌ No tag definitions available")
            return []

        # Process each row
        all_results = []

        print(f"\n🔄 Processing with ENHANCED tagging...")

        for idx, row in df.iterrows():
            print(f"\n--- Product {idx + 1}/{len(df)} ---")

            # Create product from row
            product = create_product_from_row(row, idx + 1)

            # Generate tags with enhanced logic
            tags = generate_tags_for_product_enhanced(product, tags_config)

            # Store result
            result = {
                'row_index': idx + 1,
                'product': product,
                'tags': tags,
                'tag_count': len(tags)
            }

            all_results.append(result)

            print(f"✅ Generated {len(tags)} tags")
            if tags:
                for tag in tags[:3]:  # Show top 3
                    print(f"   🏷️  {tag['name']} ({tag['confidence']:.2f})")

        return all_results

    except Exception as e:
        print(f"❌ Error processing Excel file: {e}")
        traceback.print_exc()
        return []


def display_enhanced_results(results):
    """Display enhanced tagging results"""
    print(f"\n🎯 ENHANCED TAGGING RESULTS")
    print("=" * 60)

    if not results:
        print("❌ No results to display")
        return

    # Statistics
    total_products = len(results)
    total_tags = sum(r['tag_count'] for r in results)
    products_with_tags = len([r for r in results if r['tag_count'] > 0])

    print(f"📊 Statistics:")
    print(f"   • Products processed: {total_products}")
    print(f"   • Products with tags: {products_with_tags} ({products_with_tags/total_products*100:.1f}%)")
    print(f"   • Total tags generated: {total_tags}")
    print(f"   • Average tags per product: {total_tags/total_products:.1f}")

    # Show top products
    tagged_products = [r for r in results if r['tag_count'] > 0]
    tagged_products.sort(key=lambda x: x['tag_count'], reverse=True)

    print(f"\n🏆 SUCCESSFULLY TAGGED PRODUCTS:")
    for i, result in enumerate(tagged_products[:10], 1):
        product = result['product']
        tags = result['tags']

        print(f"\n{i}. Product #{result['row_index']}: {product['title'][:80]}...")
        print(f"   💰 Price: ₹{product['price']:,.0f}" if product['price'] else "   💰 Price: Not specified")
        print(f"   🏷️  Tags ({len(tags)}):")

        for tag in tags[:6]:  # Show top 6 tags
            confidence_icon = "🟢" if tag['confidence'] > 0.7 else "🟡" if tag['confidence'] > 0.5 else "🔴"
            print(f"      {confidence_icon} {tag['name']} ({tag['confidence']:.2f}) - {tag['reasoning'][:60]}...")


def main():
    """Main function"""
    print("🔧 FIXED EXCEL AUTO-TAGGING DEMO")
    print("🎯 Fixed Confidence Thresholds & Price Parsing")
    print("=" * 70)

    excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"

    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        return False

    try:
        # Process with enhanced tagging
        results = process_excel_with_enhanced_tagging(excel_file)

        if not results:
            print("❌ No results generated")
            return False

        # Display results
        display_enhanced_results(results)

        # Export if there are tags
        tagged_results = [r for r in results if r['tag_count'] > 0]
        if tagged_results:
            try:
                import pandas as pd

                export_data = []
                for result in tagged_results:
                    product = result['product']
                    tags = result['tags']

                    row = {
                        'Row_Index': result['row_index'],
                        'Title': product['title'],
                        'Price': product['price'],
                        'City': product['city'],
                        'Tag_Count': len(tags),
                        'Top_Tags': ', '.join([tag['name'] for tag in tags[:5]]),
                        'All_Tags_Detail': ' | '.join([f"{tag['name']}({tag['confidence']:.2f})" for tag in tags])
                    }
                    export_data.append(row)

                df_export = pd.DataFrame(export_data)
                output_file = "enhanced_tagged_results.csv"
                df_export.to_csv(output_file, index=False)

                print(f"\n📤 Exported {len(tagged_results)} tagged products to: {output_file}")

            except Exception as e:
                print(f"⚠️  Export failed: {e}")

        print(f"\n🎉 TAGGING COMPLETED!")
        print(f"✅ Fixed confidence thresholds")
        print(f"✅ Fixed price parsing (handles INR suffix)")
        print(f"✅ Enhanced matching logic")
        print(f"✅ Generated tags for your real estate data")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
