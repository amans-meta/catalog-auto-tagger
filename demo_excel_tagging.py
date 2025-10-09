#!/usr/bin/env python3
"""
Excel Auto-Tagging Demo
Processes your Excel file and generates actual tags for each product
"""

import os
import sys
import traceback
import re
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def load_tag_definitions():
    """Load Indian real estate tag definitions"""
    try:
        import yaml

        tag_file = Path("config/tags/indian_real_estate.yaml")
        if not tag_file.exists():
            print("⚠️  Tag definitions file not found")
            return {}

        with open(tag_file, 'r') as f:
            tags_config = yaml.safe_load(f)

        print(f"✅ Loaded tag definitions from {tag_file}")
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

    # Parse price
    product['price'] = parse_price(product['price_text'])

    # Create full text for analysis
    product['full_text'] = f"{product['title']} {product['description']} {product['amenities']} {product['city']}"

    return product


def parse_price(price_str):
    """Parse Indian price formats"""
    if not price_str or str(price_str).lower() in ['nan', 'none', '']:
        return None

    try:
        price_str = str(price_str).lower().strip()

        # Remove currency symbols
        clean_price = (
            price_str
            .replace('₹', '')
            .replace('rs.', '')
            .replace('rs', '')
            .replace('inr', '')
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
    except:
        return None


def generate_tags_for_product(product, tags_config):
    """Generate tags for a single product"""
    generated_tags = []
    full_text = product['full_text'].lower()
    price = product['price']

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
            min_confidence = tag_info.get('min_confidence', 0.3)

            confidence = 0.0
            reasons = []

            # Check keyword matches
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in full_text:
                    keyword_matches += 1

            if keyword_matches > 0:
                confidence += keyword_matches * 0.3
                reasons.append(f"{keyword_matches} keyword matches")

            # Check regex patterns
            pattern_matches = 0
            for pattern in patterns:
                try:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        pattern_matches += 1
                except:
                    continue

            if pattern_matches > 0:
                confidence += pattern_matches * 0.4
                reasons.append(f"{pattern_matches} pattern matches")

            # Price-based tagging
            if price and 'price' in tag_name.lower():
                if 'budget' in tag_name and price < 2000000:  # < 20 lakh
                    confidence += 0.8
                    reasons.append(f"price ₹{price:,.0f} < ₹20L")
                elif 'mid_range' in tag_name and 2000000 <= price <= 10000000:
                    confidence += 0.8
                    reasons.append(f"price ₹{price:,.0f} is mid-range")
                elif 'high_end' in tag_name and 10000000 < price <= 50000000:
                    confidence += 0.8
                    reasons.append(f"price ₹{price:,.0f} is high-end")
                elif 'luxury' in tag_name and price > 50000000:
                    confidence += 0.8
                    reasons.append(f"price ₹{price:,.0f} > ₹5Cr")

            # BHK-based tagging
            if 'bhk' in tag_name.lower():
                bhk_text = f"{product['bedrooms']} {product['title']} {product['description']}".lower()
                if tag_name in bhk_text or any(k in bhk_text for k in keywords):
                    confidence += 0.7
                    reasons.append("BHK configuration match")

            # Location-based tagging
            if any(loc_word in tag_name.lower() for loc_word in ['metro', 'airport', 'it_hub', 'city_center']):
                if any(keyword.lower() in full_text for keyword in keywords):
                    confidence += 0.6
                    reasons.append("location feature detected")

            # Amenity-based tagging
            amenities_text = f"{product['amenities']} {product['description']}".lower()
            if any(amenity in tag_name.lower() for amenity in ['swimming', 'gym', 'parking', 'security', 'lift']):
                amenity_matches = sum(1 for keyword in keywords if keyword.lower() in amenities_text)
                if amenity_matches > 0:
                    confidence += amenity_matches * 0.5
                    reasons.append(f"amenity matches: {amenity_matches}")

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


def process_excel_with_tagging(excel_file):
    """Process Excel file and generate tags for each product"""
    print("🏷️  EXCEL AUTO-TAGGING PROCESSING")
    print("=" * 60)

    try:
        import pandas as pd

        # Load Excel file
        print(f"📁 Loading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)

        print(f"   • Found {len(df)} rows")
        print(f"   • Found {len(df.columns)} columns: {list(df.columns)}")

        # Load tag definitions
        tags_config = load_tag_definitions()
        if not tags_config:
            print("❌ No tag definitions available")
            return []

        total_tags = sum(len(v) for v in tags_config.values() if isinstance(v, list))
        print(f"   • Loaded {total_tags} tag definitions across {len(tags_config)} categories")

        # Process each row
        all_results = []

        print(f"\n🔄 Processing products and generating tags...")

        for idx, row in df.iterrows():
            # Create product from row
            product = create_product_from_row(row, idx + 1)

            # Generate tags
            tags = generate_tags_for_product(product, tags_config)

            # Store result
            result = {
                'row_index': idx + 1,
                'product': product,
                'tags': tags,
                'tag_count': len(tags)
            }

            all_results.append(result)

            # Progress indicator
            if (idx + 1) % 5 == 0 or idx == len(df) - 1:
                print(f"   Processed {idx + 1}/{len(df)} products...")

        print(f"✅ Completed processing {len(all_results)} products")

        return all_results

    except Exception as e:
        print(f"❌ Error processing Excel file: {e}")
        traceback.print_exc()
        return []


def display_tagging_results(results):
    """Display the tagging results in a readable format"""
    print(f"\n📊 AUTO-TAGGING RESULTS")
    print("=" * 60)

    if not results:
        print("❌ No results to display")
        return

    # Overall statistics
    total_products = len(results)
    total_tags = sum(r['tag_count'] for r in results)
    avg_tags = total_tags / total_products if total_products > 0 else 0

    print(f"📈 Overall Statistics:")
    print(f"   • Products processed: {total_products}")
    print(f"   • Total tags generated: {total_tags}")
    print(f"   • Average tags per product: {avg_tags:.1f}")

    # Products with most tags
    top_products = sorted(results, key=lambda x: x['tag_count'], reverse=True)

    print(f"\n🏆 Top Tagged Products:")
    for i, result in enumerate(top_products[:3], 1):
        product = result['product']
        tags = result['tags']
        print(f"\n   {i}. Product #{result['row_index']}: {product['title'][:60]}...")
        print(f"      💰 Price: ₹{product['price']:,.0f}" if product['price'] else "      💰 Price: Not specified")
        print(f"      🏷️  Tags ({len(tags)}):")

        for tag in tags[:8]:  # Show top 8 tags
            confidence_icon = "🟢" if tag['confidence'] > 0.7 else "🟡" if tag['confidence'] > 0.5 else "🔴"
            print(f"         {confidence_icon} {tag['name']} ({tag['confidence']:.2f}) - {tag['reasoning']}")

        if len(tags) > 8:
            print(f"         ... and {len(tags) - 8} more tags")

    # Tag category distribution
    print(f"\n📋 Tag Category Distribution:")
    category_counts = {}
    for result in results:
        for tag in result['tags']:
            category = tag.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {category}: {count} tags")

    # Most common tags
    print(f"\n🎯 Most Common Tags:")
    tag_counts = {}
    for result in results:
        for tag in result['tags']:
            tag_name = tag['name']
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

    common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (tag_name, count) in enumerate(common_tags, 1):
        percentage = (count / total_products) * 100
        print(f"   {i:2d}. {tag_name}: {count} products ({percentage:.1f}%)")


def export_results_to_csv(results, output_file):
    """Export tagging results to CSV"""
    print(f"\n📤 EXPORTING RESULTS")
    print("=" * 60)

    try:
        import pandas as pd

        # Prepare data for export
        export_data = []

        for result in results:
            product = result['product']
            tags = result['tags']

            # Create base row
            base_row = {
                'Row_Index': result['row_index'],
                'Product_ID': product['id'],
                'Title': product['title'],
                'Price_Text': product['price_text'],
                'Price_Parsed': product['price'],
                'Category': product['category'],
                'City': product['city'],
                'Bedrooms': product['bedrooms'],
                'Bathrooms': product['bathrooms'],
                'Size': product['size'],
                'Agent_Name': product['agent_name'],
                'Tag_Count': len(tags),
            }

            # Add top 5 tags as separate columns
            for i in range(5):
                if i < len(tags):
                    tag = tags[i]
                    base_row[f'Tag_{i+1}_Name'] = tag['name']
                    base_row[f'Tag_{i+1}_Confidence'] = tag['confidence']
                    base_row[f'Tag_{i+1}_Category'] = tag['category']
                else:
                    base_row[f'Tag_{i+1}_Name'] = ''
                    base_row[f'Tag_{i+1}_Confidence'] = ''
                    base_row[f'Tag_{i+1}_Category'] = ''

            # All tags as comma-separated
            all_tag_names = [tag['name'] for tag in tags]
            base_row['All_Tags'] = ', '.join(all_tag_names)

            export_data.append(base_row)

        # Create DataFrame and save
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_file, index=False)

        print(f"✅ Results exported to: {output_file}")
        print(f"   • Rows: {len(export_df)}")
        print(f"   • Columns: {len(export_df.columns)}")

        # Show sample columns
        print(f"   • Key columns: Product_ID, Title, Price_Parsed, Tag_Count, All_Tags")

    except Exception as e:
        print(f"❌ Error exporting results: {e}")


def main():
    """Main function"""
    print("🏷️  EXCEL AUTO-TAGGING DEMO")
    print("🎯 Generate Tags for Your Real Estate Data")
    print("=" * 70)

    excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"

    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Excel file not found: {excel_file}")
        return False

    try:
        # Process Excel with tagging
        results = process_excel_with_tagging(excel_file)

        if not results:
            print("❌ No results generated")
            return False

        # Display results
        display_tagging_results(results)

        # Export results
        output_file = "tagged_results.csv"
        export_results_to_csv(results, output_file)

        # Summary
        print(f"\n🎉 AUTO-TAGGING COMPLETED!")
        print("=" * 50)
        print(f"✅ Processed your Excel file successfully")
        print(f"✅ Generated tags for {len(results)} products")
        print(f"✅ Results saved to {output_file}")

        print(f"\n📖 What you got:")
        print(f"   • Automatic property type detection")
        print(f"   • Price range categorization (budget/mid/luxury)")
        print(f"   • BHK configuration tagging")
        print(f"   • Amenity and feature detection")
        print(f"   • Location-based tags")
        print(f"   • Investment potential indicators")

        print(f"\n💡 Use the tags for:")
        print(f"   • Meta/Facebook ad targeting")
        print(f"   • Property categorization")
        print(f"   • Search filters and recommendations")
        print(f"   • Market analysis and insights")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
