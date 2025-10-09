#!/usr/bin/env python3
"""
Debug Tagging - Find out why no tags are generated
"""

import os
import sys
import traceback
import re
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def debug_tag_generation():
    """Debug the tagging process step by step"""
    print("🔍 DEBUG TAGGING PROCESS")
    print("=" * 60)

    try:
        import pandas as pd
        import yaml

        # Load Excel file
        excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"
        df = pd.read_excel(excel_file)

        print(f"✅ Loaded Excel: {len(df)} rows")

        # Load tag definitions
        tag_file = Path("config/tags/indian_real_estate.yaml")
        with open(tag_file, 'r') as f:
            tags_config = yaml.safe_load(f)

        print(f"✅ Loaded tags config")
        print(f"📋 Tag categories: {list(tags_config.keys())}")

        # Count total tags
        total_tags = 0
        for category, tag_list in tags_config.items():
            if isinstance(tag_list, list):
                total_tags += len(tag_list)
                print(f"   • {category}: {len(tag_list)} tags")

        print(f"📊 Total tag definitions: {total_tags}")

        # Test with first product
        print(f"\n🧪 TESTING WITH FIRST PRODUCT")
        print("-" * 40)

        first_row = df.iloc[0]
        print(f"📝 First row columns and values:")
        for col in df.columns:
            value = first_row[col]
            if pd.notna(value):
                print(f"   • {col}: {str(value)[:60]}...")

        # Extract product fields manually
        def get_field(row, possible_names):
            for name in possible_names:
                for col in row.index:
                    if col.lower().strip() == name.lower().strip():
                        value = row[col]
                        return str(value) if pd.notna(value) else ""
            return ""

        product_title = get_field(first_row, ['title', 'name', 'property_name', 'heading'])
        product_desc = get_field(first_row, ['description', 'desc', 'details', 'summary'])
        product_price_text = get_field(first_row, ['price', 'cost', 'amount', 'value'])
        product_category = get_field(first_row, ['category', 'type', 'property_type'])

        print(f"\n📦 Extracted product fields:")
        print(f"   • Title: '{product_title}'")
        print(f"   • Description: '{product_desc[:100]}...'")
        print(f"   • Price text: '{product_price_text}'")
        print(f"   • Category: '{product_category}'")

        # Create full text
        full_text = f"{product_title} {product_desc} {product_category}".lower()
        print(f"\n🔤 Full text for matching: '{full_text[:200]}...'")

        # Test tag matching
        print(f"\n🏷️  TESTING TAG MATCHING")
        print("-" * 40)

        matches_found = 0

        for category, tag_list in tags_config.items():
            if not isinstance(tag_list, list):
                continue

            print(f"\n📂 Category: {category}")

            for tag_info in tag_list[:3]:  # Test first 3 tags in each category
                if not isinstance(tag_info, dict) or 'name' not in tag_info:
                    continue

                tag_name = tag_info['name']
                keywords = tag_info.get('keywords', [])
                min_confidence = tag_info.get('min_confidence', 0.3)

                print(f"\n   🏷️  Testing tag: '{tag_name}'")
                print(f"       Keywords: {keywords}")
                print(f"       Min confidence: {min_confidence}")

                # Check keyword matches
                keyword_matches = []
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        keyword_matches.append(keyword)

                if keyword_matches:
                    confidence = len(keyword_matches) * 0.3
                    print(f"       ✅ MATCH! Keywords found: {keyword_matches}")
                    print(f"       💪 Confidence: {confidence}")
                    if confidence >= min_confidence:
                        print(f"       🎯 WOULD GENERATE TAG!")
                        matches_found += 1
                    else:
                        print(f"       ❌ Below threshold ({min_confidence})")
                else:
                    print(f"       ❌ No keyword matches")

        print(f"\n🎯 SUMMARY")
        print(f"Total potential matches found: {matches_found}")

        # If no matches, let's check some specific things
        if matches_found == 0:
            print(f"\n🔍 DEBUGGING WHY NO MATCHES")
            print("-" * 40)

            # Check if apartment/flat keywords exist
            apartment_keywords = ['apartment', 'flat', 'residential unit', 'apt']
            print(f"Testing apartment keywords: {apartment_keywords}")
            for kw in apartment_keywords:
                if kw in full_text:
                    print(f"   ✅ Found '{kw}' in text")
                else:
                    print(f"   ❌ '{kw}' not found")

            # Check if price-based tags would work
            print(f"\nTesting price-based matching:")
            print(f"Price text: '{product_price_text}'")

            # Parse price
            try:
                price_str = str(product_price_text).lower().strip()
                clean_price = price_str.replace('₹', '').replace('rs.', '').replace('rs', '').replace(',', '').replace(' ', '')

                if 'lakh' in clean_price or 'lac' in clean_price:
                    number_part = clean_price.replace('lakh', '').replace('lac', '').strip()
                    parsed_price = float(number_part) * 100000
                elif 'crore' in clean_price or 'cr' in clean_price:
                    number_part = clean_price.replace('crore', '').replace('cr', '').strip()
                    parsed_price = float(number_part) * 10000000
                else:
                    parsed_price = float(clean_price)

                print(f"Parsed price: ₹{parsed_price:,.0f}")

                # Test price ranges
                if parsed_price < 2000000:
                    print(f"   → Would match 'budget_friendly' (< ₹20L)")
                elif 2000000 <= parsed_price <= 10000000:
                    print(f"   → Would match 'mid_range' (₹20L - ₹1Cr)")
                elif 10000000 < parsed_price <= 50000000:
                    print(f"   → Would match 'high_end' (₹1Cr - ₹5Cr)")
                elif parsed_price > 50000000:
                    print(f"   → Would match 'luxury' (> ₹5Cr)")

            except Exception as e:
                print(f"   ❌ Price parsing failed: {e}")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()


def test_simple_tag_matching():
    """Test with hardcoded simple tags"""
    print(f"\n🧪 TESTING WITH SIMPLE HARDCODED TAGS")
    print("=" * 60)

    try:
        import pandas as pd

        # Load first row
        excel_file = "/Users/amans/Desktop/catalog-auto-tagger/feeds/Book3.xlsx"
        df = pd.read_excel(excel_file)
        first_row = df.iloc[0]

        # Create simple test text
        test_text = ""
        for col in df.columns:
            value = first_row[col]
            if pd.notna(value):
                test_text += f" {str(value)}"

        test_text = test_text.lower()
        print(f"🔤 Test text: '{test_text[:200]}...'")

        # Simple hardcoded tags to test
        simple_tags = [
            {'name': 'apartment', 'keywords': ['apartment', 'flat', 'unit']},
            {'name': 'villa', 'keywords': ['villa', 'house', 'bungalow']},
            {'name': 'luxury', 'keywords': ['luxury', 'premium', 'elite']},
            {'name': 'furnished', 'keywords': ['furnished', 'ready']},
            {'name': 'parking', 'keywords': ['parking', 'garage', 'car']},
            {'name': 'pool', 'keywords': ['pool', 'swimming', 'swim']},
            {'name': 'gym', 'keywords': ['gym', 'fitness', 'exercise']},
            {'name': 'security', 'keywords': ['security', 'guard', 'safe']},
            {'name': '2bhk', 'keywords': ['2bhk', '2 bhk', '2bed', 'two bedroom']},
            {'name': '3bhk', 'keywords': ['3bhk', '3 bhk', '3bed', 'three bedroom']},
        ]

        matches = []

        for tag in simple_tags:
            tag_name = tag['name']
            keywords = tag['keywords']

            found_keywords = []
            for keyword in keywords:
                if keyword in test_text:
                    found_keywords.append(keyword)

            if found_keywords:
                matches.append({
                    'tag': tag_name,
                    'keywords_found': found_keywords,
                    'confidence': len(found_keywords) * 0.5
                })
                print(f"✅ MATCH: {tag_name} - found: {found_keywords}")

        print(f"\n🎯 Simple test results: {len(matches)} matches")

        if matches:
            print("SUCCESS! The matching logic works with simple tags")
            print("The issue is likely with the tag definition file format")
        else:
            print("No matches even with simple tags - need to check the text content")

        return matches

    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        traceback.print_exc()
        return []


def main():
    """Main debug function"""
    print("🔧 TAGGING DEBUG TOOL")
    print("🎯 Find out why no tags are generated")
    print("=" * 70)

    # Step 1: Debug full process
    debug_tag_generation()

    # Step 2: Test simple matching
    matches = test_simple_tag_matching()

    # Final recommendations
    print(f"\n💡 DEBUG RECOMMENDATIONS")
    print("=" * 60)

    if matches:
        print("✅ Text matching works - issue is in tag definition format")
        print("🔧 Solutions:")
        print("   1. Check YAML file format")
        print("   2. Verify tag structure")
        print("   3. Lower confidence thresholds")
    else:
        print("❌ No matches found - issue is deeper")
        print("🔧 Solutions:")
        print("   1. Check Excel column names")
        print("   2. Verify text extraction")
        print("   3. Add more basic keywords")
        print("   4. Check for encoding issues")

    print(f"\n🚀 Next steps:")
    print("   • Review the debug output above")
    print("   • I'll fix the tagging logic based on findings")
    print("   • Run the corrected version")


if __name__ == "__main__":
    main()
