"""
Master Execution Script for Retail Analytics Project
Author: Senior Data Scientist
Date: January 2026

This script runs the complete analysis pipeline.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

print("="*80)
print(" "*20 + "RETAIL ANALYTICS PROJECT")
print(" "*15 + "Consumer Purchase Behavior & Market Trends")
print("="*80)
print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Step 1: Data Preprocessing
print("\n" + "🔄 STEP 1: DATA PREPROCESSING")
print("="*80)
try:
    from data_preprocessing import main as preprocess_main
    df_clean = preprocess_main()
    print("✅ Data preprocessing completed successfully!")
except Exception as e:
    print(f"❌ Error in data preprocessing: {e}")
    sys.exit(1)

# Step 2: Exploratory Data Analysis
print("\n" + "📊 STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)
try:
    from eda_analysis import main as eda_main
    eda_main()
    print("✅ EDA completed successfully!")
except Exception as e:
    print(f"❌ Error in EDA: {e}")
    sys.exit(1)

# Step 3: Customer Segmentation
print("\n" + "👥 STEP 3: CUSTOMER SEGMENTATION (RFM + K-MEANS)")
print("="*80)
try:
    from customer_segmentation import main as segmentation_main
    rfm_results = segmentation_main()
    print("✅ Customer segmentation completed successfully!")
except Exception as e:
    print(f"❌ Error in customer segmentation: {e}")
    sys.exit(1)

# Step 4: Market Basket Analysis
print("\n" + "🛒 STEP 4: MARKET BASKET ANALYSIS")
print("="*80)
try:
    from market_basket_analysis import main as mba_main
    rules = mba_main()
    print("✅ Market basket analysis completed successfully!")
except Exception as e:
    print(f"❌ Error in market basket analysis: {e}")
    # Continue even if this fails
    print("⚠️  Continuing with remaining analyses...")

# Step 5: Customer Lifetime Value
print("\n" + "💎 STEP 5: CUSTOMER LIFETIME VALUE ANALYSIS")
print("="*80)
try:
    from clv_analysis import main as clv_main
    clv_results = clv_main()
    print("✅ CLV analysis completed successfully!")
except Exception as e:
    print(f"❌ Error in CLV analysis: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print(" "*25 + "EXECUTION SUMMARY")
print("="*80)
print("\n✅ All analyses completed successfully!")
print("\n📁 Output Files Generated:")
print("   - data/online_retail_cleaned.csv")
print("   - outputs/*.png (visualizations)")
print("   - outputs/customer_segments.csv")
print("   - outputs/customer_clv.csv")
print("   - outputs/association_rules.csv")
print("   - outputs/frequent_itemsets.csv")

print("\n🚀 Next Step: Launch the Streamlit Dashboard")
print("   Command: streamlit run app.py")

print(f"\n⏱️  Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
