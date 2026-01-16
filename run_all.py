

import sys
import os
from datetime import datetime



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
    print(f"❌ Data preprocessing failed: {e}")
    sys.exit(1)
print("="*80)
