"""
Project Validation Script
Checks if all components are properly set up
"""

import os
import sys
from pathlib import Path

print("="*80)
print(" "*25 + "PROJECT VALIDATION")
print("="*80)

errors = []
warnings = []

# Check Python version
print("\n✓ Checking Python version...")
if sys.version_info < (3, 9):
    errors.append("Python 3.9+ required")
else:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

# Check required files
print("\n✓ Checking required files...")
required_files = [
    'requirements.txt',
    'README.md',
    'QUICKSTART.md',
    'run_all.py',
    'app.py',
    'src/data_preprocessing.py',
    'src/eda_analysis.py',
    'src/customer_segmentation.py',
    'src/market_basket_analysis.py',
    'src/clv_analysis.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        errors.append(f"Missing file: {file}")
        print(f"  ❌ {file}")

# Check directories
print("\n✓ Checking directories...")
required_dirs = ['data', 'src', 'outputs']
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"  ✅ {directory}/")
    else:
        errors.append(f"Missing directory: {directory}")
        print(f"  ❌ {directory}/")

# Check dataset
print("\n✓ Checking dataset...")
if os.path.exists('data/online_retail.csv'):
    size = os.path.getsize('data/online_retail.csv') / (1024 * 1024)  # MB
    print(f"  ✅ Dataset found ({size:.1f} MB)")
else:
    errors.append("Dataset not found: data/online_retail.csv")
    print(f"  ❌ Dataset not found")

# Check dependencies
print("\n✓ Checking Python packages...")
required_packages = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'sklearn',
    'mlxtend',
    'streamlit'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"  ✅ {package}")
    except ImportError:
        warnings.append(f"Package not installed: {package}")
        print(f"  ⚠️  {package} (not installed)")

# Summary
print("\n" + "="*80)
print(" "*30 + "SUMMARY")
print("="*80)

if not errors and not warnings:
    print("\n🎉 All checks passed! Project is ready to run.")
    print("\n📝 Next steps:")
    print("   1. Run all analyses: python run_all.py")
    print("   2. Launch dashboard: streamlit run app.py")
elif errors:
    print(f"\n❌ Found {len(errors)} error(s):")
    for error in errors:
        print(f"   • {error}")
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   • {warning}")
    print("\n💡 Please fix errors before running the project.")
else:
    print(f"\n⚠️  Found {len(warnings)} warning(s):")
    for warning in warnings:
        print(f"   • {warning}")
    print("\n💡 Install missing packages: pip install -r requirements.txt")

print("="*80)
