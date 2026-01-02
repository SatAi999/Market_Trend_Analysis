# 🚀 Quick Start Guide

## Consumer Purchase Behavior & Market Trend Analysis

### Setup (First Time Only)

1. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Complete Analysis

**Option 1: Run All Analyses (Recommended)**
```bash
python run_all.py
```

This will execute all analysis modules in sequence:
- ✅ Data Preprocessing
- ✅ Exploratory Data Analysis
- ✅ Customer Segmentation
- ✅ Market Basket Analysis
- ✅ Customer Lifetime Value

**Option 2: Run Individual Modules**
```bash
python src/data_preprocessing.py
python src/eda_analysis.py
python src/customer_segmentation.py
python src/market_basket_analysis.py
python src/clv_analysis.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at: `http://localhost:8501`

---

## 📊 What You'll Get

### Analysis Outputs:
- **Cleaned Dataset**: `data/online_retail_cleaned.csv`
- **Customer Segments**: `outputs/customer_segments.csv`
- **CLV Rankings**: `outputs/customer_clv.csv`
- **Product Associations**: `outputs/association_rules.csv`
- **Visualizations**: `outputs/*.png`

### Dashboard Features:
- 📈 Sales trends and patterns
- 👥 Customer segmentation insights
- 🛒 Product bundling recommendations
- 💎 Customer lifetime value rankings
- 💡 Strategic business recommendations

---

## ⏱️ Estimated Execution Time

- Data Preprocessing: ~2-5 minutes
- EDA Analysis: ~1-2 minutes
- Customer Segmentation: ~2-3 minutes
- Market Basket Analysis: ~3-5 minutes (may vary by dataset size)
- CLV Analysis: ~1-2 minutes

**Total: ~10-20 minutes** (depending on dataset size)

---

## 🎯 Key Insights You'll Discover

1. **Customer Segments**: Champions, Loyal, At-Risk, Lost
2. **High-Value Customers**: Top 20% contributing 60-80% of revenue
3. **Product Associations**: Which products are frequently bought together
4. **Seasonal Trends**: Peak sales periods and patterns
5. **CLV Rankings**: Customer prioritization for retention

---

## 📝 Notes

- Dataset must be in: `data/online_retail.csv`
- All outputs saved to: `outputs/` folder
- Dashboard uses cached data for performance

---

## ⚠️ Troubleshooting

**Issue**: Module not found error
**Solution**: Ensure you're in the project root directory

**Issue**: Out of memory
**Solution**: Reduce `min_support` in market basket analysis

**Issue**: Dashboard not loading data
**Solution**: Run `run_all.py` first to generate all output files

---

## 📞 Support

For issues or questions, check the README.md for detailed documentation.

**Happy Analyzing! 📊**
