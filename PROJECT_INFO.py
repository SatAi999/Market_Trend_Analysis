"""
====================================================================================================
    CONSUMER PURCHASE BEHAVIOR & MARKET TREND ANALYSIS (RETAIL)
    End-to-End Python Data Science Project
    
    Author: Senior Data Scientist & Full-Stack Data Analytics Engineer
    Date: January 2026
    Tech Stack: Python, Pandas, Scikit-learn, mlxtend, Matplotlib, Seaborn, Streamlit
====================================================================================================

PROJECT OVERVIEW
================
This is a comprehensive retail analytics project demonstrating advanced data science techniques
including data preprocessing, exploratory analysis, customer segmentation, market basket analysis,
and customer lifetime value prediction - all presented through a professional Streamlit dashboard.

DATASET
=======
- Name: Online Retail II Dataset
- Location: data/online_retail.csv
- Size: ~90 MB
- Records: 500K+ transactions
- Attributes: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

PROJECT STRUCTURE
=================
Market_Analysis/
├── data/                           # Dataset folder
│   ├── online_retail.csv          # Raw dataset (90.5 MB)
│   └── online_retail_cleaned.csv  # Cleaned dataset (generated)
│
├── src/                            # Source code modules
│   ├── data_preprocessing.py      # Data cleaning & preprocessing
│   ├── eda_analysis.py            # Exploratory data analysis
│   ├── customer_segmentation.py   # RFM analysis & K-Means clustering
│   ├── market_basket_analysis.py  # Apriori algorithm & association rules
│   └── clv_analysis.py            # Customer lifetime value calculation
│
├── outputs/                        # Generated outputs
│   ├── *.png                      # Visualizations
│   ├── customer_segments.csv      # Segmentation results
│   ├── customer_clv.csv           # CLV rankings
│   ├── frequent_itemsets.csv      # Frequent itemsets
│   └── association_rules.csv      # Association rules
│
├── app.py                          # Streamlit dashboard (main application)
├── run_all.py                      # Master execution script
├── validate_setup.py               # Setup validation script
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── QUICKSTART.md                   # Quick start guide
└── .gitignore                      # Git ignore file

EXECUTION WORKFLOW
==================

Step 1: Validate Setup
-----------------------
python validate_setup.py
→ Checks all files, directories, and dependencies

Step 2: Run Complete Analysis
------------------------------
python run_all.py
→ Executes all 5 analysis modules in sequence:
  1. Data Preprocessing (2-5 min)
  2. EDA Analysis (1-2 min)
  3. Customer Segmentation (2-3 min)
  4. Market Basket Analysis (3-5 min)
  5. CLV Analysis (1-2 min)

Step 3: Launch Dashboard
-------------------------
streamlit run app.py
→ Opens interactive dashboard at http://localhost:8501

ANALYSIS MODULES DETAILS
=========================

1. DATA PREPROCESSING (data_preprocessing.py)
----------------------------------------------
Objectives:
- Remove canceled invoices (InvoiceNo starting with 'C')
- Remove rows with missing CustomerID
- Remove negative or zero Quantity/UnitPrice
- Convert InvoiceDate to datetime
- Create TotalPrice feature (Quantity × UnitPrice)
- Engineer temporal features (Year, Month, Quarter, DayOfWeek, Hour)

Output:
- data/online_retail_cleaned.csv (cleaned dataset)

Key Metrics:
- Initial rows: ~540,000
- Final rows: ~400,000
- Retention rate: ~75%

2. EXPLORATORY DATA ANALYSIS (eda_analysis.py)
-----------------------------------------------
Analyses:
- Total revenue and key business metrics
- Top 20 products by revenue and quantity
- Top 15 countries by revenue
- Monthly revenue trends
- Day-of-week and hourly patterns
- Comprehensive EDA dashboard

Outputs:
- outputs/top_products.png
- outputs/country_analysis.png
- outputs/temporal_trends.png
- outputs/eda_dashboard.png

Key Insights:
- Revenue trends over time
- Peak sales periods
- Geographic distribution
- Product performance

3. CUSTOMER SEGMENTATION (customer_segmentation.py)
----------------------------------------------------
Methodology:
- RFM Analysis (Recency, Frequency, Monetary)
- Feature scaling using StandardScaler
- Optimal cluster determination (Elbow + Silhouette)
- K-Means clustering
- Segment labeling and profiling

Segments Identified:
- Champions: Recent buyers, high frequency, high spending
- Loyal Customers: High frequency, high spending
- Potential Loyalists: Recent buyers, moderate spending
- At Risk: Haven't purchased recently but high value
- Lost Customers: Long time since last purchase
- Needs Attention: Others requiring engagement

Outputs:
- outputs/customer_segments.csv
- outputs/optimal_clusters.png
- outputs/customer_segments.png

Key Metrics:
- Optimal clusters: 4-6 (determined by analysis)
- Silhouette score: 0.3-0.5 (typical)
- Unique customer segments with distinct behaviors

4. MARKET BASKET ANALYSIS (market_basket_analysis.py)
------------------------------------------------------
Methodology:
- Transaction encoding to basket format
- Frequent itemset mining using Apriori algorithm
- Association rule generation
- Product bundling recommendations

Metrics:
- Support: Frequency of itemset occurrence
- Confidence: Likelihood of rule being true
- Lift: How much more likely items are bought together

Outputs:
- outputs/frequent_itemsets.csv
- outputs/association_rules.csv
- outputs/market_basket_analysis.png

Applications:
- Cross-selling strategies
- Product bundling
- Store layout optimization
- Recommendation systems

5. CUSTOMER LIFETIME VALUE (clv_analysis.py)
---------------------------------------------
Methodology:
- Average Order Value (AOV) calculation
- Purchase Frequency estimation (orders per year)
- Customer lifespan modeling (default: 3 years)
- CLV = AOV × Purchase Frequency × Lifespan

Customer Categories:
- Low Value (0-25th percentile)
- Medium Value (25-50th percentile)
- High Value (50-75th percentile)
- Very High Value (75-90th percentile)
- VIP (90-100th percentile)

Outputs:
- outputs/customer_clv.csv
- outputs/clv_analysis.png

Key Insights:
- Top 20% customers contribute 60-80% of CLV
- High-value customers requiring retention focus
- At-risk high-value customers needing re-engagement

STREAMLIT DASHBOARD (app.py)
=============================

Pages:
------
1. Dataset Overview
   - Key metrics dashboard
   - Data quality indicators
   - Sample data preview

2. Sales Trends & EDA
   - Monthly revenue trends
   - Top products and countries
   - Temporal analysis (day, hour)

3. Customer Segmentation
   - Segment distribution
   - RFM profile analysis
   - Segment visualizations

4. Market Basket Analysis
   - Association rules explorer
   - Product bundle recommendations
   - Support-Confidence-Lift metrics

5. Customer Lifetime Value
   - CLV distribution
   - Top customers ranking
   - CLV category breakdown

6. Business Insights
   - Key findings summary
   - Strategic recommendations
   - Executive summary

Features:
---------
- Wide layout with sidebar navigation
- Date range filtering
- Country filtering
- Product highlighting
- Interactive visualizations
- Metric cards and KPIs
- Professional color scheme
- Responsive design

BUSINESS INSIGHTS & RECOMMENDATIONS
====================================

Customer Retention:
- Implement VIP programs for Champions and High-Value customers
- Re-engagement campaigns for At-Risk customers
- Personalized offers based on purchase history
- Loyalty rewards for frequent buyers

Revenue Optimization:
- Focus on high-margin product categories
- Implement dynamic pricing strategies
- Expand in underserved geographic markets
- Upsell and cross-sell to existing customers

Product Strategy:
- Create product bundles based on association rules
- Stock management for high-demand items
- Seasonal inventory optimization
- New product recommendations

Marketing:
- Targeted campaigns by customer segment
- Seasonal promotions during peak periods
- Geographic expansion opportunities
- Personalized product recommendations

TECHNICAL IMPLEMENTATION
=========================

Code Quality:
- Modular, reusable functions
- Comprehensive docstrings
- Error handling and validation
- Type hints where appropriate
- Production-ready code structure

Best Practices:
- Object-oriented design
- Separation of concerns
- DRY (Don't Repeat Yourself)
- Clear naming conventions
- Extensive comments

Visualizations:
- Matplotlib and Seaborn only
- Professional color schemes
- Clear titles and labels
- Consistent styling
- Business-friendly graphs

Performance:
- Efficient pandas operations
- Vectorized computations
- Cached Streamlit data
- Optimized algorithms

DEPENDENCIES
============
pandas==2.1.4          # Data manipulation
numpy==1.26.2          # Numerical computing
matplotlib==3.8.2      # Visualization
seaborn==0.13.0        # Statistical visualization
scikit-learn==1.3.2    # Machine learning (K-Means, StandardScaler)
mlxtend==0.23.0        # Association rules (Apriori)
streamlit==1.29.0      # Web application framework

TROUBLESHOOTING
===============

Issue: Import errors
Solution: Ensure all packages installed: pip install -r requirements.txt

Issue: Dataset not found
Solution: Check data/online_retail.csv exists

Issue: Out of memory during market basket analysis
Solution: Reduce min_support parameter or filter to fewer products

Issue: Dashboard not loading data
Solution: Run run_all.py first to generate all output files

Issue: Visualizations not displaying
Solution: Check outputs/ folder has write permissions

PERFORMANCE NOTES
=================
- Dataset size: ~90 MB (500K+ transactions)
- Preprocessing time: 2-5 minutes
- Total analysis time: 10-20 minutes
- Dashboard load time: 2-5 seconds
- Memory usage: ~500 MB - 1 GB

SKILLS DEMONSTRATED
===================
✅ Data cleaning and preprocessing
✅ Exploratory Data Analysis (EDA)
✅ Feature engineering
✅ Statistical analysis
✅ Unsupervised learning (K-Means clustering)
✅ Association rule mining (Apriori algorithm)
✅ Business metrics calculation (RFM, CLV)
✅ Data visualization (Matplotlib, Seaborn)
✅ Web application development (Streamlit)
✅ Code organization and documentation
✅ Production-ready code quality

FUTURE ENHANCEMENTS
===================
- [ ] Time series forecasting for revenue prediction
- [ ] Recommendation system based on collaborative filtering
- [ ] A/B testing framework for marketing campaigns
- [ ] Customer churn prediction model
- [ ] Real-time dashboard updates
- [ ] Export functionality for reports
- [ ] Database integration
- [ ] API endpoints for external integration

CONTACT & SUPPORT
=================
This is a portfolio project demonstrating end-to-end data science capabilities.
For questions, feedback, or collaboration opportunities, please reach out.

LICENSE
=======
MIT License - See README.md for details

====================================================================================================
                               END OF PROJECT DOCUMENTATION
====================================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
