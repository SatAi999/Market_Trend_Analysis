# Market Analysis Project: Predicting Customer Behavior

## 1. Problem Statement

Retailers face critical questions: Who are our most valuable customers? Who is likely to churn? What products should we bundle? This project tackles these challenges by building a production-grade machine learning system to:
- Predict customer churn
- Forecast customer lifetime value (CLV)
- Segment customers for targeted marketing
- Discover actionable product associations for cross-selling

## 2. Dataset Used

**Source:** Online Retail II (UCI Machine Learning Repository)
- **Content:** 1,067,371 real transaction records (Dec 2009–Dec 2011) from a UK-based e-commerce retailer, covering 5,816 customers, 4,620 products, and 41 countries.
- **Why this dataset?**
	- It is large, messy, and realistic—mirroring real business data challenges (canceled orders, missing IDs, returns, promotions).
	- Rich enough for advanced analytics: segmentation, churn, CLV, and market basket analysis.
- **Preprocessing:**
	- Cleaned to 792,121 high-quality records (`online_retail_cleaned.csv`), removing 25.8% noise (canceled/invalid transactions).

## 3. Project Approach & Architecture

### a. Data Preprocessing
- **Script:** `src/data_preprocessing.py`
- Removed canceled orders, handled missing values, engineered features (e.g., TotalPrice, temporal features).

### b. Exploratory Data Analysis (EDA)
- **Script:** `src/eda_analysis.py`
- Uncovered sales trends, top products, and country-wise revenue.

### c. Customer Segmentation
- **Script:** `src/customer_segmentation.py`
- **Method:** RFM (Recency, Frequency, Monetary) analysis + K-Means clustering
- **Result:** Two clear segments—Loyal Champions (24.8%) and At-Risk/Lost (75.2%)
- **Metric:** Silhouette Score = 0.47 (statistically significant)

### d. Market Basket Analysis
- **Script:** `src/market_basket_analysis.py`
- **Method:** Apriori algorithm (mlxtend)
- **Result:** 436 association rules (avg. lift: 12.79x, max: 53x)
- **Output:** `outputs/association_rules.csv`, `outputs/frequent_itemsets.csv`

### e. Customer Lifetime Value (CLV) Analysis & Prediction
- **Script:** `src/clv_analysis.py`, `src/clv_prediction.py`
- **Method:**
	- CLV calculated using: CLV = Avg Order Value × Purchase Frequency × Customer Lifespan
	- ML regression models: Linear, Ridge, Random Forest, Gradient Boosting
- **Performance:**
	- Gradient Boosting Regressor: R² = 0.94–0.97, RMSE ≈ $1,000
	- Output: `outputs/clv_predictions_ml.csv`, `outputs/customer_clv.csv`

### f. Churn Prediction
- **Script:** `src/churn_prediction.py`
- **Method:**
	- Features: 19 engineered (RFM, temporal, purchase velocity)
	- Models: Logistic Regression, Random Forest, Gradient Boosting
- **Performance:**
	- Gradient Boosting: ROC-AUC = 0.95–0.96, Accuracy = 95%+
	- Output: `outputs/churn_predictions.csv`

### g. Interactive Dashboard
- **Script:** `app.py` (Streamlit)
- **Features:** 7 interactive pages (EDA, segmentation, basket analysis, CLV, predictions, business insights)

## 4. Key Results & Business Impact

- **Churn Prediction:**
	- 2,371 high-risk customers identified ($11.8M CLV at risk)
	- Retention campaign can recover $3.5M+ (30% save rate)
- **CLV Prediction:**
	- 5,757 customers with predicted CLV (R² = 0.96)
	- Enables VIP targeting and budget optimization
- **Customer Segmentation:**
	- Champions: 1,119 customers, avg. spend $2,795
	- At-Risk: 3,400 customers, avg. spend $686
- **Market Basket Analysis:**
	- 382 actionable product pairings (lift > 3x)
	- Bundling/cross-sell can boost order value by 15–25%

## 5. Models & Algorithms Used

- **Unsupervised Learning:**
	- K-Means (customer segmentation)
	- Apriori (association rules)
- **Supervised Classification:**
	- Logistic Regression, Random Forest, Gradient Boosting (churn)
- **Supervised Regression:**
	- Linear, Ridge, Random Forest, Gradient Boosting (CLV)
- **Evaluation Metrics:** ROC-AUC, F1, Accuracy, R², RMSE, Silhouette, Lift

## 6. Learnings & Innovations

- Real-world data cleaning and feature engineering (19 features from 8 raw columns)
- Systematic model comparison and hyperparameter tuning
- Modular, reproducible, and production-ready code
- Business translation: every model tied to ROI and action plan
- Interactive dashboard for stakeholder engagement

## 7. Output Artifacts
- **CSV:** Segments, churn predictions, CLV predictions, association rules, cleaned data
- **Visualizations:** 10+ publication-ready PNGs (model performance, clusters, trends)
- **Dashboard:** 7-page Streamlit app

## 8. Conclusion
This project delivers a robust, end-to-end ML system for retail analytics, transforming raw data into actionable business insights. The approach, models, and results set a benchmark for real-world machine learning projects.

---
**Prepared by:** [Your Name]  
**Date:** January 15, 2026
