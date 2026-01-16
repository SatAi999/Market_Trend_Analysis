

import os
import sys
import time
from datetime import datetime

# Import all modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import RetailDataPreprocessor
from src.eda_analysis import RetailEDA
from src.customer_segmentation import CustomerSegmentation
from src.market_basket_analysis import MarketBasketAnalysis
from src.clv_analysis import CLVAnalysis
from src.churn_prediction import ChurnPredictionModel
from src.clv_prediction import CLVPredictionModel

import pandas as pd
import warnings
warnings.filterwarnings('ignore')





def main():
    start_time = time.time()
    
    print("\n" + "="*70)
    print("  COMPREHENSIVE MACHINE LEARNING PROJECT")
    print("  Consumer Purchase Behavior & Market Trend Analysis")
    print("="*70)
    
    # ============================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================
    print_header("STEP 1: DATA PREPROCESSING")
    preprocessor = RetailDataPreprocessor('data/online_retail.csv')
    df = preprocessor.load_data()
    df_cleaned = preprocessor.clean_data()
    preprocessor.save_cleaned_data('data/online_retail_cleaned.csv')
    
    # ============================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ============================================================
    print_header("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    eda = RetailEDA(df_cleaned)
    eda.calculate_summary_statistics()
    eda.analyze_temporal_trends()
    
    # ============================================================
    # STEP 3: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
    # ============================================================
    print_header("STEP 3: CUSTOMER SEGMENTATION (K-Means Clustering)")
    segmentation = CustomerSegmentation(df_cleaned)
    rfm_df = segmentation.calculate_rfm()
    rfm_scaled = segmentation.scale_features()
    segmentation.find_optimal_clusters(max_k=6)
    segmentation.perform_clustering(n_clusters=2)
    segmentation.profile_segments()
    
    # ============================================================
    # STEP 4: ASSOCIATION RULE MINING - APRIORI ALGORITHM
    # ============================================================
    print_header("STEP 4: MARKET BASKET ANALYSIS (Apriori Algorithm)")
    basket_analysis = MarketBasketAnalysis(df_cleaned)
    basket_df = basket_analysis.create_basket_format()
    frequent_itemsets = basket_analysis.mine_frequent_itemsets(min_support=0.01)
    rules = basket_analysis.generate_association_rules(min_threshold=0.1)
    
    # ============================================================
    # STEP 5: BUSINESS ANALYTICS - CLV CALCULATION
    # ============================================================
    print_header("STEP 5: CUSTOMER LIFETIME VALUE ANALYSIS")
    clv_analysis = CLVAnalysis(df_cleaned)
    clv_df = clv_analysis.calculate_clv()
    clv_analysis.identify_at_risk_customers()
    clv_analysis.visualize_clv()
    
    # ============================================================
    # STEP 6: SUPERVISED LEARNING - CHURN PREDICTION (CLASSIFICATION)
    # ============================================================
    print_header("STEP 6: CHURN PREDICTION MODEL (Classification)")
    print("Training multiple classification algorithms...")
    churn_model = ChurnPredictionModel(df_cleaned)
    churn_model.engineer_features(churn_threshold_days=180)
    churn_model.prepare_data(test_size=0.2, random_state=42)
    churn_model.train_models()
    churn_results = churn_model.evaluate_models()
    churn_model.visualize_results()
    churn_model.save_predictions()
    
    # ============================================================
    # STEP 7: SUPERVISED LEARNING - CLV PREDICTION (REGRESSION)
    # ============================================================
    print_header("STEP 7: CLV PREDICTION MODEL (Regression)")
    print("Training multiple regression algorithms...")
    clv_pred_model = CLVPredictionModel(df_cleaned)
    clv_pred_model.engineer_features()
    clv_pred_model.prepare_data(test_size=0.2, random_state=42)
    clv_pred_model.train_models()
    clv_results = clv_pred_model.evaluate_models()
    clv_pred_model.visualize_results()
    clv_pred_model.save_predictions()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "="*70)
    print("  MACHINE LEARNING PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n{'='*70}")
    print("  EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"\n✓ Total Execution Time: {execution_time/60:.2f} minutes")
    print(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*70}")
    print("  ML MODELS TRAINED & EVALUATED")
    print(f"{'='*70}")
    print("\n1. UNSUPERVISED LEARNING:")
    print(f"   ✓ K-Means Clustering (Customer Segmentation)")
    print(f"   ✓ Apriori Algorithm (Association Rule Mining)")
    
    print("\n2. SUPERVISED LEARNING:")
    print(f"   ✓ Classification Models (Churn Prediction):")
    print(f"      - Logistic Regression")
    print(f"      - Random Forest Classifier")
    print(f"      - Gradient Boosting Classifier")
    
    print(f"\n   ✓ Regression Models (CLV Prediction):")
    print(f"      - Linear Regression")
    print(f"      - Ridge Regression")
    print(f"      - Random Forest Regressor")
    print(f"      - Gradient Boosting Regressor")
    
    print(f"\n{'='*70}")
    print("  OUTPUTS GENERATED")
    print(f"{'='*70}")
    print("\nDatasets:")
    print("  ✓ data/online_retail_cleaned.csv")
    print("  ✓ outputs/customer_segments.csv")
    print("  ✓ outputs/frequent_itemsets.csv")
    print("  ✓ outputs/association_rules.csv")
    print("  ✓ outputs/customer_clv.csv")
    print("  ✓ outputs/churn_predictions.csv")
    print("  ✓ outputs/clv_predictions_ml.csv")
    
    print("\nVisualizations:")
    print("  ✓ outputs/eda_analysis.png")
    print("  ✓ outputs/customer_segmentation.png")
    print("  ✓ outputs/market_basket_analysis.png")
    print("  ✓ outputs/clv_analysis.png")
    print("  ✓ outputs/churn_prediction_model.png")
    print("  ✓ outputs/clv_prediction_model.png")
    
    print(f"\n{'='*70}")
    print("  NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. View ML Model Performance:")
    print("   python ml_model_summary.py")
    
    print("\n2. Launch Interactive Dashboard:")
    print("   streamlit run app.py")
    
    print("\n3. Check Individual Model Results:")
    print("   - Churn Prediction: outputs/churn_prediction_model.png")
    print("   - CLV Prediction: outputs/clv_prediction_model.png")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
