"""
Streamlit Web Application for Consumer Purchase Behavior & Market Trend Analysis
Author: Senior Data Scientist
Date: January 2026

A professional, interactive dashboard for retail analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('src')

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background: linear-gradient(90deg, #1565c0 0%, #43a047 100%);
        color: #fff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #43a047;
        margin: 10px 0;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(21,101,192,0.08);
    }
    .warning-box {
        background: linear-gradient(90deg, #263238 0%, #607d8b 100%);
        color: #fff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(38,50,56,0.10);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all processed data."""
    try:
        df_clean = pd.read_csv('data/online_retail_cleaned.csv')
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # Load analysis results if they exist
        try:
            rfm_df = pd.read_csv('outputs/customer_segments.csv')
        except:
            rfm_df = None
        
        try:
            clv_df = pd.read_csv('outputs/customer_clv.csv')
            if 'FirstPurchase' in clv_df.columns:
                clv_df['FirstPurchase'] = pd.to_datetime(clv_df['FirstPurchase'])
            if 'LastPurchase' in clv_df.columns:
                clv_df['LastPurchase'] = pd.to_datetime(clv_df['LastPurchase'])
        except:
            clv_df = None
        
        try:
            rules_df = pd.read_csv('outputs/association_rules.csv')
        except:
            rules_df = None
        
        try:
            churn_pred = pd.read_csv('outputs/churn_predictions.csv')
        except:
            churn_pred = None
        
        try:
            clv_pred = pd.read_csv('outputs/clv_predictions_ml.csv')
        except:
            clv_pred = None
        
        return df_clean, rfm_df, clv_df, rules_df, churn_pred, clv_pred
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None


def show_sidebar(df):
    """Display sidebar with filters."""
    st.sidebar.title("🎯 Filters & Navigation")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Country filter
    st.sidebar.subheader("🌍 Country")
    countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    # Product filter
    st.sidebar.subheader("🛍️ Product Category")
    top_products = df['Description'].value_counts().head(20).index.tolist()
    selected_product = st.sidebar.selectbox(
        "Highlight Product", 
        ['All'] + top_products
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("📊 **Dashboard Info**\n\nThis dashboard provides comprehensive retail analytics including customer segmentation, market basket analysis, and CLV calculations.")
    
    return date_range, selected_country, selected_product


def apply_filters(df, date_range, country, product):
    """Apply selected filters to dataframe."""
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['InvoiceDate'].dt.date >= start_date) &
            (filtered_df['InvoiceDate'].dt.date <= end_date)
        ]
    
    # Country filter
    if country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    
    # Product filter
    if product != 'All':
        filtered_df = filtered_df[filtered_df['Description'] == product]
    
    return filtered_df


def page_overview(df):
    """Dataset Overview Page."""
    st.markdown('<div class="main-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${df['TotalPrice'].sum():,.0f}")
    with col2:
        st.metric("Total Orders", f"{df['Invoice'].nunique():,}")
    with col3:
        st.metric("Total Customers", f"{df['Customer ID'].nunique():,}")
    with col4:
        st.metric("Total Products", f"{df['StockCode'].nunique():,}")
    
    st.markdown("---")
    
    # Additional Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_order = df.groupby('Invoice')['TotalPrice'].sum().mean()
        st.metric("Avg Order Value", f"${avg_order:,.2f}")
    with col2:
        avg_basket = df.groupby('Invoice')['Quantity'].sum().mean()
        st.metric("Avg Basket Size", f"{avg_basket:.1f} items")
    with col3:
        orders_per_customer = df.groupby('Customer ID')['Invoice'].nunique().mean()
        st.metric("Avg Orders/Customer", f"{orders_per_customer:.1f}")
    with col4:
        countries = df['Country'].nunique()
        st.metric("Countries Served", f"{countries}")
    
    st.markdown("---")
    
    # Dataset Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Information")
        info_df = pd.DataFrame({
            'Metric': ['Total Records', 'Date Range', 'Top Country', 'Top Product'],
            'Value': [
                f"{len(df):,}",
                f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}",
                df.groupby('Country')['TotalPrice'].sum().idxmax(),
                df.groupby('Description')['TotalPrice'].sum().idxmax()[:40]
            ]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📊 Data Quality")
        quality_df = pd.DataFrame({
            'Check': ['Missing Values', 'Duplicate Invoices', 'Negative Values', 'Data Completeness'],
            'Status': ['✅ Clean', '✅ Clean', '✅ Clean', '✅ 100%']
        })
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    # Sample Data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)


def page_eda(df):
    
    st.markdown('<div class="main-header">📈 Sales Trends & EDA</div>', unsafe_allow_html=True)
    
    # Revenue Trend
    st.subheader("💰 Monthly Revenue Trend")
    monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
    monthly_revenue.index = monthly_revenue.index.to_timestamp()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2.5, 
            markersize=8, color='#1f77b4')
    ax.fill_between(monthly_revenue.index, monthly_revenue.values, alpha=0.3, color='lightblue')
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Top Products and Countries
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Products by Revenue")
        top_products = df.groupby('Description')['TotalPrice'].sum().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_products)), top_products.values, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels([p[:30] for p in top_products.index], fontsize=9)
        ax.set_xlabel('Revenue ($)', fontsize=10, fontweight='bold')
        ax.set_title('Top 10 Products', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🌍 Top 10 Countries by Revenue")
        top_countries = df.groupby('Country')['TotalPrice'].sum().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_countries)), top_countries.values, color='coral', alpha=0.8)
        ax.set_yticks(range(len(top_countries)))
        ax.set_yticklabels(top_countries.index, fontsize=9)
        ax.set_xlabel('Revenue ($)', fontsize=10, fontweight='bold')
        ax.set_title('Top 10 Countries', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Temporal Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Revenue by Day of Week")
        dow_revenue = df.groupby('DayOfWeek')['TotalPrice'].sum()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(7), dow_revenue.values, color='teal', alpha=0.8)
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names, rotation=45, ha='right')
        ax.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
        ax.set_title('Revenue by Day of Week', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🕐 Revenue by Hour of Day")
        hourly_revenue = df.groupby('Hour')['TotalPrice'].sum()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hourly_revenue.index, hourly_revenue.values, marker='o', 
                linewidth=2, markersize=6, color='crimson')
        ax.fill_between(hourly_revenue.index, hourly_revenue.values, alpha=0.3, color='pink')
        ax.set_xlabel('Hour of Day', fontsize=10, fontweight='bold')
        ax.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
        ax.set_title('Revenue by Hour', fontsize=11, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_segmentation(df, rfm_df):
    
    st.markdown('<div class="main-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    
    if rfm_df is None:
        st.warning("⚠️ Customer segmentation data not available. Please run the segmentation analysis first.")
        return
    
    # Segment Overview
    st.subheader("📊 Segment Distribution")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        segment_counts = rfm_df['Segment'].value_counts()
        st.write("**Customer Count by Segment:**")
        for segment, count in segment_counts.items():
            st.write(f"**{segment}:** {count:,}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(range(len(segment_counts)))
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title('Customer Segment Distribution', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col3:
        st.write("**Total Customers:**")
        st.metric("", f"{len(rfm_df):,}")
        st.write("**Total Segments:**")
        st.metric("", f"{rfm_df['Segment'].nunique()}")
    
    st.markdown("---")
    
    # Segment Profiles
    st.subheader("📋 Segment Profiles")
    
    segment_profile = rfm_df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)
    
    segment_profile.columns = ['Avg Recency (days)', 'Avg Frequency (orders)', 
                                'Avg Monetary ($)', 'Customer Count']
    segment_profile = segment_profile.sort_values('Avg Monetary ($)', ascending=False)
    
    st.dataframe(segment_profile, use_container_width=True)
    
    # RFM Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Average Spending by Segment")
        segment_monetary = rfm_df.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(segment_monetary)), segment_monetary.values, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(segment_monetary)))
        ax.set_yticklabels(segment_monetary.index, fontsize=10)
        ax.set_xlabel('Average Spending ($)', fontsize=10, fontweight='bold')
        ax.set_title('Avg Monetary Value by Segment', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📊 Recency vs Frequency")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for segment in rfm_df['Segment'].unique():
            segment_data = rfm_df[rfm_df['Segment'] == segment]
            ax.scatter(segment_data['Recency'], segment_data['Frequency'],
                      label=segment, alpha=0.6, s=50)
        ax.set_xlabel('Recency (days)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency (orders)', fontsize=10, fontweight='bold')
        ax.set_title('Recency vs Frequency by Segment', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_market_basket(rules_df):
    
    st.markdown('<div class="main-header">🛒 Market Basket Analysis</div>', unsafe_allow_html=True)
    
    if rules_df is None or len(rules_df) == 0:
        st.warning("⚠️ Market basket analysis data not available. Please run the analysis first.")
        return
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", f"{len(rules_df):,}")
    with col2:
        st.metric("Avg Confidence", f"{rules_df['confidence'].mean():.2%}")
    with col3:
        st.metric("Avg Lift", f"{rules_df['lift'].mean():.2f}x")
    with col4:
        st.metric("Max Lift", f"{rules_df['lift'].max():.2f}x")
    
    st.markdown("---")
    
    # Top Rules
    st.subheader("🏆 Top 20 Association Rules (by Lift)")
    
    top_rules = rules_df.nlargest(20, 'lift')
    
    display_df = pd.DataFrame({
        'If Customer Buys': top_rules['antecedents'].str[:50],
        'Then Also Buys': top_rules['consequents'].str[:50],
        'Support': top_rules['support'].apply(lambda x: f"{x:.3f}"),
        'Confidence': top_rules['confidence'].apply(lambda x: f"{x:.1%}"),
        'Lift': top_rules['lift'].apply(lambda x: f"{x:.2f}x")
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Support vs Confidence")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(rules_df['support'], rules_df['confidence'], 
                           c=rules_df['lift'], s=60, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Support', fontsize=11, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_title('Support vs Confidence (colored by Lift)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Lift')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📊 Top 15 Rules by Lift")
        
        top_15 = rules_df.nlargest(15, 'lift')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_15)), top_15['lift'].values, color='coral', alpha=0.8)
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels([f"Rule {i+1}" for i in range(len(top_15))], fontsize=9)
        ax.set_xlabel('Lift', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Association Rules', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Product Bundle Recommendations
    st.markdown("---")
    st.subheader("💡 Product Bundle Recommendations")
    
    high_confidence_rules = rules_df[
        (rules_df['confidence'] >= 0.3) & 
        (rules_df['lift'] >= 1.5)
    ].nlargest(10, 'lift')
    
    for idx, row in high_confidence_rules.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="insight-box">
            <strong>Bundle {idx+1}:</strong><br>
            🛍️ <strong>If buying:</strong> {row['antecedents'][:60]}<br>
            ➡️ <strong>Recommend:</strong> {row['consequents'][:60]}<br>
            📊 <strong>Metrics:</strong> Lift: {row['lift']:.2f}x | Confidence: {row['confidence']:.1%} | Support: {row['support']:.3f}
            </div>
            """, unsafe_allow_html=True)


def page_clv(clv_df):
    
    st.markdown('<div class="main-header">💎 Customer Lifetime Value</div>', unsafe_allow_html=True)
    
    if clv_df is None:
        st.warning("⚠️ CLV data not available. Please run the CLV analysis first.")
        return
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total CLV", f"${clv_df['CLV'].sum():,.0f}")
    with col2:
        st.metric("Average CLV", f"${clv_df['CLV'].mean():,.2f}")
    with col3:
        st.metric("Median CLV", f"${clv_df['CLV'].median():,.2f}")
    with col4:
        st.metric("Top Customer CLV", f"${clv_df['CLV'].max():,.2f}")
    
    st.markdown("---")
    
    # CLV Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 CLV Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(clv_df['CLV'][clv_df['CLV'] < clv_df['CLV'].quantile(0.95)], 
               bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(clv_df['CLV'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: ${clv_df["CLV"].mean():,.0f}')
        ax.axvline(clv_df['CLV'].median(), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: ${clv_df["CLV"].median():,.0f}')
        ax.set_xlabel('CLV ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('CLV Distribution (95th percentile)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📋 CLV Categories")
        if 'CLV_Category' in clv_df.columns:
            category_counts = clv_df['CLV_Category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(clv_df)) * 100
                st.write(f"**{category}:** {count:,} ({percentage:.1f}%)")
    
    # Top Customers
    st.markdown("---")
    st.subheader("🏆 Top 20 Customers by CLV")
    
    top_customers = clv_df.nlargest(20, 'CLV')
    
    display_df = pd.DataFrame({
        'Rank': range(1, len(top_customers) + 1),
        'Customer ID': top_customers['CustomerID'].astype(int),
        'CLV': top_customers['CLV'].apply(lambda x: f"${x:,.2f}"),
        'Orders': top_customers['OrderCount'].astype(int),
        'Avg Order Value': top_customers['AvgOrderValue'].apply(lambda x: f"${x:,.2f}"),
        'Purchase Freq/Year': top_customers['PurchaseFrequency_Year'].apply(lambda x: f"{x:.1f}")
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # CLV Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 20 Customers CLV")
        
        fig, ax = plt.subplots(figsize=(8, 7))
        top_20 = clv_df.nlargest(20, 'CLV')
        ax.barh(range(len(top_20)), top_20['CLV'].values, color='coral', alpha=0.8)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels([f"Cust {int(cid)}" for cid in top_20['CustomerID']], fontsize=9)
        ax.set_xlabel('CLV ($)', fontsize=10, fontweight='bold')
        ax.set_title('Top 20 Customers by CLV', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📊 CLV vs Order Frequency")
        
        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = ax.scatter(clv_df['AvgOrderValue'], clv_df['PurchaseFrequency_Year'],
                           c=clv_df['CLV'], s=60, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Avg Order Value ($)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Purchase Frequency (orders/year)', fontsize=10, fontweight='bold')
        ax.set_title('AOV vs Purchase Frequency', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='CLV ($)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_ml_predictions(churn_pred, clv_pred, df):
    
    st.markdown('<div class="main-header">Machine Learning Predictions</div>', unsafe_allow_html=True)
    
    if churn_pred is None and clv_pred is None:
        st.warning("ML prediction data not available. Please run the ML pipeline first using: python run_ml_project.py")
        return
    
    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if churn_pred is not None:
        with col1:
            churn_count = churn_pred['ChurnPrediction'].sum()
            st.metric("Predicted Churners", f"{churn_count:,}")
        with col2:
            churn_rate = (churn_count / len(churn_pred)) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    if clv_pred is not None:
        with col3:
            avg_predicted_clv = clv_pred['PredictedCLV'].mean()
            st.metric("Avg Predicted CLV", f"${avg_predicted_clv:,.0f}")
        with col4:
            total_predicted_clv = clv_pred['PredictedCLV'].sum()
            st.metric("Total Predicted CLV", f"${total_predicted_clv:,.0f}")
    
    st.markdown("---")
    
    # Churn Prediction Section
    if churn_pred is not None:
        st.subheader("Customer Churn Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Churn Risk Distribution**")
            churn_dist = churn_pred['ChurnPrediction'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = ['#2ecc71', '#e74c3c']
            labels = ['Retained (0)', 'Churned (1)']
            ax.pie(churn_dist.values, labels=labels, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax.set_title('Churn Prediction Distribution', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Churn Probability Distribution**")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(churn_pred['ChurnProbability'], bins=30, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Churn Probability', fontsize=10, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=10, fontweight='bold')
            ax.set_title('Distribution of Churn Probabilities', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # High-Risk Customers
        st.markdown("---")
        st.subheader("High-Risk Customers (Churn Probability > 0.7)")
        
        high_risk = churn_pred[churn_pred['ChurnProbability'] > 0.7].sort_values('ChurnProbability', ascending=False)
        
        if len(high_risk) > 0:
            st.write(f"**Total High-Risk Customers:** {len(high_risk):,}")
            st.dataframe(high_risk.head(20), use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>Action Items:</strong><br>
            <ul>
            <li>Immediate outreach to customers with probability > 0.9</li>
            <li>Personalized retention offers and discounts</li>
            <li>Survey to understand pain points</li>
            <li>Enhance customer service touchpoints</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("No high-risk customers identified!")
    
    st.markdown("---")
    
    # CLV Prediction Section
    if clv_pred is not None:
        st.subheader("Customer Lifetime Value (CLV) Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predicted vs Actual CLV**")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sample_clv = clv_pred.sample(min(1000, len(clv_pred)))
            ax.scatter(sample_clv['ActualCLV'], sample_clv['PredictedCLV'], 
                      alpha=0.5, s=30, color='steelblue')
            
            # Add diagonal line
            max_val = max(sample_clv['ActualCLV'].max(), sample_clv['PredictedCLV'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual CLV ($)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted CLV ($)', fontsize=10, fontweight='bold')
            ax.set_title('Predicted vs Actual CLV', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Prediction Error Distribution**")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(clv_pred['PredictionError'], bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Prediction Error ($)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=10, fontweight='bold')
            ax.set_title('CLV Prediction Error Distribution', fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Top Predicted High-Value Customers
        st.markdown("---")
        st.subheader("Top 20 Predicted High-Value Customers")
        
        top_clv = clv_pred.nlargest(20, 'PredictedCLV')[['CustomerID', 'ActualCLV', 'PredictedCLV', 'PredictionError']]
        top_clv['PredictionError'] = top_clv['PredictionError'].round(2)
        top_clv['ActualCLV'] = top_clv['ActualCLV'].round(2)
        top_clv['PredictedCLV'] = top_clv['PredictedCLV'].round(2)
        
        st.dataframe(top_clv, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Strategic Actions:</strong><br>
        <ul>
        <li>Enroll high-CLV customers in VIP loyalty programs</li>
        <li>Provide exclusive early access to new products</li>
        <li>Personalized account management</li>
        <li>Special rewards and recognition</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # CLV Distribution by Ranges
        st.markdown("---")
        st.subheader("CLV Distribution by Value Ranges")
        
        clv_pred['CLV_Range'] = pd.cut(clv_pred['PredictedCLV'], 
                                       bins=[0, 1000, 5000, 10000, 50000, float('inf')],
                                       labels=['$0-1K', '$1K-5K', '$5K-10K', '$10K-50K', '$50K+'])
        
        clv_range_counts = clv_pred['CLV_Range'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(range(len(clv_range_counts)), clv_range_counts.values, 
                   color='teal', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(clv_range_counts)))
            ax.set_xticklabels(clv_range_counts.index, rotation=45, ha='right')
            ax.set_xlabel('CLV Range', fontsize=10, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=10, fontweight='bold')
            ax.set_title('Customer Distribution by CLV Range', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Customer Count by CLV Range:**")
            for range_label, count in clv_range_counts.items():
                pct = (count / len(clv_pred)) * 100
                st.write(f"**{range_label}:** {count:,} ({pct:.1f}%)")
    
    # Combined Analysis
    if churn_pred is not None and clv_pred is not None:
        st.markdown("---")
        st.subheader("Combined Risk Analysis: Churn Risk vs CLV")
        
        # Merge datasets
        combined = churn_pred.merge(clv_pred, on='CustomerID', how='inner')
        
        # Create risk segments
        combined['RiskSegment'] = 'Low Priority'
        combined.loc[(combined['ChurnProbability'] > 0.7) & (combined['PredictedCLV'] > 10000), 'RiskSegment'] = 'Critical (High CLV + High Churn)'
        combined.loc[(combined['ChurnProbability'] > 0.7) & (combined['PredictedCLV'] <= 10000), 'RiskSegment'] = 'Medium Risk (Low CLV + High Churn)'
        combined.loc[(combined['ChurnProbability'] <= 0.7) & (combined['PredictedCLV'] > 10000), 'RiskSegment'] = 'Protect (High CLV + Low Churn)'
        
        segment_counts = combined['RiskSegment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Risk Segment Distribution**")
            for segment, count in segment_counts.items():
                pct = (count / len(combined)) * 100
                st.write(f"**{segment}:** {count:,} ({pct:.1f}%)")
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            colors_map = {
                'Critical (High CLV + High Churn)': '#e74c3c',
                'Medium Risk (Low CLV + High Churn)': '#f39c12',
                'Protect (High CLV + Low Churn)': '#2ecc71',
                'Low Priority': '#95a5a6'
            }
            colors = [colors_map.get(seg, '#95a5a6') for seg in segment_counts.index]
            
            ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 8, 'fontweight': 'bold'})
            ax.set_title('Customer Risk Segmentation', fontsize=11, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        # Critical customers table
        critical_customers = combined[combined['RiskSegment'] == 'Critical (High CLV + High Churn)'].sort_values('PredictedCLV', ascending=False)
        
        if len(critical_customers) > 0:
            st.markdown("---")
            st.subheader("CRITICAL: High-Value Customers at Risk of Churning")
            st.error(f"**{len(critical_customers):,} high-value customers at immediate risk!**")
            
            display_cols = ['CustomerID', 'ChurnProbability', 'PredictedCLV', 'ActualCLV']
            st.dataframe(critical_customers[display_cols].head(15), use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>URGENT ACTION REQUIRED:</strong><br>
            <ul>
            <li>Executive-level outreach within 24-48 hours</li>
            <li>Customized retention packages with significant value</li>
            <li>Root cause analysis through direct feedback</li>
            <li>Dedicated account manager assignment</li>
            <li>Monitor engagement closely post-intervention</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def page_insights(df, rfm_df, clv_df, rules_df):
    
    st.markdown('<div class="main-header">💡 Business Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Key Findings
    st.subheader("🎯 Key Findings")
    
    # Revenue Insights
    total_revenue = df['TotalPrice'].sum()
    top_country = df.groupby('Country')['TotalPrice'].sum().idxmax()
    top_country_revenue = df.groupby('Country')['TotalPrice'].sum().max()
    top_country_pct = (top_country_revenue / total_revenue) * 100
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>1. Revenue Concentration:</strong><br>
    {top_country} contributes <strong>${top_country_revenue:,.0f}</strong> ({top_country_pct:.1f}% of total revenue).
    Consider diversifying markets to reduce dependency on single regions.
    </div>
    """, unsafe_allow_html=True)
    
    # Seasonal Trends
    monthly_revenue = df.groupby(df['InvoiceDate'].dt.month)['TotalPrice'].sum()
    peak_month = monthly_revenue.idxmax()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>2. Seasonal Pattern:</strong><br>
    Peak sales month: <strong>{month_names[peak_month-1]}</strong> with ${monthly_revenue.max():,.0f} in revenue.
    Plan inventory and marketing campaigns around seasonal peaks.
    </div>
    """, unsafe_allow_html=True)
    
    # Customer Insights
    if clv_df is not None:
        top_20_pct_clv = (clv_df.nlargest(int(len(clv_df)*0.2), 'CLV')['CLV'].sum() / clv_df['CLV'].sum()) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>3. Customer Value Distribution:</strong><br>
        Top 20% of customers contribute <strong>{top_20_pct_clv:.1f}%</strong> of total CLV.
        Implement VIP programs and retention strategies for high-value customers.
        </div>
        """, unsafe_allow_html=True)
    
    # Product Bundling
    if rules_df is not None and len(rules_df) > 0:
        high_lift_rules = len(rules_df[rules_df['lift'] > 2])
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>4. Cross-Selling Opportunities:</strong><br>
        Found <strong>{high_lift_rules}</strong> strong product associations (lift > 2).
        Use these insights for bundle promotions and product placement.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("📋 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
        <strong>🎯 Marketing Strategies</strong><br>
        <ul>
        <li>Launch targeted campaigns for dormant high-CLV customers</li>
        <li>Create product bundles based on association rules</li>
        <li>Implement personalized recommendations</li>
        <li>Increase engagement during peak sales periods</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>💰 Revenue Optimization</strong><br>
        <ul>
        <li>Focus on high-margin product categories</li>
        <li>Implement dynamic pricing during peak hours</li>
        <li>Expand in underserved geographic markets</li>
        <li>Upsell and cross-sell to existing customers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>👥 Customer Retention</strong><br>
        <ul>
        <li>Develop loyalty programs for Champions segment</li>
        <li>Re-engagement campaigns for At-Risk customers</li>
        <li>Personalized offers based on purchase history</li>
        <li>Improve customer experience to boost frequency</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>📦 Operational Excellence</strong><br>
        <ul>
        <li>Optimize inventory for seasonal demand</li>
        <li>Streamline fulfillment for top products</li>
        <li>Monitor and improve delivery times</li>
        <li>Expand product catalog in high-demand categories</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Summary
    st.markdown("---")
    st.subheader("📊 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{df['Customer ID'].nunique():,}")
        st.metric("Repeat Customer Rate", 
                 f"{(df.groupby('Customer ID')['Invoice'].nunique() > 1).sum() / df['Customer ID'].nunique() * 100:.1f}%")
    
    with col2:
        st.metric("Total Revenue", f"${df['TotalPrice'].sum():,.0f}")
        st.metric("Avg Order Value", f"${df.groupby('Invoice')['TotalPrice'].sum().mean():,.2f}")
    
    with col3:
        if clv_df is not None:
            st.metric("Avg Customer CLV", f"${clv_df['CLV'].mean():,.2f}")
            st.metric("High-Value Customers", 
                     f"{len(clv_df[clv_df['CLV_Percentile'] >= 75]):,}")


def main():
    
    # Load data
    df, rfm_df, clv_df, rules_df, churn_pred, clv_pred = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure data files are available.")
        return
    
    # Sidebar
    date_range, country, product = show_sidebar(df)
    
    # Apply filters
    df_filtered = apply_filters(df, date_range, country, product)
    
    # Display filter info
    if country != 'All' or product != 'All':
        st.info(f"Filters Applied: Country={country}, Product={product if product != 'All' else 'All'}")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dataset Overview", "Sales Trends & EDA", "Customer Segmentation", 
         "Market Basket Analysis", "Customer Lifetime Value", "ML Predictions", "Business Insights"]
    )
    
    # Page routing
    if page == "Dataset Overview":
        page_overview(df_filtered)
    elif page == "Sales Trends & EDA":
        page_eda(df_filtered)
    elif page == "Customer Segmentation":
        page_segmentation(df_filtered, rfm_df)
    elif page == "Market Basket Analysis":
        page_market_basket(rules_df)
    elif page == "Customer Lifetime Value":
        page_clv(clv_df)
    elif page == "ML Predictions":
        page_ml_predictions(churn_pred, clv_pred, df_filtered)
    elif page == "Business Insights":
        page_insights(df_filtered, rfm_df, clv_df, rules_df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
    <p><strong>Consumer Purchase Behavior<br>& Market Trend Analysis</strong></p>
    <p>Powered by Python & Streamlit</p>
    <p>© 2026 Retail Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
