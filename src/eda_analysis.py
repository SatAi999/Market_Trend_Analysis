"""
Exploratory Data Analysis Module for Retail Analysis
Author: Senior Data Scientist
Date: January 2026

This module performs comprehensive EDA on retail transaction data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RetailEDA:
    """
    A class to perform Exploratory Data Analysis on retail data.
    
    Includes analysis for:
    - Revenue metrics
    - Top products
    - Country-wise performance
    - Temporal trends
    """
    
    def __init__(self, df):
        """
        Initialize EDA with cleaned dataframe.
        
        Args:
            df (pd.DataFrame): Cleaned retail transaction data
        """
        self.df = df
        self.summary_stats = {}
        
    def calculate_summary_statistics(self):
        """Calculate key business metrics."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        total_revenue = self.df['TotalPrice'].sum()
        total_orders = self.df['Invoice'].nunique()
        total_customers = self.df['Customer ID'].nunique()
        total_products = self.df['StockCode'].nunique()
        avg_order_value = total_revenue / total_orders
        avg_basket_size = self.df.groupby('Invoice')['Quantity'].sum().mean()
        
        self.summary_stats = {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'total_customers': total_customers,
            'total_products': total_products,
            'avg_order_value': avg_order_value,
            'avg_basket_size': avg_basket_size
        }
        
        print(f"Total Revenue:        ${total_revenue:,.2f}")
        print(f"Total Orders:         {total_orders:,}")
        print(f"Total Customers:      {total_customers:,}")
        print(f"Total Products:       {total_products:,}")
        print(f"Avg Order Value:      ${avg_order_value:.2f}")
        print(f"Avg Basket Size:      {avg_basket_size:.2f} items")
        
        return self.summary_stats
    
    def analyze_top_products(self, top_n=20):
        """
        Analyze and visualize top products by revenue and quantity.
        
        Args:
            top_n (int): Number of top products to display
        """
        print(f"\n{'='*60}")
        print(f"TOP {top_n} PRODUCTS ANALYSIS")
        print("="*60)
        
        # Top products by revenue
        product_revenue = self.df.groupby('Description').agg({
            'TotalPrice': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        product_revenue = product_revenue.sort_values('TotalPrice', ascending=False).head(top_n)
        
        print("\nTop Products by Revenue:")
        for idx, row in product_revenue.head(10).iterrows():
            print(f"  {row['Description'][:40]:40s} - ${row['TotalPrice']:,.2f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Revenue plot
        axes[0].barh(range(len(product_revenue)), product_revenue['TotalPrice'], color='steelblue')
        axes[0].set_yticks(range(len(product_revenue)))
        axes[0].set_yticklabels([desc[:30] for desc in product_revenue['Description']], fontsize=9)
        axes[0].set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Top {top_n} Products by Revenue', fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Quantity plot
        product_qty = self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(top_n)
        axes[1].barh(range(len(product_qty)), product_qty.values, color='coral')
        axes[1].set_yticks(range(len(product_qty)))
        axes[1].set_yticklabels([desc[:30] for desc in product_qty.index], fontsize=9)
        axes[1].set_xlabel('Quantity Sold', fontsize=11, fontweight='bold')
        axes[1].set_title(f'Top {top_n} Products by Quantity', fontsize=13, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/top_products.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/top_products.png")
        plt.close()
        
        return product_revenue
    
    def analyze_country_performance(self, top_n=15):
        """
        Analyze and visualize country-wise revenue.
        
        Args:
            top_n (int): Number of top countries to display
        """
        print(f"\n{'='*60}")
        print(f"TOP {top_n} COUNTRIES ANALYSIS")
        print("="*60)
        
        country_revenue = self.df.groupby('Country').agg({
            'TotalPrice': 'sum',
            'Invoice': 'nunique',
            'Customer ID': 'nunique'
        }).reset_index()
        
        country_revenue.columns = ['Country', 'Revenue', 'Orders', 'Customers']
        country_revenue = country_revenue.sort_values('Revenue', ascending=False).head(top_n)
        
        print("\nTop Countries by Revenue:")
        for idx, row in country_revenue.head(10).iterrows():
            print(f"  {row['Country']:25s} - ${row['Revenue']:,.2f} ({row['Orders']:,} orders)")
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Revenue by country
        axes[0].bar(range(len(country_revenue)), country_revenue['Revenue'], color='seagreen', alpha=0.8)
        axes[0].set_xticks(range(len(country_revenue)))
        axes[0].set_xticklabels(country_revenue['Country'], rotation=45, ha='right', fontsize=10)
        axes[0].set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
        axes[0].set_title('Revenue by Country', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Orders and Customers
        x = np.arange(len(country_revenue))
        width = 0.35
        axes[1].bar(x - width/2, country_revenue['Orders'], width, label='Orders', color='royalblue', alpha=0.8)
        axes[1].bar(x + width/2, country_revenue['Customers'], width, label='Customers', color='orange', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(country_revenue['Country'], rotation=45, ha='right', fontsize=10)
        axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[1].set_title('Orders and Customers by Country', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/country_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/country_analysis.png")
        plt.close()
        
        return country_revenue
    
    def analyze_temporal_trends(self):
        """Analyze and visualize sales trends over time."""
        print(f"\n{'='*60}")
        print("TEMPORAL TRENDS ANALYSIS")
        print("="*60)
        
        # Monthly revenue trend
        monthly_revenue = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M')).agg({
            'TotalPrice': 'sum',
            'Invoice': 'nunique',
            'Customer ID': 'nunique'
        }).reset_index()
        
        monthly_revenue['InvoiceDate'] = monthly_revenue['InvoiceDate'].dt.to_timestamp()
        
        print(f"\nMonthly revenue range: ${monthly_revenue['TotalPrice'].min():,.2f} to ${monthly_revenue['TotalPrice'].max():,.2f}")
        
        # Create comprehensive temporal visualizations
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Monthly Revenue Trend
        axes[0].plot(monthly_revenue['InvoiceDate'], monthly_revenue['TotalPrice'], 
                    marker='o', linewidth=2, markersize=6, color='darkblue')
        axes[0].fill_between(monthly_revenue['InvoiceDate'], monthly_revenue['TotalPrice'], 
                            alpha=0.3, color='lightblue')
        axes[0].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
        axes[0].set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Daily of Week Analysis
        dow_revenue = self.df.groupby('DayOfWeek')['TotalPrice'].sum()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        axes[1].bar(range(7), dow_revenue.values, color='teal', alpha=0.8)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(dow_names, fontsize=10)
        axes[1].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
        axes[1].set_title('Revenue by Day of Week', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. Hourly Analysis
        hourly_revenue = self.df.groupby('Hour')['TotalPrice'].sum()
        axes[2].plot(hourly_revenue.index, hourly_revenue.values, 
                    marker='o', linewidth=2, markersize=5, color='crimson')
        axes[2].fill_between(hourly_revenue.index, hourly_revenue.values, alpha=0.3, color='pink')
        axes[2].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
        axes[2].set_title('Revenue by Hour of Day', fontsize=13, fontweight='bold')
        axes[2].set_xticks(range(0, 24))
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/temporal_trends.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/temporal_trends.png")
        plt.close()
        
        # Quarterly analysis
        quarterly_revenue = self.df.groupby(['Year', 'Quarter'])['TotalPrice'].sum().reset_index()
        print("\nQuarterly Revenue:")
        for idx, row in quarterly_revenue.iterrows():
            print(f"  {int(row['Year'])} Q{int(row['Quarter'])}: ${row['TotalPrice']:,.2f}")
        
        return monthly_revenue
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive EDA dashboard."""
        print(f"\n{'='*60}")
        print("CREATING COMPREHENSIVE DASHBOARD")
        print("="*60)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Revenue Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        monthly_rev = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
        monthly_rev.index = monthly_rev.index.to_timestamp()
        ax1.plot(monthly_rev.index, monthly_rev.values, marker='o', linewidth=2, color='navy')
        ax1.fill_between(monthly_rev.index, monthly_rev.values, alpha=0.3)
        ax1.set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Revenue ($)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Top 10 Countries
        ax2 = fig.add_subplot(gs[0, 2])
        top_countries = self.df.groupby('Country')['TotalPrice'].sum().nlargest(10)
        ax2.barh(range(len(top_countries)), top_countries.values, color='coral')
        ax2.set_yticks(range(len(top_countries)))
        ax2.set_yticklabels([c[:15] for c in top_countries.index], fontsize=8)
        ax2.set_title('Top 10 Countries', fontsize=10, fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Top 10 Products
        ax3 = fig.add_subplot(gs[1, :2])
        top_products = self.df.groupby('Description')['TotalPrice'].sum().nlargest(10)
        ax3.barh(range(len(top_products)), top_products.values, color='seagreen')
        ax3.set_yticks(range(len(top_products)))
        ax3.set_yticklabels([p[:40] for p in top_products.index], fontsize=8)
        ax3.set_title('Top 10 Products by Revenue', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Revenue ($)', fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Order Value Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        order_values = self.df.groupby('Invoice')['TotalPrice'].sum()
        ax4.hist(order_values[order_values < order_values.quantile(0.95)], 
                bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_title('Order Value Distribution', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Order Value ($)', fontsize=9)
        ax4.set_ylabel('Frequency', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Day of Week
        ax5 = fig.add_subplot(gs[2, 0])
        dow_rev = self.df.groupby('DayOfWeek')['TotalPrice'].sum()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax5.bar(range(7), dow_rev.values, color='teal', alpha=0.8)
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(dow_labels, fontsize=9)
        ax5.set_title('Revenue by Day', fontsize=10, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Hour of Day
        ax6 = fig.add_subplot(gs[2, 1])
        hourly_rev = self.df.groupby('Hour')['TotalPrice'].sum()
        ax6.plot(hourly_rev.index, hourly_rev.values, marker='o', color='crimson', linewidth=2)
        ax6.fill_between(hourly_rev.index, hourly_rev.values, alpha=0.3)
        ax6.set_title('Revenue by Hour', fontsize=10, fontweight='bold')
        ax6.set_xlabel('Hour', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Customer Activity
        ax7 = fig.add_subplot(gs[2, 2])
        customer_orders = self.df.groupby('Customer ID')['Invoice'].nunique()
        ax7.hist(customer_orders[customer_orders < customer_orders.quantile(0.95)], 
                bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax7.set_title('Customer Order Frequency', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Number of Orders', fontsize=9)
        ax7.set_ylabel('Customers', fontsize=9)
        ax7.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Retail Analytics - Comprehensive EDA Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('outputs/eda_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Dashboard saved: outputs/eda_dashboard.png")
        plt.close()


def main():
    """Main execution function."""
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize EDA
    eda = RetailEDA(df)
    
    # Run analyses
    eda.calculate_summary_statistics()
    eda.analyze_top_products(top_n=20)
    eda.analyze_country_performance(top_n=15)
    eda.analyze_temporal_trends()
    eda.create_comprehensive_dashboard()
    
    print("\n✓ EDA completed successfully!")


if __name__ == "__main__":
    main()
