

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CLVAnalysis:
    
    def __init__(self, df):
        self.df = df
        self.clv_df = None
        
    def calculate_clv(self, lifespan_years=3):
        print("\n" + "="*60)
        print("CUSTOMER LIFETIME VALUE CALCULATION")
        print("="*60)
        
        print(f"\nEstimated customer lifespan: {lifespan_years} years")
        
        # Calculate metrics per customer
        customer_metrics = self.df.groupby('Customer ID').agg({
            'Invoice': 'nunique',           # Number of orders (frequency)
            'TotalPrice': 'sum',             # Total spending
            'InvoiceDate': ['min', 'max']    # First and last purchase dates
        }).reset_index()
        
        customer_metrics.columns = ['CustomerID', 'OrderCount', 'TotalSpending', 
                                    'FirstPurchase', 'LastPurchase']
        
        # Calculate Average Order Value
        customer_metrics['AvgOrderValue'] = customer_metrics['TotalSpending'] / customer_metrics['OrderCount']
        
        # Calculate days between first and last purchase
        customer_metrics['CustomerAge_Days'] = (
            customer_metrics['LastPurchase'] - customer_metrics['FirstPurchase']
        ).dt.days
        
        # Calculate Purchase Frequency (orders per year)
        # Avoid division by zero
        customer_metrics['PurchaseFrequency_Year'] = customer_metrics.apply(
            lambda row: (row['OrderCount'] / (row['CustomerAge_Days'] / 365)) 
            if row['CustomerAge_Days'] > 0 else row['OrderCount'], 
            axis=1
        )
        
        # For new customers with only one purchase, use average frequency
        avg_frequency = customer_metrics[customer_metrics['OrderCount'] > 1]['PurchaseFrequency_Year'].mean()
        customer_metrics['PurchaseFrequency_Year'] = customer_metrics['PurchaseFrequency_Year'].replace(
            [np.inf, -np.inf], avg_frequency
        ).fillna(avg_frequency)
        
        # Calculate CLV
        customer_metrics['CLV'] = (
            customer_metrics['AvgOrderValue'] * 
            customer_metrics['PurchaseFrequency_Year'] * 
            lifespan_years
        )
        
        # Add percentile rank
        customer_metrics['CLV_Percentile'] = customer_metrics['CLV'].rank(pct=True) * 100
        
        # Categorize customers by CLV
        customer_metrics['CLV_Category'] = pd.cut(
            customer_metrics['CLV_Percentile'],
            bins=[0, 25, 50, 75, 90, 100],
            labels=['Low Value', 'Medium Value', 'High Value', 'Very High Value', 'VIP']
        )
        
        self.clv_df = customer_metrics.sort_values('CLV', ascending=False)
        
        print(f"\n✓ CLV calculated for {len(customer_metrics):,} customers")
        print(f"\nCLV Summary Statistics:")
        print(f"  Mean CLV:        ${customer_metrics['CLV'].mean():,.2f}")
        print(f"  Median CLV:      ${customer_metrics['CLV'].median():,.2f}")
        print(f"  Min CLV:         ${customer_metrics['CLV'].min():,.2f}")
        print(f"  Max CLV:         ${customer_metrics['CLV'].max():,.2f}")
        print(f"  Std Dev:         ${customer_metrics['CLV'].std():,.2f}")
        
        print(f"\nCustomer Distribution by CLV Category:")
        category_counts = customer_metrics['CLV_Category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(customer_metrics)) * 100
            avg_clv = customer_metrics[customer_metrics['CLV_Category'] == category]['CLV'].mean()
            print(f"  {category:18s}: {count:,} customers ({percentage:.1f}%) - Avg CLV: ${avg_clv:,.2f}")
        
        return customer_metrics
    
    def display_top_customers(self, top_n=20):
        if self.clv_df is None:
            raise ValueError("CLV not calculated. Run calculate_clv() first.")
        
        print("\n" + "="*60)
        print(f"TOP {top_n} CUSTOMERS BY CLV")
        print("="*60)
        
        top_customers = self.clv_df.head(top_n)
        
        print(f"\n{'Rank':<6}{'CustomerID':<15}{'CLV':<15}{'Orders':<10}{'Avg Order':<15}{'Category':<20}")
        print("-" * 90)
        
        for idx, (rank, row) in enumerate(top_customers.iterrows(), 1):
            print(f"{idx:<6}{int(row['CustomerID']):<15}${row['CLV']:<14,.2f}{int(row['OrderCount']):<10}"
                  f"${row['AvgOrderValue']:<14,.2f}{row['CLV_Category']:<20}")
        
        return top_customers
    
    def analyze_clv_segments(self):
        if self.clv_df is None:
            raise ValueError("CLV not calculated. Run calculate_clv() first.")
        
        print("\n" + "="*60)
        print("CLV SEGMENT ANALYSIS")
        print("="*60)
        
        segment_analysis = self.clv_df.groupby('CLV_Category').agg({
            'CustomerID': 'count',
            'CLV': ['mean', 'median', 'sum'],
            'AvgOrderValue': 'mean',
            'PurchaseFrequency_Year': 'mean',
            'OrderCount': 'mean',
            'TotalSpending': 'sum'
        }).round(2)
        
        print("\nDetailed Segment Metrics:")
        print(segment_analysis)
        
        # Calculate revenue contribution
        total_clv = self.clv_df['CLV'].sum()
        print(f"\nRevenue Contribution by Segment:")
        for category in self.clv_df['CLV_Category'].cat.categories:
            segment_clv = self.clv_df[self.clv_df['CLV_Category'] == category]['CLV'].sum()
            contribution = (segment_clv / total_clv) * 100
            customer_count = len(self.clv_df[self.clv_df['CLV_Category'] == category])
            print(f"  {category:18s}: ${segment_clv:,.2f} ({contribution:.1f}%) - {customer_count:,} customers")
        
        return segment_analysis
    
    def visualize_clv(self):
        print("\n" + "="*60)
        print("CREATING CLV VISUALIZATIONS")
        print("="*60)
        
        if self.clv_df is None:
            raise ValueError("CLV not calculated. Run calculate_clv() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. CLV Distribution
        axes[0, 0].hist(self.clv_df['CLV'][self.clv_df['CLV'] < self.clv_df['CLV'].quantile(0.95)], 
                       bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.clv_df['CLV'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: ${self.clv_df["CLV"].mean():,.0f}')
        axes[0, 0].axvline(self.clv_df['CLV'].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: ${self.clv_df["CLV"].median():,.0f}')
        axes[0, 0].set_xlabel('CLV ($)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('CLV Distribution (95th percentile)', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Top 20 Customers
        top_20 = self.clv_df.head(20)
        axes[0, 1].barh(range(len(top_20)), top_20['CLV'], color='coral', alpha=0.8)
        axes[0, 1].set_yticks(range(len(top_20)))
        axes[0, 1].set_yticklabels([f"Cust {int(cid)}" for cid in top_20['CustomerID']], fontsize=8)
        axes[0, 1].set_xlabel('CLV ($)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Top 20 Customers by CLV', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. CLV Category Distribution
        category_counts = self.clv_df['CLV_Category'].value_counts()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(category_counts)))
        axes[0, 2].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 2].set_title('Customer Distribution by CLV Category', fontsize=11, fontweight='bold')
        
        # 4. Average Order Value vs Purchase Frequency
        scatter = axes[1, 0].scatter(self.clv_df['AvgOrderValue'], 
                                     self.clv_df['PurchaseFrequency_Year'],
                                     c=self.clv_df['CLV'], s=50, alpha=0.6, cmap='viridis')
        axes[1, 0].set_xlabel('Avg Order Value ($)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Purchase Frequency (orders/year)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('AOV vs Purchase Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='CLV ($)')
        
        # 5. CLV by Category (Box Plot)
        category_order = ['Low Value', 'Medium Value', 'High Value', 'Very High Value', 'VIP']
        axes[1, 1].boxplot([self.clv_df[self.clv_df['CLV_Category'] == cat]['CLV'].values 
                            for cat in category_order if cat in self.clv_df['CLV_Category'].values],
                          labels=[cat.replace(' ', '\n') for cat in category_order 
                                 if cat in self.clv_df['CLV_Category'].values],
                          patch_artist=True)
        axes[1, 1].set_ylabel('CLV ($)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('CLV Distribution by Category', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', labelsize=8)
        
        # 6. Cumulative CLV
        sorted_clv = self.clv_df['CLV'].sort_values(ascending=False).reset_index(drop=True)
        cumulative_clv = sorted_clv.cumsum()
        cumulative_pct = (cumulative_clv / cumulative_clv.iloc[-1]) * 100
        customer_pct = (np.arange(len(sorted_clv)) / len(sorted_clv)) * 100
        
        axes[1, 2].plot(customer_pct, cumulative_pct, linewidth=2, color='navy')
        axes[1, 2].fill_between(customer_pct, cumulative_pct, alpha=0.3)
        axes[1, 2].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% of CLV')
        axes[1, 2].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% of CLV')
        axes[1, 2].set_xlabel('% of Customers', fontsize=11, fontweight='bold')
        axes[1, 2].set_ylabel('% of Total CLV', fontsize=11, fontweight='bold')
        axes[1, 2].set_title('Cumulative CLV Distribution (Pareto)', fontsize=12, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Customer Lifetime Value Analysis', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig('outputs/clv_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/clv_analysis.png")
        plt.close()
    
    def identify_at_risk_customers(self, recency_threshold_days=180):
        if self.clv_df is None:
            raise ValueError("CLV not calculated. Run calculate_clv() first.")
        
        print("\n" + "="*60)
        print("AT-RISK HIGH-VALUE CUSTOMERS")
        print("="*60)
        
        # Calculate days since last purchase
        reference_date = self.df['InvoiceDate'].max()
        self.clv_df['DaysSinceLastPurchase'] = (
            reference_date - self.clv_df['LastPurchase']
        ).dt.days
        
        # Identify at-risk customers (high CLV but inactive)
        at_risk = self.clv_df[
            (self.clv_df['CLV_Percentile'] >= 75) &  # Top 25% by CLV
            (self.clv_df['DaysSinceLastPurchase'] > recency_threshold_days)
        ].copy()
        
        at_risk = at_risk.sort_values('CLV', ascending=False)
        
        print(f"\nFound {len(at_risk):,} at-risk high-value customers")
        print(f"(CLV >= 75th percentile, inactive > {recency_threshold_days} days)")
        
        if len(at_risk) > 0:
            print(f"\nAt-Risk Customer Statistics:")
            print(f"  Total CLV at Risk:   ${at_risk['CLV'].sum():,.2f}")
            print(f"  Avg CLV:             ${at_risk['CLV'].mean():,.2f}")
            print(f"  Avg Days Inactive:   {at_risk['DaysSinceLastPurchase'].mean():.0f} days")
            
            print(f"\nTop 10 At-Risk Customers:")
            print(f"{'CustomerID':<15}{'CLV':<15}{'Days Inactive':<20}{'Last Purchase':<20}")
            print("-" * 70)
            for _, row in at_risk.head(10).iterrows():
                print(f"{int(row['CustomerID']):<15}${row['CLV']:<14,.2f}{int(row['DaysSinceLastPurchase']):<20}"
                      f"{row['LastPurchase'].strftime('%Y-%m-%d'):<20}")
        
        return at_risk
    
    def save_results(self, output_path='outputs/customer_clv.csv'):
        if self.clv_df is None:
            raise ValueError("No results to save.")
        
        self.clv_df.to_csv(output_path, index=False)
        print(f"\n✓ CLV results saved to: {output_path}")


def main():
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize CLV analysis
    clv_analysis = CLVAnalysis(df)
    
    # Execute CLV workflow
    clv_analysis.calculate_clv(lifespan_years=3)
    clv_analysis.display_top_customers(top_n=20)
    clv_analysis.analyze_clv_segments()
    clv_analysis.visualize_clv()
    clv_analysis.identify_at_risk_customers(recency_threshold_days=180)
    clv_analysis.save_results()
    
    print("\n✓ CLV analysis completed successfully!")
    
    return clv_analysis.clv_df


if __name__ == "__main__":
    clv_results = main()
