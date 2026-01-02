"""
Market Basket Analysis Module using Apriori Algorithm
Author: Senior Data Scientist
Date: January 2026

This module performs market basket analysis to discover product associations and bundling patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


class MarketBasketAnalysis:
    """
    A class to perform market basket analysis using Apriori algorithm.
    
    Features:
    - Transaction encoding
    - Frequent itemset mining
    - Association rule generation
    - Product bundling recommendations
    """
    
    def __init__(self, df):
        """
        Initialize market basket analysis with cleaned dataframe.
        
        Args:
            df (pd.DataFrame): Cleaned retail transaction data
        """
        self.df = df
        self.basket_df = None
        self.frequent_itemsets = None
        self.rules = None
        
    def create_basket_format(self, min_support_items=50):
        """
        Convert transaction data to basket format for Apriori algorithm.
        
        Args:
            min_support_items (int): Minimum number of transactions for a product
        
        Returns:
            pd.DataFrame: Basket dataframe with one-hot encoding
        """
        print("\n" + "="*60)
        print("CREATING BASKET FORMAT")
        print("="*60)
        
        # Filter products that appear in at least min_support_items transactions
        product_counts = self.df.groupby('Description')['Invoice'].nunique()
        popular_products = product_counts[product_counts >= min_support_items].index
        
        print(f"\nFiltering products with >= {min_support_items} transactions")
        print(f"Popular products: {len(popular_products):,}")
        
        # Filter dataframe
        df_filtered = self.df[self.df['Description'].isin(popular_products)].copy()
        
        # Create basket format (one-hot encoding)
        basket = df_filtered.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
        
        # Convert to binary (1 if purchased, 0 if not)
        basket_binary = basket.map(lambda x: 1 if x > 0 else 0)
        
        self.basket_df = basket_binary
        
        print(f"\n✓ Basket created:")
        print(f"  Transactions: {basket_binary.shape[0]:,}")
        print(f"  Products:     {basket_binary.shape[1]:,}")
        print(f"  Sparsity:     {(1 - basket_binary.sum().sum() / basket_binary.size) * 100:.2f}%")
        
        return basket_binary
    
    def mine_frequent_itemsets(self, min_support=0.01):
        """
        Mine frequent itemsets using Apriori algorithm.
        
        Args:
            min_support (float): Minimum support threshold (0-1)
        
        Returns:
            pd.DataFrame: Frequent itemsets
        """
        print("\n" + "="*60)
        print("MINING FREQUENT ITEMSETS")
        print("="*60)
        
        if self.basket_df is None:
            raise ValueError("Basket format not created. Run create_basket_format() first.")
        
        print(f"\nApplying Apriori with min_support={min_support}")
        print("This may take a few moments...")
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(self.basket_df, min_support=min_support, use_colnames=True)
        
        # Add itemset length
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        # Sort by support
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        
        self.frequent_itemsets = frequent_itemsets
        
        print(f"\n✓ Found {len(frequent_itemsets):,} frequent itemsets")
        print(f"\nItemset Length Distribution:")
        for length in sorted(frequent_itemsets['length'].unique()):
            count = len(frequent_itemsets[frequent_itemsets['length'] == length])
            print(f"  Length {length}: {count:,} itemsets")
        
        print(f"\nTop 10 Most Frequent Single Items:")
        singles = frequent_itemsets[frequent_itemsets['length'] == 1].head(10)
        for idx, row in singles.iterrows():
            item = list(row['itemsets'])[0]
            print(f"  {item[:50]:50s} - Support: {row['support']:.4f}")
        
        return frequent_itemsets
    
    def generate_association_rules(self, metric='lift', min_threshold=1.0):
        """
        Generate association rules from frequent itemsets.
        
        Args:
            metric (str): Metric to use ('lift', 'confidence', 'support')
            min_threshold (float): Minimum threshold for the metric
        
        Returns:
            pd.DataFrame: Association rules
        """
        print("\n" + "="*60)
        print("GENERATING ASSOCIATION RULES")
        print("="*60)
        
        if self.frequent_itemsets is None:
            raise ValueError("Frequent itemsets not mined. Run mine_frequent_itemsets() first.")
        
        print(f"\nGenerating rules with {metric} >= {min_threshold}")
        
        # Generate association rules
        rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        if len(rules) == 0:
            print(f"\n⚠ No rules found with {metric} >= {min_threshold}")
            print("Trying with lower threshold...")
            rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold/2)
        
        # Add readable formats
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)[:2]))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)[:2]))
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        self.rules = rules
        
        print(f"\n✓ Generated {len(rules):,} association rules")
        print(f"\nRule Statistics:")
        print(f"  Support:     {rules['support'].min():.4f} to {rules['support'].max():.4f}")
        print(f"  Confidence:  {rules['confidence'].min():.4f} to {rules['confidence'].max():.4f}")
        print(f"  Lift:        {rules['lift'].min():.4f} to {rules['lift'].max():.4f}")
        
        return rules
    
    def display_top_rules(self, top_n=20):
        """
        Display top association rules.
        
        Args:
            top_n (int): Number of top rules to display
        """
        if self.rules is None or len(self.rules) == 0:
            print("\n⚠ No rules available to display")
            return
        
        print("\n" + "="*60)
        print(f"TOP {top_n} ASSOCIATION RULES (by Lift)")
        print("="*60)
        
        top_rules = self.rules.head(top_n)
        
        for idx, row in top_rules.iterrows():
            antecedents = ', '.join([str(x)[:30] for x in list(row['antecedents'])[:2]])
            consequents = ', '.join([str(x)[:30] for x in list(row['consequents'])[:2]])
            
            print(f"\n{idx+1}. IF: {antecedents}")
            print(f"   THEN: {consequents}")
            print(f"   Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}")
    
    def extract_product_bundles(self, top_n=15):
        """
        Extract top product bundles based on association rules.
        
        Args:
            top_n (int): Number of top bundles to extract
        
        Returns:
            pd.DataFrame: Top product bundles
        """
        print("\n" + "="*60)
        print(f"TOP {top_n} PRODUCT BUNDLES")
        print("="*60)
        
        if self.rules is None or len(self.rules) == 0:
            print("\n⚠ No rules available for bundling")
            return None
        
        # Filter rules with good metrics
        bundles = self.rules[
            (self.rules['confidence'] >= 0.3) & 
            (self.rules['lift'] >= 1.5)
        ].copy()
        
        bundles = bundles.sort_values('lift', ascending=False).head(top_n)
        
        print(f"\nFound {len(bundles)} high-quality bundles")
        print("\nTop Product Bundles:")
        print("-" * 100)
        
        for idx, row in bundles.head(10).iterrows():
            antecedents = ' + '.join([str(x)[:25] for x in list(row['antecedents'])])
            consequents = ' + '.join([str(x)[:25] for x in list(row['consequents'])])
            
            print(f"\nBundle {idx+1}:")
            print(f"  Buy: {antecedents}")
            print(f"  Get: {consequents}")
            print(f"  Lift: {row['lift']:.2f}x | Confidence: {row['confidence']*100:.1f}% | Support: {row['support']:.4f}")
        
        return bundles
    
    def visualize_rules(self):
        """Create comprehensive visualizations for association rules."""
        print("\n" + "="*60)
        print("CREATING MARKET BASKET VISUALIZATIONS")
        print("="*60)
        
        if self.rules is None or len(self.rules) == 0:
            print("\n⚠ No rules available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Support vs Confidence
        scatter = axes[0, 0].scatter(self.rules['support'], self.rules['confidence'], 
                                     c=self.rules['lift'], s=50, alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel('Support', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Support vs Confidence (colored by Lift)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Lift')
        
        # 2. Top Rules by Lift
        top_rules = self.rules.nlargest(15, 'lift')
        rule_labels = [f"{', '.join([str(x)[:15] for x in list(row['antecedents'])[:1]])} → "
                       f"{', '.join([str(x)[:15] for x in list(row['consequents'])[:1]])}"
                       for _, row in top_rules.iterrows()]
        
        axes[0, 1].barh(range(len(top_rules)), top_rules['lift'], color='steelblue', alpha=0.8)
        axes[0, 1].set_yticks(range(len(top_rules)))
        axes[0, 1].set_yticklabels(rule_labels, fontsize=8)
        axes[0, 1].set_xlabel('Lift', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Top 15 Rules by Lift', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Lift Distribution
        axes[1, 0].hist(self.rules['lift'], bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.rules['lift'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.rules["lift"].mean():.2f}')
        axes[1, 0].set_xlabel('Lift', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Lift Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Confidence Distribution
        axes[1, 1].hist(self.rules['confidence'], bins=50, color='seagreen', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(self.rules['confidence'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.rules["confidence"].mean():.2f}')
        axes[1, 1].set_xlabel('Confidence', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Market Basket Analysis - Association Rules', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig('outputs/market_basket_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/market_basket_analysis.png")
        plt.close()
    
    def save_results(self):
        """Save analysis results to CSV files."""
        if self.frequent_itemsets is not None:
            # Convert frozenset to string for CSV
            freq_save = self.frequent_itemsets.copy()
            freq_save['itemsets'] = freq_save['itemsets'].apply(lambda x: ', '.join(list(x)))
            freq_save.to_csv('outputs/frequent_itemsets.csv', index=False)
            print(f"\n✓ Frequent itemsets saved to: outputs/frequent_itemsets.csv")
        
        if self.rules is not None and len(self.rules) > 0:
            # Convert frozenset to string for CSV
            rules_save = self.rules.copy()
            rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_save['consequents'] = rules_save['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_save.to_csv('outputs/association_rules.csv', index=False)
            print(f"✓ Association rules saved to: outputs/association_rules.csv")


def main():
    """Main execution function."""
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize market basket analysis
    mba = MarketBasketAnalysis(df)
    
    # Execute analysis workflow
    mba.create_basket_format(min_support_items=50)
    mba.mine_frequent_itemsets(min_support=0.01)
    mba.generate_association_rules(metric='lift', min_threshold=1.0)
    mba.display_top_rules(top_n=20)
    mba.extract_product_bundles(top_n=15)
    mba.visualize_rules()
    mba.save_results()
    
    print("\n✓ Market basket analysis completed successfully!")
    
    return mba.rules


if __name__ == "__main__":
    rules = main()
