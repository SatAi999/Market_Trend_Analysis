

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    
    def __init__(self, df):
        self.df = df
        self.rfm_df = None
        self.rfm_scaled = None
        self.optimal_k = None
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def calculate_rfm(self, reference_date=None):
        print("\n" + "="*60)
        print("RFM CALCULATION")
        print("="*60)
        
        # Use max date + 1 day as reference if not provided
        if reference_date is None:
            reference_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        print(f"Reference Date: {reference_date}")
        
        # Calculate RFM metrics
        rfm = self.df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'Invoice': 'nunique',  # Frequency
            'TotalPrice': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Remove any outliers
        for col in ['Recency', 'Frequency', 'Monetary']:
            Q1 = rfm[col].quantile(0.05)
            Q3 = rfm[col].quantile(0.95)
            rfm = rfm[(rfm[col] >= Q1) & (rfm[col] <= Q3)]
        
        self.rfm_df = rfm
        
        print(f"\n✓ RFM calculated for {len(rfm):,} customers")
        print(f"\nRFM Summary Statistics:")
        print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
        
        return rfm
    
    def scale_features(self):
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)
        
        if self.rfm_df is None:
            raise ValueError("RFM not calculated. Run calculate_rfm() first.")
        
        # Scale the RFM features
        self.rfm_scaled = self.scaler.fit_transform(
            self.rfm_df[['Recency', 'Frequency', 'Monetary']]
        )
        
        print("✓ Features scaled using StandardScaler")
        print(f"  Scaled shape: {self.rfm_scaled.shape}")
        
        return self.rfm_scaled
    
    def find_optimal_clusters(self, max_k=10):
        print("\n" + "="*60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60)
        
        if self.rfm_scaled is None:
            raise ValueError("Features not scaled. Run scale_features() first.")
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        print(f"\nTesting K from 2 to {max_k}...")
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.rfm_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.rfm_scaled, kmeans.labels_))
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='navy')
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Inertia', fontsize=11, fontweight='bold')
        axes[0].set_title('Elbow Method - Inertia vs K', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(K_range)
        
        # Silhouette plot
        axes[1].plot(K_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='darkgreen')
        axes[1].set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        axes[1].set_title('Silhouette Score vs K', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(K_range)
        
        # Mark optimal K (highest silhouette score)
        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_k = K_range[optimal_idx]
        axes[1].axvline(x=self.optimal_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Optimal K={self.optimal_k}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/optimal_clusters.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/optimal_clusters.png")
        plt.close()
        
        print(f"\n✓ Optimal number of clusters: {self.optimal_k}")
        
        return self.optimal_k
    
    def perform_clustering(self, n_clusters=None):
        print("\n" + "="*60)
        print("PERFORMING K-MEANS CLUSTERING")
        print("="*60)
        
        if self.rfm_scaled is None:
            raise ValueError("Features not scaled. Run scale_features() first.")
        
        if n_clusters is None:
            if self.optimal_k is None:
                print("Finding optimal clusters first...")
                self.find_optimal_clusters()
            n_clusters = self.optimal_k
        
        print(f"\nClustering with K={n_clusters}...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df['Cluster'] = self.kmeans.fit_predict(self.rfm_scaled)
        
        print(f"✓ Clustering completed")
        print(f"\nCluster Distribution:")
        cluster_counts = self.rfm_df['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(self.rfm_df)) * 100
            print(f"  Cluster {cluster}: {count:,} customers ({percentage:.1f}%)")
        
        return self.rfm_df['Cluster'].values
    
    def profile_segments(self):
        print("\n" + "="*60)
        print("CUSTOMER SEGMENT PROFILING")
        print("="*60)
        
        if 'Cluster' not in self.rfm_df.columns:
            raise ValueError("Clustering not performed. Run perform_clustering() first.")
        
        # Calculate segment statistics
        segment_profile = self.rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).reset_index()
        
        segment_profile.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 
                                   'Avg_Monetary', 'Customer_Count']
        
        # Assign segment labels based on RFM characteristics
        def assign_label(row):
            if row['Avg_Recency'] < 50 and row['Avg_Monetary'] > segment_profile['Avg_Monetary'].median():
                return 'Champions'
            elif row['Avg_Frequency'] > segment_profile['Avg_Frequency'].median() and \
                 row['Avg_Monetary'] > segment_profile['Avg_Monetary'].median():
                return 'Loyal Customers'
            elif row['Avg_Recency'] < segment_profile['Avg_Recency'].median() and \
                 row['Avg_Monetary'] < segment_profile['Avg_Monetary'].median():
                return 'Potential Loyalists'
            elif row['Avg_Recency'] > segment_profile['Avg_Recency'].median() and \
                 row['Avg_Monetary'] > segment_profile['Avg_Monetary'].median():
                return 'At Risk'
            elif row['Avg_Recency'] > segment_profile['Avg_Recency'].median():
                return 'Lost Customers'
            else:
                return 'Needs Attention'
        
        segment_profile['Segment_Label'] = segment_profile.apply(assign_label, axis=1)
        
        # Add to rfm_df
        label_map = dict(zip(segment_profile['Cluster'], segment_profile['Segment_Label']))
        self.rfm_df['Segment'] = self.rfm_df['Cluster'].map(label_map)
        
        print("\nSegment Profiles:")
        print("="*100)
        for idx, row in segment_profile.iterrows():
            print(f"\n{row['Segment_Label']} (Cluster {row['Cluster']}):")
            print(f"  Customers:       {row['Customer_Count']:,}")
            print(f"  Avg Recency:     {row['Avg_Recency']:.1f} days")
            print(f"  Avg Frequency:   {row['Avg_Frequency']:.1f} orders")
            print(f"  Avg Monetary:    ${row['Avg_Monetary']:,.2f}")
        
        return segment_profile
    
    def visualize_segments(self):
        print("\n" + "="*60)
        print("CREATING SEGMENT VISUALIZATIONS")
        print("="*60)
        
        if 'Segment' not in self.rfm_df.columns:
            raise ValueError("Segments not profiled. Run profile_segments() first.")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 3D Scatter Plot
        ax1 = fig.add_subplot(gs[:2, :2], projection='3d')
        for cluster in sorted(self.rfm_df['Cluster'].unique()):
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            ax1.scatter(cluster_data['Recency'], 
                       cluster_data['Frequency'], 
                       cluster_data['Monetary'],
                       label=cluster_data['Segment'].iloc[0],
                       s=50, alpha=0.6)
        ax1.set_xlabel('Recency (days)', fontweight='bold')
        ax1.set_ylabel('Frequency (orders)', fontweight='bold')
        ax1.set_zlabel('Monetary ($)', fontweight='bold')
        ax1.set_title('Customer Segments in 3D RFM Space', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        
        # 2. Segment Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        segment_counts = self.rfm_df['Segment'].value_counts()
        colors = plt.cm.Set3(range(len(segment_counts)))
        ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Segment Distribution', fontsize=11, fontweight='bold')
        
        # 3. Monetary by Segment
        ax3 = fig.add_subplot(gs[1, 2])
        segment_monetary = self.rfm_df.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
        ax3.barh(range(len(segment_monetary)), segment_monetary.values, color='steelblue')
        ax3.set_yticks(range(len(segment_monetary)))
        ax3.set_yticklabels(segment_monetary.index, fontsize=9)
        ax3.set_xlabel('Avg Monetary ($)', fontsize=9, fontweight='bold')
        ax3.set_title('Average Spending by Segment', fontsize=10, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Recency vs Frequency
        ax4 = fig.add_subplot(gs[2, 0])
        for cluster in sorted(self.rfm_df['Cluster'].unique()):
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            ax4.scatter(cluster_data['Recency'], cluster_data['Frequency'],
                       label=cluster_data['Segment'].iloc[0], alpha=0.6, s=30)
        ax4.set_xlabel('Recency (days)', fontsize=9, fontweight='bold')
        ax4.set_ylabel('Frequency (orders)', fontsize=9, fontweight='bold')
        ax4.set_title('Recency vs Frequency', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
        
        # 5. Frequency vs Monetary
        ax5 = fig.add_subplot(gs[2, 1])
        for cluster in sorted(self.rfm_df['Cluster'].unique()):
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            ax5.scatter(cluster_data['Frequency'], cluster_data['Monetary'],
                       label=cluster_data['Segment'].iloc[0], alpha=0.6, s=30)
        ax5.set_xlabel('Frequency (orders)', fontsize=9, fontweight='bold')
        ax5.set_ylabel('Monetary ($)', fontsize=9, fontweight='bold')
        ax5.set_title('Frequency vs Monetary', fontsize=10, fontweight='bold')
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)
        
        # 6. Customer Count by Segment
        ax6 = fig.add_subplot(gs[2, 2])
        segment_counts = self.rfm_df['Segment'].value_counts().sort_values(ascending=False)
        ax6.bar(range(len(segment_counts)), segment_counts.values, color='coral', alpha=0.8)
        ax6.set_xticks(range(len(segment_counts)))
        ax6.set_xticklabels(segment_counts.index, rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Customer Count', fontsize=9, fontweight='bold')
        ax6.set_title('Customers per Segment', fontsize=10, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('outputs/customer_segments.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/customer_segments.png")
        plt.close()
    
    def save_results(self, output_path='outputs/customer_segments.csv'):
        if self.rfm_df is None:
            raise ValueError("No results to save.")
        
        self.rfm_df.to_csv(output_path, index=False)
        print(f"\n✓ Segmentation results saved to: {output_path}")


def main():
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize segmentation
    segmentation = CustomerSegmentation(df)
    
    # Execute segmentation workflow
    segmentation.calculate_rfm()
    segmentation.scale_features()
    segmentation.find_optimal_clusters(max_k=8)
    segmentation.perform_clustering()
    segment_profile = segmentation.profile_segments()
    segmentation.visualize_segments()
    segmentation.save_results()
    
    print("\n✓ Customer segmentation completed successfully!")
    
    return segmentation.rfm_df


if __name__ == "__main__":
    rfm_results = main()
