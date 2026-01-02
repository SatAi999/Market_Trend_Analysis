"""
Machine Learning Model Performance Evaluation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING MODEL PERFORMANCE".center(80))
print("="*80)

# Load data
rfm = pd.read_csv('outputs/customer_segments.csv')
rules = pd.read_csv('outputs/association_rules.csv')

# ============================================================================
# 1. K-MEANS CLUSTERING PERFORMANCE
# ============================================================================
print("\n🔹 K-MEANS CLUSTERING MODEL")
print("-" * 80)

# Prepare data
X = rfm[['Recency', 'Frequency', 'Monetary']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
labels = rfm['Cluster'].values

# Calculate performance metrics
silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)
inertia = kmeans.inertia_

print(f"\n📊 Clustering Quality Metrics:")
print(f"  Silhouette Score:          {silhouette:.4f}")
print(f"    ├─ Range: [-1, 1]")
print(f"    ├─ Interpretation: {silhouette:.4f} = {'Excellent' if silhouette > 0.7 else 'Good' if silhouette > 0.5 else 'Fair' if silhouette > 0.25 else 'Poor'}")
print(f"    └─ Cluster separation is {'strong' if silhouette > 0.4 else 'moderate'}")
print(f"\n  Davies-Bouldin Index:      {davies_bouldin:.4f}")
print(f"    ├─ Range: [0, ∞] (lower is better)")
print(f"    ├─ Interpretation: {davies_bouldin:.4f} = {'Excellent' if davies_bouldin < 0.5 else 'Good' if davies_bouldin < 1.0 else 'Fair' if davies_bouldin < 1.5 else 'Needs improvement'}")
print(f"    └─ Cluster compactness is {'excellent' if davies_bouldin < 1.0 else 'good'}")
print(f"\n  Calinski-Harabasz Score:   {calinski:.2f}")
print(f"    ├─ Range: [0, ∞] (higher is better)")
print(f"    ├─ Interpretation: {calinski:.2f} = {'Excellent' if calinski > 1000 else 'Good' if calinski > 500 else 'Fair'}")
print(f"    └─ Variance ratio is {'excellent' if calinski > 1000 else 'good'}")
print(f"\n  Inertia (Within-SS):       {inertia:.2f}")
print(f"    └─ Sum of squared distances to centroids")

# Cluster statistics
print(f"\n📈 Cluster Distribution:")
for cluster in sorted(rfm['Cluster'].unique()):
    count = len(rfm[rfm['Cluster'] == cluster])
    pct = count / len(rfm) * 100
    segment = rfm[rfm['Cluster'] == cluster]['Segment'].iloc[0]
    print(f"  Cluster {cluster} ({segment:20s}): {count:,} samples ({pct:.1f}%)")

# Feature importance in clustering
print(f"\n📊 Feature Statistics by Cluster:")
print(f"{'Cluster':<10} {'Recency':>12} {'Frequency':>12} {'Monetary':>15}")
print("-" * 52)
for cluster in sorted(rfm['Cluster'].unique()):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    r = cluster_data['Recency'].mean()
    f = cluster_data['Frequency'].mean()
    m = cluster_data['Monetary'].mean()
    print(f"{cluster:<10} {r:>12.1f} {f:>12.1f} ${m:>14,.2f}")

# Separation metrics
print(f"\n🎯 Cluster Separation Analysis:")
cluster0 = rfm[rfm['Cluster'] == 0][['Recency', 'Frequency', 'Monetary']].mean()
cluster1 = rfm[rfm['Cluster'] == 1][['Recency', 'Frequency', 'Monetary']].mean()
print(f"  Mean Recency Difference:   {abs(cluster0['Recency'] - cluster1['Recency']):.1f} days")
print(f"  Mean Frequency Difference: {abs(cluster0['Frequency'] - cluster1['Frequency']):.1f} orders")
print(f"  Mean Monetary Difference:  ${abs(cluster0['Monetary'] - cluster1['Monetary']):,.2f}")

# ============================================================================
# 2. ASSOCIATION RULES PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("🔹 ASSOCIATION RULES MODEL (APRIORI)")
print("-" * 80)

print(f"\n📊 Rule Generation Metrics:")
print(f"  Total Rules Generated:     {len(rules):,}")
print(f"  Unique Antecedents:        {rules['antecedents'].nunique():,}")
print(f"  Unique Consequents:        {rules['consequents'].nunique():,}")

print(f"\n📈 Support Metrics:")
print(f"  Mean Support:              {rules['support'].mean():.4f}")
print(f"  Median Support:            {rules['support'].median():.4f}")
print(f"  Min Support:               {rules['support'].min():.4f}")
print(f"  Max Support:               {rules['support'].max():.4f}")
print(f"  Std Dev:                   {rules['support'].std():.4f}")

print(f"\n📈 Confidence Metrics:")
print(f"  Mean Confidence:           {rules['confidence'].mean():.4f} ({rules['confidence'].mean()*100:.1f}%)")
print(f"  Median Confidence:         {rules['confidence'].median():.4f} ({rules['confidence'].median()*100:.1f}%)")
print(f"  Min Confidence:            {rules['confidence'].min():.4f} ({rules['confidence'].min()*100:.1f}%)")
print(f"  Max Confidence:            {rules['confidence'].max():.4f} ({rules['confidence'].max()*100:.1f}%)")
print(f"  Std Dev:                   {rules['confidence'].std():.4f}")

print(f"\n📈 Lift Metrics:")
print(f"  Mean Lift:                 {rules['lift'].mean():.2f}x")
print(f"  Median Lift:               {rules['lift'].median():.2f}x")
print(f"  Min Lift:                  {rules['lift'].min():.2f}x")
print(f"  Max Lift:                  {rules['lift'].max():.2f}x")
print(f"  Std Dev:                   {rules['lift'].std():.2f}")

print(f"\n🎯 Rule Quality Distribution:")
excellent = len(rules[rules['lift'] > 10])
good = len(rules[(rules['lift'] > 3) & (rules['lift'] <= 10)])
fair = len(rules[(rules['lift'] > 1.5) & (rules['lift'] <= 3)])
weak = len(rules[rules['lift'] <= 1.5])
print(f"  Excellent (Lift > 10):     {excellent:,} rules ({excellent/len(rules)*100:.1f}%)")
print(f"  Good (Lift 3-10):          {good:,} rules ({good/len(rules)*100:.1f}%)")
print(f"  Fair (Lift 1.5-3):         {fair:,} rules ({fair/len(rules)*100:.1f}%)")
print(f"  Weak (Lift ≤ 1.5):         {weak:,} rules ({weak/len(rules)*100:.1f}%)")

print(f"\n🎯 Confidence Quality Distribution:")
very_high = len(rules[rules['confidence'] >= 0.8])
high = len(rules[(rules['confidence'] >= 0.6) & (rules['confidence'] < 0.8)])
medium = len(rules[(rules['confidence'] >= 0.4) & (rules['confidence'] < 0.6)])
low = len(rules[rules['confidence'] < 0.4])
print(f"  Very High (≥80%):          {very_high:,} rules ({very_high/len(rules)*100:.1f}%)")
print(f"  High (60-80%):             {high:,} rules ({high/len(rules)*100:.1f}%)")
print(f"  Medium (40-60%):           {medium:,} rules ({medium/len(rules)*100:.1f}%)")
print(f"  Low (<40%):                {low:,} rules ({low/len(rules)*100:.1f}%)")

# Actionable rules
actionable = len(rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.5)])
print(f"\n💡 Actionable Rules:")
print(f"  (Confidence ≥30% & Lift ≥1.5): {actionable:,} rules ({actionable/len(rules)*100:.1f}%)")

# ============================================================================
# 3. OVERALL MODEL PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✨ OVERALL MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print(f"\n🏆 Key Performance Indicators:")
print(f"\n  K-Means Clustering:")
print(f"    ✓ Silhouette Score:        {silhouette:.4f} ({'GOOD' if silhouette > 0.4 else 'FAIR'})")
print(f"    ✓ Davies-Bouldin:          {davies_bouldin:.4f} ({'GOOD' if davies_bouldin < 1.0 else 'FAIR'})")
print(f"    ✓ Calinski-Harabasz:       {calinski:.2f} ({'EXCELLENT' if calinski > 1000 else 'GOOD'})")
print(f"    ✓ Model Quality:           {'✅ PRODUCTION-READY' if silhouette > 0.4 else '⚠️ NEEDS TUNING'}")

print(f"\n  Association Rules:")
print(f"    ✓ Rules Generated:         {len(rules):,}")
print(f"    ✓ High-Quality Rules:      {excellent + good:,} ({(excellent+good)/len(rules)*100:.1f}%)")
print(f"    ✓ Max Lift Achieved:       {rules['lift'].max():.2f}x ({'EXCELLENT' if rules['lift'].max() > 10 else 'GOOD'})")
print(f"    ✓ Mean Confidence:         {rules['confidence'].mean()*100:.1f}% ({'GOOD' if rules['confidence'].mean() > 0.35 else 'FAIR'})")
print(f"    ✓ Model Quality:           {'✅ PRODUCTION-READY' if rules['lift'].max() > 10 else '⚠️ NEEDS TUNING'}")

print(f"\n🎯 Model Recommendations:")
if silhouette > 0.4 and rules['lift'].max() > 10:
    print(f"  ✅ Both models show GOOD performance")
    print(f"  ✅ Ready for production deployment")
    print(f"  ✅ High-confidence actionable insights available")
else:
    print(f"  ⚠️ Consider parameter tuning for optimization")

print("\n" + "="*80)
