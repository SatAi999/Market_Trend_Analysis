

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, accuracy_score)
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictionModel:
    
    def __init__(self, df):
        self.df = df
        self.feature_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def engineer_features(self, churn_threshold_days=180):
        print("\n" + "="*60)
        print("FEATURE ENGINEERING FOR CHURN PREDICTION")
        print("="*60)
        
        reference_date = self.df['InvoiceDate'].max()
        
        # Customer-level aggregations
        features = self.df.groupby('Customer ID').agg({
            'Invoice': 'nunique',
            'InvoiceDate': ['min', 'max'],
            'Quantity': ['sum', 'mean', 'std'],
            'TotalPrice': ['sum', 'mean', 'std', 'max', 'min'],
            'StockCode': 'nunique',
            'Country': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        features.columns = ['CustomerID', 'TotalOrders', 'FirstPurchase', 'LastPurchase',
                           'TotalQuantity', 'AvgQuantity', 'StdQuantity',
                           'TotalRevenue', 'AvgRevenue', 'StdRevenue', 'MaxRevenue', 'MinRevenue',
                           'UniqueProducts', 'Country']
        
        # Temporal features
        features['CustomerLifespan'] = (features['LastPurchase'] - features['FirstPurchase']).dt.days
        features['DaysSinceLastPurchase'] = (reference_date - features['LastPurchase']).dt.days
        features['DaysSinceFirstPurchase'] = (reference_date - features['FirstPurchase']).dt.days
        
        # RFM features
        features['Recency'] = features['DaysSinceLastPurchase']
        features['Frequency'] = features['TotalOrders']
        features['Monetary'] = features['TotalRevenue']
        
        # Derived features
        features['AvgOrderValue'] = features['TotalRevenue'] / features['TotalOrders']
        features['PurchaseFrequency'] = features['TotalOrders'] / (features['CustomerLifespan'] + 1)
        features['AvgDaysBetweenOrders'] = features['CustomerLifespan'] / (features['TotalOrders'] + 1)
        
        # Fill NaN values
        features['StdQuantity'].fillna(0, inplace=True)
        features['StdRevenue'].fillna(0, inplace=True)
        features['PurchaseFrequency'].fillna(0, inplace=True)
        
        # Target variable: Churn (1 if inactive > threshold, 0 otherwise)
        features['Churn'] = (features['DaysSinceLastPurchase'] > churn_threshold_days).astype(int)
        
        # Encode categorical
        features['IsUK'] = (features['Country'] == 'United Kingdom').astype(int)
        
        self.feature_df = features
        
        print(f"\n✓ Features engineered for {len(features):,} customers")
        print(f"✓ Total features: {len(features.columns)}")
        print(f"\nChurn Distribution:")
        churn_counts = features['Churn'].value_counts()
        print(f"  Not Churned (0): {churn_counts[0]:,} ({churn_counts[0]/len(features)*100:.1f}%)")
        print(f"  Churned (1):     {churn_counts[1]:,} ({churn_counts[1]/len(features)*100:.1f}%)")
        
        return features
    
    def prepare_data(self, test_size=0.2, random_state=42):
        print("\n" + "="*60)
        print("PREPARING TRAIN/TEST SPLIT")
        print("="*60)
        
        # Select features
        feature_cols = ['TotalOrders', 'TotalQuantity', 'AvgQuantity', 'StdQuantity',
                       'TotalRevenue', 'AvgRevenue', 'StdRevenue', 'MaxRevenue', 'MinRevenue',
                       'UniqueProducts', 'CustomerLifespan', 'DaysSinceLastPurchase',
                       'Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
                       'PurchaseFrequency', 'AvgDaysBetweenOrders', 'IsUK']
        
        X = self.feature_df[feature_cols]
        y = self.feature_df['Churn']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Train set: {len(self.X_train):,} samples")
        print(f"✓ Test set:  {len(self.X_test):,} samples")
        print(f"✓ Features:  {len(feature_cols)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr
        print("   ✓ Trained")
        
        # 2. Random Forest
        print("\n2. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("   ✓ Trained")
        
        # 3. Gradient Boosting
        print("\n3. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        print("   ✓ Trained")
        
        print(f"\n✓ Trained {len(self.models)} models successfully")
    
    def evaluate_models(self):
        print("\n" + "="*60)
        print("MODEL EVALUATION & COMPARISON")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"{'='*60}")
            
            # Predictions
            if name == 'Logistic Regression':
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Not Churned', 'Churned']))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
        
        # Compare models
        results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"\n{results_df.to_string(index=False)}")
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
        
        return results_df
    
    def visualize_results(self):
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate', fontweight='bold')
        axes[0, 0].set_ylabel('True Positive Rate', fontweight='bold')
        axes[0, 0].set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf = self.models['Random Forest']
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-15:]
            
            axes[0, 1].barh(range(len(indices)), importances[indices], color='steelblue')
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([self.X_train.columns[i] for i in indices], fontsize=9)
            axes[0, 1].set_xlabel('Importance', fontweight='bold')
            axes[0, 1].set_title('Top 15 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Confusion Matrix Heatmap (Best Model)
        best_name = list(self.models.keys())[0]
        best_model = self.models[best_name]
        
        if best_name == 'Logistic Regression':
            y_pred = best_model.predict(self.X_test_scaled)
        else:
            y_pred = best_model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        axes[1, 0].set_ylabel('True Label', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Label', fontweight='bold')
        axes[1, 0].set_title(f'Confusion Matrix - {best_name}', fontsize=12, fontweight='bold')
        
        # 4. Model Performance Comparison
        metrics = []
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            metrics.append({
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
            })
        
        metrics_df = pd.DataFrame(metrics)
        x = np.arange(len(metrics_df))
        width = 0.25
        
        axes[1, 1].bar(x - width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        axes[1, 1].bar(x, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        axes[1, 1].bar(x + width, metrics_df['ROC-AUC'], width, label='ROC-AUC', alpha=0.8)
        
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
        axes[1, 1].set_ylabel('Score', fontweight='bold')
        axes[1, 1].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('outputs/churn_prediction_model.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/churn_prediction_model.png")
        plt.close()
    
    def save_predictions(self):
        # Get predictions for all customers
        predictions = self.feature_df[['CustomerID']].copy()
        
        if 'Random Forest' in self.models:
            model = self.models['Random Forest']
            feature_cols = self.X_train.columns
            X_all = self.feature_df[feature_cols]
            predictions['ChurnProbability'] = model.predict_proba(X_all)[:, 1]
            predictions['ChurnPrediction'] = model.predict(X_all)
        
        predictions.to_csv('outputs/churn_predictions.csv', index=False)
        print(f"\n✓ Predictions saved: outputs/churn_predictions.csv")


def main():
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize model
    churn_model = ChurnPredictionModel(df)
    
    # Execute ML pipeline
    churn_model.engineer_features(churn_threshold_days=180)
    churn_model.prepare_data(test_size=0.2, random_state=42)
    churn_model.train_models()
    results = churn_model.evaluate_models()
    churn_model.visualize_results()
    churn_model.save_predictions()
    
    print("\n✓ Churn prediction model completed successfully!")
    
    return churn_model


if __name__ == "__main__":
    model = main()
