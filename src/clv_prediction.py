

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class CLVPredictionModel:
    
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
        
    def engineer_features(self):
        print("\n" + "="*60)
        print("FEATURE ENGINEERING FOR CLV PREDICTION")
        print("="*60)
        
        reference_date = self.df['InvoiceDate'].max()
        
        # Customer-level features
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
        
        # RFM features
        features['Recency'] = features['DaysSinceLastPurchase']
        features['Frequency'] = features['TotalOrders']
        features['Monetary'] = features['TotalRevenue']
        
        # Derived features
        features['AvgOrderValue'] = features['TotalRevenue'] / features['TotalOrders']
        features['PurchaseFrequency'] = features['TotalOrders'] / (features['CustomerLifespan'] + 1)
        features['RevenuePerDay'] = features['TotalRevenue'] / (features['CustomerLifespan'] + 1)
        
        # Calculate actual CLV (target variable)
        features['CLV'] = features['AvgOrderValue'] * features['Frequency'] * 36  # 3-year projection
        
        # Fill NaN
        features['StdQuantity'].fillna(0, inplace=True)
        features['StdRevenue'].fillna(0, inplace=True)
        features['PurchaseFrequency'].fillna(0, inplace=True)
        features['RevenuePerDay'].fillna(0, inplace=True)
        
        # Encode categorical
        features['IsUK'] = (features['Country'] == 'United Kingdom').astype(int)
        
        # Remove outliers (top 1% CLV for stability)
        clv_99th = features['CLV'].quantile(0.99)
        features = features[features['CLV'] <= clv_99th]
        
        self.feature_df = features
        
        print(f"\n✓ Features engineered for {len(features):,} customers")
        print(f"✓ CLV Statistics:")
        print(f"  Mean:   ${features['CLV'].mean():,.2f}")
        print(f"  Median: ${features['CLV'].median():,.2f}")
        print(f"  Std:    ${features['CLV'].std():,.2f}")
        print(f"  Min:    ${features['CLV'].min():,.2f}")
        print(f"  Max:    ${features['CLV'].max():,.2f}")
        
        return features
    
    def prepare_data(self, test_size=0.2, random_state=42):
        print("\n" + "="*60)
        print("PREPARING TRAIN/TEST SPLIT")
        print("="*60)
        
        # Select features (exclude CLV and CustomerID)
        feature_cols = ['TotalOrders', 'TotalQuantity', 'AvgQuantity', 'StdQuantity',
                       'TotalRevenue', 'AvgRevenue', 'StdRevenue', 'MaxRevenue', 'MinRevenue',
                       'UniqueProducts', 'CustomerLifespan', 'DaysSinceLastPurchase',
                       'Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
                       'PurchaseFrequency', 'RevenuePerDay', 'IsUK']
        
        X = self.feature_df[feature_cols]
        y = self.feature_df['CLV']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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
        print("TRAINING REGRESSION MODELS")
        print("="*60)
        
        # 1. Linear Regression
        print("\n1. Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr
        print("   ✓ Trained")
        
        # 2. Ridge Regression
        print("\n2. Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(self.X_train_scaled, self.y_train)
        self.models['Ridge'] = ridge
        print("   ✓ Trained")
        
        # 3. Random Forest
        print("\n3. Training Random Forest Regressor...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("   ✓ Trained")
        
        # 4. Gradient Boosting
        print("\n4. Training Gradient Boosting Regressor...")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        print("   ✓ Trained")
        
        print(f"\n✓ Trained {len(self.models)} regression models successfully")
    
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
            if name in ['Linear Regression', 'Ridge']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            
            print(f"\nPerformance Metrics:")
            print(f"  R² Score:  {r2:.4f}")
            print(f"  RMSE:      ${rmse:,.2f}")
            print(f"  MAE:       ${mae:,.2f}")
            print(f"  MAPE:      {mape:.2f}%")
            
            results.append({
                'Model': name,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })
        
        # Compare models
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"\n{results_df.to_string(index=False)}")
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  R² Score: {results_df.iloc[0]['R²']:.4f}")
        
        return results_df
    
    def visualize_results(self):
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted (Best Model)
        best_name = list(self.models.keys())[-1]  # Gradient Boosting
        best_model = self.models[best_name]
        
        if best_name in ['Linear Regression', 'Ridge']:
            y_pred = best_model.predict(self.X_test_scaled)
        else:
            y_pred = best_model.predict(self.X_test)
        
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5, s=20)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual CLV ($)', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted CLV ($)', fontweight='bold')
        axes[0, 0].set_title(f'Actual vs Predicted CLV - {best_name}', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Residual Plot
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted CLV ($)', fontweight='bold')
        axes[0, 1].set_ylabel('Residuals ($)', fontweight='bold')
        axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf = self.models['Random Forest']
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-15:]
            
            axes[1, 0].barh(range(len(indices)), importances[indices], color='steelblue')
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([self.X_train.columns[i] for i in indices], fontsize=9)
            axes[1, 0].set_xlabel('Importance', fontweight='bold')
            axes[1, 0].set_title('Top 15 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Model Performance Comparison
        metrics = []
        for name, model in self.models.items():
            if name in ['Linear Regression', 'Ridge']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            r2 = r2_score(self.y_test, y_pred)
            metrics.append({'Model': name, 'R² Score': r2})
        
        metrics_df = pd.DataFrame(metrics).sort_values('R² Score', ascending=True)
        
        axes[1, 1].barh(metrics_df['Model'], metrics_df['R² Score'], color='steelblue')
        axes[1, 1].set_xlabel('R² Score', fontweight='bold')
        axes[1, 1].set_title('Model Performance Comparison (R² Score)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig('outputs/clv_prediction_model.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: outputs/clv_prediction_model.png")
        plt.close()
    
    def save_predictions(self):
        predictions = self.feature_df[['CustomerID', 'CLV']].copy()
        predictions.rename(columns={'CLV': 'ActualCLV'}, inplace=True)
        
        # Get predictions from best model
        if 'Gradient Boosting' in self.models:
            model = self.models['Gradient Boosting']
            feature_cols = self.X_train.columns
            X_all = self.feature_df[feature_cols]
            predictions['PredictedCLV'] = model.predict(X_all)
            predictions['PredictionError'] = predictions['ActualCLV'] - predictions['PredictedCLV']
        
        predictions.to_csv('outputs/clv_predictions_ml.csv', index=False)
        print(f"\n✓ Predictions saved: outputs/clv_predictions_ml.csv")


def main():
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('data/online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Initialize model
    clv_model = CLVPredictionModel(df)
    
    # Execute ML pipeline
    clv_model.engineer_features()
    clv_model.prepare_data(test_size=0.2, random_state=42)
    clv_model.train_models()
    results = clv_model.evaluate_models()
    clv_model.visualize_results()
    clv_model.save_predictions()
    
    print("\n✓ CLV prediction model completed successfully!")
    
    return clv_model


if __name__ == "__main__":
    model = main()
