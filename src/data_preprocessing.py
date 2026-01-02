"""
Data Preprocessing Module for Retail Analysis
Author: Senior Data Scientist
Date: January 2026

This module handles data cleaning and preprocessing for the Online Retail II dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RetailDataPreprocessor:
    """
    A class to preprocess retail transaction data.
    
    This class handles data cleaning including:
    - Removing canceled invoices
    - Handling missing values
    - Filtering invalid records
    - Feature engineering
    """
    
    def __init__(self, filepath):
        """
        Initialize the preprocessor with the dataset filepath.
        
        Args:
            filepath (str): Path to the CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load the dataset from CSV file."""
        try:
            print("Loading dataset...")
            self.df = pd.read_csv(self.filepath, encoding='ISO-8859-1')
            print(f"✓ Dataset loaded successfully: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            raise
    
    def display_info(self):
        """Display basic information about the dataset."""
        if self.df is None:
            print("✗ No data loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        print(f"\nShape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nSample Data:\n{self.df.head()}")
        
    def clean_data(self):
        """
        Clean the dataset according to business rules.
        
        Steps:
        1. Remove canceled invoices (InvoiceNo starting with 'C')
        2. Remove rows with missing CustomerID
        3. Remove negative or zero Quantity
        4. Remove negative or zero UnitPrice
        5. Convert InvoiceDate to datetime
        6. Create TotalPrice feature
        7. Remove outliers
        """
        if self.df is None:
            print("✗ No data loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATA CLEANING PROCESS")
        print("="*60)
        
        # Start with a copy
        df_clean = self.df.copy()
        initial_rows = len(df_clean)
        
        # 1. Remove canceled invoices
        print(f"\n1. Removing canceled invoices (InvoiceNo starting with 'C')...")
        before = len(df_clean)
        df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
        after = len(df_clean)
        print(f"   Removed {before - after:,} canceled transactions")
        
        # 2. Remove missing CustomerID
        print(f"\n2. Removing rows with missing CustomerID...")
        before = len(df_clean)
        df_clean = df_clean[df_clean['Customer ID'].notna()]
        after = len(df_clean)
        print(f"   Removed {before - after:,} rows with missing CustomerID")
        
        # 3. Remove negative or zero Quantity
        print(f"\n3. Removing negative or zero Quantity...")
        before = len(df_clean)
        df_clean = df_clean[df_clean['Quantity'] > 0]
        after = len(df_clean)
        print(f"   Removed {before - after:,} rows")
        
        # 4. Remove negative or zero UnitPrice
        print(f"\n4. Removing negative or zero UnitPrice...")
        before = len(df_clean)
        df_clean = df_clean[df_clean['Price'] > 0]
        after = len(df_clean)
        print(f"   Removed {before - after:,} rows")
        
        # 5. Convert InvoiceDate to datetime
        print(f"\n5. Converting InvoiceDate to datetime...")
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        print(f"   ✓ Conversion successful")
        
        # 6. Create TotalPrice feature
        print(f"\n6. Creating TotalPrice feature...")
        df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']
        print(f"   ✓ TotalPrice = Quantity × Price")
        
        # 7. Remove extreme outliers in TotalPrice
        print(f"\n7. Removing extreme outliers...")
        Q1 = df_clean['TotalPrice'].quantile(0.01)
        Q3 = df_clean['TotalPrice'].quantile(0.99)
        before = len(df_clean)
        df_clean = df_clean[(df_clean['TotalPrice'] >= Q1) & (df_clean['TotalPrice'] <= Q3)]
        after = len(df_clean)
        print(f"   Removed {before - after:,} extreme outliers")
        
        # Additional feature engineering
        print(f"\n8. Engineering additional features...")
        df_clean['Year'] = df_clean['InvoiceDate'].dt.year
        df_clean['Month'] = df_clean['InvoiceDate'].dt.month
        df_clean['Quarter'] = df_clean['InvoiceDate'].dt.quarter
        df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
        df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
        print(f"   ✓ Created temporal features: Year, Month, Quarter, DayOfWeek, Hour")
        
        # Clean CustomerID (ensure it's integer)
        df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
        
        self.df_clean = df_clean
        
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"Initial rows:     {initial_rows:,}")
        print(f"Final rows:       {len(df_clean):,}")
        print(f"Rows removed:     {initial_rows - len(df_clean):,}")
        print(f"Retention rate:   {(len(df_clean)/initial_rows)*100:.2f}%")
        print(f"\nDate range:       {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")
        print(f"Unique customers: {df_clean['Customer ID'].nunique():,}")
        print(f"Unique products:  {df_clean['StockCode'].nunique():,}")
        print(f"Unique countries: {df_clean['Country'].nunique()}")
        
        return self.df_clean
    
    def save_cleaned_data(self, output_path='data/online_retail_cleaned.csv'):
        """
        Save the cleaned dataset to a CSV file.
        
        Args:
            output_path (str): Path where the cleaned data will be saved
        """
        if self.df_clean is None:
            print("✗ No cleaned data available. Please run clean_data() first.")
            return
        
        try:
            self.df_clean.to_csv(output_path, index=False)
            print(f"\n✓ Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving cleaned data: {e}")
    
    def get_cleaned_data(self):
        """Return the cleaned dataframe."""
        return self.df_clean


def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = RetailDataPreprocessor('data/online_retail.csv')
    
    # Load data
    preprocessor.load_data()
    
    # Display info
    preprocessor.display_info()
    
    # Clean data
    df_clean = preprocessor.clean_data()
    
    # Save cleaned data
    preprocessor.save_cleaned_data('data/online_retail_cleaned.csv')
    
    print("\n✓ Data preprocessing completed successfully!")
    
    return df_clean


if __name__ == "__main__":
    df_cleaned = main()
