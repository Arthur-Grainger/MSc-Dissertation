import os
import pandas as pd
from typing import Dict, List
from functools import reduce

# Set working directory
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")

def process_claimant_count() -> pd.DataFrame:
    """Process claimant count data."""
    print("Processing Claimant Count Data...")
    
    df = pd.read_excel("claimant_count.xlsx", skiprows=5, usecols=[0, 1])
    df.columns = ['Date', 'claimant_count']
    df['Date'] = pd.to_datetime(df['Date'], format='%B %Y')
    df = df.dropna(subset=['Date'])
    
    print(f"✓ {len(df)} rows | Range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    return df

def process_epu_index() -> pd.DataFrame:
    """Process EPU Index data."""
    print("Processing EPU Index Data...")
    
    df = pd.read_excel('EPU_Index.xlsx')
    df = df[pd.to_numeric(df['year'], errors='coerce').notna()]
    df = df[pd.to_numeric(df['month'], errors='coerce').notna()]
    
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.drop(columns=['year', 'month'])
    
    print(f"✓ {len(df)} rows | Range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    return df

def process_ftse_data() -> pd.DataFrame:
    """Process FTSE-100 data to monthly returns."""
    print("Processing FTSE-100 Data...")
    
    df = pd.read_csv('FTSE 100 Historical Results Price Data.csv', usecols=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    
    # Convert to monthly returns
    df = df.set_index('Date').sort_index()
    monthly_prices = df['Price'].resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    result = pd.DataFrame({
        'Date': monthly_returns.index.to_period('M').to_timestamp(),
        'ftse_returns': monthly_returns.values
    }).reset_index(drop=True)
    
    print(f"✓ {len(result)} monthly obs | Mean return: {result['ftse_returns'].mean():.4f}")
    return result

def process_retail_sales() -> pd.DataFrame:
    """Process retail sales data."""
    print("Processing Retail Sales Data...")
    
    df = pd.read_excel(
        'retail sales.xlsx',
        sheet_name='CPSA3',
        skiprows=199,
        nrows=257,
        usecols=[0, 1],
        names=['Date', 'retail_sales_index']
    )
    
    # Remove rows with NaN dates (footer rows with revision notes etc.)
    df = df.dropna(subset=['Date'])
    
    # Convert dates with automatic parsing (more robust)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove any dates that couldn't be parsed
    df = df.dropna(subset=['Date'])
    
    print(f"✓ {len(df)} rows | Range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    return df

def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all datasets on Date column, keeping only complete observations."""
    print("\nCombining datasets...")
    
    # Start with first dataset
    dataset_list = list(datasets.values())
    combined = dataset_list[0]
    
    # Merge remaining datasets using inner joins (only keep dates with all data)
    for df in dataset_list[1:]:
        combined = pd.merge(combined, df, on='Date', how='inner')
    
    # Sort by date
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Combined dataset: {len(combined)} rows × {len(combined.columns)} columns")
    print(f"  Date range: {combined['Date'].min().strftime('%Y-%m')} to {combined['Date'].max().strftime('%Y-%m')}")
    
    # Check for any remaining missing values (should be none with inner join)
    missing_counts = combined.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"  Warning: {total_missing} missing values found after inner join:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"    {col}: {count}")
        
        # Drop rows with any missing values to ensure complete dataset
        combined = combined.dropna()
        print(f"  After dropping missing: {len(combined)} rows remain")
    else:
        print(f"  ✓ No missing values - all observations complete")
    
    return combined

def main() -> pd.DataFrame:
    """Main processing and combining function."""
    print("Processing Traditional Indicators Data")
    print("=" * 50)
    
    # Process all datasets
    datasets = {
        'claimant_count': process_claimant_count(),
        'epu_index': process_epu_index(),
        'ftse_returns': process_ftse_data(),
        'retail_sales': process_retail_sales()
    }
    
    # Combine all datasets
    combined_data = combine_datasets(datasets)
    
    # Save final combined dataset
    combined_data.to_excel('combined_traditional_indicators.xlsx', index=False)
    print(f"\n✓ Final combined dataset saved as 'combined_traditional_indicators.xlsx'")
    
    return combined_data

if __name__ == "__main__":
    final_dataset = main()
    
    # Optional: Quick preview
    print("\nPreview of combined data:")
    print(final_dataset.head())
    print(f"\nColumns: {list(final_dataset.columns)}")