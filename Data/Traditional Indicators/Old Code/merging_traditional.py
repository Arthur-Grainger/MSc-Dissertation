import os
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")
import pandas as pd

# Load all four datasets
claimant_df = pd.read_excel('cleaned_claimant_count.xlsx')
epu_df = pd.read_excel('EPU_Index_with_date.xlsx')
ftse_df = pd.read_excel('ftse_monthly_returns.xlsx')
retail_df = pd.read_excel('cleaned_retail_sales.xlsx')

# Ensure Date columns are in datetime format
claimant_df['Date'] = pd.to_datetime(claimant_df['Date'])
epu_df['Date'] = pd.to_datetime(epu_df['Date'])
ftse_df['Date'] = pd.to_datetime(ftse_df['Date'])
retail_df['Date'] = pd.to_datetime(retail_df['Date'])

# Check the date ranges before merging
print("Claimant Count date range:")
print(f"  Start: {claimant_df['Date'].min()}")
print(f"  End: {claimant_df['Date'].max()}")
print(f"  Number of observations: {len(claimant_df)}")

print("\nEPU Index date range:")
print(f"  Start: {epu_df['Date'].min()}")
print(f"  End: {epu_df['Date'].max()}")
print(f"  Number of observations: {len(epu_df)}")

print("\nFTSE 100 Returns date range:")
print(f"  Start: {ftse_df['Date'].min()}")
print(f"  End: {ftse_df['Date'].max()}")
print(f"  Number of observations: {len(ftse_df)}")

print("\nRetail Sales Index date range:")
print(f"  Start: {retail_df['Date'].min()}")
print(f"  End: {retail_df['Date'].max()}")
print(f"  Number of observations: {len(retail_df)}")

# Merge all four datasets on Date (inner join keeps only matching dates)
# First merge claimant and EPU
merged_df = pd.merge(
    claimant_df,
    epu_df,
    on='Date',
    how='inner',
    validate='one_to_one'
)

# Then merge with FTSE data
merged_df = pd.merge(
    merged_df,
    ftse_df,
    on='Date',
    how='inner',
    validate='one_to_one'
)

# Finally merge with retail sales data
merged_df = pd.merge(
    merged_df,
    retail_df,
    on='Date',
    how='inner',
    validate='one_to_one'
)

print(f"\nAfter merging all four datasets:")
print(f"  Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
print(f"  Number of observations: {len(merged_df)}")
print(f"  Columns: {list(merged_df.columns)}")

# Display first few rows
print("\nFirst 10 rows of merged dataset:")
print(merged_df.head(10))

# Display basic statistics for all numeric variables
print(f"\nBasic Statistics:")
numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Mean: {merged_df[col].mean():.4f}")
    print(f"  Std Dev: {merged_df[col].std():.4f}")
    print(f"  Min: {merged_df[col].min():.4f}")
    print(f"  Max: {merged_df[col].max():.4f}")

# Save the merged dataset
merged_df.to_excel('merged_traditional.xlsx', index=False)
print("\nMerged dataset saved as 'merged_traditional.xlsx'")

# Check for any missing values
print(f"\nMissing values check:")
print(merged_df.isnull().sum())

# Show correlation between variables (optional but useful)
print(f"\nCorrelation matrix:")
print(merged_df[numeric_cols].corr().round(3))