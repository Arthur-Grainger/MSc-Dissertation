import os
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")

import pandas as pd

# Load the claimant count data
# Adjust the file path and skiprows as needed based on your actual file structure
claimant_df = pd.read_excel("claimant_count.xlsx", skiprows=5)  # Skip header rows

# Keep only the first two columns and rename them
claimant_clean = claimant_df.iloc[:, [0, 1]].copy()
claimant_clean.columns = ['Date', 'claimant_count']

# Convert the Date column to proper datetime format
# The dates appear to be in "Month Year" format like "January 2004"
claimant_clean['Date'] = pd.to_datetime(claimant_clean['Date'], format='%B %Y')

# Remove any rows with missing dates
claimant_clean = claimant_clean.dropna(subset=['Date'])

# Display the cleaned data
print("Cleaned Claimant Count Data:")
print(claimant_clean.head(10))
print(f"\nData shape: {claimant_clean.shape}")
print(f"Date range: {claimant_clean['Date'].min()} to {claimant_clean['Date'].max()}")

# Save the cleaned data
claimant_clean.to_excel('cleaned_claimant_count.xlsx', index=False)
print("\nCleaned data saved to 'cleaned_claimant_count.xlsx'")

# Check the date format matches your EPU dataset
print(f"\nSample dates in cleaned format:")
print(claimant_clean['Date'].head().dt.strftime('%Y-%m-%d'))