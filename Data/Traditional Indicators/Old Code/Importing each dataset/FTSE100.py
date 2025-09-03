import os
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")

import pandas as pd

# Read your CSV file
ftse_raw = pd.read_csv('FTSE 100 Historical Results Price Data.csv')  # Replace with your filename

# Process the data using Price column (Column B)
ftse_clean = ftse_raw[['Date', 'Price']].copy()

# Convert Date to datetime
ftse_clean['Date'] = pd.to_datetime(ftse_clean['Date'], format='%d/%m/%Y')

# Remove commas from Price if they exist and convert to numeric
ftse_clean['Price'] = ftse_clean['Price'].astype(str).str.replace(',', '').astype(float)

# Set date as index and sort (oldest first)
ftse_clean = ftse_clean.set_index('Date').sort_index()

# Convert to monthly data (last trading day of each month)
ftse_monthly_prices = ftse_clean['Price'].resample('ME').last()

# Calculate monthly returns
ftse_monthly_returns = ftse_monthly_prices.pct_change().dropna()

# Create final DataFrame
ftse_final = pd.DataFrame({
    'Date': ftse_monthly_returns.index.to_period('M').to_timestamp(),
    'ftse_returns': ftse_monthly_returns.values
}).reset_index(drop=True)

print(f"Successfully processed {len(ftse_final)} monthly observations")
print("Date range:", ftse_final['Date'].min(), "to", ftse_final['Date'].max())
print(f"Mean monthly return: {ftse_final['ftse_returns'].mean():.4f}")
print(f"Standard deviation: {ftse_final['ftse_returns'].std():.4f}")

print("\nFirst few observations:")
print(ftse_final.head())

# Save to Excel
ftse_final.to_excel('ftse_monthly_returns.xlsx', index=False)
print("\nData saved to 'ftse_monthly_returns.xlsx'")