import os
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")
import pandas as pd

# Import the retail sales data
retail_sales = pd.read_excel(
    'retail sales.xlsx',
    sheet_name='CPSA3',
    skiprows=200,  # Skip first 200 rows (to start from row 201)
    nrows=256,  # Import 257 rows (from row 201 to 457 inclusive)
    usecols=[0, 1],  # Only columns A and B (0-indexed)
    header=0,  # Use first row as header after skipping
    names=['Date', 'retail_sales_index']  # Rename columns
)

# Convert the date format from "1988 Jan" to datetime
retail_sales['Date'] = pd.to_datetime(retail_sales['Date'], format='%Y %b')

# Display basic info about the dataset
print("Dataset shape:", retail_sales.shape)
print("\nFirst few rows:")
print(retail_sales.head())
print("\nData types:")
print(retail_sales.dtypes)
print("\nAny missing values:")
print(retail_sales.isnull().sum())

# Save the cleaned dataset (optional)
retail_sales.to_excel('cleaned_retail_sales.xlsx', index=False)