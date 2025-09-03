import os
os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Traditional Indicators")

import pandas as pd
from datetime import datetime

# %% Sorting EPU data

# Load data and clean non-numeric rows
df = pd.read_excel('EPU_Index.xlsx')
df = df[pd.to_numeric(df['year'], errors='coerce').notna()]
df = df[pd.to_numeric(df['month'], errors='coerce').notna()]

# Create Date column and remove year/month
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.drop(columns=['year', 'month'])

# Reorder columns and save
df = df[['Date'] + [col for col in df.columns if col != 'Date']]
df.to_excel('EPU_Index_with_date.xlsx', index=False)

print(f"Converted {len(df)} rows. Date range: {df['Date'].min()} to {df['Date'].max()}")

# %% Sorting Claimant Count Data

