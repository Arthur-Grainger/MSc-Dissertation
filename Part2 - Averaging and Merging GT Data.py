import os
import pandas as pd
import glob
from typing import List, Dict

# %% Define directories and configuration

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

# Configuration for averaging Google Trends data
SAMPLE_FOLDER_NAMES = [
    "Automated GT Data 1",
    "Automated GT Data 2",
    "Automated GT Data 3",
    "Automated GT Data 4",
    "Automated GT Data 5",
    "Automated GT Data 6",
    "Automated GT Data 7",
    "Automated GT Data 8",
    "Automated GT Data 9",
    "Automated GT Data 10"
]

AVERAGED_OUTPUT_FOLDER_NAME = "Averaged GT Data"

# Configuration for merging datasets
TRADITIONAL_INDICATORS_PATH = os.path.join(BASE_DIR, "Traditional Indicators")
UNEMPLOYMENT_DATA_PATH = os.path.join(BASE_DIR, "Unemployment")
FINAL_DATASET_PATH = os.path.join(BASE_DIR, "Final Dataset")

# %% Functions - Google Trends Averaging

def average_trends_data(base_path: str, sample_folders: List[str], output_folder: str):
    """
    Finds corresponding Google Trends CSVs across multiple sample folders,
    averages their values, and saves the results to a new directory.
    """
    print("--- Starting Google Trends Averaging Script ---")

    sample_dir_paths = [os.path.join(base_path, folder) for folder in sample_folders]
    output_dir_path = os.path.join(base_path, output_folder)

    print("\n1. Validating sample directories...")
    valid_sample_dirs = []
    for path in sample_dir_paths:
        if os.path.exists(path):
            print(f"   Found: {path}")
            valid_sample_dirs.append(path)
        else:
            print(f" Warning: Directory not found, skipping: {path}")
    
    if not valid_sample_dirs:
        print("\nError: No valid sample directories found.")
        return

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"\nCreated output directory: {output_dir_path}")

    print("\n2. Discovering unique keywords to process...")
    try:
        master_file_list = [f for f in os.listdir(valid_sample_dirs[0]) if f.endswith('.csv')]
        print(f"  Found {len(master_file_list)} unique keywords to process.")
    except IndexError:
        print("\nError: Cannot find any files in the first valid directory.")
        return

    print("\n3. Averaging data for each keyword...")
    total_files = len(master_file_list)
    for i, filename in enumerate(master_file_list, 1):
        print(f"  ({i}/{total_files}) Processing: {filename}")
        
        keyword_dfs = []
        for sample_dir in valid_sample_dirs:
            file_path = os.path.join(sample_dir, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'date' in df.columns:
                        df.rename(columns={'date': 'Date'}, inplace=True)
                    
                    if 'isPartial' in df.columns:
                        df.drop(columns=['isPartial'], inplace=True)

                    data_col_name = [col for col in df.columns if col != 'Date'][0]
                    df[data_col_name] = pd.to_numeric(df[data_col_name], errors='coerce').fillna(0)

                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    keyword_dfs.append(df)
                except Exception as e:
                    print(f" Could not read or process file: {file_path}. Error: {e}")

        if len(keyword_dfs) > 1:
            try:
                # Concatenate all dataframes side-by-side, aligning on the 'Date' index.
                combined_df = pd.concat(keyword_dfs, axis=1)
                
                # Calculate the mean for each date, resulting in a pandas Series.
                averaged_series = combined_df.mean(axis=1)
                
                # Get the clean keyword name from the filename.
                keyword_name = os.path.splitext(filename)[0].replace('_term', '').replace('_topic', '').replace('_website', '')

                # Explicitly create the final DataFrame from the index (dates) and values.
                averaged_df = pd.DataFrame({
                    'Date': averaged_series.index,
                    keyword_name: averaged_series.values
                })
        
                output_file_path = os.path.join(output_dir_path, filename)
                averaged_df.to_csv(output_file_path, index=False)
                print(f" Averaged {len(keyword_dfs)} samples and saved to '{output_folder}'")

            except Exception as e:
                print(f"    - Error averaging data for {filename}. Error: {e}")
        elif len(keyword_dfs) == 1:
            df_single = keyword_dfs[0].reset_index() # Turn 'Date' index back into a column.
            output_file_path = os.path.join(output_dir_path, filename)
            df_single.to_csv(output_file_path, index=False)
            print(f" Only one sample found. Copied file to '{output_folder}'.")
        else:
            print(" No valid files found for this keyword across any sample directory.")

    print("\n--- Google Trends Averaging Complete ---")
    print(f" Averaged data has been saved to: {output_dir_path}")

# %% Functions - Traditional Indicators Processing

def process_claimant_count() -> pd.DataFrame:
    """Process claimant count data."""
    print("Processing Claimant Count Data...")
    path = os.path.join(TRADITIONAL_INDICATORS_PATH, "claimant_count.xlsx")
    df = pd.read_excel(path, skiprows=5, usecols=[0, 1])
    df.columns = ['Date', 'claimant_count']
    df['Date'] = pd.to_datetime(df['Date'], format='%B %Y')
    df = df.dropna(subset=['Date'])
    return df

def process_epu_index() -> pd.DataFrame:
    """Process EPU Index data."""
    print("Processing EPU Index Data...")
    path = os.path.join(TRADITIONAL_INDICATORS_PATH, 'EPU_Index.xlsx')
    df = pd.read_excel(path)
    df = df[pd.to_numeric(df['year'], errors='coerce').notna()]
    df = df[pd.to_numeric(df['month'], errors='coerce').notna()]
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.drop(columns=['year', 'month'])
    return df

def process_ftse_data() -> pd.DataFrame:
    """Process FTSE-100 data to monthly returns."""
    print("Processing FTSE-100 Data...")
    path = os.path.join(TRADITIONAL_INDICATORS_PATH, 'FTSE 100 Historical Results Price Data.csv')
    df = pd.read_csv(path, usecols=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    df = df.set_index('Date').sort_index()
    monthly_prices = df['Price'].resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    result = pd.DataFrame({
        'Date': monthly_returns.index.to_period('M').to_timestamp(),
        'ftse_returns': monthly_returns.values
    }).reset_index(drop=True)
    return result

def process_retail_sales() -> pd.DataFrame:
    """Process retail sales data."""
    print("Processing Retail Sales Data...")
    path = os.path.join(TRADITIONAL_INDICATORS_PATH, 'retail sales.xlsx')
    df = pd.read_excel(path, sheet_name='CPSA3', skiprows=199, nrows=257, usecols=[0, 1], names=['Date', 'retail_sales_index'])
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

def process_unemployment_rate() -> pd.DataFrame:
    """Process unemployment rate data."""
    print("Processing Unemployment Rate Data...")
    path = os.path.join(UNEMPLOYMENT_DATA_PATH, 'Unemployment_Rate.csv')
    df = pd.read_csv(path, skiprows=279, header=None, names=['Date', 'unemp_rate'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b', errors='coerce')
    df = df.dropna()
    return df

# %% Functions - Merging Functions

def process_and_merge_google_trends(google_trends_input_path: str) -> pd.DataFrame:
    """
    Reads all averaged Google Trends CSVs, resamples them to a monthly frequency,
    and merges them into a single, clean DataFrame.
    """
    print("\nProcessing and Merging Averaged Google Trends Data...")
    
    file_paths = glob.glob(os.path.join(google_trends_input_path, "*.csv"))
    if not file_paths:
        print("  - Error: No CSV files found in the averaged data directory. Stopping.")
        return pd.DataFrame()
        
    print(f"  Found {len(file_paths)} averaged CSV files to merge.")
    
    list_of_dfs = []
    for file_path in file_paths:
        try:
            # Read the raw averaged data
            df = pd.read_csv(file_path)
            
            # Convert 'Date' column to datetime objects and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Correctly resample from weekly to monthly frequency ('MS' for Month Start)
            df_monthly = df.resample('MS').mean()
            
            # Clean the keyword name to use as the final column header
            col_name = os.path.splitext(os.path.basename(file_path))[0]
            col_name = col_name.replace('_term', '').replace('_topic', '').replace('_website', '')
            
            # Rename the data column to the clean name
            if not df_monthly.empty:
                df_monthly.rename(columns={df_monthly.columns[0]: col_name}, inplace=True)
                list_of_dfs.append(df_monthly)
            else:
                print(f"  - Warning: Skipping {os.path.basename(file_path)} as it was empty after resampling.")

        except Exception as e:
            print(f"  - Warning: Could not process file {os.path.basename(file_path)}. Skipping. Error: {e}")

    if not list_of_dfs:
        print("  - Error: No Google Trends files were successfully loaded. Cannot proceed.")
        return pd.DataFrame()

    # Concatenate all dataframes
    merged_df = pd.concat(list_of_dfs, axis=1)
    
    # Reset the index to turn the 'Date' index back into a column
    merged_df.reset_index(inplace=True)
    
    print(f" Merged to {merged_df.shape[0]} rows and {merged_df.shape[1]-1} GT variables.")
    return merged_df

def combine_traditional_indicators(traditional_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all traditional indicators into one dataset."""
    print("\nCombining traditional indicators...")
    combined_traditional = list(traditional_datasets.values())[0]
    
    for name, df in list(traditional_datasets.items())[1:]:
        combined_traditional = pd.merge(combined_traditional, df, on='Date', how='inner')
    
    combined_traditional = combined_traditional.sort_values('Date').reset_index(drop=True)
    print(f" Final traditional indicators dataset: {len(combined_traditional)} rows.")
    return combined_traditional

# %% Main execution function

def main():
    """Main function combining both parts of the data processing pipeline."""
    print("=" * 80)
    print("COMBINED GOOGLE TRENDS AND TRADITIONAL INDICATORS PROCESSING")
    print("=" * 80)
    
    # Average Google Trends Data
    print("\n" + "=" * 40)
    print("AVERAGING GOOGLE TRENDS DATA")
    print("=" * 40)
    
    average_trends_data(
        base_path=BASE_DIR,
        sample_folders=SAMPLE_FOLDER_NAMES,
        output_folder=AVERAGED_OUTPUT_FOLDER_NAME
    )
    
    # Merge Datasets
    print("\n" + "=" * 40)
    print("MERGING DATASETS")
    print("=" * 40)
    
    # Create final dataset directory if it doesn't exist
    if not os.path.exists(FINAL_DATASET_PATH):
        os.makedirs(FINAL_DATASET_PATH)
        print(f"Created directory: {FINAL_DATASET_PATH}")
    
    # Process traditional indicators
    print("\nProcessing traditional indicators...")
    traditional_datasets = {
        'claimant_count': process_claimant_count(),
        'epu_index': process_epu_index(),
        'ftse_returns': process_ftse_data(),
        'retail_sales': process_retail_sales(),
        'unemployment_rate': process_unemployment_rate()
    }
    
    # Combine traditional indicators
    traditional_combined = combine_traditional_indicators(traditional_datasets)
    
    # Process and merge the averaged Google Trends data
    google_trends_input_path = os.path.join(BASE_DIR, AVERAGED_OUTPUT_FOLDER_NAME)
    gt_data_raw = process_and_merge_google_trends(google_trends_input_path)
    
    # Save final datasets
    print("\nSaving final datasets...")
    if not traditional_combined.empty:
        traditional_path = os.path.join(FINAL_DATASET_PATH, 'traditional_indicators_dataset.xlsx')
        traditional_combined.to_excel(traditional_path, index=False)
        print(f" Traditional indicators dataset saved to '{traditional_path}'")
    
    if not gt_data_raw.empty:
        gt_path = os.path.join(FINAL_DATASET_PATH, 'google_trends_raw_dataset.xlsx')
        gt_data_raw.to_excel(gt_path, index=False)
        print(f" Raw Google Trends dataset saved to '{gt_path}'")
    
    print("\n" + "=" * 80)
    print(" COMPLETE DATA PROCESSING PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal outputs saved to: {FINAL_DATASET_PATH}")
    print("- traditional_indicators_dataset.xlsx")
    print("- google_trends_raw_dataset.xlsx")

if __name__ == "__main__":
    main()
