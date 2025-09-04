import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# %% Define the base path to your final dataset directory

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project" 

INPUT_FILENAME = 'final_dataset_ar_only_lags.xlsx'
OUTPUT_DIR = 'random_forest_analysis'
EXCEL_RESULTS_PATH = os.path.join(BASE_DIR, 'out-of-sample', 'Model_Out_of_Sample_Analysis_Results.xlsx')

# %% Main Analysis Functions

def load_data(base_path, filename):
    """Loads and prepares the final dataset for analysis."""
    print("--- Loading Data ---")
    file_path = os.path.join(base_path, filename)
    try:
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f" Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f" ERROR: Input file not found at '{file_path}'. Please ensure the file exists.")
        return None

def run_rolling_forecast_random_forest(df, target_var, predictors, train_length=48):
    """
    Performs a rolling-window forecast using a Random Forest model.
    In each window, it trains on 'train_length' months and predicts the next month.
    """
    print("\n--- Starting Rolling-Window Forecast with Random Forest ---")
    print(f"  Training window size: {train_length} months")
    
    all_forecasts = []
    total_obs = len(df)
    
    # The loop starts after the first full training window and goes to the end
    for i in range(train_length, total_obs):
        # Define the training window for this iteration
        train_start_idx = i - train_length
        train_end_idx = i
        
        # Define the test point 
        test_idx = i

        # Slice the data
        train_df = df.iloc[train_start_idx:train_end_idx]
        test_df = df.iloc[test_idx:test_idx+1]

        # Prepare X and y for scikit-learn
        X_train = train_df[predictors]
        y_train = train_df[target_var]
        X_test = test_df[predictors]
        y_actual = test_df[target_var].iloc[0]

        # --- Model Training and Prediction ---
        # Initialize the Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Train the model on the current window's data
        rf_model.fit(X_train, y_train)
        
        # Make a single prediction for the next time step
        prediction = rf_model.predict(X_test)[0]
        
        # Store the results
        all_forecasts.append({
            'forecast_date': test_df['Date'].iloc[0],
            'y_actual': y_actual,
            'y_pred_rf': prediction
        })
        
        if (i - train_length + 1) % 12 == 0:
            print(f"  ... completed forecast for {test_df['Date'].iloc[0].strftime('%Y-%m')} ({i - train_length + 1}/{total_obs - train_length} forecasts)")

    print(" Rolling forecast complete.")
    return pd.DataFrame(all_forecasts)

def load_benchmark_forecasts(excel_path):
    """Load benchmark forecasts from the results Excel file."""
    print("\n--- Loading Benchmark Forecasts ---")
    try:
        # Read the All_Forecasts sheet
        all_forecasts_df = pd.read_excel(excel_path, sheet_name='All_Forecasts')
        
        # Filter for 1-month ahead forecasts only
        benchmark_1m = all_forecasts_df[all_forecasts_df['horizon'] == 1].copy()
        
        if benchmark_1m.empty:
            print(" ERROR: No 1-month ahead forecasts found in the Excel file.")
            return None
        
        # Convert forecast_date to datetime for proper matching
        benchmark_1m['forecast_date'] = pd.to_datetime(benchmark_1m['forecast_date'])
        
        # Select only the columns we need
        benchmark_data = benchmark_1m[['forecast_date', 'y_actual', 'forecast_benchmark_lasso_ols']].copy()
        benchmark_data = benchmark_data.dropna()
        
        print(f" Loaded {len(benchmark_data)} benchmark forecasts for 1-month ahead.")
        print(f" Date range: {benchmark_data['forecast_date'].min().strftime('%Y-%m')} to {benchmark_data['forecast_date'].max().strftime('%Y-%m')}")
        
        return benchmark_data
        
    except FileNotFoundError:
        print(f" ERROR: Excel file not found at '{excel_path}'. Please run the first script first.")
        return None
    except Exception as e:
        print(f" ERROR loading benchmark forecasts: {str(e)}")
        return None

def merge_forecasts(rf_forecasts, benchmark_forecasts):
    """Merge Random Forest and Benchmark forecasts on forecast_date."""
    print("\n--- Merging Forecasts ---")
    
    # Ensure both dataframes have datetime format
    rf_forecasts['forecast_date'] = pd.to_datetime(rf_forecasts['forecast_date'])
    benchmark_forecasts['forecast_date'] = pd.to_datetime(benchmark_forecasts['forecast_date'])
    
    # Merge on forecast_date
    merged_df = pd.merge(rf_forecasts, benchmark_forecasts, 
                        on='forecast_date', 
                        how='inner', 
                        suffixes=('_rf', '_bench'))
    
    print(f" Successfully merged {len(merged_df)} forecasts.")
    print(f" Common forecast period: {merged_df['forecast_date'].min().strftime('%Y-%m')} to {merged_df['forecast_date'].max().strftime('%Y-%m')}")
    
    # Use actual values from Random Forest data
    merged_df['y_actual'] = merged_df['y_actual_rf']
    
    # Clean up column names
    merged_df = merged_df.rename(columns={
        'y_pred_rf': 'random_forest',
        'forecast_benchmark_lasso_ols': 'benchmark_lasso'
    })
    
    return merged_df[['forecast_date', 'y_actual', 'random_forest', 'benchmark_lasso']]

def diebold_mariano_test(actual, forecast1, forecast2, h=1):
    """
    Perform Diebold-Mariano test for forecast accuracy comparison.
    
    Parameters:
    actual: array of actual values
    forecast1: array of forecasts from model 1
    forecast2: array of forecasts from model 2
    h: forecast horizon (default=1)
    
    Returns:
    dm_stat: DM test statistic
    p_value: p-value of the test
    """
    # Calculate forecast errors
    e1 = actual - forecast1
    e2 = actual - forecast2
    
    # Calculate squared errors (MSE loss function)
    d = e1**2 - e2**2
    
    # Mean of the loss differential
    d_mean = np.mean(d)
    
    # Variance of the loss differential
    d_var = np.var(d, ddof=1)
    
    # Number of observations
    n = len(d)
    
    # DM test statistic
    if d_var > 0:
        dm_stat = d_mean / np.sqrt(d_var / n)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value
    else:
        return 0.0, 1.0

def evaluate_and_compare_models(merged_df, output_dir):
    """Calculate performance metrics and run Diebold-Mariano test."""
    print("\n--- Evaluating Performance and Comparing Models ---")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created output directory: {output_dir}")

    # Extract data
    actual = merged_df['y_actual'].values
    rf_forecasts = merged_df['random_forest'].values
    benchmark_forecasts = merged_df['benchmark_lasso'].values
    
    # Calculate metrics for Random Forest
    rf_rmse = np.sqrt(mean_squared_error(actual, rf_forecasts))
    rf_mae = np.mean(np.abs(actual - rf_forecasts))
    rf_r2 = r2_score(actual, rf_forecasts)
    
    # Calculate metrics for Benchmark
    bench_rmse = np.sqrt(mean_squared_error(actual, benchmark_forecasts))
    bench_mae = np.mean(np.abs(actual - benchmark_forecasts))
    bench_r2 = r2_score(actual, benchmark_forecasts)
    
    print("\nModel Performance Comparison:")
    print("="*50)
    print("Random Forest:")
    print(f"  - RMSE: {rf_rmse:.4f}")
    print(f"  - MAE:  {rf_mae:.4f}")
    print(f"  - R²:   {rf_r2:.4f}")
    print("\nBenchmark (AR-LASSO):")
    print(f"  - RMSE: {bench_rmse:.4f}")
    print(f"  - MAE:  {bench_mae:.4f}")
    print(f"  - R²:   {bench_r2:.4f}")
    
    # Run Diebold-Mariano test
    dm_stat, p_value = diebold_mariano_test(actual, rf_forecasts, benchmark_forecasts)
    
    print("\nDiebold-Mariano Test Results:")
    print("="*50)
    print(f"DM Statistic: {dm_stat:.4f}")
    print(f"p-value:      {p_value:.4f}")
    
    # Interpretation
    if p_value < 0.05:
        if dm_stat > 0:
            print("Conclusion: Random Forest is significantly better than Benchmark (p < 0.05)")
        else:
            print("Conclusion: Benchmark is significantly better than Random Forest (p < 0.05)")
    else:
        print("Conclusion: No significant difference between models (p >= 0.05)")
    
    # Save comparison results
    comparison_results = {
        'Model': ['Random Forest', 'Benchmark (AR-LASSO)'],
        'RMSE': [rf_rmse, bench_rmse],
        'MAE': [rf_mae, bench_mae],
        'R²': [rf_r2, bench_r2]
    }
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.loc[len(comparison_df)] = ['DM Test', f'Statistic: {dm_stat:.4f}', f'p-value: {p_value:.4f}', '']
    
    results_path = os.path.join(output_dir, 'model_comparison_results.xlsx')
    with pd.ExcelWriter(results_path) as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        merged_df.to_excel(writer, sheet_name='Forecasts', index=False)
    
    print(f"\n Results saved to '{results_path}'")
    
    return merged_df

def plot_forecast_comparison(merged_df, output_dir):
    """Create plots comparing Random Forest and Benchmark forecasts."""
    print("\n--- Creating Comparison Plots ---")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1.plot(merged_df['forecast_date'], merged_df['y_actual'], 
             label='Actual Values', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(merged_df['forecast_date'], merged_df['random_forest'], 
             label='Random Forest', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.plot(merged_df['forecast_date'], merged_df['benchmark_lasso'], 
             label='Benchmark (AR-LASSO)', color='blue', linestyle=':', linewidth=1.5, alpha=0.8)
    
    ax1.set_title('Forecast Comparison: Random Forest vs Benchmark', fontsize=16, pad=15)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Change in Unemployment Rate', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot 2: Forecast errors comparison
    rf_errors = merged_df['y_actual'] - merged_df['random_forest']
    bench_errors = merged_df['y_actual'] - merged_df['benchmark_lasso']
    
    ax2.plot(merged_df['forecast_date'], rf_errors, 
             label='Random Forest Errors', color='red', alpha=0.7)
    ax2.plot(merged_df['forecast_date'], bench_errors, 
             label='Benchmark Errors', color='blue', alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_title('Forecast Errors Comparison', fontsize=16, pad=15)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Forecast Error (Actual - Predicted)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'forecast_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" Forecast comparison plot saved to '{plot_path}'")
    plt.show()

# %% Main Execution Block
if __name__ == "__main__":
    # Load the dataset
    df = load_data(BASE_DIR, INPUT_FILENAME)
    
    if df is not None:
        # Define the target variable and the top 8 predictors from ElasticNet analysis
        TARGET_VARIABLE = 'd_unemp_rate'
        TOP_8_PREDICTORS = [
            'redundancy',          # GT
            'ftse_vol_3m_L1',      # Traditional
            'd_unemp_rate_L2',     # AR
            'retraining',          # GT
            'd_finance_jobs',      # GT
            'learn_new_skills',    # GT
            'd_reed',              # GT
            'd_claimant_count_L1'  # Traditional
        ]
        
        # Verify that all selected predictors exist in the dataframe
        missing_vars = [var for var in TOP_8_PREDICTORS if var not in df.columns]
        if missing_vars:
            print(f" ERROR: The following predictor variables were not found in the dataset: {missing_vars}")
        else:
            print("\n All selected predictor variables are present in the dataset.")
            
            # Run the rolling forecast with Random Forest
            rf_results = run_rolling_forecast_random_forest(df, TARGET_VARIABLE, TOP_8_PREDICTORS)
            
            # Load benchmark forecasts from Excel file
            benchmark_data = load_benchmark_forecasts(EXCEL_RESULTS_PATH)
            
            if benchmark_data is not None:
                # Merge the forecasts
                merged_forecasts = merge_forecasts(rf_results, benchmark_data)
                
                if not merged_forecasts.empty:
                    # Evaluate and compare models
                    final_data = evaluate_and_compare_models(merged_forecasts, OUTPUT_DIR)
                    
                    # Create comparison plots
                    plot_forecast_comparison(final_data, OUTPUT_DIR)
                    
                    print("\n--- Analysis Complete ---")
                else:
                    print(" ERROR: No overlapping forecasts found between Random Forest and Benchmark models.")
            else:
                print(" ERROR: Could not load benchmark forecasts. Please ensure the first script has been run successfully.")