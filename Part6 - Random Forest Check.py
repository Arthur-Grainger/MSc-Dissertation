import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# %% Define the base path to your final dataset directory

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

INPUT_FILENAME = 'final_dataset_ar_only_lags.xlsx'
OUTPUT_DIR = 'random_forest_analysis'

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
        
        # Define the test point (the single observation to predict)
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
        # n_estimators=100 is a good default. random_state ensures reproducibility.
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

def evaluate_and_plot_results(results_df, output_dir):
    """Calculates performance metrics and plots the results."""
    print("\n--- Evaluating Performance and Plotting Results ---")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created output directory: {output_dir}")

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results_df['y_actual'], results_df['y_pred_rf']))
    r2 = r2_score(results_df['y_actual'], results_df['y_pred_rf'])
    
    print("\nRandom Forest Out-of-Sample Performance:")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - R-squared: {r2:.4f}")

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(results_df['forecast_date'], results_df['y_actual'], label='Actual Values', color='black', linewidth=1.5, alpha=0.8)
    ax.plot(results_df['forecast_date'], results_df['y_pred_rf'], label='Random Forest Forecast', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_title('Random Forest Forecast vs. Actual Values', fontsize=16, pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Change in Unemployment Rate (d_unemp_rate)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'random_forest_forecast_vs_actual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n Forecast plot saved to '{plot_path}'")
    plt.show()

# %% Main Execution Block
if __name__ == "__main__":
    # Load the dataset
    df = load_data(BASE_DIR, INPUT_FILENAME)
    
    if df is not None:
        # Define the target variable and the top 8 predictors from your ElasticNet analysis
        TARGET_VARIABLE = 'd_unemp_rate'
        TOP_8_PREDICTORS = [
            'redundancy',          # GT
            'ftse_vol_3m_L1',      # Traditional (Note: Use the lagged version from your final dataset)
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
            # Run the rolling forecast
            forecast_results = run_rolling_forecast_random_forest(df, TARGET_VARIABLE, TOP_8_PREDICTORS)
            
            # Evaluate and visualize the results
            evaluate_and_plot_results(forecast_results, OUTPUT_DIR)
            
            print("\n--- Analysis Complete ---")
            
# %% Diebold and mariano test

from dieboldmariano import dm_test

# --- STEP 1: Load Excel data ---
file_path = "forecast_output.xlsx"  # Update path if needed
df = pd.read_excel(file_path)

# Extract columns (adjust names to match your Excel file!)
actual = df["y_actual"]              # True values
forecast1 = df["random_forest"]  # Forecasts from Model 1
forecast2 = df["forecast_benchmark_lasso_ols"]  # Forecasts from Model 2

# --- STEP 2: Run Diebold-Mariano test ---
# Choose loss function: 'mae' (power=1) or 'mse' (power=2)
dm_result = dm_test(
    actual, forecast1, forecast2,
    h=2      # 1 for MAE, 2 for MSE
)

# --- STEP 3: Print and interpret results ---
print("\nDiebold-Mariano Test Results:")
print(f"DM Statistic: {dm_result[0]:.3f}")
print(f"p-value:      {dm_result[1]:.3f}")

# Interpretation guide
if dm_result[1] < 0.05:
    if dm_result[0] > 0:
        print("Conclusion: Model 1 is significantly better (p < 0.05)")
    else:
        print("Conclusion: Model 2 is significantly better (p < 0.05)")
else:
    print("Conclusion: No significant difference between models (p >= 0.05)")
