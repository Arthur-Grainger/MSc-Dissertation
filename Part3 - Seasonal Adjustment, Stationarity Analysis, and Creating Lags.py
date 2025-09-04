import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
from scipy.stats import linregress

# Import optional dependencies with fallbacks
try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    warnings.warn("statsmodels STL not available. Install with: pip install statsmodels")

try:
    from statsmodels.tsa.stattools import adfuller, coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

warnings.filterwarnings('ignore')

# %% Configuration and Constants

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

# Input files
INPUT_TRADITIONAL_DATA = 'traditional_indicators_dataset.xlsx'
INPUT_GT_DATA_RAW = 'google_trends_raw_dataset.xlsx'

# Intermediate files
GT_SEASONALLY_ADJUSTED = 'google_trends_seasonally_adjusted.xlsx'
COMPLETE_DATASET_STL = 'complete_economic_dataset_stl.xlsx'
STATIONARITY_RESULTS_JSON = 'stationarity_results.json'
STATIONARITY_EXCEL_REPORT = 'stationarity_and_cointegration_report.xlsx'

# Final output
FINAL_DATASET = 'final_dataset_ar_only_lags.xlsx'

# Plot directory
PLOT_OUTPUT_DIR = 'trend_plots'

# Analysis configuration
NO_SEASONAL_ADJUSTMENT = ['furlough', 'brexit', 'financial_crisis']
VARS_WITH_TREND = ['employment_rights', 'learn_new_skills']
EVENT_VARS = ['furlough', 'brexit', 'Financial_crisis']

# %% STL Seasonal Adjustment Functions

def adaptive_stl_seasonal_adjustment(series, variable_name=None, min_nonzero_threshold=1):
    """Applies STL seasonal adjustment, handling periods of zero values."""
    if not STL_AVAILABLE:
        return series
    if series.isna().any():
        series = series.fillna(0)
    if len(series) < 24 or series.std() < 0.1:
        return series
    if isinstance(series.index, pd.RangeIndex):
        start_date = pd.Timestamp('2004-01-01')
        series.index = pd.date_range(start=start_date, periods=len(series), freq='MS')
    
    nonzero_mask = series > min_nonzero_threshold
    if not nonzero_mask.any():
        return series
        
    first_nonzero_idx = nonzero_mask.idxmax()
    zero_period = series[series.index < first_nonzero_idx]
    nonzero_period = series[series.index >= first_nonzero_idx]
    
    if len(zero_period) > 6 and len(nonzero_period) >= 24:
        try:
            stl = STL(nonzero_period, seasonal=13)
            result = stl.fit()
            adjusted_nonzero = result.trend + result.resid
            return pd.concat([zero_period, adjusted_nonzero]).sort_index()
        except Exception:
            return series
    else:
        try:
            stl = STL(series, seasonal=13)
            result = stl.fit()
            return result.trend + result.resid
        except Exception:
            return series

def apply_seasonal_adjustment_to_gt_data(gt_data_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply STL seasonal adjustment to all Google Trends variables except specified exclusions."""
    print("Applying STL seasonal adjustment to Google Trends data...")
    
    if not STL_AVAILABLE:
        print("\nWARNING: STL not available! Returning original data.")
        return gt_data_raw.copy()
    
    gt_data_adjusted = gt_data_raw.copy()
    gt_columns = [col for col in gt_data_adjusted.columns if 'date' not in col.lower()]
    
    adjustment_vars = [col for col in gt_columns if col.lower() not in NO_SEASONAL_ADJUSTMENT]
    
    print(f"\nProcessing {len(gt_columns)} Google Trends variables...")
    
    for i, col in enumerate(adjustment_vars, 1):
        print(f"  [{i}/{len(adjustment_vars)}] Adjusting {col}...")
        date_col = [c for c in gt_data_adjusted.columns if 'date' in c.lower()][0]
        ts = pd.Series(gt_data_adjusted[col].values, index=gt_data_adjusted[date_col], name=col)
        adjusted_series = adaptive_stl_seasonal_adjustment(ts, variable_name=col)
        gt_data_adjusted[col] = adjusted_series.values
    
    print("\nSeasonal adjustment completed.")
    return gt_data_adjusted

def combine_final_datasets(traditional_data: pd.DataFrame, gt_data_adjusted: pd.DataFrame) -> pd.DataFrame:
    """Combine traditional indicators with seasonally adjusted Google Trends data."""
    print("\nCombining traditional indicators with seasonally adjusted Google Trends data...")

    # Ensure the date column in the traditional data is in datetime format.
    traditional_data['Date'] = pd.to_datetime(traditional_data['Date'])
    
    # Find the date column in the Google Trends data.
    date_col_gt = [col for col in gt_data_adjusted.columns if 'date' in col.lower()][0]
    
    # Rename the Google Trends date column to 'Date' to ensure it matches.
    gt_data_adjusted.rename(columns={date_col_gt: 'Date'}, inplace=True)
    gt_data_adjusted['Date'] = pd.to_datetime(gt_data_adjusted['Date'])

    # Merge datasets
    final_combined = pd.merge(traditional_data, gt_data_adjusted, on='Date', how='inner')
  
    # Create the 3-month rolling volatility feature from ftse_returns
    if 'ftse_returns' in final_combined.columns:
        final_combined['ftse_vol_3m'] = final_combined['ftse_returns'].rolling(window=3).std()
        print("  Created base feature: ftse_vol_3m")
        
    # Sort and clean the final dataframe
    final_combined = final_combined.sort_values('Date').reset_index(drop=True)
    
    if final_combined.isnull().sum().sum() > 0:
        final_combined = final_combined.dropna()
        print("  Dropped rows with missing values.")

    print(f"Final combined dataset created: {final_combined.shape}")
    return final_combined

def load_datasets_from_part1() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the datasets created in Part 1."""
    print("Loading datasets from Part 1...")
    os.chdir(BASE_DIR)
    
    traditional_data = pd.read_excel(INPUT_TRADITIONAL_DATA)
    gt_data_raw = pd.read_excel(INPUT_GT_DATA_RAW)
    
    print("Datasets loaded.")
    return traditional_data, gt_data_raw

# %% Stationarity Analysis Functions

def plot_trend_variables(df: pd.DataFrame, vars_to_plot: list, output_dir: str):
    """
    Plots variables with their linear trend line using a consistent academic style
    and saves them to files.
    """
    print("\n--- Plotting Variables with Potential Trends ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created output directory: {output_dir}")

    df['Date'] = pd.to_datetime(df['Date'])

    for var in vars_to_plot:
        if var not in df.columns:
            print(f"  Warning: Variable '{var}' not found in DataFrame. Skipping plot.")
            continue

        series = df[['Date', var]].dropna()
        if series.empty:
            print(f"  Warning: No data for '{var}'. Skipping plot.")
            continue

        x_numeric = np.arange(len(series))
        y = series[var]

        slope, intercept, r_value, p_value, std_err = linregress(x_numeric, y)
        trend_line = slope * x_numeric + intercept

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(series['Date'], y, label='Original Series', color='royalblue', linewidth=1.5)
        ax.plot(series['Date'], trend_line, color='red', linestyle='--', label=f'Linear Trend (R²: {r_value**2:.2f})', linewidth=2)
        title_var = var.replace("_", " ").title()
        ax.set_title(f'Time Series and Trend for: {title_var}', fontsize=16, pad=15)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()

        plot_filename = os.path.join(output_dir, f'{var}_trend_plot.png')
        try:
            plt.savefig(plot_filename, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"   Academic-style plot for '{var}' saved.")
        except Exception as e:
            print(f"   ERROR saving plot for '{var}': {e}")
        finally:
            plt.close(fig)

    print("--- Trend Plotting Complete ---")

def plot_event_indicators(df: pd.DataFrame, vars_to_plot: list, output_dir: str):
    """
    Plots multiple event indicator variables on separate, vertically-stacked subplots
    within a single figure, using academic style.
    """
    print("\n--- Plotting Event Indicator Variables ---")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create 3 subplots stacked vertically, sharing the same x-axis
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

    # Ensure axes is always a list for consistent handling
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, var in zip(axes, vars_to_plot):
        if var in df.columns:
            # Plot each variable on its own subplot
            ax.plot(df['Date'], df[var], linewidth=1.5, alpha=0.9, color='darkslateblue')
            # Use the y-label to identify the variable clearly
            ax.set_ylabel(var.replace("_", " ").title(), fontsize=12)
            # Add grid to each subplot
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.tick_params(axis='y', labelsize=10)
        else:
            print(f"  Warning: Event variable '{var}' not found. Skipping.")
            # Display a message on the subplot itself if data is missing
            ax.text(0.5, 0.5, f"'{var}' not found", ha='center', va='center', transform=ax.transAxes)

    # Set a single main title for the entire figure
    fig.suptitle('Event Indicator Variables Over Time', fontsize=16)
    # Set the x-axis label only on the bottom-most plot
    axes[-1].set_xlabel('Date', fontsize=14)
    axes[-1].tick_params(axis='x', labelsize=12)

    # Adjust layout to prevent the main title from overlapping subplots
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_filename = os.path.join(output_dir, 'event_indicators_subplots.png')
    try:
        plt.savefig(plot_filename, dpi=600, bbox_inches='tight', facecolor='white')
        print("   Stacked event indicator plot saved.")
    except Exception as e:
        print(f"   ERROR saving event plot: {e}")
    finally:
        plt.close(fig)

    print("--- Event Plotting Complete ---")

def perform_adf_test(series: pd.Series, variable_name: str, test_round: str, regression_type: str) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    if not STATSMODELS_AVAILABLE:
        return None
    
    try:
        clean_series = series.dropna()
        if len(clean_series) < 20: 
            return None
        result = adfuller(clean_series, regression=regression_type, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        return {
            'Variable': variable_name, 'Test_Round': test_round,
            'Test_Type': 'Constant' if regression_type == 'c' else 'Constant & Trend',
            'ADF_Statistic': round(result[0], 4), 'P_Value': round(p_value, 4),
            'Lags_Used': result[2], 'Is_Stationary_5%': 'Yes' if is_stationary else 'No',
            'Critical_Value_1%': round(result[4]['1%'], 4), 'Critical_Value_5%': round(result[4]['5%'], 4),
        }
    except Exception: 
        return None

def analyze_variable_stationarity(df: pd.DataFrame) -> Tuple[Dict, List]:
    """Analyze stationarity of all variables using ADF tests."""
    print("\n--- Starting Stationarity Analysis (ADF Tests) ---")
    
    if not STATSMODELS_AVAILABLE:
        print("WARNING: Statsmodels not available! Skipping stationarity analysis.")
        return {'I(0)': [], 'I(1)': [], 'I(2)': []}, []
    
    print(f"  Manual specification: Testing {VARS_WITH_TREND} with a trend component.")
    integration_orders = {'I(0)': [], 'I(1)': [], 'I(2)': []}
    adf_test_results = []
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

    for i, var in enumerate(numeric_vars):
        print(f"  ({i+1}/{len(numeric_vars)}) Testing: {var}")
        is_stationary_in_levels = False
        result = perform_adf_test(df[var], var, 'Initial', 'ct' if var in VARS_WITH_TREND else 'c')
        if result:
            adf_test_results.append(result)
            if result['Is_Stationary_5%'] == 'Yes': 
                is_stationary_in_levels = True

        if is_stationary_in_levels:
            integration_orders['I(0)'].append(var)
            print("    -> Result: I(0) - Stationary in levels.")
            continue

        result_d = perform_adf_test(df[var].diff(), var, 'First Difference', 'c')
        if result_d: 
            adf_test_results.append(result_d)
        if result_d and result_d['Is_Stationary_5%'] == 'Yes':
            integration_orders['I(1)'].append(var)
            print("    -> Result: I(1) - Stationary after first difference.")
            continue

        result_d2 = perform_adf_test(df[var].diff().diff(), var, 'Second Difference', 'c')
        if result_d2: 
            adf_test_results.append(result_d2)
        integration_orders['I(2)'].append(var)
        print(" -> Result: I(2) - Stationary after second difference." if result_d2 and result_d2['Is_Stationary_5%'] == 'Yes' else "    -> Result: I(2) - Non-stationary (fallback).")
    
    print("--- Stationarity Analysis Complete ---")
    return integration_orders, adf_test_results

def analyze_cointegration(df: pd.DataFrame, i1_vars: list) -> Tuple[List, List]:
    """Analyze cointegration relationships using Engle-Granger test."""
    print("\n--- Starting Cointegration Analysis (Engle-Granger) ---")
    
    if not STATSMODELS_AVAILABLE:
        print("WARNING: Statsmodels not available! Skipping cointegration analysis.")
        return [], []
    
    target_var = 'unemp_rate'
    cointegrated_vars, coint_test_results = [], []
    if target_var not in i1_vars:
        print(f"  Warning: Target variable '{target_var}' is not I(1). Skipping.")
        return [], []
    
    for i, var in enumerate(i1_vars):
        if var == target_var: 
            continue
        print(f"  ({i+1}/{len(i1_vars)-1}) Testing '{target_var}' vs '{var}'...")
        try:
            y, x = df[target_var], df[var]
            merged = pd.concat([y, x], axis=1).dropna()
            coint_stat, p_value, crit_values = coint(merged.iloc[:, 0], merged.iloc[:, 1], trend='c', autolag='AIC')
            is_cointegrated = p_value < 0.05
            coint_test_results.append({
                'Target': target_var, 'Variable': var, 'EG_Statistic': round(coint_stat, 4),
                'P_Value': round(p_value, 4), 'Is_Cointegrated_5%': 'Yes' if is_cointegrated else 'No',
                'Critical_Value_1%': round(crit_values[0], 4), 'Critical_Value_5%': round(crit_values[1], 4),
            })
            if is_cointegrated:
                cointegrated_vars.append(var)
                print(f"    -> Result: Cointegrated (p-value: {p_value:.4f})")
        except Exception as e:
            print(f"    -> ERROR testing cointegration for '{var}': {e}")
            continue
    
    print("--- Cointegration Analysis Complete ---")
    return cointegrated_vars, coint_test_results

def save_excel_report(adf_results: list, coint_results: list, integration_orders: dict):
    """Save detailed stationarity and cointegration results to Excel."""
    report_path = os.path.join(BASE_DIR, STATIONARITY_EXCEL_REPORT)
    print(f"\nSaving detailed report to '{report_path}'...")
    try:
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            pd.DataFrame(adf_results).to_excel(writer, sheet_name='ADF_Test_Details', index=False)
            if coint_results: 
                pd.DataFrame(coint_results).to_excel(writer, sheet_name='Cointegration_Test_Details', index=False)
            pd.DataFrame.from_dict(integration_orders, orient='index').transpose().to_excel(writer, sheet_name='Integration_Order_Summary', index=False)
        print(" Excel report saved successfully.")
    except Exception as e:
        print(f" ERROR: Could not save Excel report. Error: {e}")

# %% PART 3: Feature Engineering Functions

def create_differences_and_drop_cointegrated(df: pd.DataFrame, stationarity_results: dict) -> pd.DataFrame:
    """
    Drops cointegrated variables, then creates first/second differences for I(1)/I(2) variables.
    """
    print("\nCreating differences and dropping cointegrated variables...")
    
    i1_vars = stationarity_results.get('I1_variables', [])
    i2_vars = stationarity_results.get('I2_variables', [])
    cointegrated_vars = stationarity_results.get('cointegrated_variables', [])
    
    df_copy = df.copy()
    
    # First, remove the cointegrated variables entirely from the dataset
    if cointegrated_vars:
        df_copy.drop(columns=cointegrated_vars, inplace=True, errors='ignore')
        print(f"  - Dropped {len(cointegrated_vars)} cointegrated variables: {cointegrated_vars}")
    
    # Now, proceed with differencing the remaining non-stationary variables
    i1_to_difference = [v for v in i1_vars if v not in cointegrated_vars]
    i2_to_difference = [v for v in i2_vars if v not in cointegrated_vars]

    print(f"  - Differencing {len(i1_to_difference)} I(1) variables.")
    print(f"  - Differencing {len(i2_to_difference)} I(2) variables.")

    # Create first differences for the remaining I(1) variables
    for var in i1_to_difference:
        if var in df_copy.columns:
            df_copy[f"d_{var}"] = df_copy[var].diff()
    
    # Create second differences for the remaining I(2) variables
    for var in i2_to_difference:
        if var in df_copy.columns:
            df_copy[f"d2_{var}"] = df_copy[var].diff().diff()
            
    # Drop the original levels of the variables that were differenced
    vars_to_drop = i1_to_difference + i2_to_difference
    df_transformed = df_copy.drop(columns=vars_to_drop, errors='ignore')
    
    # We will do a final dropna after all lagging is complete
    print(f" Transformation complete. Shape before lagging: {df_transformed.shape}")
    return df_transformed

def create_predictor_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Creates 1-month lags for specific predictors to account for publication delays."""
    print("\nCreating 1-month lags for key economic predictors...")
    
    df_lagged = df.copy()
    vars_to_lag_config = {
        'claimant_count': 1,
        'retail_sales_index': 1,
        'UK_EPU_Index': 1,
        'ftse_vol_3m': 1
    }
    vars_to_drop = []
    
    for base_var, lag_period in vars_to_lag_config.items():
        # Check for differenced version first, then original
        var_to_lag = f"d_{base_var}"
        if var_to_lag not in df_lagged.columns:
            var_to_lag = base_var

        if var_to_lag in df_lagged.columns:
            new_col_name = f"{var_to_lag}_L{lag_period}"
            df_lagged[new_col_name] = df_lagged[var_to_lag].shift(lag_period)
            vars_to_drop.append(var_to_lag)
            print(f"  - Created '{new_col_name}' from '{var_to_lag}' and marked for removal.")
        else:
            print(f"  - Warning: Neither '{base_var}' nor 'd_{base_var}' found for lagging.")
            
    df_final = df_lagged.drop(columns=vars_to_drop, errors='ignore')
    print(" Predictor lagging complete.")
    return df_final

def create_ar_only_lags(df: pd.DataFrame, num_lags: int = 3) -> pd.DataFrame:
    """Create lags ONLY for the differenced unemployment variable."""
    print(f"\nCreating {num_lags} auto-regressive lags for unemployment rate...")
    
    unemployment_var = 'd_unemp_rate'
    if unemployment_var not in df.columns:
        print(f"  ! Warning: '{unemployment_var}' not found. Cannot create AR lags.")
        return df

    result_df = df.copy()
    for lag in range(1, num_lags + 1):
        result_df[f"{unemployment_var}_L{lag}"] = result_df[unemployment_var].shift(lag)
    
    # Reorder columns to have the target and its lags first
    other_vars = [col for col in df.columns if col not in [unemployment_var, 'Date']]
    new_order = ['Date', unemployment_var] + [f"{unemployment_var}_L{lag}" for lag in range(1, num_lags + 1)] + other_vars
    new_order_existing = [col for col in new_order if col in result_df.columns]
    result_df = result_df[new_order_existing]

    print(" AR Lag creation complete.")
    return result_df

# %% Main Pipeline Function

def main():
    """Main function to execute the complete pipeline."""
    print("\n" + "="*80)
    print("COMPLETE ECONOMIC DATASET PROCESSING PIPELINE")
    print("="*80)
    
    # Set up matplotlib for academic plotting
    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        print("Using LaTeX for academic plot rendering.")
    except RuntimeError:
        print("Warning: LaTeX distribution not found. Using default font settings.")
        plt.rcdefaults()
    
    # Change to data directory
    os.chdir(BASE_DIR)
    
    # PART 1: SEASONAL ADJUSTMENT AND DATASET COMBINATION
    print("\nPART 1: SEASONAL ADJUSTMENT AND DATASET COMBINATION")
    print("-" * 60)
    
    try:
        # Load initial datasets
        traditional_data, gt_data_raw = load_datasets_from_part1()
        
        # Apply seasonal adjustment to Google Trends data
        gt_data_adjusted = apply_seasonal_adjustment_to_gt_data(gt_data_raw)
        
        # Combine datasets
        final_combined = combine_final_datasets(traditional_data, gt_data_adjusted)
        
        # Save intermediate files
        gt_data_adjusted.to_excel(GT_SEASONALLY_ADJUSTED, index=False)
        final_combined.to_excel(COMPLETE_DATASET_STL, index=False)
        print("Part 1 datasets saved.")
        
    except Exception as e:
        print(f"ERROR in Part 1: {e}")
        return
    
    # PART 2: STATIONARITY AND COINTEGRATION ANALYSIS
    print("\nPART 2: STATIONARITY AND COINTEGRATION ANALYSIS")
    print("-" * 60)
    
    try:
        # Set up plot directory
        plot_dir_path = os.path.join(BASE_DIR, PLOT_OUTPUT_DIR)
        
        # Generate plots
        plot_trend_variables(final_combined, VARS_WITH_TREND, plot_dir_path)
        plot_event_indicators(final_combined, EVENT_VARS, plot_dir_path)
        
        # Run stationarity analysis
        integration_orders, adf_results = analyze_variable_stationarity(final_combined)
        
        # Run cointegration analysis
        i1_vars = integration_orders.get('I(1)', [])
        cointegrated_vars, coint_results = analyze_cointegration(final_combined, i1_vars)
        
        # Prepare results for next stage
        stationarity_results = {
            'I0_variables': integration_orders.get('I(0)', []),
            'I1_variables': i1_vars,
            'I2_variables': integration_orders.get('I(2)', []),
            'cointegrated_variables': cointegrated_vars
        }
        
        # Save results
        json_output_path = os.path.join(BASE_DIR, STATIONARITY_RESULTS_JSON)
        with open(json_output_path, 'w') as f:
            json.dump(stationarity_results, f, indent=4)
        print("Stationarity results saved to JSON.")
        
        save_excel_report(adf_results, coint_results, stationarity_results)
        
        # Print summary
        print("\n--- Summary of Stationarity Analysis ---")
        print(f"  I(0) Variables: {len(stationarity_results['I0_variables'])}")
        print(f"  I(1) Variables: {len(stationarity_results['I1_variables'])}")
        print(f"  I(2) Variables: {len(stationarity_results['I2_variables'])}")
        print(f"  Cointegrated with unemp_rate: {len(stationarity_results['cointegrated_variables'])}")
        
    except Exception as e:
        print(f"ERROR in Part 2: {e}")
        # Create fallback results if stationarity analysis fails
        stationarity_results = {'I0_variables': [], 'I1_variables': [], 'I2_variables': [], 'cointegrated_variables': []}

    # PART 3: FEATURE ENGINEERING (DIFFERENCES AND LAGS)
    print("\nPART 3: FEATURE ENGINEERING (DIFFERENCES AND LAGS)")
    print("-" * 60)
    
    try:
        # Create differenced variables and drop cointegrated ones
        df_with_diffs = create_differences_and_drop_cointegrated(final_combined, stationarity_results)
        
        # Create lags for economic predictors
        df_with_predictor_lags = create_predictor_lags(df_with_diffs)
        
        # Create autoregressive lags for unemployment rate
        df_with_all_lags = create_ar_only_lags(df_with_predictor_lags)
        
        # Final cleanup - drop all NaN values
        final_dataset = df_with_all_lags.dropna().reset_index(drop=True)
        print(f"\nFinal dataset shape after dropping all NaNs: {final_dataset.shape}")
        
        # Save final dataset
        save_path = os.path.join(BASE_DIR, FINAL_DATASET)
        final_dataset.to_excel(save_path, index=False)
        print(f"Final dataset saved to '{FINAL_DATASET}'")
        
    except Exception as e:
        print(f"ERROR in Part 3: {e}")
        return
    
    # PIPELINE COMPLETION
    print("\n" + "="*80)
    print("PIPELINE COMPLETION SUMMARY")
    print("="*80)
    
    print("\nSUCCESSFULLY COMPLETED ALL STAGES:")
    print("  - Part 1: Seasonal adjustment and dataset combination")
    print("  - Part 2: Stationarity and cointegration analysis") 
    print("  - Part 3: Feature engineering (differences and lags)")
    
    print("\nFILES CREATED:")
    print(f"  * {GT_SEASONALLY_ADJUSTED}")
    print(f"  * {COMPLETE_DATASET_STL}")
    print(f"  * {STATIONARITY_RESULTS_JSON}")
    print(f"  * {STATIONARITY_EXCEL_REPORT}")
    print(f"  * {FINAL_DATASET}")
    print(f"  * Plots in '{PLOT_OUTPUT_DIR}/' directory")
    
    print("\nFINAL DATASET READY FOR ANALYSIS:")
    print(f"  * Shape: {final_dataset.shape}")
    print(f"  * Variables: {final_dataset.shape[1] - 1} (excluding Date)")
    print(f"  * Time period: {final_dataset['Date'].min()} to {final_dataset['Date'].max()}")
    
    # Reset matplotlib settings
    plt.rcdefaults()
    
    print("\nCOMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("Dataset is ready for LASSO analysis and modeling.")
    print("="*80)
    
    return final_dataset

if __name__ == "__main__":
    try:
        final_dataset = main()
        print("\nPipeline executed successfully!")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()