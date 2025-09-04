import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

# %% MODEL ESTIMATION AND ANALYSIS

def load_and_prepare_data(filename):
    """Loads and prepares data from the specified CSV file."""
    try:
        # Check if the file exists before attempting to change directory or load
        if not os.path.exists(filename):
             raise FileNotFoundError(f"File {filename} not found in the current directory '{os.getcwd()}'. Please ensure the file is in the correct directory or provide a full path.")
        
        df = pd.read_excel(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"Warning: Found missing values in columns: {missing_data[missing_data > 0].to_dict()}")
        print(f"Dataset loaded: {df.shape}")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        return df
    except FileNotFoundError as e:
        raise e # Re-raise the specific file not found error
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
        
def analyze_dependent_variable_variation(df, target_var):
    """
    Analyzes and prints the variation (std dev, variance) of the dependent variable
    both overall and across different economic periods.
    """
    print("\n Analyzing Variation of the Dependent Variable ")
    if target_var not in df.columns:
        print(f"   Warning: Target variable '{target_var}' not found. Skipping analysis.")
        return

    # Define economic periods
    periods = {
        'Pre-Recession': ('2004-01-01', '2007-11-30'),
        'Great Recession': ('2007-12-01', '2009-06-30'),
        'Post-Recession Recovery': ('2009-07-01', '2016-05-31'),
        'Brexit Uncertainty': ('2016-06-01', '2020-02-29'),
        'COVID-19 & Aftermath': ('2020-03-01', '2025-12-31')
    }

    results_data = []

    # Overall statistics
    results_data.append({
        'Period': 'Overall Sample',
        'Start Date': df['Date'].min().strftime('%Y-%m'),
        'End Date': df['Date'].max().strftime('%Y-%m'),
        'Mean': df[target_var].mean(),
        'Std. Dev.': df[target_var].std(),
        'Variance': df[target_var].var()
    })

    # Per-period statistics
    for name, (start, end) in periods.items():
        period_df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        if not period_df.empty:
            results_data.append({
                'Period': name,
                'Start Date': pd.to_datetime(start).strftime('%Y-%m'),
                'End Date': period_df['Date'].max().strftime('%Y-%m'),
                'Mean': period_df[target_var].mean(),
                'Std. Dev.': period_df[target_var].std(),
                'Variance': period_df[target_var].var()
            })

    summary_df = pd.DataFrame(results_data).set_index('Period')
    print(f"  Variation analysis for: '{target_var}'")
    print(summary_df[['Start Date', 'End Date', 'Mean', 'Std. Dev.', 'Variance']].round(4).to_string())
    print("-" * 65)
    return summary_df

def prepare_variables(df):
    """Prepare variables for analysis by separating them into sets for each model."""
    print("\nPreparing variables for analysis...")
    target_var = 'd_unemp_rate'
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataset")

    # Define variable groups 
    benchmark_ar_vars = ['d_unemp_rate_L1', 'd_unemp_rate_L2', 'd_unemp_rate_L3']
    other_econ_vars = [
        'd_claimant_count_L1',
        'd_UK_EPU_Index_L1',
        'ftse_returns',
        'retail_sales_index_L1',
        'ftse_vol_3m_L1'
    ]
    all_traditional_econ_vars = benchmark_ar_vars + other_econ_vars

    # Filter to existing columns and identify GT variables 
    predictors_benchmark = [v for v in benchmark_ar_vars if v in df.columns]
    existing_all_econ = [v for v in all_traditional_econ_vars if v in df.columns]
    all_potential_predictors = [c for c in df.columns if c not in ['Date', target_var]]
    gt_vars = [v for v in all_potential_predictors if v not in existing_all_econ]

    # Define final predictor sets for each model type 
    predictors_ar_gt = predictors_benchmark + gt_vars
    predictors_all = existing_all_econ + gt_vars

    # Print summary 
    print(f"  Target variable: {target_var}")
    print("\nVariable sets for models:")
    print(f"  1. Benchmark Model Predictors (AR terms): {len(predictors_benchmark)}")
    print(f"  2. AR+GT Model Predictors: {len(predictors_ar_gt)}")
    print(f"  3. All Predictors Model (AR+Econ+GT): {len(predictors_all)}")
    print(f"     - Google Trends variables identified: {len(gt_vars)}")

    return target_var, predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars

def create_rolling_windows(df, train_length=48, step_size=6):
    """Create rolling windows for in-sample analysis, dropping final partial window."""
    print("\nCreating rolling windows:")
    print(f"  Training window: {train_length} months, Step size: {step_size} months")
    total_obs = len(df)
    in_sample_windows = []

    for i in range(0, total_obs - train_length + 1, step_size):
        in_sample_windows.append({
            'window_id': len(in_sample_windows) + 1, 'type': 'in_sample', 'start_idx': i, 'end_idx': i + train_length,
            'start_date': df.iloc[i]['Date'], 'end_date': df.iloc[i + train_length - 1]['Date'], 'n_obs': train_length
        })

    if not in_sample_windows:
        print("Warning: No full windows could be created with the given parameters.")
        return []

    last_window_end_idx = in_sample_windows[-1]['end_idx']
    dropped_obs_count = total_obs - last_window_end_idx

    if dropped_obs_count > 0:
        print(f"  Note: Dropped the final {dropped_obs_count} observations that did not fit into a full window of size {train_length}.")
        print(f"  Data from {df.iloc[last_window_end_idx]['Date'].strftime('%Y-%m-%d')} to {df.iloc[-1]['Date'].strftime('%Y-%m-%d')} will not be used.")
    else:
        print("  Note: All observations were used, fitting perfectly into the windows.")

    print(f"Created {len(in_sample_windows)} in-sample windows from {in_sample_windows[0]['start_date'].strftime('%Y-%m')} to {in_sample_windows[-1]['end_date'].strftime('%Y-%m')}")
    return in_sample_windows

def estimate_global_parameters(df, target_var, predictor_vars):
    """Estimate optimal LASSO and ElasticNet parameters for a GIVEN set of predictors."""
    X = df[predictor_vars].copy().fillna(method='ffill').fillna(method='bfill')
    y = df[target_var].copy().fillna(method='ffill').fillna(method='bfill')
    
    variable_vars = X.columns[X.std() > 1e-10].tolist()
    X = X[variable_vars]
    
    if X.empty: raise ValueError("No valid predictor variables after removing constants")
    
    X_scaled = StandardScaler().fit_transform(X)
    n_splits = min(15, len(df) // 15)
    if n_splits < 2: n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    alpha_range = np.logspace(-8, 2, 200)

    lasso_cv = LassoCV(alphas=alpha_range, cv=tscv, max_iter=10000, random_state=42, n_jobs=-1).fit(X_scaled, y)
    
    if X.shape[1] > 1:
        elasticnet_cv = ElasticNetCV(alphas=alpha_range, l1_ratio=np.linspace(0.1, 0.9, 9), cv=tscv, max_iter=10000, random_state=42, n_jobs=-1).fit(X_scaled, y)
        el_lambda, el_l1_ratio = elasticnet_cv.alpha_, elasticnet_cv.l1_ratio_
    else: 
        el_lambda, el_l1_ratio = lasso_cv.alpha_, 1.0

    print(f"  Optimal LASSO λ: {lasso_cv.alpha_:.6f}")
    print(f"  Optimal ElasticNet λ: {el_lambda:.6f}, l1_ratio: {el_l1_ratio:.3f}")
    
    return lasso_cv.alpha_, el_lambda, el_l1_ratio, variable_vars

def _fit_and_evaluate_model(X_data, y_data, model, is_ols=False):
    """Helper function to fit a model, predict, and calculate metrics."""
    if is_ols:
        X_data = sm.add_constant(X_data)
        fitted_model = sm.OLS(y_data, X_data).fit()
        y_pred = fitted_model.fittedvalues
        coeffs = dict(zip(X_data.columns, fitted_model.params))
        pvalues = dict(zip(X_data.columns, fitted_model.pvalues))
        tvalues = dict(zip(X_data.columns, fitted_model.tvalues))
        r2_adj = fitted_model.rsquared_adj
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        fitted_model = model.fit(X_scaled, y_data)
        y_pred = fitted_model.predict(X_scaled)
        coeffs = dict(zip(X_data.columns, fitted_model.coef_))
        pvalues, tvalues, r2_adj = None, None, None
    
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    r2 = r2_score(y_data, y_pred)
    return rmse, r2, r2_adj, coeffs, pvalues, tvalues

def estimate_window_in_sample(window_data, target_var, model_predictors, hyperparams, window_id):
    """Estimate all model sets for a single window using their specific hyperparameters."""
    y = window_data[target_var].copy().fillna(method='ffill').fillna(method='bfill')
    results = {'window_id': window_id}
    
    for model_name, predictors in model_predictors.items():
        X = window_data[predictors].copy().fillna(method='ffill').fillna(method='bfill')
        X = X[X.columns[X.std() > 1e-8]]
        if X.empty: continue

        model_params = hyperparams[model_name]
        
        lasso_model = Lasso(alpha=model_params['lasso_lambda'], max_iter=5000, random_state=42)
        _, _, _, shrunken_coeffs_lasso, _, _ = _fit_and_evaluate_model(X, y, lasso_model)
        selected_vars = X.columns[np.abs(list(shrunken_coeffs_lasso.values())) > 1e-10].tolist()
        
        results.update({
            f'lasso_shrunken_coeffs_{model_name}': shrunken_coeffs_lasso,
            f'lasso_n_selected_{model_name}': len(selected_vars), 
            f'lasso_selected_vars_{model_name}': selected_vars
        })
        
        if selected_vars:
            rmse_ols, r2_ols, r2_adj_ols, ols_coeffs, ols_pvalues, _ = _fit_and_evaluate_model(X[selected_vars], y, None, is_ols=True)
        else:
            mean_pred = np.full(len(y), y.mean())
            rmse_ols, r2_ols, r2_adj_ols = np.sqrt(mean_squared_error(y, mean_pred)), 0, 0
            ols_coeffs, ols_pvalues = {}, {}
            
        results.update({
            f'rmse_lasso_ols_{model_name}': rmse_ols, 
            f'r2_lasso_ols_{model_name}': r2_ols, 
            f'r2_adj_lasso_ols_{model_name}': r2_adj_ols,
            f'lasso_ols_coeffs_{model_name}': ols_coeffs,
            f'lasso_ols_pvalues_{model_name}': ols_pvalues
        })
        
        if model_name == 'benchmark': continue

        en_model = ElasticNet(alpha=model_params['elasticnet_lambda'], l1_ratio=model_params['elasticnet_l1_ratio'], max_iter=5000, random_state=42)
        _, _, _, shrunken_coeffs_en, _, _ = _fit_and_evaluate_model(X, y, en_model)
        selected_vars_en = X.columns[np.abs(list(shrunken_coeffs_en.values())) > 1e-10].tolist()

        results.update({
            f'elasticnet_shrunken_coeffs_{model_name}': shrunken_coeffs_en,
            f'elasticnet_n_selected_{model_name}': len(selected_vars_en),
            f'elasticnet_selected_vars_{model_name}': selected_vars_en
        })
        
        if selected_vars_en:
            rmse_ols, r2_ols, r2_adj_ols, ols_coeffs, ols_pvalues, _ = _fit_and_evaluate_model(X[selected_vars_en], y, None, is_ols=True)
        else:
            mean_pred = np.full(len(y), y.mean())
            rmse_ols, r2_ols, r2_adj_ols = np.sqrt(mean_squared_error(y, mean_pred)), 0, 0
            ols_coeffs, ols_pvalues = {}, {}

        results.update({
            f'rmse_elasticnet_ols_{model_name}': rmse_ols, 
            f'r2_elasticnet_ols_{model_name}': r2_ols, 
            f'r2_adj_elasticnet_ols_{model_name}': r2_adj_ols,
            f'elasticnet_ols_coeffs_{model_name}': ols_coeffs,
            f'elasticnet_ols_pvalues_{model_name}': ols_pvalues
        })
                                
    return results

def run_rolling_analysis(df, in_sample_windows, target_var, model_predictors, hyperparams):
    """Run in-sample rolling window analysis for all model sets."""
    print(f"\nRunning Rolling Window Analysis for {len(in_sample_windows)} windows...")
    all_results = []
    for i, window in enumerate(in_sample_windows):
        if (i + 1) % 5 == 0: print(f"  ... processing window {i+1}/{len(in_sample_windows)}")
        window_data = df.iloc[window['start_idx']:window['end_idx']].copy()
        try:
            window_results = estimate_window_in_sample(
                window_data, target_var, model_predictors, hyperparams, window['window_id']
            )
            if window_results:
                window_results.update({'start_date': window['start_date'], 'end_date': window['end_date']})
                all_results.append(window_results)
        except Exception as e:
            print(f"   Error in window {window['window_id']}: {str(e)}")
            continue
    print(f"\nRolling Analysis Complete! {len(all_results)} successful estimations.")
    return all_results

def save_model_results(all_results, target_var, model_predictors, gt_vars, hyperparams, filename):
    """Save all model results and metadata to a pickle file."""
    print(f"\nSaving model results to {filename}...")
    model_data = {
        'results': all_results,
        'metadata': {
            'target_var': target_var,
            'model_predictors': model_predictors,
            'gt_vars': gt_vars,
            'hyperparameters_by_model': hyperparams,
            'n_windows': len(all_results)
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(" Model results saved successfully.")

def main_lasso_elasticnet_analysis(filename, output_filename):
    """Main function to run the complete multi-model analysis with model-specific hyperparameters."""
    print("="*80)
    print("MULTI-MODEL ANALYSIS with MODEL-SPECIFIC HYPERPARAMETERS")
    print("="*80)
    try:
        df = load_and_prepare_data(filename)
        in_sample_windows = create_rolling_windows(df, train_length=48, step_size=6)
        
        target_var, predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars = prepare_variables(df)
        
        analyze_dependent_variable_variation(df, target_var)

        model_predictors = {
            "benchmark": predictors_benchmark,
            "ar_gt": predictors_ar_gt,
            "all": predictors_all
        }
        
        hyperparams = {}
        for model_name, predictors in model_predictors.items():
            print(f"\n Estimating Global Hyperparameters for '{model_name.upper()}' Model ")
            l_lambda, e_lambda, e_l1, _ = estimate_global_parameters(df, target_var, predictors)
            hyperparams[model_name] = {'lasso_lambda': l_lambda, 'elasticnet_lambda': e_lambda, 'elasticnet_l1_ratio': e_l1}

        all_results = run_rolling_analysis(
            df, in_sample_windows, target_var, model_predictors, hyperparams
        )

        if not all_results:
            print("No successful window estimations completed!")
            return None

        save_model_results(all_results, target_var, model_predictors, gt_vars, hyperparams, output_filename)
        
        print("\n" + "="*80 + "\nANALYSIS COMPLETE!\n" + "="*80)

        df_results = pd.DataFrame(all_results)
        benchmark_rmse_col = 'rmse_lasso_ols_benchmark'

        print("Average Variables Selected (Post-LASSO):")
        for model_name in model_predictors.keys():
            col = f'lasso_n_selected_{model_name}'
            if col in df_results.columns:
                print(f"  - {model_name.upper()}: {df_results[col].mean():.1f} variables")

        for model_type in ['ar_gt', 'all']:
            for reg_type in ['lasso', 'elasticnet']:
                model_rmse_col = f'rmse_{reg_type}_ols_{model_type}'
                if model_rmse_col not in df_results.columns: continue
                
                improvement = 100 * (df_results[benchmark_rmse_col] - df_results[model_rmse_col]) / df_results[benchmark_rmse_col]
                
                print(f"\nPerformance: {reg_type.upper()} ({model_type.upper()}) vs. Benchmark (AR)")
                print(f"  Outperforms Benchmark: {np.sum(improvement > 0)}/{len(improvement)} windows ({np.mean(improvement > 0):.1%})")
                print(f"  Average RMSE Reduction: {np.mean(improvement):.2f}%")

        return all_results

    except Exception as e:
        print(f" CRITICAL ERROR in model estimation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# %% SPREADSHEET GENERATION

def get_economic_period(date):
    """Classifies a date into a predefined economic period."""
    if pd.to_datetime(date).year < 2008: return "Pre-Recession"
    if pd.to_datetime(date).year < 2010: return "Great Recession"
    if pd.to_datetime(date).year < 2016: return "Post-Recession Recovery"
    if pd.to_datetime(date).year < 2020: return "Brexit Uncertainty"
    return "COVID-19 Pandemic"

def get_variable_type(var, metadata):
    """Identifies if a variable is AR, Traditional, or Google Trends."""
    if var in metadata['model_predictors']['benchmark']:
        return 'AR Lag'
    if var in metadata['gt_vars']:
        return 'Google Trends'
    if var in metadata['model_predictors']['all']:
        return 'Traditional'
    return 'Other'

def create_performance_summary(results):
    """Creates a summary DataFrame of model performance metrics."""
    summary_data = []
    # Define models and their keys
    models = {
        '1. Benchmark (AR)': ('benchmark', 'lasso'),
        '2. AR+GT (LASSO)': ('ar_gt', 'lasso'),
        '3. AR+GT (ElasticNet)': ('ar_gt', 'elasticnet'),
        '4. All Predictors (LASSO)': ('all', 'lasso'),
        '5. All Predictors (ElasticNet)': ('all', 'elasticnet'),
    }
    
    for model_name, (model_key, reg_type) in models.items():
        summary_data.append({
            'Model': model_name,
            'Avg. Vars Selected': np.mean([r.get(f'{reg_type}_n_selected_{model_key}', np.nan) for r in results]),
            'Std. Dev. of Vars Selected': np.std([r.get(f'{reg_type}_n_selected_{model_key}', np.nan) for r in results]),
            'Avg. R-Squared': np.mean([r.get(f'r2_{reg_type}_ols_{model_key}', np.nan) for r in results]),
            'Std. Dev. of R-Squared': np.std([r.get(f'r2_{reg_type}_ols_{model_key}', np.nan) for r in results]),
            'Avg. Adj. R-Squared': np.mean([r.get(f'r2_adj_{reg_type}_ols_{model_key}', np.nan) for r in results]),
            'Std. Dev. of Adj. R-Squared': np.std([r.get(f'r2_adj_{reg_type}_ols_{model_key}', np.nan) for r in results]),
            'Avg. RMSE': np.mean([r.get(f'rmse_{reg_type}_ols_{model_key}', np.nan) for r in results]),
            'Std. Dev. of RMSE': np.std([r.get(f'rmse_{reg_type}_ols_{model_key}', np.nan) for r in results]),
        })
    df = pd.DataFrame(summary_data)
    # Reorder columns for clarity
    column_order = [
        'Model', 
        'Avg. Vars Selected', 'Std. Dev. of Vars Selected',
        'Avg. RMSE', 'Std. Dev. of RMSE',
        'Avg. R-Squared', 'Std. Dev. of R-Squared',
        'Avg. Adj. R-Squared', 'Std. Dev. of Adj. R-Squared'
    ]
    return df[column_order]

def create_window_results_summary(results):
    """Creates a detailed DataFrame with results for each rolling window."""
    window_data = []
    for r in results:
        ar_rmse = r.get('rmse_lasso_ols_benchmark')
        
        row = {
            'Window_ID': r['window_id'],
            'Start_Date': r['start_date'],
            'End_Date': r['end_date'],
            'Period': get_economic_period(r['start_date']),
            'Benchmark_RMSE': ar_rmse,
            'Benchmark_R2': r.get('r2_lasso_ols_benchmark'),
            'Benchmark_Adj_R2': r.get('r2_adj_lasso_ols_benchmark'),
        }
        
        for model_key in ['ar_gt', 'all']:
            for reg_type in ['lasso', 'elasticnet']:
                model_rmse = r.get(f'rmse_{reg_type}_ols_{model_key}')
                if model_rmse is not None:
                    improvement = 100 * (ar_rmse - model_rmse) / ar_rmse if ar_rmse else 0
                    prefix = f"{model_key.upper()}_{reg_type.upper()}"
                    row[f'{prefix}_RMSE'] = model_rmse
                    row[f'{prefix}_R2'] = r.get(f'r2_{reg_type}_ols_{model_key}')
                    row[f'{prefix}_Adj_R2'] = r.get(f'r2_adj_{reg_type}_ols_{model_key}')
                    row[f'{prefix}_Improvement_%'] = improvement

        window_data.append(row)
    return pd.DataFrame(window_data)

def create_variable_selection_summary(results, metadata):
    """Creates a summary of variable selection frequency."""
    all_vars = sorted(list(set(metadata['model_predictors']['all'])))
    var_summary = []
    total_windows = len(results)
    
    for var in all_vars:
        var_info = {'Variable': var, 'Type': get_variable_type(var, metadata)}
        for model in ['benchmark', 'ar_gt', 'all']:
            for reg_type in ['lasso', 'elasticnet']:
                key = f'{reg_type}_selected_vars_{model}'
                if key not in results[0]: continue
                
                selection_count = sum(1 for r in results if var in r.get(key, []))
                var_info[f'{model.upper()}_{reg_type.upper()}_Selection_Freq'] = selection_count / total_windows
        var_summary.append(var_info)
        
    df = pd.DataFrame(var_summary)
    # Sort by selection frequency in the most complex LASSO model
    df = df.sort_values('ALL_LASSO_Selection_Freq', ascending=False)
    return df

def save_to_excel(dfs, filename='Model_In_Sample_Analysis_Results.xlsx'):
    """Saves multiple DataFrames to sheets in a single Excel file."""
    print(f"\nSaving analysis to Excel file: '{filename}'...")
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f" Successfully created Excel file with sheets: {', '.join(dfs.keys())}")
    except Exception as e:
        print(f" Error saving to Excel: {e}")

# %% VISUALIZATIONS

def add_crisis_shading(ax):
    """Adds shaded regions for major economic crises."""
    crisis_periods = {
        'Great Recession': ('2007-12-01', '2009-06-01'),
        'Brexit Uncertainty': ('2016-06-01', '2020-01-01'),
        'COVID-19 Pandemic': ('2020-05-01', '2023-05-01')
    }
    colors = ['salmon', 'lightskyblue', 'mediumpurple'] 
    for (period, (start, end)), color in zip(crisis_periods.items(), colors):
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.3, zorder=0)

def extract_coefficient_evolution(results, metadata):
    """Extracts shrunken coeffs and OLS coeffs for ElasticNet models across rolling windows."""
    print("Extracting coefficient evolution for Benchmark, AR+GT, and Full (Post-ElasticNet) models...")
    model_keys = {'benchmark': 'benchmark', 'ar_gt': 'ar_gt', 'full': 'all'}
    shrunken_coeff_evolution = {key: defaultdict(list) for key in model_keys}
    ols_coeff_evolution = {key: defaultdict(list) for key in model_keys}
    window_dates = [r['end_date'] for r in results]
    all_vars_by_model = {
        'benchmark': metadata['model_predictors']['benchmark'],
        'ar_gt': metadata['model_predictors']['ar_gt'],
        'full': metadata['model_predictors']['all']
    }
    for result in results:
        for key, model_name in model_keys.items():
            reg_type = 'lasso' if key == 'benchmark' else 'elasticnet'
            shrunken_coeffs = result.get(f'{reg_type}_shrunken_coeffs_{model_name}', {})
            ols_coeffs = result.get(f'{reg_type}_ols_coeffs_{model_name}', {})
            for var in all_vars_by_model[key]:
                shrunken_coeff_evolution[key][var].append(shrunken_coeffs.get(var, 0.0))
                ols_coeff_evolution[key][var].append(ols_coeffs.get(var, 0.0))
            ols_coeff_evolution[key]['const'].append(ols_coeffs.get('const', 0.0))
    print(" Coefficient extraction complete.")
    return shrunken_coeff_evolution, ols_coeff_evolution, window_dates

def get_top_variables_by_model(results, metadata):
    """Identify and rank ALL variables based on their selection frequency in the initial ElasticNet model."""
    print("Ranking variables based on initial ElasticNet selection frequency...")
    ranked_variables = {}
    total_windows = len(results)
    for model_key_map, predictors in {'benchmark': 'benchmark', 'ar_gt': 'ar_gt', 'full': 'all'}.items():
        var_importance = {}
        reg_type = 'lasso' if model_key_map == 'benchmark' else 'elasticnet'
        lookup_key = f'{reg_type}_selected_vars_{predictors}'
        all_potential_vars = metadata['model_predictors'][predictors]
        for var in all_potential_vars:
            selection_count = sum(1 for r in results if var in r.get(lookup_key, []))
            selection_frequency = selection_count / total_windows
            if selection_frequency > 0:
                var_importance[var] = selection_frequency
        sorted_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
        ranked_variables[model_key_map] = [var for var, _ in sorted_vars]
        print(f"  {model_key_map.capitalize()} model: Ranked {len(ranked_variables[model_key_map])} variables.")
        for i, (var, importance) in enumerate(sorted_vars[:5], 1):
            print(f"    Rank {i}. {var}: {importance:.4f}")
    return ranked_variables

def format_variable_name_latex(var_name, is_traditional):
    """Cleans and formats a variable name for LaTeX display in plots."""
    prefix = ""
    if var_name.startswith('d_'):
        prefix = r'\Delta '
        var_name = var_name[2:]
    if var_name.startswith('d2_'):
        prefix = r'\Delta_2 '
        var_name = var_name[3:]

    lag_match = re.search('_L(\d+)', 'var_name')
    subscript = f'_{{t-{lag_match.group(1)}}}' if lag_match else ""
    if lag_match:
        var_name = var_name[:lag_match.start()]

    var_name = var_name.replace('_', ' ').title()
    replacements = {
        'Unemp Rate': 'Unemployment Rate', 'Uk Epu Index': 'EPU Index', 
        'Ftse Returns': 'FTSE Returns', 'Claimant Count': 'Claimant Count', 
        'Cv Library': 'CV-Library', 'Job Centre': 'Job Centre', 
        'Jobseekers Allowance': 'JSA', 'Universal Credit': 'Universal Credit', 
        'Work From Home Jobs': 'WFH Jobs', 'Part Time Work': 'Part-Time Work', 
        'Retail Sales Index': 'Retail Sales Index', 'Ftse Vol 3M': 'FTSE Volatility'
    }
    if var_name in replacements:
        var_name = replacements[var_name]

    asterisk = '^*' if is_traditional else ''
    
    # Build the LaTeX formatted string 
    texttt_part = f'\\texttt{{{var_name}}}'
    return f'${prefix}{texttt_part}{subscript}{asterisk}$'

def plot_variable_selection_evolution(results, window_dates, output_dir):
    """
    Plots the evolution of the number of variables selected by ElasticNet/Lasso
    for each model across all rolling windows.
    """
    print("Generating plot for the evolution of selected variables...")

    n_vars_benchmark = [r.get('lasso_n_selected_benchmark', 0) for r in results]
    n_vars_ar_gt = [r.get('elasticnet_n_selected_ar_gt', 0) for r in results]
    n_vars_full = [r.get('elasticnet_n_selected_all', 0) for r in results]

    try:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
    except RuntimeError:
        print("Warning: LaTeX distribution not found. Plot labels will not be formatted.")
        plt.rcdefaults()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(window_dates, n_vars_benchmark, label='Benchmark (AR)', color='#2ca02c', linestyle=':', marker='x', markersize=5)
    ax.plot(window_dates, n_vars_ar_gt, label='AR+GT (ElasticNet)', color='#1f77b4', linestyle='--', marker='s', markersize=4)
    ax.plot(window_dates, n_vars_full, label='Full (ElasticNet)', color='#d62728', linestyle='-', marker='^', markersize=4)

    add_crisis_shading(ax)

    ax.set_title('Evolution of Number of Variables Selected per Window', fontsize=16, pad=15)
    ax.set_xlabel('End Date of Rolling Window', fontsize=12)
    ax.set_ylabel('Number of Selected Variables', fontsize=12)
    ax.legend(title='Model', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    max_vars = max(max(n_vars_benchmark), max(n_vars_ar_gt), max(n_vars_full))
    ax.set_ylim(0, max_vars + 3)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'variable_selection_evolution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcdefaults() 
    print(f" Variable selection evolution plot saved to {save_path}")

def plot_single_model_importance(results, metadata, output_dir, model_key, model_display_name, top_n=15):
    """Creates a bar chart for top N variables for a single model based on initial ElasticNet selection."""
    print(f"Generating ElasticNet variable importance plot for {model_display_name} model...")
    traditional_vars_set = set(metadata['model_predictors']['benchmark'] + [
        'd_claimant_count_L1', 'd_UK_EPU_Index_L1', 'ftse_returns',
        'retail_sales_index_L1', 'ftse_vol_3m_L1'
    ])
    selection_frequencies = {}
    total_windows = len(results)
    results_model_key = 'all' if model_key == 'full' else model_key
    reg_type = 'lasso' if model_key == 'benchmark' else 'elasticnet'
    lookup_key = f'{reg_type}_selected_vars_{results_model_key}'
    all_potential_vars = metadata['model_predictors'][results_model_key]

    for var in all_potential_vars:
        selection_count = sum(1 for r in results if var in r.get(lookup_key, []))
        frequency = selection_count / total_windows
        if frequency > 0:
            selection_frequencies[var] = frequency

    top_vars_series = pd.Series(selection_frequencies).nlargest(top_n)
    if top_vars_series.empty:
        print(f"No variables selected for {model_display_name} model.")
        return

    plot_df = top_vars_series.reset_index()
    plot_df.columns = ['Variable', 'Selection Frequency']
    variable_order = plot_df.sort_values('Selection Frequency', ascending=True)['Variable'].tolist()
    formatted_labels = [format_variable_name_latex(var, var in traditional_vars_set) for var in variable_order]
    
    if 'AR+GT' in model_display_name:
        color = '#1f77b4'
    elif 'Full' in model_display_name:
        color = '#d62728'
    else:
        color = '#808080'

    try:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
    except RuntimeError:
        print("Warning: LaTeX distribution not found. Plot labels will not be formatted.")
        plt.rcdefaults()

    plt.figure(figsize=(8, 9))
    ax = sns.barplot(data=plot_df, y='Variable', x='Selection Frequency', order=variable_order, color=color, dodge=False)
    ax.set_yticklabels(formatted_labels, fontsize=18) 
    ax.tick_params(axis='x', labelsize=11)
    ax.set_ylabel("")
    ax.set_xlabel('Selection Frequency', fontsize=18)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'variable_importance_{model_key}_elasticnet.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.rcdefaults()
    print(f" {model_display_name} ElasticNet importance plot saved to {save_path}")

def plot_coefficient_evolution(coeff_data, variables_to_plot, window_dates, metadata, output_dir, model_name, filename_suffix, figure_title, rank_start=1):
    """Creates a 4x2 grid of coefficient evolution plots for a SINGLE coefficient type."""
    print(f"Generating 4x2 coefficient evolution plot for {model_name} - {filename_suffix}...")
    traditional_vars_set = set(metadata['model_predictors']['benchmark'] + [
        'd_claimant_count_L1', 'd_UK_EPU_Index_L1', 'ftse_returns',
        'retail_sales_index_L1', 'ftse_vol_3m_L1'
    ])
    color_map = {
        'Benchmark': '#2ca02c', 'Full': '#d62728', 'AR_GT': '#1f77b4'
    }
    model_type = 'Benchmark' if 'Benchmark' in model_name else 'Full' if 'Full' in model_name else 'AR_GT'
    color = color_map.get(model_type, '#8c564b')

    try:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
    except RuntimeError:
        print("Warning: LaTeX distribution not found. Plot labels will not be formatted.")
        plt.rcdefaults()

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()

    for i, var in enumerate(variables_to_plot):
        if i >= 8: break
        ax = axes[i]
        coeffs = coeff_data.get(var, [])
        ax.plot(window_dates, coeffs, color=color, linestyle='-')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.7)
        add_crisis_shading(ax)
        formatted_title = format_variable_name_latex(var, var in traditional_vars_set)
        ax.set_title(f"Rank \\#{i + rank_start}: {formatted_title}", fontsize=20) 
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.tick_params(axis='y', labelsize=16) 
        ax.tick_params(axis='x', rotation=45, labelsize=16) 

        if i % 2 == 0: 
            ax.set_ylabel('Coefficient Value', fontsize=16)

    for j in range(len(variables_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(figure_title, fontsize=20)
    save_path = os.path.join(output_dir, f'{model_name.lower()}_coefficient_evolution_{filename_suffix}_elasticnet.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.rcdefaults()
    print(f" Coefficient evolution plot saved to {save_path}")

# %% MAIN EXECUTION FUNCTION

def run_complete_analysis(BASE_DIR, input_filename, run_estimation=True, run_spreadsheet=True, run_visualizations=True):
    """
    Main function to run the complete analysis pipeline.
    """
    print("="*80)
    print("COMPLETE MULTI-MODEL ANALYSIS PIPELINE")
    print("="*80)
    
    # Set working directory
    os.chdir(BASE_DIR)
    
    # Define file paths
    input_file = input_filename
    results_file = 'final_in_sample_results.pkl'
    excel_output_dir = os.path.join(BASE_DIR, 'in-sample')
    plots_dir = os.path.join(excel_output_dir, 'in_sample_visualisations')  
    
    # Create output directories
    for path in [excel_output_dir, plots_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created output directory: {path}")
            
    results = None
    metadata = None
    
    # PART 1: MODEL ESTIMATION
    if run_estimation:
        print("\n" + "="*80)
        print("PART 1: MODEL ESTIMATION")
        print("="*80)
        results = main_lasso_elasticnet_analysis(input_file, results_file)
        if results is None:
            print("Model estimation failed. Stopping analysis.")
            return None
    
    # Load results if not running estimation but need them for other parts
    if (run_spreadsheet or run_visualizations) and results is None:
        print(f"\nLoading existing results from {results_file}...")
        try:
            with open(results_file, 'rb') as f:
                model_data = pickle.load(f)
            results, metadata = model_data['results'], model_data['metadata']
            print(" Results loaded successfully.")
        except FileNotFoundError:
            print(f"Results file {results_file} not found. Please run estimation first.")
            return None
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return None
    
    # Extract metadata if we ran estimation
    if run_estimation and results is not None:
        with open(results_file, 'rb') as f:
            model_data = pickle.load(f)
        metadata = model_data['metadata']
    
    # PART 2: SPREADSHEET GENERATION
    if run_spreadsheet and results is not None:
        print("\n" + "="*80)
        print("PART 2: SPREADSHEET GENERATION")
        print("="*80)
        
        print("Creating analysis summaries...")
        performance_summary_df = create_performance_summary(results)
        window_results_df = create_window_results_summary(results)
        variable_summary_df = create_variable_selection_summary(results, metadata)
        print(" Summaries created.")

        all_dfs = {
            'Model_Performance_Summary': performance_summary_df,
            'Variable_Selection': variable_summary_df,
            'Window_Results': window_results_df
        }
        
        excel_output_path = os.path.join(excel_output_dir, 'Model_In_Sample_Analysis_Results.xlsx')
        save_to_excel(all_dfs, filename=excel_output_path)
        print(" Spreadsheet generation complete!")
    
    if run_visualizations and results is not None:
        print("\n" + "="*80)
        print("PART 3: VISUALIZATIONS")
        print("="*80)
    
    # Extract data for visualizations
    ranked_variables = get_top_variables_by_model(results, metadata)
    shrunken_coeffs, ols_coeffs, window_dates = extract_coefficient_evolution(results, metadata)
    
    # Generate all visualizations
    print("\nGenerating variable selection evolution plot...")
    plot_variable_selection_evolution(results, window_dates, plots_dir)
    
    print("\nGenerating variable importance plots...")
    plot_single_model_importance(results, metadata, plots_dir, model_key='ar_gt', model_display_name='AR+GT', top_n=15)
    plot_single_model_importance(results, metadata, plots_dir, model_key='full', model_display_name='Full', top_n=15)
    
    print("\nGenerating coefficient evolution plots...")
    
    # Full model coefficient evolution plots
    full_model_ranked_vars = ranked_variables.get('full', [])
    chunk_size = 8 
    for i in range(0, len(full_model_ranked_vars), chunk_size):
        variable_chunk = full_model_ranked_vars[i:i + chunk_size]
        start_rank, end_rank = i + 1, i + len(variable_chunk)
        
        plot_coefficient_evolution(
            shrunken_coeffs['full'], variable_chunk, window_dates, metadata, 
            plots_dir, model_name='Full_Model', 
            filename_suffix=f'Ranks_{start_rank}-{end_rank}_Shrunken', 
            figure_title=f'Evolution of Shrunken Full ElasticNet Coefficients (Ranks {start_rank}-{end_rank})', 
            rank_start=start_rank
        )
        
        plot_coefficient_evolution(
            ols_coeffs['full'], variable_chunk, window_dates, metadata, 
            plots_dir, model_name='Full_Model', 
            filename_suffix=f'Ranks_{start_rank}-{end_rank}_OLS', 
            figure_title=f'Evolution of Post-ElasticNet OLS Full Model Coefficients (Ranks {start_rank}-{end_rank})', 
            rank_start=start_rank
        )

    # AR+GT model coefficient evolution plots
    argt_model_ranked_vars = ranked_variables.get('ar_gt', [])
    for i in range(0, len(argt_model_ranked_vars), chunk_size):
        variable_chunk = argt_model_ranked_vars[i:i+chunk_size]
        start_rank, end_rank = i + 1, i + len(variable_chunk)
        
        plot_coefficient_evolution(
            shrunken_coeffs['ar_gt'], variable_chunk, window_dates, metadata, 
            plots_dir, model_name='AR_GT_Model', 
            filename_suffix=f'Ranks_{start_rank}-{end_rank}_Shrunken', 
            figure_title=f'Evolution of Shrunken AR+GT ElasticNet Coefficients (Ranks {start_rank}-{end_rank})', 
            rank_start=start_rank
        )
        
        plot_coefficient_evolution(
            ols_coeffs['ar_gt'], variable_chunk, window_dates, metadata, 
            plots_dir, model_name='AR_GT_Model', 
            filename_suffix=f'Ranks_{start_rank}-{end_rank}_OLS', 
            figure_title=f'Evolution of Post-ElasticNet OLS AR+GT Model Coefficients (Ranks {start_rank}-{end_rank})', 
            rank_start=start_rank
        )

    # Benchmark model coefficient evolution plots
    benchmark_ranked_vars = ranked_variables.get('benchmark', [])
    if benchmark_ranked_vars:
        plot_coefficient_evolution(
            shrunken_coeffs['benchmark'], benchmark_ranked_vars[:8], window_dates, metadata, 
            plots_dir, model_name='Benchmark_Model', 
            filename_suffix='Top_8_Shrunken', 
            figure_title='Evolution of Top 8 Shrunken Benchmark Coefficients', 
            rank_start=1
        )
        
        plot_coefficient_evolution(
            ols_coeffs['benchmark'], benchmark_ranked_vars[:8], window_dates, metadata, 
            plots_dir, model_name='Benchmark_Model', 
            filename_suffix='Top_8_OLS', 
            figure_title='Evolution of Top 8 Post-Lasso OLS Benchmark Coefficients', 
            rank_start=1
        )
    
    print(" Visualization generation complete!")
    
    # Update the final print statement
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS PIPELINE FINISHED!")
    print("="*80)
    
    if run_estimation and results is not None:
        print(f" Model estimation completed ({len(results)} windows)")
    if run_spreadsheet:
        print(f" Excel spreadsheet saved to: {excel_output_dir}")
    if run_visualizations:
        print(f" Visualizations saved to: {plots_dir}")  # Updated message
    
    return results

# %% SCRIPT EXECUTION

if __name__ == "__main__":
    # Configuration
    INPUT_FILENAME = 'final_dataset_ar_only_lags.xlsx'
    
    # Control which parts to run
    RUN_ESTIMATION = True      # Set to False if you already have results and just want spreadsheet/visualizations
    RUN_SPREADSHEET = True     # Generate Excel spreadsheet
    RUN_VISUALIZATIONS = True  # Generate all plots
    
    # Run the complete analysis
    results = run_complete_analysis(
        BASE_DIR=BASE_DIR,
        input_filename=INPUT_FILENAME,
        run_estimation=RUN_ESTIMATION,
        run_spreadsheet=RUN_SPREADSHEET,
        run_visualizations=RUN_VISUALIZATIONS
    )
    
    if results:
        print("\n Analysis completed successfully!")
    else:
        print("\n Analysis failed!")