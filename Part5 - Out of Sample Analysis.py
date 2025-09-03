import os
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

# ==== CHANGE THIS TO YOUR OWN DIRECTORY ====
BASE_DIR = r"C:\Path\To\Your\Project"

# %% PART 1: ESTIMATION (from part5a)

# Load and Prepare Data

def load_and_prepare_data(filename):
    """Loads and prepares data from the specified CSV file."""
    os.chdir(BASE_DIR)
    try:
        df = pd.read_excel(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"Warning: Found missing values in columns: {missing_data[missing_data > 0].to_dict()}")

        print(f"Dataset loaded: {df.shape}")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def prepare_variables(df):
    """Prepare variables for forecasting analysis based on the three-model approach."""
    print("\nPreparing variables for the three-model forecasting analysis...")

    target_var = 'd_unemp_rate'
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataset")

    # Define variable groups based on in-sample analysis 
    benchmark_ar_vars = ['d_unemp_rate_L1', 'd_unemp_rate_L2', 'd_unemp_rate_L3']
    other_econ_vars = [
        'd_claimant_count_L1',
        'd_UK_EPU_Index_L1',
        'ftse_returns',
        'retail_sales_index_L1',
        'ftse_vol_3m_L1'
    ]

    # Filter to existing columns and identify GT variables 
    existing_benchmark_ar = [v for v in benchmark_ar_vars if v in df.columns]
    existing_other_econ = [v for v in other_econ_vars if v in df.columns]
    all_traditional_econ_vars = existing_benchmark_ar + existing_other_econ
    
    all_potential_predictors = [c for c in df.columns if c not in ['Date', target_var]]
    gt_vars = [v for v in all_potential_predictors if v not in all_traditional_econ_vars]
    
    # Define final predictor sets for each model 
    # Model 1: Benchmark (AR only)
    predictors_benchmark = existing_benchmark_ar
    
    # Model 2: AR + Google Trends
    predictors_ar_gt = existing_benchmark_ar + gt_vars
    
    # Model 3: All Predictors (AR + Traditional Econ + GT)
    predictors_all = all_traditional_econ_vars + gt_vars

    # Print summary 
    print(f"  Target variable: {target_var}")
    print("\nVariable sets for forecasting models:")
    print(f"  1. Benchmark Model Predictors (AR terms): {len(predictors_benchmark)}")
    print(f"  2. AR+GT Model Predictors: {len(predictors_ar_gt)}")
    print(f"     - Google Trends variables identified: {len(gt_vars)}")
    print(f"  3. All Predictors Model (Full): {len(predictors_all)}")
    print(f"     - Other traditional economic variables: {len(existing_other_econ)}")

    return target_var, predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars

# Multi-Horizon Window Creation

def create_multi_horizon_windows(df, train_length=48, forecast_horizons=[1, 3, 6, 12], step_size=1):
    """Create rolling windows for multiple forecasting horizons."""
    print("\nCreating multi-horizon forecasting windows:")
    print(f"  Training window: {train_length} months")
    print(f"  Forecast horizons: {forecast_horizons} months ahead")
    print(f"  Step size: {step_size} month")

    total_obs = len(df)
    all_windows = {}
    
    for horizon in forecast_horizons:
        horizon_windows = []
        
        # Create windows for this specific horizon
        for i in range(0, total_obs - train_length - horizon + 1, step_size):
            window_info = {
                'window_id': len(horizon_windows) + 1,
                'horizon': horizon,
                'train_start_idx': i,
                'train_end_idx': i + train_length,
                'forecast_idx': i + train_length + horizon - 1, # h-step ahead
                'train_start_date': df.iloc[i]['Date'],
                'train_end_date': df.iloc[i + train_length - 1]['Date'],
                'forecast_date': df.iloc[i + train_length + horizon - 1]['Date'],
                'train_n_obs': train_length
            }
            horizon_windows.append(window_info)
        
        all_windows[horizon] = horizon_windows
        print(f"  {horizon}-month ahead: {len(horizon_windows)} windows")
        
        if horizon_windows:
            print(f"    Forecast range: {horizon_windows[0]['forecast_date'].strftime('%Y-%m')} to {horizon_windows[-1]['forecast_date'].strftime('%Y-%m')}")

    return all_windows

# Model Parameter Optimization

def optimize_model_parameters(X_train, y_train):
    """Optimize LASSO and ElasticNet parameters using time series cross-validation."""
    
    # Remove constant variables
    constant_threshold = 1e-10
    variable_cols = X_train.columns[X_train.std() > constant_threshold]
    X_train_filtered = X_train[variable_cols]
    
    if X_train_filtered.empty:
        return None, None, None, []
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    
    # Cross-validation setup
    n_splits = min(8, len(X_train) // 8)
    if n_splits < 2: n_splits = 2 # Ensure at least 2 splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Parameter ranges
    alpha_range = np.logspace(-6, 1, 50)
    l1_ratio_range = np.linspace(0.1, 0.9, 9)
    
    try:
        # LASSO CV
        lasso_cv = LassoCV(
            alphas=alpha_range, cv=tscv, max_iter=5000,
            random_state=42, n_jobs=-1
        )
        lasso_cv.fit(X_train_scaled, y_train)
        
        # ElasticNet CV
        elasticnet_cv = ElasticNetCV(
            alphas=alpha_range, l1_ratio=l1_ratio_range, cv=tscv,
            max_iter=5000, random_state=42, n_jobs=-1
        )
        elasticnet_cv.fit(X_train_scaled, y_train)
        
        return lasso_cv.alpha_, elasticnet_cv.alpha_, elasticnet_cv.l1_ratio_, variable_cols.tolist()
        
    except Exception as e:
        print(f"Error in parameter optimization: {str(e)}")
        return None, None, None, []

# Model Estimation and Forecasting

def _fit_predict_post_regularization(model_type, X_train, y_train, X_forecast, hyperparams):
    """Helper to fit a regularized model, select variables, and predict with OLS."""
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_forecast_scaled = scaler.transform(X_forecast)

    # Fit the regularized model (LASSO or ElasticNet)
    if model_type == 'lasso':
        model = Lasso(alpha=hyperparams['lambda'], max_iter=5000, random_state=42)
    else: # elasticnet
        model = ElasticNet(alpha=hyperparams['lambda'], l1_ratio=hyperparams['l1_ratio'], max_iter=5000, random_state=42)
    
    model.fit(X_train_scaled, y_train)

    # Select variables where coefficients are non-zero
    selected_vars = X_train.columns[np.abs(model.coef_) > 1e-10].tolist()
    
    # Fit Post-Regularization OLS and forecast
    if selected_vars:
        try:
            X_train_selected = sm.add_constant(X_train[selected_vars])
            X_forecast_selected = sm.add_constant(X_forecast[selected_vars])
            
            ols_model = sm.OLS(y_train, X_train_selected).fit()
            forecast_ols = ols_model.predict(X_forecast_selected)[0]
        except Exception:
            # Fallback to the regularized model's forecast if OLS fails
            forecast_ols = model.predict(X_forecast_scaled)[0]
    else:
        # If no variables are selected, forecast the mean
        forecast_ols = y_train.mean()
        
    return forecast_ols, selected_vars

def estimate_models_and_forecast(train_data, forecast_data, target_var, 
                                 predictors_benchmark, predictors_ar_gt, predictors_all,
                                 window_id, horizon, 
                                 lasso_lambda, elasticnet_lambda, elasticnet_l1_ratio):
    """Estimate all three models and generate h-step ahead forecasts."""
    
    y_train = train_data[target_var].copy().fillna(method='ffill').fillna(method='bfill')
    y_actual = forecast_data[target_var].iloc[0]
    
    if y_train.isnull().any() or len(y_train) < 10:
        return None

    results = {
        'window_id': window_id, 'horizon': horizon, 'y_actual': y_actual,
        'train_start_date': train_data['Date'].iloc[0], 'train_end_date': train_data['Date'].iloc[-1],
        'forecast_date': forecast_data['Date'].iloc[0]
    }

    model_specs = {
        'benchmark': {'predictors': predictors_benchmark, 'methods': ['lasso']},
        'ar_gt': {'predictors': predictors_ar_gt, 'methods': ['lasso', 'elasticnet']},
        'all': {'predictors': predictors_all, 'methods': ['lasso', 'elasticnet']}
    }
    
    for model_name, spec in model_specs.items():
        X_train_raw = train_data[spec['predictors']].copy().fillna(method='ffill').fillna(method='bfill')
        X_forecast_raw = forecast_data[spec['predictors']].copy().fillna(method='ffill').fillna(method='bfill')

        # Remove constant variables within the window
        variable_cols = X_train_raw.columns[X_train_raw.std() > 1e-8]
        X_train = X_train_raw[variable_cols]
        X_forecast = X_forecast_raw[variable_cols]
        
        if X_train.empty:
            # If no variables remain, skip this model for this window
            for method in spec['methods']:
                results[f'forecast_{model_name}_{method}_ols'] = y_train.mean()
                results[f'{model_name}_{method}_n_selected'] = 0
                results[f'{model_name}_{method}_selected_vars'] = []
            continue

        # Fit models
        for method in spec['methods']:
            try:
                if method == 'lasso':
                    hparams = {'lambda': lasso_lambda}
                else: # elasticnet
                    hparams = {'lambda': elasticnet_lambda, 'l1_ratio': elasticnet_l1_ratio}
                
                forecast, selected_vars = _fit_predict_post_regularization(
                    method, X_train, y_train, X_forecast, hparams
                )
                
                results[f'forecast_{model_name}_{method}_ols'] = forecast
                results[f'{model_name}_{method}_n_selected'] = len(selected_vars)
                results[f'{model_name}_{method}_selected_vars'] = selected_vars

            except Exception as e:
                print(f"Error in model '{model_name}_{method}' (window {window_id}, h={horizon}): {e}")
                results[f'forecast_{model_name}_{method}_ols'] = y_train.mean()
                results[f'{model_name}_{method}_n_selected'] = 0
                results[f'{model_name}_{method}_selected_vars'] = []
    
    return results

# Forecast Evaluation Metrics

def calculate_forecast_metrics(forecasts_df):
    """Calculate comprehensive forecast evaluation metrics."""
    
    models = [
        'benchmark_lasso_ols', 'ar_gt_lasso_ols', 'ar_gt_elasticnet_ols',
        'all_lasso_ols', 'all_elasticnet_ols'
    ]
    metrics = {}
    
    for model in models:
        forecast_col = f'forecast_{model}'
        if forecast_col in forecasts_df.columns:
            actual = forecasts_df['y_actual']
            predicted = forecasts_df[forecast_col]
            
            mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
            actual_clean, predicted_clean = actual[mask], predicted[mask]
            
            if len(actual_clean) > 0:
                errors = actual_clean - predicted_clean
                metrics[model] = {
                    'n_forecasts': len(actual_clean),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'mae': np.mean(np.abs(errors)),
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'hit_rate_10': np.mean(np.abs(errors) <= 0.1) * 100,
                    'hit_rate_20': np.mean(np.abs(errors) <= 0.2) * 100,
                }
    return metrics

# Statistical Tests

def diebold_mariano_test(errors1, errors2):
    """Perform Diebold-Mariano test for forecast accuracy comparison."""
    try:
        d = errors1**2 - errors2**2
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        if d_var > 1e-10:
            dm_stat = d_mean / np.sqrt(d_var / len(d))
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
            return dm_stat, p_value
        else:
            return np.nan, np.nan
    except:
        return np.nan, np.nan

def compare_forecast_accuracy(forecasts_df):
    """Compare forecast accuracy between all models using statistical tests."""
    
    baseline = 'benchmark_lasso_ols'
    competitor_models = [
        'ar_gt_lasso_ols', 'ar_gt_elasticnet_ols',
        'all_lasso_ols', 'all_elasticnet_ols'
    ]
    comparisons = {}
    actual = forecasts_df['y_actual']

    # Compare each competitor to the baseline
    for model in competitor_models:
        if f'forecast_{model}' in forecasts_df.columns and f'forecast_{baseline}' in forecasts_df.columns:
            
            pred_model = forecasts_df[f'forecast_{model}']
            pred_baseline = forecasts_df[f'forecast_{baseline}']
            
            mask = ~(np.isnan(actual) | np.isnan(pred_model) | np.isnan(pred_baseline))
            
            if mask.sum() > 10:
                errors_model = actual[mask] - pred_model[mask]
                errors_baseline = actual[mask] - pred_baseline[mask]
                
                # Note: DM test assumes errors1 (baseline) is being tested against errors2 (model)
                # A positive DM stat means the second model's errors are smaller (i.e., it's better)
                dm_stat, p_value = diebold_mariano_test(errors_baseline, errors_model)
                
                comparisons[f'{model}_vs_{baseline}'] = {
                    'dm_statistic': dm_stat, 'p_value': p_value,
                    'model_better': dm_stat > 0 if not np.isnan(dm_stat) else None,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else None
                }
    return comparisons

# Single Horizon Analysis

def run_single_horizon_analysis(df, windows, target_var, 
                                predictors_benchmark, predictors_ar_gt, predictors_all, 
                                horizon):
    """Run forecasting analysis for a single horizon."""
    
    print(f"\nRunning {horizon}-month ahead forecasting:")
    print(f"  - {len(windows)} forecasting windows")
    
    all_forecasts = []
    failed_windows = []
    
    for i, window in enumerate(windows):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(windows)} windows processed")
        
        train_data = df.iloc[window['train_start_idx']:window['train_end_idx']].copy()
        forecast_data = df.iloc[window['forecast_idx']:window['forecast_idx']+1].copy()
        
        try:
            # MODIFIED: Optimize parameters on the most complex model (predictors_all).
            X_train_for_optim = train_data[predictors_all]
            y_train = train_data[target_var]
            
            lasso_lambda, elasticnet_lambda, elasticnet_l1_ratio, _ = optimize_model_parameters(X_train_for_optim, y_train)
            
            if lasso_lambda is None:
                failed_windows.append(window['window_id'])
                continue
            
            window_results = estimate_models_and_forecast(
                train_data, forecast_data, target_var, 
                predictors_benchmark, predictors_ar_gt, predictors_all,
                window['window_id'], horizon, 
                lasso_lambda, elasticnet_lambda, elasticnet_l1_ratio
            )
            
            if window_results is not None:
                window_results.update({
                    'lasso_lambda': lasso_lambda,
                    'elasticnet_lambda': elasticnet_lambda,
                    'elasticnet_l1_ratio': elasticnet_l1_ratio
                })
                all_forecasts.append(window_results)
            else:
                failed_windows.append(window['window_id'])
                
        except Exception as e:
            print(f"  Error in window {window['window_id']}: {str(e)}")
            failed_windows.append(window['window_id'])
            continue
    
    print(f"  - {len(all_forecasts)} successful forecasts")
    print(f"  - {len(failed_windows)} failed windows")
    
    return all_forecasts, failed_windows

# Multi-Horizon Analysis

def run_multi_horizon_analysis(df, target_var, predictors_benchmark, predictors_ar_gt, predictors_all,
                               train_length=48, forecast_horizons=[1, 3, 6, 12]):
    """Run forecasting analysis across multiple horizons."""
    
    print("MULTI-HORIZON FORECASTING ANALYSIS")
    print("="*80)
    
    all_windows = create_multi_horizon_windows(df, train_length, forecast_horizons)
    horizon_results = {}
    
    for horizon in forecast_horizons:
        print(f"\n{'='*20} {horizon}-MONTH AHEAD FORECASTING {'='*20}")
        
        windows = all_windows[horizon]
        if not windows:
            print(f"No windows available for {horizon}-month horizon")
            continue
            
        forecasts, failed = run_single_horizon_analysis(
            df, windows, target_var, 
            predictors_benchmark, predictors_ar_gt, predictors_all, 
            horizon
        )
        
        if forecasts:
            forecasts_df = pd.DataFrame(forecasts)
            metrics = calculate_forecast_metrics(forecasts_df)
            comparisons = compare_forecast_accuracy(forecasts_df)
            
            horizon_results[horizon] = {
                'forecasts': forecasts, 'metrics': metrics, 'comparisons': comparisons,
                'n_successful': len(forecasts), 'n_failed': len(failed)
            }
            
            print(f"\n{horizon}-Month Ahead Results:")
            print(f"  Successful forecasts: {len(forecasts)}")
            
            if metrics:
                print("  RMSE Comparison:")
                model_names = {
                    'benchmark_lasso_ols': 'Benchmark (AR)',
                    'ar_gt_lasso_ols': 'AR+GT (LASSO)',
                    'ar_gt_elasticnet_ols': 'AR+GT (ElasticNet)',
                    'all_lasso_ols': 'Full Model (LASSO)',
                    'all_elasticnet_ols': 'Full Model (ElasticNet)'
                }
                
                for model, model_metrics in metrics.items():
                    display_name = model_names.get(model, model)
                    print(f"    {display_name:<25}: {model_metrics['rmse']:.4f}")
                
                best_model = min(metrics.keys(), key=lambda x: metrics[x]['rmse'])
                print(f"  Best model: {model_names.get(best_model, best_model)}")
                
                if comparisons:
                    print("  Statistical Tests (vs Benchmark):")
                    for comp_name, result in comparisons.items():
                        model_name = model_names.get(comp_name.split('_vs_')[0], comp_name)
                        if result.get('significant'):
                            significance = "**" if result['p_value'] < 0.01 else "*"
                            status = "outperforms" if result['model_better'] else "underperforms"
                            print(f"    {model_name:<25}: {status} Benchmark {significance} (p={result['p_value']:.3f})")
        else:
            print(f"No successful forecasts for {horizon}-month horizon")
            horizon_results[horizon] = None
    
    return horizon_results

# Cross-Horizon Analysis

def compare_across_horizons(horizon_results):
    """Compare model performance across different forecasting horizons."""
    
    print("\n" + "="*80)
    print("CROSS-HORIZON PERFORMANCE COMPARISON")
    print("="*80)
    
    summary_data = []
    model_names = {
        'benchmark_lasso_ols': 'Benchmark (AR)',
        'ar_gt_lasso_ols': 'AR+GT (LASSO)',
        'ar_gt_elasticnet_ols': 'AR+GT (ElasticNet)',
        'all_lasso_ols': 'Full Model (LASSO)',
        'all_elasticnet_ols': 'Full Model (ElasticNet)'
    }

    for horizon, results in sorted(horizon_results.items()):
        if results and results['metrics']:
            metrics = results['metrics']
            for model, model_metrics in metrics.items():
                summary_data.append({
                    'Horizon': horizon,
                    'Model': model_names.get(model, model),
                    'RMSE': model_metrics['rmse'],
                    'MAE': model_metrics['mae'],
                    'Hit Rate (10bp)': model_metrics.get('hit_rate_10', np.nan)
                })
    
    if not summary_data:
        print("No results to compare.")
        return None

    summary_df = pd.DataFrame(summary_data)
    
    print("\nRMSE by Horizon and Model:")
    print("-" * 60)
    for horizon in sorted(horizon_results.keys()):
        if horizon_results[horizon]:
            print(f"\n{horizon}-month ahead:")
            horizon_data = summary_df[summary_df['Horizon'] == horizon]
            for _, row in horizon_data.iterrows():
                print(f"  {row['Model']:<25}: RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")
                
    print("\nBest Model by Horizon (Lowest RMSE):")
    print("-" * 40)
    for horizon in sorted(horizon_results.keys()):
        if horizon_results[horizon]:
            metrics = horizon_results[horizon]['metrics']
            if metrics:
                best_model_key = min(metrics.keys(), key=lambda x: metrics[x]['rmse'])
                best_name = model_names.get(best_model_key, best_model_key)
                best_rmse = metrics[best_model_key]['rmse']
                print(f"  {horizon:2d}-month: {best_name:<25} (RMSE={best_rmse:.4f})")
    
    return summary_df

# Save Results

def save_multi_horizon_results(horizon_results, summary_df, target_var, 
                               predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars,
                               forecast_horizons, filename='final_out_of_sample_3_model_results.pkl'):
    """Save multi-horizon forecasting results."""
    
    print(f"\nSaving multi-horizon results to {filename}...")
    
    multi_horizon_data = {
        'horizon_results': horizon_results,
        'summary_table': summary_df.to_dict() if summary_df is not None else None,
        'metadata': {
            'forecast_horizons': forecast_horizons,
            'target_var': target_var,
            'predictors_benchmark': predictors_benchmark,
            'predictors_ar_gt': predictors_ar_gt,
            'predictors_all': predictors_all,
            'gt_vars': gt_vars,
            'analysis_type': 'multi_horizon_forecasting_3_model'
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(multi_horizon_data, f)
    
    print(f" Multi-horizon results saved successfully to {filename}")

# Main Multi-Horizon Execution

def main_multi_horizon_analysis(filename, forecast_horizons=[1, 3, 6, 12], train_length=48):
    """Main function for the three-model multi-horizon forecasting analysis."""
    
    try:
        df = load_and_prepare_data(filename)
        target_var, predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars = prepare_variables(df)
        
        horizon_results = run_multi_horizon_analysis(
            df, target_var, predictors_benchmark, predictors_ar_gt, predictors_all,
            train_length=train_length, forecast_horizons=forecast_horizons
        )
        
        summary_df = compare_across_horizons(horizon_results)
        
        save_multi_horizon_results(
            horizon_results, summary_df, target_var,
            predictors_benchmark, predictors_ar_gt, predictors_all, gt_vars,
            forecast_horizons
        )
        
        print("\n" + "="*80)
        print("MULTI-HORIZON FORECASTING ANALYSIS COMPLETE!")
        print("="*80)
        
        return horizon_results, summary_df
        
    except Exception as e:
        print(f" Critical error in multi-horizon analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Execute Multi-Horizon Analysis

if __name__ == "__main__":
    horizons = [1, 3, 6, 12] # Example: 1 and 3 months ahead
    
    print("3-MODEL MULTI-HORIZON UNEMPLOYMENT FORECASTING ANALYSIS")
    print("="*80)
    print(f"Testing forecast horizons: {horizons} months")
    print("Models: Benchmark (AR), AR+GT (LASSO/ElasticNet), Full (LASSO/ElasticNet)")
    print("="*80)
    
    results, summary = main_multi_horizon_analysis(
        'final_dataset_ar_only_lags.xlsx', 
        forecast_horizons=horizons,
        train_length=48
    )
    
    if results:
        print("\n Analysis completed successfully!")
    else:
        print("\n Analysis failed!") 

# %% PART 2: SPREADSHEET CREATION (from part5b)

# Load Results

def load_results(filename='final_out_of_sample_3_model_results.pkl'):
    """Loads the 3-model out-of-sample analysis results."""
    print(f"Loading results from '{filename}'...")
    os.chdir(r"C:\Users\arthu\OneDrive - University of Surrey\Documents\Surrey\Semester 2\Dissertation\Data\Final Dataset")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}. Please run the out-of-sample estimation script first.")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(" Results loaded.")
    # Returns the main dictionary of results and the metadata
    return data.get('horizon_results', {}), data.get('metadata', {})

# Helper Functions

def format_significance(p_value):
    """Formats p-value with significance stars."""
    if pd.isna(p_value): return "N/A"
    if p_value < 0.01: return f"{p_value:.3f}***"
    if p_value < 0.05: return f"{p_value:.3f}**"
    if p_value < 0.10: return f"{p_value:.3f}*"
    return f"{p_value:.3f}"

# Analysis Functions

def create_performance_summary(horizon_results, full_forecasts_df):
    """Creates a summary DataFrame of model performance and statistical tests for all horizons."""
    summary_data = []
    
    model_names = {
        'benchmark_lasso_ols': 'Benchmark (AR)',
        'ar_gt_lasso_ols': 'AR+GT (LASSO)',
        'ar_gt_elasticnet_ols': 'AR+GT (ElasticNet)',
        'all_lasso_ols': 'Full (LASSO)',
        'all_elasticnet_ols': 'Full (ElasticNet)'
    }

    for h, data in sorted(horizon_results.items()):
        if not data: continue
        
        metrics = data.get('metrics', {})
        comparisons = data.get('comparisons', {})
        
        # Filter forecasts for the current horizon to get final CSFE
        horizon_forecasts = full_forecasts_df[full_forecasts_df['horizon'] == h] if not full_forecasts_df.empty else pd.DataFrame()

        # Start with performance metrics
        row = {'Horizon': h}
        for key, name in model_names.items():
            row[f'{name} RMSE'] = metrics.get(key, {}).get('rmse')
            row[f'{name} MAE'] = metrics.get(key, {}).get('mae')
            # NEW: Add the final CSFE value
            csfe_col = f'csfe_{key}'
            if not horizon_forecasts.empty and csfe_col in horizon_forecasts.columns:
                # The final CSFE is the max value for that horizon
                row[f'{name} CSFE'] = horizon_forecasts[csfe_col].max()
            else:
                row[f'{name} CSFE'] = np.nan


        # Add Diebold-Mariano test results (vs. Benchmark)
        for model_key, model_name in model_names.items():
            if model_key == 'benchmark_lasso_ols': continue # Skip comparing benchmark to itself
            
            comp_key = f'{model_key}_vs_benchmark_lasso_ols'
            dm_test = comparisons.get(comp_key, {})
            row[f'DM p-value ({model_name} vs Benchmark)'] = format_significance(dm_test.get('p_value'))

        summary_data.append(row)
        
    df = pd.DataFrame(summary_data)
    # Reorder columns for clarity
    if not df.empty:
        cols = ['Horizon']
        for name in model_names.values():
            # NEW: Added CSFE to the column list
            cols.extend([f'{name} RMSE', f'{name} MAE', f'{name} CSFE'])
        for name in model_names.values():
            if name != 'Benchmark (AR)':
                cols.append(f'DM p-value ({name} vs Benchmark)')
        # Ensure all columns exist before reordering
        df = df[[c for c in cols if c in df.columns]]

    return df

def create_full_forecast_sheet(horizon_results):
    """Creates a comprehensive sheet with all individual forecast data from all horizons."""
    all_forecasts_list = []
    for h, data in horizon_results.items():
        if data and 'forecasts' in data:
            # Sort forecasts by date within each horizon before extending
            sorted_forecasts = sorted(data['forecasts'], key=lambda x: x['forecast_date'])
            all_forecasts_list.extend(sorted_forecasts)
    
    if not all_forecasts_list:
        return pd.DataFrame()

    df = pd.DataFrame(all_forecasts_list)
    
    # Define model keys used in the forecast results
    model_keys = [
        'benchmark_lasso_ols', 'ar_gt_lasso_ols', 'ar_gt_elasticnet_ols',
        'all_lasso_ols', 'all_elasticnet_ols'
    ]
    
    # Calculate errors and squared errors for each model's forecast
    for key in model_keys:
        forecast_col = f'forecast_{key}'
        if forecast_col in df.columns:
            df[f'error_{key}'] = df['y_actual'] - df[forecast_col]
            df[f'sq_error_{key}'] = df[f'error_{key}']**2 # Calculate squared error
            
    # Calculate Cumulative Squared Forecast Error (CSFE) for each model
    for key in model_keys:
        sq_error_col = f'sq_error_{key}'
        csfe_col = f'csfe_{key}' # Cumulative Squared Forecast Error column
        if sq_error_col in df.columns:
            df[csfe_col] = df.groupby('horizon')[sq_error_col].cumsum()
            
    return df

def create_variable_selection_summary(horizon_results, metadata):
    """Analyzes the frequency of variable selection for each model and horizon."""
    summary_list = []
    
    # Models that perform variable selection
    selectable_models = {
        'ar_gt_lasso': 'AR+GT (LASSO)',
        'ar_gt_elasticnet': 'AR+GT (ElasticNet)',
        'all_lasso': 'Full (LASSO)',
        'all_elasticnet': 'Full (ElasticNet)',
    }
    
    for h, data in sorted(horizon_results.items()):
        if not data or 'forecasts' not in data: continue
        
        forecast_list = data['forecasts']
        num_windows = len(forecast_list)
        if num_windows == 0: continue

        selection_counts = {key: {} for key in selectable_models}

        # Count selections in each window
        for forecast in forecast_list:
            for key in selectable_models:
                vars_key = f'{key}_selected_vars'
                if vars_key in forecast:
                    for var in forecast[vars_key]:
                        selection_counts[key][var] = selection_counts[key].get(var, 0) + 1
        
        # Convert counts to a flat list for the DataFrame
        for key, var_counts in selection_counts.items():
            for var, count in var_counts.items():
                summary_list.append({
                    'Horizon': h,
                    'Model': selectable_models[key],
                    'Variable': var,
                    'Selection_Rate': count / num_windows,
                    'Selection_Count': count
                })
                
    if not summary_list:
        return pd.DataFrame()

    df = pd.DataFrame(summary_list)
    return df.sort_values(['Horizon', 'Model', 'Selection_Rate'], ascending=[True, True, False])


def main_spreadsheet(results_file, output_file):
    """Main function to load results, create summaries, and save to Excel."""
    print("GENERATING SPREADSHEET FROM 3-MODEL OUT-OF-SAMPLE ANALYSIS")
    print("=" * 60)
    try:
        horizon_results, metadata = load_results(results_file)
        print("Creating analysis summaries...")
        full_forecasts_df = create_full_forecast_sheet(horizon_results)
        performance_df = create_performance_summary(horizon_results, full_forecasts_df)
        variable_selection_df = create_variable_selection_summary(horizon_results, metadata)
        print(" Summaries created.")

        print(f"Saving analysis to Excel file: '{output_file}'...")
        os.chdir(r"C:\\Users\\arthu\\OneDrive - University of Surrey\\Documents\\Surrey\\Semester 2\\Dissertation\\Data\\Final Dataset\\3 model approach")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            performance_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            variable_selection_df.to_excel(writer, sheet_name='Variable_Selection', index=False)
            full_forecasts_df.to_excel(writer, sheet_name='All_Forecasts', index=False)
        print(" Excel file saved successfully.")

        print("\n" + "="*60)
        print("SPREADSHEET CREATION COMPLETE!")

    except Exception as e:
        print(f"\n  An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

# %% PART 3: VISUALISATIONS (from part5c)

# Core Function Definitions

def load_results(file_path, file_name):
    """Loads the saved forecasting results from a pickle file."""
    full_path = os.path.join(file_path, file_name)
    try:
        with open(full_path, 'rb') as f:
            results = pickle.load(f)
        print(f" Successfully loaded '{file_name}'")
        return results
    except FileNotFoundError:
        print(f" Error: '{file_name}' not found at '{file_path}'. Please run the main analysis script first.")
        return None

def select_best_models_by_final_cse(df):
    """
    Selects the best performing model (LASSO or ElasticNet) based on the lowest
    final Cumulative Squared Error (CSE).
    """
    if df is None or df.empty:
        return None

    final_cse = {}
    competing_models = [
        'ar_gt_lasso_ols', 'ar_gt_elasticnet_ols',
        'all_lasso_ols', 'all_elasticnet_ols'
    ]
    for model in competing_models:
        col_name = f'cumulative_se_{model}'
        if col_name in df.columns:
            final_cse[model] = df[col_name].iloc[-1]
        else:
            final_cse[model] = float('inf')

    best_ar_gt_model = ('ar_gt_elasticnet_ols'
                        if final_cse['ar_gt_elasticnet_ols'] < final_cse['ar_gt_lasso_ols']
                        else 'ar_gt_lasso_ols')

    best_all_model = ('all_elasticnet_ols'
                      if final_cse['all_elasticnet_ols'] < final_cse['all_lasso_ols']
                      else 'all_lasso_ols')

    return ['benchmark_lasso_ols', best_ar_gt_model, best_all_model]


def prepare_plot_data(results, horizon):
    """Prepares a full DataFrame for a given horizon, calculating errors for ALL model variants."""
    if not results or horizon not in results.get('horizon_results', {}):
        print(f"Warning: Horizon {horizon} not found in results.")
        return None
    forecast_data = results['horizon_results'][horizon].get('forecasts')
    if not forecast_data:
        print(f"Warning: No forecast data found for horizon {horizon}.")
        return None
    df = pd.DataFrame(forecast_data)
    if df.empty:
        return None
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df.sort_values('forecast_date', inplace=True)
    all_model_keys = [
        'benchmark_lasso_ols', 'ar_gt_lasso_ols', 'ar_gt_elasticnet_ols',
        'all_lasso_ols', 'all_elasticnet_ols'
    ]
    for model in all_model_keys:
        forecast_col = f'forecast_{model}'
        if forecast_col in df.columns and not df[forecast_col].isnull().all():
            df[f'error_{model}'] = df['y_actual'] - df[forecast_col]
            df[f'se_{model}'] = df[f'error_{model}']**2
            df[f'cumulative_se_{model}'] = df[f'se_{model}'].cumsum()
    benchmark_se_col = 'se_benchmark_lasso_ols'
    if benchmark_se_col in df.columns:
        for model in all_model_keys:
            if model != 'benchmark_lasso_ols' and f'se_{model}' in df.columns:
                df[f'se_diff_{model}'] = df[benchmark_se_col] - df[f'se_{model}']
    return df

def generate_dynamic_model_meta(model_keys):
    """Creates a metadata dictionary for plotting based on the selected model keys."""
    base_meta = {
        'benchmark_lasso_ols': {'name': 'Benchmark (AR)', 'color': '#1f77b4', 'style': ':'},
        'ar_gt_lasso_ols': {'name': 'AR+GT (LASSO)', 'color': '#ff7f0e', 'style': '-'},
        'ar_gt_elasticnet_ols': {'name': 'AR+GT (ElasticNet)', 'color': '#ff7f0e', 'style': '--'},
        'all_lasso_ols': {'name': 'Full (LASSO)', 'color': '#d62728', 'style': '-'},
        'all_elasticnet_ols': {'name': 'Full (ElasticNet)', 'color': '#d62728', 'style': '--'}
    }
    dynamic_meta = {key: base_meta[key] for key in model_keys if key in base_meta}
    return dynamic_meta

def create_rmse_vs_horizon_plot(horizon_results, output_dir):
    """Generates a plot comparing model RMSE across all forecast horizons."""
    print("\nGenerating RMSE vs. Horizon plot...")

    plot_data = []
    model_names = {
        'benchmark_lasso_ols': 'Benchmark (AR)',
        'ar_gt_lasso_ols': 'AR+GT (LASSO)',
        'ar_gt_elasticnet_ols': 'AR+GT (ElasticNet)',
        'all_lasso_ols': 'Full (LASSO)',
        'all_elasticnet_ols': 'Full (ElasticNet)'
    }
    model_styles = {
        'Benchmark (AR)': {'color': '#ff7f0e', 'style': ':', 'marker': 'o'},
        'AR+GT (LASSO)': {'color': '#1f77b4', 'style': '-', 'marker': 's'},
        'AR+GT (ElasticNet)': {'color': '#1f77b4', 'style': '--', 'marker': '^'},
        'Full (LASSO)': {'color': '#d62728', 'style': '-', 'marker': 'D'},
        'Full (ElasticNet)': {'color': '#d62728', 'style': '--', 'marker': 'v'}
    }

    for h, results in sorted(horizon_results.items()):
        if not results or 'metrics' not in results:
            continue
        for key, name in model_names.items():
            rmse = results['metrics'].get(key, {}).get('rmse')
            if rmse is not None:
                plot_data.append({'Horizon': h, 'Model': name, 'RMSE': rmse})

    if not plot_data:
        print(" Could not generate RMSE vs. Horizon plot: No RMSE data found.")
        return

    df = pd.DataFrame(plot_data)
    df_pivot = df.pivot(index='Horizon', columns='Model', values='RMSE')

    plt.figure(figsize=(11, 7))
    for model_name in sorted(df_pivot.columns):
        style_info = model_styles.get(model_name, {'color': 'gray', 'style': '-', 'marker': '.'})
        plt.plot(df_pivot.index, df_pivot[model_name],
                 label=model_name,
                 color=style_info['color'],
                 linestyle=style_info['style'],
                 marker=style_info['marker'],
                 linewidth=1.5,
                 markersize=7)

    plt.title('Model RMSE vs. Forecast Horizon', fontsize=18, pad=15)
    plt.xlabel('Forecast Horizon (Months)', fontsize=17)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=17)
    plt.xticks(sorted(df['Horizon'].unique()))
    plt.legend(title='Model', fontsize=14, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, 'rmse_vs_horizon_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" Saved plot to '{plot_filename}'")
    plt.show()
    plt.close()


def create_full_performance_plots(df, horizon, model_meta, output_dir):
    """Generates an academic-style plot comparing the best models for the full forecast period."""
    if df is None or df.empty: return
    print(f"\nGenerating full performance plot for {horizon}-month horizon...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    event_periods = {
        'Great Recession': ('2007-12-01', '2009-06-01', 'lightcoral', 0.2),
        'Brexit Uncertainty': ('2016-06-01', '2020-01-01', 'lightskyblue', 0.3),
        'COVID-19 Pandemic': ('2020-03-01', '2023-05-01', 'mediumpurple', 0.2) 
    }

    ax1 = axes[0]
    for key, meta in model_meta.items():
        if key == 'benchmark_lasso_ols': continue
        diff_col = f'se_diff_{key}'
        if diff_col in df.columns:
            ax1.plot(df['forecast_date'], df[diff_col], label=f"{meta['name']} vs. Benchmark", color=meta['color'], linestyle=meta['style'], alpha=0.9)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax1.set_title(r'(a) Squared Forecast Error Difference vs. Benchmark (AR)', fontsize=16, pad=10)
    ax1.set_ylabel(r'SE(Benchmark) - SE(Model)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=11)

    ax2 = axes[1]
    for key, meta in model_meta.items():
        cum_se_col = f'cumulative_se_{key}'
        if cum_se_col in df.columns:
            ax2.plot(df['forecast_date'], df[cum_se_col], label=meta['name'], color=meta['color'], linestyle=meta['style'], alpha=0.9, linewidth=2 if key == 'benchmark_lasso_ols' else 1.5)
    ax2.set_title(r'(b) Cumulative Squared Forecast Error', fontsize=16, pad=10)
    ax2.set_ylabel('Cumulative Squared Forecast Error', fontsize=14)
    ax2.set_xlabel('Forecast Date', fontsize=14)
    ax2.legend(title='Model', fontsize=11)

    forecast_start_date = df['forecast_date'].min()
    forecast_end_date = df['forecast_date'].max()

    for ax in axes:
        for label, (start, end, color, alpha_value) in event_periods.items():
            event_start = pd.Timestamp(start)
            event_end = pd.Timestamp(end)

            if event_start < forecast_end_date and event_end > forecast_start_date:
                ax.axvspan(event_start, event_end, alpha=alpha_value, color=color, ec='none')
        
        ax.set_xlim(forecast_start_date, forecast_end_date)
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=12)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(output_dir, f'full_performance_best_models_{horizon}m_horizon.png')
    plt.savefig(plot_filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f" Saved plot to '{plot_filename}'")
    plt.show()
    plt.close(fig)

def create_recent_error_plots(df, horizon, years_to_plot, model_meta, output_dir):
    """Generates a two-panel plot for the most recent N years of forecast errors."""
    if df is None or df.empty: return
    print(f"\nGenerating recent error plot for {horizon}-month horizon ({years_to_plot} years)...")
    end_date = df['forecast_date'].max()
    start_date = end_date - pd.DateOffset(years=years_to_plot)
    filtered_df = df[df['forecast_date'] >= start_date].copy()
    if filtered_df.empty:
        print(f"Warning: No data available in the last {years_to_plot} years to plot.")
        return
    for key in model_meta:
        if f'se_{key}' in filtered_df.columns:
            filtered_df[f'cumulative_se_recent_{key}'] = filtered_df[f'se_{key}'].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    ax1 = axes[0]
    for key, meta in model_meta.items():
        if key == 'benchmark_lasso_ols': continue
        diff_col = f'se_diff_{key}'
        if diff_col in filtered_df.columns:
            ax1.plot(filtered_df['forecast_date'], filtered_df[diff_col], label=f"{meta['name']} vs. Benchmark", color=meta['color'], linestyle=meta['style'], alpha=0.9)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax1.set_title(fr'(a) Squared Forecast Error Difference (Last {years_to_plot} Years)', fontsize=16, pad=10)
    ax1.set_ylabel(r'SE(Benchmark) - SE(Model)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=11)
    ax2 = axes[1]
    for key, meta in model_meta.items():
        cum_se_col = f'cumulative_se_recent_{key}'
        if cum_se_col in filtered_df.columns:
            ax2.plot(filtered_df['forecast_date'], filtered_df[cum_se_col], label=meta['name'], color=meta['color'], linestyle=meta['style'], alpha=0.9, linewidth=2 if key == 'benchmark_lasso_ols' else 1.5)
    ax2.set_title(fr'(b) Cumulative Squared Forecast Error (Last {years_to_plot} Years)', fontsize=16, pad=10)
    ax2.set_ylabel('Cumulative Squared Error', fontsize=14)
    ax2.set_xlabel('Forecast Date', fontsize=14)
    ax2.legend(title='Model', fontsize=11)
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 
    plot_filename = os.path.join(output_dir, f'recent_error_analysis_best_models_{years_to_plot}yrs_{horizon}m.png')
    plt.savefig(plot_filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f" Saved plot to '{plot_filename}'")
    plt.show()
    plt.close(fig)

def create_crisis_period_plots(df, crisis_periods, horizon, model_meta, output_dir):
    """Generates a separate plot of cumulative squared error for each defined crisis period."""
    if df is None or df.empty: return
    print(f"\nGenerating plots for crisis periods for {horizon}-month horizon...")
    for crisis_name, (start_date, end_date, color, alpha) in crisis_periods.items():
        crisis_df = df[(df['forecast_date'] >= start_date) & (df['forecast_date'] <= end_date)].copy()
        if crisis_df.empty:
            print(f"  - Skipping '{crisis_name}' plot: No forecast data in this period.")
            continue
        for key in model_meta:
            if f'se_{key}' in crisis_df.columns:
                crisis_df[f'cumulative_se_crisis_{key}'] = crisis_df[f'se_{key}'].cumsum()
        plt.figure(figsize=(10, 6))
        for key, meta in model_meta.items():
            cum_se_col = f'cumulative_se_crisis_{key}'
            if cum_se_col in crisis_df.columns:
                plt.plot(crisis_df['forecast_date'], crisis_df[cum_se_col], label=meta['name'], color=meta['color'], linestyle=meta['style'], alpha=0.9, linewidth=2 if key == 'benchmark_lasso_ols' else 1.5)
        plt.title(f'Cumulative Squared Forecast Error during {crisis_name.replace("_", " ")}\n({horizon}-Month Horizon)', fontsize=16, pad=10)
        plt.ylabel('Cumulative Squared Forecast Error', fontsize=15)
        plt.xlabel('Forecast Date', fontsize=15)
        plt.legend(title='Model', fontsize=15, title_fontsize=15)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(axis='both', labelsize=15)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'crisis_plot_best_models_{crisis_name}_{horizon}m_horizon.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved crisis plot to '{plot_filename}'")
        plt.show()
        plt.close()

CRISIS_PERIODS = {
    'Great Recession': ('2007-12-01', '2009-06-01', 'lightcoral', 0.2),
    'Brexit Uncertainty': ('2016-06-01', '2020-01-01', 'lightskyblue', 0.3),
    'COVID-19 Pandemic': ('2020-03-01', '2023-05-01', 'mediumpurple', 0.2)
}

def main_visualisations():
    """Main function to generate visualisations from saved results."""
    print("\n--- Generating Visualisations ---")
    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        print(" Using LaTeX for plot rendering.")
    except RuntimeError:
        print("Warning: LaTeX distribution not found. Using default font settings.")
        plt.rcdefaults()

    RESULTS_FILE_PATH = r"C:\\Users\\arthu\\OneDrive - University of Surrey\\Documents\\Surrey\\Semester 2\\Dissertation\\Data\\Final Dataset"
    RESULTS_FILE_NAME = 'final_out_of_sample_3_model_results.pkl'
    OUTPUT_DIRECTORY = "out_of_sample_visualisations_best_by_cum_se"

    all_results = load_results(RESULTS_FILE_PATH, RESULTS_FILE_NAME)
    if all_results:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        print(f"\nVisualisations will be saved to: '{os.path.abspath(OUTPUT_DIRECTORY)}'")

        available_horizons = sorted(list(all_results.get('horizon_results', {}).keys()))
        if available_horizons:
            create_rmse_vs_horizon_plot(all_results['horizon_results'], OUTPUT_DIRECTORY)
            for h in available_horizons:
                plot_df = prepare_plot_data(all_results, h)
                if plot_df is None or plot_df.empty:
                    continue
                best_model_keys = select_best_models_by_final_cse(plot_df)
                if not best_model_keys:
                    continue
                model_meta = generate_dynamic_model_meta(best_model_keys)
                create_full_performance_plots(plot_df, h, model_meta, OUTPUT_DIRECTORY)
                create_crisis_period_plots(plot_df, CRISIS_PERIODS, h, model_meta, OUTPUT_DIRECTORY)
                if h == 1:
                    create_recent_error_plots(plot_df, h, 5, model_meta, OUTPUT_DIRECTORY)

    plt.rcdefaults()
    print("\nVisualisations complete!")

# %% MAIN SEQUENTIAL EXECUTION

if __name__ == "__main__":
    horizons = [1, 3, 6, 12]
    print("3-MODEL MULTI-HORIZON UNEMPLOYMENT FORECASTING ANALYSIS")
    print("="*80)

    # === Run Estimation ===
    results, summary = main_multi_horizon_analysis(
        'final_dataset_ar_only_lags.xlsx',
        forecast_horizons=horizons,
        train_length=48
    )

    if results:
        print("\n Estimation completed successfully!")
    else:
        print("\n Estimation failed!")
        exit()

    # === Spreadsheet Export ===
    print("\n--- Generating Spreadsheet ---")
    RESULTS_FILE = 'final_out_of_sample_3_model_results.pkl'
    OUTPUT_EXCEL = 'Model_Out_of_Sample_Analysis_Results.xlsx'
    main_spreadsheet(results_file=RESULTS_FILE, output_file=OUTPUT_EXCEL)

    # === Visualisations ===
    main_visualisations()

    print("\nAll tasks completed successfully!")

