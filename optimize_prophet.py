import pandas as pd
import numpy as np
import os
import pickle
import warnings
from pathlib import Path
from itertools import product
from datetime import datetime

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore')


# ============================================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================================

PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'changepoint_range': [0.8, 0.85, 0.9, 0.95]
}
CV_CONFIG = {
    'initial': '3285 days',  # ~9 years for initial training
    'period': '365 days',     # ~1 year period between validation folds
    'horizon': '30 days',     # 30-day forecast horizon
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_stock_data(stock_symbol, data_folder='dataset', years_to_use=10):
    """
    Load stock data from CSV file.
    
    Parameters
    ----------
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    data_folder : str
        Path to folder containing stock data CSVs
    years_to_use : int
        Number of recent years to use for training
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Close, Volume, etc.
    
    Raises
    ------
    FileNotFoundError
        If stock data file not found
    """
    filepath = os.path.join(data_folder, f"{stock_symbol}_stock_data.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stock data not found: {filepath}")
    
    df = pd.read_csv(filepath, sep='|')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter to last N years if specified
    if years_to_use:
        cutoff_date = df['Date'].max() - pd.DateOffset(years=years_to_use)
        df = df[df['Date'] >= cutoff_date].reset_index(drop=True)
    
    return df


def add_technical_features(df):
    """
    Add technical indicator features to stock data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock data with 'Close' and 'Volume' columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added technical features
    """
    df = df.copy()
    
    # Exponential moving averages
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Volatility (smooth)
    df['Volatility_21'] = df['Close'].rolling(window=21, min_periods=1).std()
    df['Volatility_21'] = df['Volatility_21'].ewm(span=5, adjust=False).mean()
    
    # Momentum (smooth)
    df['Momentum_14'] = df['Close'].pct_change(periods=14).fillna(0)
    df['Momentum_14'] = df['Momentum_14'].ewm(span=5, adjust=False).mean()
    
    # Volume indicators
    df['Volume_EMA_21'] = df['Volume'].ewm(span=21, adjust=False).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_EMA_21']
    df['Volume_Ratio'] = df['Volume_Ratio'].ewm(span=5, adjust=False).mean()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def fit_prophet_with_features(train_df, params):
    """
    Fit Prophet model with technical features as regressors.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns: ds, y, and optional regressor columns
    params : dict
        Prophet hyperparameters (changepoint_prior_scale, seasonality_prior_scale, etc.)
    
    Returns
    -------
    Prophet
        Fitted Prophet model
    """
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        holidays_prior_scale=params.get('holidays_prior_scale', 0.1),
        changepoint_range=params.get('changepoint_range', 0.8),
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    
    # Add regressors (technical features)
    regressor_cols = ['EMA_7', 'EMA_21', 'EMA_50', 'Volatility_21', 'Momentum_14', 'Volume_Ratio']
    for col in regressor_cols:
        if col in train_df.columns:
            model.add_regressor(col)
    
    model.fit(train_df)
    return model


def prepare_prophet_data(df):
    """
    Prepare stock data for Prophet training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock data with 'Date' and 'Close' columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame formatted for Prophet (columns: ds, y, and regressors)
    """
    train = pd.DataFrame({
        'ds': df['Date'],
        'y': df['Close']
    })
    
    # Add technical features as regressors
    regressor_cols = ['EMA_7', 'EMA_21', 'EMA_50', 'Volatility_21', 'Momentum_14', 'Volume_Ratio']
    for col in regressor_cols:
        if col in df.columns:
            train[col] = df[col]
    
    return train


def evaluate_model(model, train_df, params_key, cv_results=None):
    """
    Evaluate Prophet model using cross-validation metrics.
    
    Parameters
    ----------
    model : Prophet
        Fitted Prophet model
    train_df : pd.DataFrame
        Training data
    params_key : str
        Parameter combination identifier
    cv_results : pd.DataFrame, optional
        Pre-computed cross-validation results
    
    Returns
    -------
    dict
        Dictionary with evaluation metrics (mape, rmse, mdape)
    """
    metrics = {}
    
    if cv_results is not None:
        # Use provided cross-validation results
        mape_values = cv_results['mape'].dropna()
        rmse_values = cv_results['rmse'].dropna()
        mdape_values = cv_results['mdape'].dropna()
        
        metrics['mape'] = mape_values.mean()
        metrics['rmse'] = rmse_values.mean()
        metrics['mdape'] = mdape_values.mean()
    else:
        # Fallback: fit on 80% of data, test on 20%
        split_idx = int(len(train_df) * 0.8)
        train_subset = train_df.iloc[:split_idx].copy()
        test_subset = train_df.iloc[split_idx:].copy()
        
        model.fit(train_subset)
        future = model.make_future_dataframe(periods=len(test_subset))
        
        # Add regressors to future dataframe
        regressor_cols = [col for col in train_df.columns if col not in ['ds', 'y']]
        for col in regressor_cols:
            if col in train_df.columns:
                future[col] = np.interp(
                    range(len(future)),
                    range(len(train_df)),
                    train_df[col].values
                )
        
        forecast = model.predict(future)
        test_forecast = forecast.iloc[split_idx:]['yhat'].values
        test_actual = test_subset['y'].values
        
        mape = mean_absolute_percentage_error(test_actual, test_forecast)
        rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
        
        metrics['mape'] = mape
        metrics['rmse'] = rmse
        metrics['mdape'] = mape  # Simplified: use MAPE as MDAPE proxy
    
    return metrics


def optimize_prophet_for_stock(
    stock_symbol,
    data_folder='dataset',
    model_folder='SaveModels',
    years_to_use=10,
    param_grid=None,
    cv_config=None,
    verbose=True,
    use_cross_validation=False,
    early_stop=None,
):
    """
    Optimize Prophet model hyperparameters for a specific stock using grid search.
    
    This function:
    1. Loads stock data
    2. Prepares technical features
    3. Tests multiple parameter combinations
    4. Evaluates each model using cross-validation or train/test split
    5. Selects and saves the best model
    
    Parameters
    ----------
    stock_symbol : str
        Stock ticker (e.g., 'AAPL', 'MSFT')
    data_folder : str
        Path to folder containing stock CSVs
    model_folder : str
        Path to folder for saving optimized models
    years_to_use : int
        Number of recent years to use for training
    param_grid : dict, optional
        Prophet hyperparameters to test. If None, uses PROPHET_PARAM_GRID
    cv_config : dict, optional
        Cross-validation configuration. If None, uses CV_CONFIG
    verbose : bool
        Print progress information
    use_cross_validation : bool
        If True, use Prophet cross_validation (slower but more thorough).
        If False, use simple train/test split (faster).
    early_stop : dict or int or None
        Early stopping configuration for grid search. If None, early stopping
        is disabled. If an int is provided, it is treated as `patience` (number
        of consecutive non-improving trials before stopping). If a dict is
        provided it can include:
            - 'patience' (int): number of non-improving trials to tolerate
            - 'min_delta' (float): minimum improvement required to reset patience
            - 'metric' (str): metric to monitor (default: 'mape')
    
    Returns
    -------
    dict
        Dictionary containing:
            'best_model': Prophet model object
            'best_params': Best hyperparameter dict
            'best_metrics': Evaluation metrics of best model
            'all_results': DataFrame with all parameter combinations tested
    
    Example
    -------
    >>> result = optimize_prophet_for_stock(
    ...     'AAPL',
    ...     data_folder='dataset',
    ...     model_folder='SaveModels',
    ...     use_cross_validation=True
    ... )
    >>> print(f"Best MAPE: {result['best_metrics']['mape']:.4f}")
    >>> print(f"Best parameters: {result['best_params']}")
    """
    
    if param_grid is None:
        param_grid = PROPHET_PARAM_GRID
    if cv_config is None:
        cv_config = CV_CONFIG
    
    # Create model folder if it doesn't exist
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing Prophet for {stock_symbol}")
        print(f"{'='*70}")
    
    # Load and prepare data
    if verbose:
        print(f"[1/5] Loading stock data for {stock_symbol}...")
    df = load_stock_data(stock_symbol, data_folder=data_folder, years_to_use=years_to_use)
    df = add_technical_features(df)
    train_df = prepare_prophet_data(df)
    
    if verbose:
        print(f"      Loaded {len(df)} trading days ({df['Date'].min().date()} to {df['Date'].max().date()})")
    
    # Generate parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    if verbose:
        print(f"[2/5] Testing {len(param_combinations)} parameter combinations...")
    
    # Grid search
    results = []
    best_mape = float('inf')
    best_model = None
    best_params = None
    best_metrics = None
    # Early stopping setup (grid-search level)
    es_enabled = False
    es_patience = None
    es_min_delta = 0.0
    es_metric = 'mape'
    es_no_improve = 0
    if early_stop is not None:
        es_enabled = True
        if isinstance(early_stop, int):
            es_patience = int(early_stop)
        elif isinstance(early_stop, dict):
            es_patience = int(early_stop.get('patience', 3))
            es_min_delta = float(early_stop.get('min_delta', 0.0))
            es_metric = early_stop.get('metric', 'mape')
        else:
            es_patience = 3
    
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        try:
            # Fit model
            model = fit_prophet_with_features(train_df.copy(), params)
            
            # Evaluate model
            if use_cross_validation and len(train_df) > 365:
                # Use cross-validation for evaluation (slower but more robust)
                if verbose:
                    print(f"      [{i+1}/{len(param_combinations)}] Testing {params}...")
                cv_results = cross_validation(
                    model,
                    initial=cv_config['initial'],
                    period=cv_config['period'],
                    horizon=cv_config['horizon'],
                    parallel="processes",
                    verbose=False,
                )
                metrics_df = performance_metrics(cv_results)
                metrics = evaluate_model(model, train_df, f"{i}", metrics_df)
            else:
                # Use simple train/test split (faster)
                metrics = evaluate_model(model, train_df, f"{i}")
            
            results.append({
                'param_combination': str(params),
                'mape': metrics['mape'],
                'rmse': metrics['rmse'],
                'mdape': metrics['mdape'],
                **params
            })
            
            if verbose:
                print(f"      [{i+1}/{len(param_combinations)}] MAPE: {metrics['mape']:.4f}, RMSE: {metrics['rmse']:.4f}")

            # Determine current metric value to compare
            current_metric_value = metrics.get(es_metric, metrics.get('mape'))

            # Track best model with early stopping
            if current_metric_value + es_min_delta < best_mape:
                best_mape = current_metric_value
                best_model = model
                best_params = params
                best_metrics = metrics
                # reset early-stop counter
                es_no_improve = 0

                if verbose:
                    print(f"      >>> NEW BEST! {es_metric.upper()}: {best_mape:.4f}")
            else:
                if es_enabled:
                    es_no_improve += 1
                    if verbose:
                        print(f"      Early-stop counter: {es_no_improve}/{es_patience}")
                    if es_patience is not None and es_no_improve >= es_patience:
                        if verbose:
                            print(f"      Early stopping: no improvement in {es_patience} consecutive trials.")
                        break
        
        except Exception as e:
            if verbose:
                print(f"      [{i+1}/{len(param_combinations)}] ERROR: {str(e)[:50]}")
            results.append({
                'param_combination': str(params),
                'mape': float('inf'),
                'rmse': float('inf'),
                'mdape': float('inf'),
                **params
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mape')
    
    if verbose:
        print(f"[3/5] Optimization complete!")
        print(f"\nTop 5 Parameter Combinations:")
        print(results_df[['mape', 'rmse', 'changepoint_prior_scale', 'seasonality_prior_scale']].head())
    
    # Save best model
    if verbose:
        print(f"[4/5] Saving optimized model for {stock_symbol}...")
    
    model_path = os.path.join(model_folder, f"{stock_symbol}_model.pkl")
    params_path = os.path.join(model_folder, f"{stock_symbol}_best_params.pkl")
    metrics_path = os.path.join(model_folder, f"{stock_symbol}_metrics.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    with open(metrics_path, 'wb') as f:
        pickle.dump(best_metrics, f)
    
    if verbose:
        print(f"      Model saved to: {model_path}")
        print(f"      Parameters saved to: {params_path}")
    
    if verbose:
        print(f"[5/5] Best Parameters:")
        for key, value in best_params.items():
            print(f"      {key}: {value}")
        print(f"\nBest Metrics:")
        for key, value in best_metrics.items():
            print(f"      {key}: {value:.6f}")
        print(f"\n{'='*70}\n")
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'all_results': results_df,
        'stock_symbol': stock_symbol,
        'timestamp': datetime.now()
    }


def optimize_all_stocks(
    stock_symbols=None,
    data_folder='dataset',
    model_folder='SaveModels',
    years_to_use=10,
    use_cross_validation=True,
    verbose=True,
    early_stop=None,
):
    """
    Optimize Prophet models for multiple stocks.
    
    Parameters
    ----------
    stock_symbols : list, optional
        List of stock symbols to optimize. If None, uses all CSV files in data_folder.
    data_folder : str
        Path to folder containing stock CSVs
    model_folder : str
        Path to folder for saving optimized models
    years_to_use : int
        Number of recent years to use for training
    use_cross_validation : bool
        Use Prophet cross-validation (slower but more robust)
    verbose : bool
        Print progress information
    
    Returns
    -------
    dict
        Dictionary mapping stock symbols to optimization results
    """
    
    if stock_symbols is None:
        # Auto-detect stocks from CSV files
        csv_files = [f.replace('_stock_data.csv', '') for f in os.listdir(data_folder) 
                     if f.endswith('_stock_data.csv')]
        stock_symbols = sorted(csv_files)
    
    results = {}
    
    for stock in stock_symbols:
        try:
            result = optimize_prophet_for_stock(
                stock,
                data_folder=data_folder,
                model_folder=model_folder,
                years_to_use=years_to_use,
                use_cross_validation=use_cross_validation,
                verbose=verbose,
                early_stop=early_stop,
            )
            results[stock] = result
        except Exception as e:
            if verbose:
                print(f"ERROR optimizing {stock}: {str(e)}")
            results[stock] = {'error': str(e)}
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    Option 1: Optimize a single stock
    >>> result = optimize_prophet_for_stock('AAPL')
    
    Option 2: Optimize all stocks (fast method)
    >>> results = optimize_all_stocks(use_cross_validation=False)
    
    Option 3: Optimize all stocks with cross-validation (slower but more thorough)
    results = optimize_all_stocks(use_cross_validation=True)
    
    Option 4: Optimize specific stocks only
    >>> results = optimize_all_stocks(
    ...     stock_symbols=['AAPL', 'MSFT', 'NVDA'],
    ...     use_cross_validation=True
    ... )
    """
    
    # Example: Optimize a subset of stocks with cross-validation disabled (faster)
    print("Starting Prophet Model Optimization...")
    print("="*70)
    
    stock_symbols=['SHW','TRV', 'UNH', 'V']
    
    results = optimize_all_stocks(
        stock_symbols=stock_symbols,
        use_cross_validation=False,
        verbose=True,
        early_stop= 100
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    for stock, result in results.items():
        if 'error' not in result:
            print(f"\n{stock}:")
            print(f"  Best MAPE: {result['best_metrics']['mape']:.6f}")
            print(f"  Best RMSE: {result['best_metrics']['rmse']:.6f}")
            print(f"  Best Parameters: {result['best_params']}")
        else:
            print(f"\n{stock}: ERROR - {result['error']}")
