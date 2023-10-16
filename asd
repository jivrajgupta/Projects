https://www.tejwin.com/en/insight/arima-garch-modelpart-1/#Database
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product

def find_best_exponential_smoothing_params(data, seasonal_periods=12):
    """
    Find the best parameters for exponential smoothing using grid search and generate test results.
    
    Parameters:
        data (pd.Series): Univariate time series data.
        seasonal_periods (int): Number of periods in a season (default is 12 for monthly data).
        
    Returns:
        best_params (dict): Dictionary containing the best parameters for exponential smoothing.
        test_results (pd.Series): Series containing test results from the best model.
    """
    best_aic = float("inf")
    best_params = {}
    test_results = None
    
    lambda_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Smoothing parameter for Box-Cox transformation
    alpha_range = [0.1, 0.3, 0.5, 0.7, 0.9]  # Smoothing parameter for level
    beta_range = [0.1, 0.3, 0.5, 0.7, 0.9]  # Smoothing parameter for trend
    
    trend_options = ['add', 'multiplicative']
    seasonal_options = ['add', 'multiplicative']
    
    for lambda_val, alpha, beta, trend, seasonal in product(lambda_range, alpha_range, beta_range, trend_options, seasonal_options):
        model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        try:
            result = model.fit(smoothing_level=alpha, smoothing_slope=beta, damping_slope=1.0, use_boxcox=True, lamda=lambda_val)
            aic = result.aic
            if aic < best_aic:
                best_aic = aic
                best_params = {
                    'lambda': lambda_val,
                    'alpha': alpha,
                    'beta': beta,
                    'trend': trend,
                    'seasonal': seasonal
                }
                # Generate test results using the best model
                test_results = result.forecast(len(data))
        except:
            continue
    
    return best_params, test_results

# Example usage:
# Load your univariate time series data into a pandas Series (replace 'your_data.csv' with your actual data file path)
# data = pd.read_csv('your_data.csv', header=None, names=['value_column'])['value_column']

# Call the function to find the best parameters for exponential smoothing and generate test results
# best_params, test_results = find_best_exponential_smoothing_params(data)
# print("Best Exponential Smoothing Parameters:", best_params)
