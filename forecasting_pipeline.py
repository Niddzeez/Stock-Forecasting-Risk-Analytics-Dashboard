import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Prophet
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

# Metrics & preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Streamlit
import streamlit as st

# Warnings & plotting styles
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 8)


def build_arima_model_main(train_data, test_data, order=(1, 1, 1), seasonal_order=None,
                      threshold=0.004, retrain_freq=1, verbose=True):
    """Walk-forward ARIMA/SARIMA forecasting using provided orders."""

    use_seasonal = seasonal_order is not None
    predictions = []
    current_train = train_data.copy()

    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    for i in range(len(test_data)):
        if (i % retrain_freq == 0) or i == 0:
            if use_seasonal:
                model = SARIMAX(
                    current_train['Close'],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(current_train['Close'], order=order)

            fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.get_forecast(steps=1)
        pred = float(forecast.predicted_mean.iloc[-1])  # make sure this is a float
        ci = forecast.conf_int().iloc[-1]                # get last row of CI
        ci_lower = float(ci[0])                          # convert to float
        ci_upper = float(ci[1])
        current_price = float(current_train['Close'].iloc[-1])  # ensure scalar float
        actual = float(test_data['Close'].iloc[i])              # also make scalar
        

        if (pred > current_price * (1 + threshold)) and (ci_lower > current_price):
            direction = 'BUY'
        elif (pred < current_price * (1 - threshold)) and (ci_upper < current_price):
            direction = 'SELL'
        else:
            direction = 'HOLD'

        if verbose:
            date_val = pd.to_datetime(test_data.index[i]).date()
            print(f"{date_val} | CP: {current_price:.2f} | Pred: {pred:.2f} | CI: ({ci[0]:.2f}, {ci[1]:.2f}) | Dir: {direction}")

        predictions.append({
            'Date': test_data.index[i],
            'Current_Price': current_price,
            'Predicted_Price': pred,
            'Actual_Price': actual,
            'Direction': direction,
            'Error': abs(pred - actual),
            'CI_Lower': ci[0],
            'CI_Upper': ci[1]
        })

        # Expand training set
        current_train = pd.concat([current_train, test_data.iloc[i:i+1][['Close']]])
    
    if not predictions:
        print("⚠️ No predictions were made. Check training data size or model fit step.")
        return pd.DataFrame(), None

    results_df = pd.DataFrame(predictions).set_index('Date')

    # Safe type handling
    results_df['Direction'] = results_df['Direction'].astype(str)
    results_df['Actual_Price'] = pd.to_numeric(results_df['Actual_Price'], errors='coerce')
    results_df['Current_Price'] = pd.to_numeric(results_df['Current_Price'], errors='coerce')
    results_df.dropna(subset=['Direction', 'Actual_Price', 'Current_Price'], inplace=True)

    # Fix for ambiguous truth value error
    buy_correct = (results_df['Direction'] == 'BUY') & (results_df['Actual_Price'] > results_df['Current_Price'])
    sell_correct = (results_df['Direction'] == 'SELL') & (results_df['Actual_Price'] < results_df['Current_Price'])
    results_df['Correct_Direction'] = buy_correct | sell_correct

    results_df['Signal_Return'] = results_df['Actual_Price'].pct_change().shift(-1)

    # Metrics
    trade_mask = results_df['Direction'] != 'HOLD'
    accuracy = results_df.loc[trade_mask, 'Correct_Direction'].mean()
    mae = mean_absolute_error(results_df['Actual_Price'], results_df['Predicted_Price'])
    avg_return = results_df.loc[trade_mask, 'Signal_Return'].mean()
    trade_freq = trade_mask.mean()
    rmse = np.sqrt(mean_squared_error(results_df['Actual_Price'], results_df['Predicted_Price']))
    mape = np.mean(np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / results_df['Actual_Price'])) * 100

    if verbose:
        print(f"\n📊 Model Performance:")
        print(f"• MAE                : ${mae:.2f}")
        print(f"• RMSE               : ${rmse:.2f}")
        print(f"• MAPE               : {mape:.2f}%")
        print(f"• Direction Accuracy : {accuracy:.2%}")
        print(f"• Trade Frequency    : {trade_freq:.2%}")
        print(f"• Avg Signal Return  : {avg_return:.2%}")

    return results_df, fitted_model




def walk_forward_sarima_final(train_data, test_data, 
                        order=(1,1,1), seasonal_order=(1,0,1,5), 
                        threshold=0.001, retrain_freq=1, 
                        forecast_horizon=1, verbose=True):
    """
    Walk-forward SARIMA forecasting with direction-based signal generation.
    Supports configurable retraining and forecast horizon.
    """
    predictions = []
    current_train = train_data.copy()

    for i in range(len(test_data)):
        # Refit model at intervals
        if (i % retrain_freq == 0) or (i == 0):
            model = SARIMAX(
                current_train['Close'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)

        # Forecast
        forecast = fitted_model.get_forecast(steps=forecast_horizon)
        pred = float(forecast.predicted_mean.iloc[-1])
        ci = forecast.conf_int().iloc[-1]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

        current_price = float(current_train['Close'].iloc[-1])
        actual = float(test_data['Close'].iloc[i])

        # Signal Logic
        if (pred > current_price * (1 + threshold)) and (ci_lower > current_price):
            direction = 'BUY'
        elif (pred < current_price * (1 - threshold)) and (ci_upper < current_price):
            direction = 'SELL'
        else:
            direction = 'HOLD'

        if verbose:
            print(f"{test_data.index[i].date()} | CP: {current_price:.2f} | Pred: {pred:.2f} | "
                  f"CI: ({ci_lower:.2f}, {ci_upper:.2f}) | Dir: {direction} | Actual: {actual:.2f}")

        # Store prediction
        predictions.append({
            'Date': test_data.index[i],
            'Current_Price': current_price,
            'Predicted_Price': pred,
            'Actual_Price': actual,
            'Direction': direction,
            'Error': abs(pred - actual),
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })

        # Update training data
        current_train = pd.concat([
            current_train,
            test_data.iloc[i:i+1][['Close']]
        ])

    # Convert to DataFrame
    results_df = pd.DataFrame(predictions).set_index('Date')

    # Evaluation columns
    results_df['Correct_Direction'] = np.where(
        (results_df['Direction'] == 'BUY') & (results_df['Actual_Price'] > results_df['Current_Price']) |
        (results_df['Direction'] == 'SELL') & (results_df['Actual_Price'] < results_df['Current_Price']),
        True, False
    )
    results_df['Signal_Return'] = results_df['Actual_Price'].pct_change().shift(-1)

    # Metrics
    trade_mask = results_df['Direction'] != 'HOLD'
    accuracy = results_df.loc[trade_mask, 'Correct_Direction'].mean()
    trade_freq = trade_mask.mean()
    avg_return = results_df.loc[trade_mask, 'Signal_Return'].mean()
    mae = mean_absolute_error(results_df['Actual_Price'], results_df['Predicted_Price'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual_Price'], results_df['Predicted_Price']))
    mape = np.mean(np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / results_df['Actual_Price'])) * 100

    # Optional printout
    if verbose:
        print(f"\n📊 Final Model Performance:")
        print(f"• Direction Accuracy : {accuracy:.2%}")
        print(f"• Trade Frequency    : {trade_freq:.2%}")
        print(f"• Avg Signal Return  : {avg_return:.2%}")
        print(f"• MAE                : ${mae:.2f}")
        print(f"• RMSE               : ${rmse:.2f}")
        print(f"• MAPE               : {mape:.2f}%")

    # Return results + metrics
    return results_df, fitted_model


def compute_mape(y_true, y_pred):
    """Calculate MAPE safely with zero handling"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def simple_sarimax_forecast(train_data, test_data, features=None, order=(1,1,1), seasonal_order=(1,0,1,5), verbose=False):
    """
    Simplified SARIMAX forecasting with exogenous variables
    
    Parameters:
    - train: Training DataFrame (must include 'Close' and features)
    - test: Test DataFrame (same structure as train)
    - features: List of exogenous feature columns
    - order: SARIMAX (p,d,q) order
    - seasonal_order: SARIMAX seasonal order
    
    Returns:
    - results: DataFrame with predictions
    - model: Trained SARIMAX model
    - metrics: Dictionary of performance metrics
    """

    train = train_data.copy()
    test = test_data.copy()
    
    # Set default features if not provided
    if features is None:
        features = ['LogVolume', 'Volume_MA5', 'RollingVolatility', 'RSI', 'MA5', 'MA20', 'Daily_Return']
    
    # Clean data (handle inf/nan)
    def clean(df):
        return df[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    exog_train = clean(train)
    exog_test = clean(test)
    
    # Train model
    model = SARIMAX(train['Close'], exog=exog_train, 
                   order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Generate predictions
    preds = []
    for i in range(len(test)):
        forecast = model_fit.get_forecast(steps=1, exog=exog_test.iloc[[i]])
        pred = float(forecast.predicted_mean.iloc[0])
        ci = forecast.conf_int().iloc[0].values

        
        current_price = float(train['Close'].iloc[-1] if i == 0 else preds[-1]['Actual_Price'])
        actual = float(test['Close'].iloc[i])

        ci_lower, ci_upper = float(ci[0]), float(ci[1])
        
        # Simple trading signal
        if pred > current_price and ci_lower > current_price:
            direction = 'BUY'
            ret = (actual - current_price) / current_price
        elif pred < current_price and ci_upper < current_price:
            direction = 'SELL'
            ret = (current_price - actual) / current_price
        else:
            direction = 'HOLD'
            ret = 0
            
        preds.append({
            'Date': test.index[i],
            'Predicted_Price': pred,
            'Actual_Price': actual,
            'Current_Price': current_price,
            'Direction': direction,
            'Signal_Return': ret,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })
    
    # Create results DataFrame
    results = pd.DataFrame(preds).set_index('Date')
    
    # Calculate metrics
    trades = results[results['Direction'] != 'HOLD']
    correct = ((trades['Direction'] == 'BUY') & (trades['Actual_Price'] > trades['Current_Price'])) | \
              ((trades['Direction'] == 'SELL') & (trades['Actual_Price'] < trades['Current_Price']))
    
    metrics = {
        'MAE': mean_absolute_error(results['Actual_Price'], results['Predicted_Price']),
        'RMSE': np.sqrt(mean_squared_error(results['Actual_Price'], results['Predicted_Price'])),
        'MAPE': compute_mape(results['Actual_Price'].values, results['Predicted_Price'].values),
        'Accuracy': correct.mean() if len(trades) > 0 else np.nan,
        'Trade_Rate': len(trades) / len(results),
        'Avg_Return': trades['Signal_Return'].mean() if len(trades) > 0 else 0,
        'Total_Return': trades['Signal_Return'].sum()
    }

    
    return results, model_fit

def compute_mape(y_true, y_pred):
    """Simpler MAPE calculation with zero division handling"""
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape if np.isfinite(mape) else np.nan

def build_sarimax_rolling_forecast(train_data, test_data, 
                                  order=(1, 1, 1), 
                                  seasonal_order=(1, 0, 1, 5), 
                                  exog_features=None,
                                  retrain_freq=15, 
                                  window_size=120,
                                  verbose=True):
    """
    Simplified SARIMAX model with rolling window forecasting.
    
    Parameters:
    - train_data, test_data: DataFrames with 'Close' price and exogenous features
    - order: SARIMAX order parameters
    - seasonal_order: SARIMAX seasonal order parameters
    - exog_features: List of exogenous feature columns
    - retrain_freq: How often to retrain model (in periods)
    - window_size: Rolling window size for training
    - verbose: Whether to print performance metrics
    
    Returns:
    - results_df: DataFrame with predictions and metrics
    - model: The final trained model
    """
    # Default exogenous features
    if exog_features is None:
        exog_features = ['LogVolume', 'Volume_MA5', 'RollingVolatility', 'RSI', 'MA5', 'MA20', 'Daily_Return']
    
    # Clean and prepare data
    def prepare_data(data):
        data = data.copy()
        exog = data[exog_features].replace([np.inf, -np.inf], np.nan)
        exog = exog.ffill().bfill()
        return data['Close'], exog
    
    y_train, exog_train = prepare_data(train_data)
    y_test, exog_test = prepare_data(test_data)
    
    # Initialize model
    model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    try:
        fitted_model = model.fit(disp=False)
    except Exception as e:
        if verbose:
            print(f"Initial model fitting failed: {str(e)}")
        return None, None
    
    # Prepare rolling forecast
    predictions = []
    current_train = y_train.copy()
    current_exog = exog_train.copy()
    
    for i in range(len(y_test)):
        # Get current exogenous variables
        exog_i = exog_test.iloc[[i]]
        
        try:
            # Forecast next step
            forecast = fitted_model.get_forecast(steps=1, exog=exog_i)
            pred = float(forecast.predicted_mean.iloc[0])
            ci = forecast.conf_int().iloc[0].values
            
            # Generate trading signal
            current_price = float(current_train.iloc[-1]) if len(predictions) == 0 else predictions[-1]['Actual_Price']
            actual = float(y_test.iloc[i])

            ci_lower, ci_upper = map(float, forecast.conf_int().iloc[0].values)

            # Simplified signal logic
            if pred > current_price and ci_lower > current_price:
                direction = 'BUY'
                signal_return = (actual - current_price) / current_price
            elif pred < current_price and ci_upper < current_price:
                direction = 'SELL'
                signal_return = (current_price - actual) / current_price
            else:
                direction = 'HOLD'
                signal_return = 0
                
            predictions.append({
                'Date': y_test.index[i],
                'Current_Price': current_price,
                'Predicted_Price': pred,
                'Actual_Price': actual,
                'Direction': direction,
                'Error': abs(pred - actual),
                'CI_Lower': ci[0],
                'CI_Upper': ci[1],
                'Signal_Return': signal_return
            })
            
            # Update training window
            current_train = pd.concat([current_train, y_test.iloc[[i]]])
            current_exog = pd.concat([current_exog, exog_test.iloc[[i]]])
            
            # Maintain window size
            if len(current_train) > window_size:
                current_train = current_train.iloc[-window_size:]
                current_exog = current_exog.iloc[-window_size:]
            
            # Retrain periodically
            if (i + 1) % retrain_freq == 0:
                model = SARIMAX(current_train, exog=current_exog,
                              order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit(disp=False)
                
        except Exception as e:
            if verbose:
                print(f"Error at step {i}: {str(e)}")
            continue
    
    # Process results
    results_df = pd.DataFrame(predictions).set_index('Date')
    
    if len(results_df) == 0:
        if verbose:
            print("No successful predictions generated")
        return None, None
    
    # Calculate metrics
    trade_mask = results_df['Direction'] != 'HOLD'
    metrics = {
        'mae': mean_absolute_error(results_df['Actual_Price'], results_df['Predicted_Price']),
        'rmse': np.sqrt(mean_squared_error(results_df['Actual_Price'], results_df['Predicted_Price'])),
        'mape': compute_mape(results_df['Actual_Price'].values, results_df['Predicted_Price'].values),
        'direction_accuracy': ((
            (results_df['Direction'] == 'BUY') & (results_df['Actual_Price'] > results_df['Current_Price']) |
            (results_df['Direction'] == 'SELL') & (results_df['Actual_Price'] < results_df['Current_Price'])
        )[trade_mask].mean()),
        'trade_freq': trade_mask.mean(),
        'avg_return': results_df.loc[trade_mask, 'Signal_Return'].mean()
    }
    
    if verbose:
        print("\n📊 Model Performance:")
        print(f"• Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"• Trade Frequency: {metrics['trade_freq']:.2%}")
        print(f"• Avg Signal Return: {metrics['avg_return']:.2%}")
        print(f"• MAE: ${metrics['mae']:.2f}")
        print(f"• RMSE: ${metrics['rmse']:.2f}")
        print(f"• MAPE: {metrics['mape']:.2f}%")
    
    return results_df, fitted_model

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_best_arima_params(train_series, seasonal=True, m=5, exog=None):
    warnings.filterwarnings("ignore")

    print("🔍 Starting ARIMA hyperparameter grid search...\n")

    # Basic grid setup
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, [1], q))  # d=1 as a rule
    seasonal_pdq = [(0, 0, 0, m)]

    if seasonal:
        seasonal_pdq += list(itertools.product(range(0, 2), range(0, 2), range(0, 2), [m]))

    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    trial_count = 0
    for order in pdq:
        for s_order in seasonal_pdq:
            trial_count += 1
            try:
                model = sm.tsa.statespace.SARIMAX(
                    train_series,
                    exog=exog,
                    order=order,
                    seasonal_order=s_order if seasonal else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

                aic = results.aic
                print(f" Trying {trial_count:03d}: ARIMA{order} x Seasonal{s_order} | AIC = {aic:.2f}", end="")

                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = s_order
                else:
                    print("")

            except Exception as e:
                print(f"❌ Failed ({type(e).__name__})")

    print("\n✅ Best Model Selection Completed:")
    print(f"   - Best ARIMA order: {best_order}")
    print(f"   - Best Seasonal order: {best_seasonal_order}")
    print(f"   - Best AIC: {best_aic:.2f}")
    
    return best_order, best_seasonal_order


def run_all_models_with_auto_params_new(ticker_name, df, train_data, test_data):
    try:
        # ----- Data Validation -----
        if train_data.empty or test_data.empty:
            raise ValueError("Empty train/test data provided")
            
        required_cols = ['Close', 'Volume']
        if not all(col in train_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in train_data.columns]
            raise ValueError(f"Missing columns: {missing}")

        # ----- Feature Engineering -----
        exog_features = ['LogVolume', 'Volume_MA5', 'RollingVolatility', 
                        'RSI', 'MA5', 'MA20', 'Daily_Return']
        
        def process_data(dataset):
            dataset = dataset.copy()
            
            # Feature calculations
            dataset['LogVolume'] = np.log1p(dataset['Volume'])
            dataset['Volume_MA5'] = dataset['Volume'].rolling(5, min_periods=1).mean()
            dataset['RollingVolatility'] = dataset['Close'].pct_change().rolling(5, min_periods=1).std()
            dataset['RSI'] = dataset['Close'].pct_change().rolling(14, min_periods=1).mean()
            dataset['MA5'] = dataset['Close'].rolling(5, min_periods=1).mean()
            dataset['MA20'] = dataset['Close'].rolling(20, min_periods=1).mean()
            dataset['Daily_Return'] = dataset['Close'].pct_change()


            
            # Handle missing values
            if dataset.isnull().values.any():
                dataset.ffill(inplace=True)
                dataset.bfill(inplace=True)
                if dataset.isnull().values.any():
                    raise ValueError("Could not fill all missing values")
                                        
            return dataset

        train_data = process_data(train_data)
        test_data = process_data(test_data)

        # ----- Feature Scaling -----
        scaler = StandardScaler()
        try:
            train_data[exog_features] = scaler.fit_transform(train_data[exog_features])
            test_data[exog_features] = scaler.transform(test_data[exog_features])
        except ValueError as e:
            print(f"Scaling failed: {str(e)} - using unscaled features")
            exog_features = [f for f in exog_features if train_data[f].var() > 0]  # Remove constant features

        # ----- Auto-ARIMA Parameters -----
        try:
            exog_train = train_data[exog_features]
            order, seasonal = get_best_arima_params(
                train_series=train_data['Close'],
                exog=exog_train,
            )
            
            print(f"Order : {order} Seasonal : {seasonal}")
        except:
            print("Auto-ARIMA failed, using default parameters")
            order, seasonal = (1,1,1), (1,0,1,5)

        # ----- Model Execution -----
        model_results = []
        
        def safe_model_run(model_func, name, **kwargs):
            try:
                result, fitted_model = model_func(**kwargs)
                return result, fitted_model
            except Exception as e:
                print(f"🔴 {name} failed: {str(e)}")
                return pd.DataFrame(), None


        models = [
            (build_arima_model_main, "ARIMA"),
            (walk_forward_sarima_final, "SARIMA"),
            (simple_sarimax_forecast, "SARIMAX + exog"),
            (build_sarimax_rolling_forecast, "SARIMAX + exog + rolling")
        ]
        
        for model_func, name in models:
            result, fitted_model = safe_model_run(
                model_func,
                name,
                train_data = train_data.copy(),
                test_data = test_data.copy(),
                order=order,
                seasonal_order=seasonal,
                verbose=False
            )
            
            if result is not None:
                try:
                    trades = result[result['Direction'] != 'HOLD']
                    correct = ((trades['Direction'] == 'BUY') & (trades['Actual_Price'] > trades['Current_Price'])) | \
                             ((trades['Direction'] == 'SELL') & (trades['Actual_Price'] < trades['Current_Price']))
                    
                    model_results.append({
                    "Ticker": ticker_name,
                    "Model": name,
                    "MAE": mean_absolute_error(result['Actual_Price'], result['Predicted_Price']),
                    "RMSE": np.sqrt(mean_squared_error(result['Actual_Price'], result['Predicted_Price'])),
                    "MAPE": compute_mape(result['Actual_Price'], result['Predicted_Price']),
                    "Direction_Acc": correct.mean() if len(trades) > 0 else np.nan,
                    "Trade_Freq": len(trades)/len(result),
                    "Avg_Return": trades['Signal_Return'].mean() if len(trades) > 0 else 0,
                    "fitted_model": fitted_model,
                    "scaler": scaler if name.startswith("SARIMAX") else None
                })

                except KeyError as e:
                    print(f"Metric calculation failed for {name}: missing column {str(e)}")

                
        return (
                pd.DataFrame([{k: v for k, v in res.items() if k != 'fitted_model'} for res in model_results]),
                    model_results,
                    train_data,
                    test_data,
                    order,
                    seasonal
        )
    
    except Exception as e:
        print(f"🔴 Fatal error processing {ticker_name}: {str(e)}")
        traceback.print_exc()  # Print full traceback to know exactly what went wrong
        return pd.DataFrame(), [], None, None, None, None  # Return 6 values so unpacking doesn’t crash

def forecast_arima(model, df, steps=[1, 7, 30]):
    import pandas as pd
    import numpy as np

    forecast_obj = model.get_forecast(steps=max(steps))
    forecast_frame = forecast_obj.summary_frame()
    last_date = df.index.max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max(steps))
    selected = forecast_frame.iloc[:max(steps)][["mean", "mean_ci_lower", "mean_ci_upper"]].copy()
    selected.reset_index()
    selected["Date"] = future_dates
    selected["Days"] = np.arange(1, max(steps) + 1)

    # ✅ Rename columns for clarity
    selected.rename(columns={
        "mean": "Forecast",
        "mean_ci_lower": "Lower_CI",
        "mean_ci_upper": "Upper_CI"
    }, inplace=True)

    return selected  


def forecast_sarima(model, df, steps=[1, 7, 30]):
    return forecast_arima(model, df, steps)

def forecast_sarimax(model, df, steps=[1, 7, 30], scaler=None):
    forecast_list = []
    exog_cols = ['LogVolume', 'Volume_MA5', 'RollingVolatility', 
                 'RSI', 'MA5', 'MA20', 'Daily_Return']

    df = df.copy()

    # Create features
    df['LogVolume'] = np.log1p(df['Volume'])
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['RollingVolatility'] = df['Close'].pct_change().rolling(5, min_periods=1).std()
    df['RSI'] = df['Close'].pct_change().rolling(14, min_periods=1).mean()
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['Daily_Return'] = df['Close'].pct_change()

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Extract last row of exog
    last_known = df[exog_cols].iloc[-1].copy().astype(float).fillna(0).replace([np.inf, -np.inf], 0)

    # Build repeated exog
    future_exog = pd.DataFrame([last_known.values] * max(steps), columns=exog_cols)

    # Apply same scaler as training
    if scaler is not None:
        try:
            future_exog = pd.DataFrame(scaler.transform(future_exog), columns=exog_cols)
        except Exception as e:
            raise RuntimeError(f"Scaler transformation failed: {e}")

    # Forecast
    forecast_obj = model.get_forecast(steps=max(steps), exog=future_exog)
    forecast_df = forecast_obj.summary_frame()
    forecast_df.reset_index(drop=True, inplace=True)
    forecast_df['Days'] = np.arange(1, max(steps) + 1)
    
    filtered = forecast_df[forecast_df['Days'].isin(steps)].copy()
    filtered.rename(columns={
        'index': 'Date',
        'mean': 'Forecast',
        'mean_ci_lower': 'Lower_CI',
        'mean_ci_upper': 'Upper_CI'
    }, inplace=True)

    return filtered[['Date', 'Forecast', 'Lower_CI', 'Upper_CI', 'Days']]


def forecast_sarimax_rolling(model, df, steps=[1, 7, 30], scaler=None):
    # Same logic as forecast_sarimax for now
    return forecast_sarimax(model, df, steps, scaler)

def download_and_prepare_ticker_data(ticker, start="2022-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print(f"❌ No data downloaded for {ticker}")
        return None

    df = df[['Close', 'Volume']].copy()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds'}, inplace=True)  # Keep 'Close' as-is
    df.set_index('ds', inplace=True)
    return df



def full_model_pipeline_with_forecasts_for_single(ticker, df):
    all_comparisons = []
    all_forecasts = []

    print(f"\n📈 Running models for: {ticker}")

    # Split data
    test_size = 30
    train_data = df.iloc[:-test_size].copy()
    test_data = df.iloc[-test_size:].copy()

    # Run all models and get comparison
    comparison_df, model_results, train_data, test_data, order, seasonal = run_all_models_with_auto_params_new(
        ticker_name=ticker,
        df=df,
        train_data=train_data,
        test_data=test_data
    )

    if comparison_df.empty:
        print(f"⚠️ No valid models for {ticker}")
        return None, None

    all_comparisons.append(comparison_df)

    # --- Choose Best Model by MAPE ---
    best_model_row = comparison_df.loc[comparison_df['MAPE'].idxmin()]
    best_model_name = best_model_row['Model']

    # --- Get the corresponding fitted model ---
    best_model_entry = next((m for m in model_results if m['Model'] == best_model_name), None)
    fitted_model = best_model_entry['fitted_model'] if best_model_entry else None
    scaler = best_model_entry.get('scaler', None)

    full_df = pd.concat([train_data, test_data])
    full_df.columns = ['Close', 'Volume', 'LogVolume', 'Volume_MA5', 'RollingVolatility', 'RSI',  'MA5', 'MA20', 'Daily_Return']
    print(full_df.columns)
    print(full_df.head())

    exog_full = ['LogVolume', 'Volume_MA5', 'RollingVolatility', 
                 'RSI', 'MA5', 'MA20', 'Daily_Return']

    # --- Forecast using best model ---
    if best_model_name == "ARIMA":
        retrained_model = ARIMA(full_df['Close'], order=order).fit()
        forecast = forecast_arima(retrained_model, df, steps=[1, 7, 30])
    
    elif best_model_name == "SARIMA":
        retrained_model = SARIMAX(full_df['Close'], order=order, seasonal_order=seasonal).fit()
        forecast = forecast_sarima(retrained_model, df, steps=[1, 7, 30])
    
    elif best_model_name == "SARIMAX + exog":
        retrained_model = SARIMAX(full_df['Close'], exog=full_df[exog_full],
                                  order=order, seasonal_order=seasonal).fit()
        forecast = forecast_sarimax(retrained_model, df, steps=[1, 7, 30], scaler=scaler)
    
    elif best_model_name == "SARIMAX + exog + rolling":
        retrained_model = SARIMAX(full_df['Close'], exog=full_df[exog_full],
                                  order=order, seasonal_order=seasonal).fit()
        forecast = forecast_sarimax(retrained_model, df, steps=[1, 7, 30], scaler=scaler)
    
    else:
        print(f"❓ Forecast not implemented for {best_model_name}")
        return None, None

    forecast['Ticker'] = ticker
    forecast['Model'] = best_model_name
    all_forecasts.append(forecast)

    # Combine all
    full_comparison_df = pd.concat(all_comparisons, ignore_index=True)
    full_forecast_df = pd.concat(all_forecasts, ignore_index=True)


    return full_comparison_df, full_forecast_df, train_data, test_data


def plot_forecast_with_confidence(forecast_df, ticker):
    # Filter forecast data
    forecast_sub = forecast_df.copy()

    # Sanity check
    if forecast_sub.empty:
        print(f"No forecast data found for {ticker} with model.")
        return

    # Ensure numeric types
    forecast_sub['Days'] = pd.to_numeric(forecast_sub['Days'], errors='coerce')
    forecast_sub['Forecast'] = pd.to_numeric(forecast_sub['Forecast'], errors='coerce')
    forecast_sub['Lower_CI'] = pd.to_numeric(forecast_sub['Lower_CI'], errors='coerce')
    forecast_sub['Upper_CI'] = pd.to_numeric(forecast_sub['Upper_CI'], errors='coerce')
    forecast_sub.dropna(subset=['Days', 'Forecast', 'Lower_CI', 'Upper_CI'], inplace=True)

    # Plot setup
    plt.figure(figsize=(12, 6))

    # Forecast line
    sns.lineplot(data=forecast_sub, x='Days', y='Forecast', label='Forecast', color='blue')

    # Confidence interval
    plt.fill_between(
        forecast_sub['Days'],
        forecast_sub['Lower_CI'],
        forecast_sub['Upper_CI'],
        color='blue',
        alpha=0.2,
        label='Confidence Interval'
    )

    # Styling
    plt.title(f"Forecast for {ticker}")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#----PROPHET---------#

def load_data(ticker, start_date='2022-01-01'):
    df = yf.download(ticker, start=start_date, auto_adjust=True)
    df.ffill(inplace=True)
    df = df.reset_index()[['Date', 'Close', 'Volume']]
    df.columns = ['ds', 'y', 'volume']
    return df


def add_features(df):
    df = df.copy()
    df['ma7'] = df['y'].rolling(window=7, min_periods=1).mean()
    df['volatility'] = df['y'].rolling(window=7, min_periods=1).std().fillna(0)

    delta = df['y'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'].fillna(method='bfill', inplace=True)
    df['rsi'].fillna(method='ffill', inplace=True)
    return df


def train_full_model(df):
    years = list(range(df['ds'].dt.year.min(), df['ds'].dt.year.max() + 1))
    holidays = make_holidays_df(year_list=years, country='US')
    m = Prophet(
        daily_seasonality=True,
        changepoint_prior_scale=0.1,
        holidays=holidays
    )
    for reg in ['volume', 'ma7', 'volatility', 'rsi']:
        m.add_regressor(reg)
    m.fit(df)
    return m


def prepare_future(df, model, days):
    future = model.make_future_dataframe(periods=days, freq='B')
    features = df.set_index('ds')[['volume', 'ma7', 'volatility', 'rsi']]
    features = features.reindex(future['ds']).ffill().reset_index()
    features.columns = ['ds', 'volume', 'ma7', 'volatility', 'rsi']
    return pd.merge(future, features, on='ds', how='left')


def forecast_horizon(model, df, horizon):
    future_df = prepare_future(df, model, horizon)
    forecast = model.predict(future_df)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
    return result


def evaluate_pointwise(df, forecast_df, horizon):
    merged = pd.merge(df[['ds', 'y']], forecast_df, on='ds', how='inner')
    if merged.empty or len(merged) < horizon:
        return None

    y_true = merged['y'].values[-horizon:]
    y_pred = merged['yhat'].values[-horizon:]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'Horizon': horizon,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE': round(mape, 2)
    }

def plot_prophet_forecast(forecast_df, h):
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='blue')
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], 
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.title(f'Prophet Forecast for {h} Day(s)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)


def run_forecast_for_horizons(ticker='AAPL', start_date='2022-01-01', horizons=[1, 7, 30]):
    print(f"\n🚀 Running full forecast pipeline for {ticker}...\n")
    df = load_data(ticker, start_date)
    df = add_features(df)
    model = train_full_model(df)

    all_metrics = []
    for h in horizons:
        forecast_df = forecast_horizon(model, df, h)
        print(f"\n📅 Forecast for next {h} business day(s):")
        print(forecast_df[['ds', 'yhat']])


# === Title ===
st.title("📈 Stock Forecasting Dashboard")

# === Sidebar Inputs ===
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ("ARIMA", "SARIMA", "SARIMAX", "Prophet")
)

ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").strip().upper()
run_button = st.sidebar.button("Run Forecast")

# === Main Logic ===
if run_button and ticker:
    st.subheader(f"📊 Forecast Results for **{ticker}** using **{model_choice}**")

    if model_choice in ("ARIMA", "SARIMA", "SARIMAX"):
        with st.spinner("Running SARIMA-based model..."):
            df = download_and_prepare_ticker_data(ticker)
            result_df, forecast_df, train_data, test_data = full_model_pipeline_with_forecasts_for_single(ticker, df)

            st.write("📈 Forecast Table:")
            st.dataframe(result_df.head())
            st.dataframe(forecast_df)
            

            st.write("📉 Forecast Plot:")
            plot_forecast_with_confidence(forecast_df, ticker)
            st.pyplot()

    elif model_choice == "Prophet":
        with st.spinner("Running Prophet model..."):
            df = load_data(ticker, start_date="2022-01-01")
            df = add_features(df)
            model = train_full_model(df)

            horizons = [30]
            for h in horizons:
                forecast_df = forecast_horizon(model, df, h)
                st.write(f"📅 Forecast for next {h} business day(s):")
                st.dataframe(forecast_df[['ds', 'yhat']])

                st.write(f"📉 Forecast Plot ({h} day horizon):")
                plot_prophet_forecast(forecast_df, h)
                st.pyplot(plt.gcf())

# Footer
st.markdown("---")
st.caption("🔮 Built with Streamlit | Forecasts using ARIMA/SARIMA/SARIMAX & Prophet")
