import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def time_series_forecast(df, date_col, value_col, forecast_steps=12, model_type='arima', freq='ME'):
    """
    Fits a time series forecasting model and predicts future values.
    model_type: 'moving_average', 'arima', 'exponential_smoothing'
    freq: 'D' (daily), 'W' (weekly), 'ME' (monthly), etc.
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("Date or Value column not found in dataset.")
        
    # Prepare data: sort, set index, resample to standard frequency
    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    ts_df = ts_df.sort_values(by=date_col)
    
    # Group by date and sum/mean values
    ts_df = ts_df.groupby(date_col)[value_col].sum().reset_index()
    ts_df.set_index(date_col, inplace=True)
    
    # Resample to ensure fixed interval frequency
    ts_data = ts_df[value_col].resample(freq).sum().ffill()
    
    if len(ts_data) < 5:
        raise ValueError("Time series must have at least 5 observations after resampling for modeling.")
        
    # Split into train/validation for metric calculations
    train_size = int(len(ts_data) * 0.8)
    if train_size < 3:
        train_size = len(ts_data)
        
    train = ts_data.iloc[:train_size]
    test = ts_data.iloc[train_size:]
    
    fitted_model = None
    y_pred_val = []
    
    # Fit model on training set to compute metrics (if test exists)
    has_test = len(test) > 0
    
    # Fit model on entire series for final forecast
    if model_type == 'moving_average':
        # 3-period moving average fallback
        window = 3
        if has_test:
            # Predict test set using rolling average
            history = list(train.values)
            for i in range(len(test)):
                pred = np.mean(history[-window:])
                y_pred_val.append(pred)
                history.append(test.iloc[i]) # add true value for next forecast step
                
        # Final forecast
        history = list(ts_data.values)
        forecast_vals = []
        for _ in range(forecast_steps):
            pred = np.mean(history[-window:])
            forecast_vals.append(pred)
            history.append(pred)
            
    elif model_type == 'exponential_smoothing':
        try:
            if has_test:
                model_train = ExponentialSmoothing(train, seasonal_periods=min(12, len(train)//2), trend='add', seasonal='add').fit()
                y_pred_val = model_train.forecast(len(test))
            
            # Final model
            fitted_model = ExponentialSmoothing(ts_data, seasonal_periods=min(12, len(ts_data)//2), trend='add', seasonal='add').fit()
            forecast_vals = fitted_model.forecast(forecast_steps)
        except Exception:
            # Fallback to simple exponential smoothing
            if has_test:
                model_train = ExponentialSmoothing(train, initialization_method="estimated").fit()
                y_pred_val = model_train.forecast(len(test))
            fitted_model = ExponentialSmoothing(ts_data, initialization_method="estimated").fit()
            forecast_vals = fitted_model.forecast(forecast_steps)
            
    elif model_type == 'prophet':
        try:
            from prophet import Prophet
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            # Prepare training data
            train_df = train.reset_index()
            train_df.columns = ['ds', 'y']
            if train_df['ds'].dt.tz is not None:
                train_df['ds'] = train_df['ds'].dt.tz_localize(None)
                
            m = Prophet()
            m.fit(train_df)
            
            if has_test:
                future_test = pd.DataFrame({'ds': test.index})
                if future_test['ds'].dt.tz is not None:
                    future_test['ds'] = future_test['ds'].dt.tz_localize(None)
                forecast_test = m.predict(future_test)
                y_pred_val = forecast_test['yhat'].values
                
            # Fit final model
            full_df = ts_data.reset_index()
            full_df.columns = ['ds', 'y']
            if full_df['ds'].dt.tz is not None:
                full_df['ds'] = full_df['ds'].dt.tz_localize(None)
                
            m_full = Prophet()
            m_full.fit(full_df)
            
            # Generate future predictions
            future_dates = pd.DataFrame({'ds': pd.date_range(start=ts_data.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=forecast_steps, freq=freq)})
            if future_dates['ds'].dt.tz is not None:
                future_dates['ds'] = future_dates['ds'].dt.tz_localize(None)
            forecast_full = m_full.predict(future_dates)
            forecast_vals = forecast_full['yhat'].values
        except Exception as e:
            # Fallback to ARIMA if Prophet fails
            print(f"[Warning] Prophet model failed (possibly because package is not installed). Falling back to ARIMA: {str(e)}")
            model_type = 'arima'  # fall through to arima below
            
    if model_type == 'arima':
        try:
            # Fit simple ARIMA(1, 1, 1) or auto
            if has_test:
                model_train = ARIMA(train, order=(1, 1, 1)).fit()
                y_pred_val = model_train.forecast(len(test))
            
            fitted_model = ARIMA(ts_data, order=(1, 1, 1)).fit()
            forecast_vals = fitted_model.forecast(forecast_steps)
        except Exception as e:
            # Fallback to ARIMA(0, 1, 0)
            if has_test:
                model_train = ARIMA(train, order=(0, 1, 0)).fit()
                y_pred_val = model_train.forecast(len(test))
            fitted_model = ARIMA(ts_data, order=(0, 1, 0)).fit()
            forecast_vals = fitted_model.forecast(forecast_steps)
            
    # Calculate performance metrics
    metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    if has_test and len(y_pred_val) == len(test):
        metrics["RMSE"] = np.sqrt(mean_squared_error(test, y_pred_val))
        metrics["MAE"] = mean_absolute_error(test, y_pred_val)
        metrics["MAPE"] = calculate_mape(test, y_pred_val)
        
    # Generate future dates
    future_index = pd.date_range(start=ts_data.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=forecast_steps, freq=freq)
    forecast_series = pd.Series(forecast_vals, index=future_index)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=ts_data.index, y=ts_data.values,
        mode='lines+markers', name='Historical Data',
        line=dict(color='#1f77b4', width=2.5)
    ))
    
    # Forecasted data
    fig.add_trace(go.Scatter(
        x=forecast_series.index, y=forecast_series.values,
        mode='lines+markers', name='Forecasted Future',
        line=dict(color='#ff7f0e', dash='dash', width=2.5)
    ))
    
    fig.update_layout(
        title=f"Time Series Forecast ({model_type.upper()}) for {value_col}",
        xaxis_title="Date",
        yaxis_title=value_col,
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif"),
        hovermode="x unified"
    )
    
    forecast_df = pd.DataFrame({
        "Date": forecast_series.index,
        "Forecasted_Value": forecast_series.values
    }).reset_index(drop=True)
    
    return {
        "forecast_df": forecast_df,
        "metrics": metrics,
        "forecast_fig": fig
    }
