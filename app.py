import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import gaussian_filter1d
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Prophet backend for Streamlit Cloud
os.environ['PROPHET_STAN_BACKEND'] = 'CMDSTANPY'

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📈 Stock Price Forecasting with Prophet")
st.markdown("---")

# Available stocks
AVAILABLE_STOCKS = [
    'AAPL', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 
    'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MMM', 'MSFT', 
    'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V'
]

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

selected_stock = st.sidebar.selectbox(
    "Select Stock:",
    AVAILABLE_STOCKS,
    index=AVAILABLE_STOCKS.index('AAPL')
)

forecast_days = st.sidebar.slider(
    "Forecast Days:",
    min_value=30,
    max_value=365,
    value=91,
    step=1
)

smoothing_strength = st.sidebar.slider(
    "Smoothing Strength:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

use_hybrid = st.sidebar.checkbox("Use Hybrid Model (Prophet + ARIMA)", value=True)

# Folder paths
MODEL_FOLDER = 'SaveModels'
DATA_FOLDER = 'dataset'

# Helper functions
def add_technical_features(df):
    """Add technical indicators"""
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility_21'] = df['Close'].rolling(window=21, min_periods=1).std()
    df['Volatility_21'] = df['Volatility_21'].ewm(span=5, adjust=False).mean()
    df['Momentum_14'] = df['Close'].pct_change(periods=14).fillna(0)
    df['Momentum_14'] = df['Momentum_14'].ewm(span=5, adjust=False).mean()
    df['Volume_EMA_21'] = df['Volume'].ewm(span=21, adjust=False).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_EMA_21']
    df['Volume_Ratio'] = df['Volume_Ratio'].ewm(span=5, adjust=False).mean()
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

@st.cache_data
def load_stock_data(stock_name):
    """Load stock data from dataset folder"""
    try:
        filepath = os.path.join(DATA_FOLDER, f"{stock_name}_stock_data.csv")
        
        if not os.path.exists(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, DATA_FOLDER, f"{stock_name}_stock_data.csv")
        
        if not os.path.exists(filepath):
            return None
        
        df = pd.read_csv(filepath, sep='|')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_or_train_model(stock_name, train_data):
    """Load saved forecast or train new model"""
    
    # Try to load saved forecast first
    forecast_path = os.path.join(MODEL_FOLDER, f"{stock_name}_forecast.pkl")
    params_path = os.path.join(MODEL_FOLDER, f"{stock_name}_best_params.pkl")
    
    # Check if files exist
    if os.path.exists(forecast_path):
        try:
            with open(forecast_path, 'rb') as f:
                saved_forecast = pickle.load(f)
            
            # Convert string dates back to datetime if needed
            if isinstance(saved_forecast['ds'].iloc[0], str):
                saved_forecast['ds'] = pd.to_datetime(saved_forecast['ds'])
            
            st.success(f"✅ Loaded pre-trained forecast for {stock_name}")
            return saved_forecast, "loaded"
            
        except Exception as e:
            st.warning(f"Could not load saved forecast: {e}")
    
    # If no saved forecast, train the model
    st.info(f"⚙️ Training model for {stock_name}... This may take a few minutes.")
    
    try:
        # Load best parameters if available
        best_params = None
        if os.path.exists(params_path):
            try:
                with open(params_path, 'rb') as f:
                    best_params = pickle.load(f)
            except:
                pass
        
        # Create Prophet model
        if best_params:
            model = Prophet(
                changepoint_prior_scale=float(best_params['changepoint_prior_scale']),
                seasonality_prior_scale=float(best_params['seasonality_prior_scale']),
                seasonality_mode=str(best_params['seasonality_mode']),
                changepoint_range=float(best_params.get('changepoint_range', 0.8)),
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                interval_width=0.95
            )
        else:
            # Use default parameters
            volatility = train_data['y'].pct_change().std()
            
            if volatility > 0.03:
                model = Prophet(
                    changepoint_prior_scale=0.03,
                    seasonality_prior_scale=3,
                    seasonality_mode='multiplicative',
                    changepoint_range=0.85,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
                    interval_width=0.95
                )
            else:
                model = Prophet(
                    changepoint_prior_scale=0.02,
                    seasonality_prior_scale=5,
                    seasonality_mode='additive',
                    changepoint_range=0.82,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
                    interval_width=0.95
                )
        
        # Add seasonality and regressors
        model.add_country_holidays(country_name='US')
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=4)
        
        regressor_cols = ['EMA_7', 'EMA_21', 'EMA_50', 'Volatility_21', 'Momentum_14', 'Volume_Ratio']
        for col in regressor_cols:
            if col in train_data.columns:
                model.add_regressor(col, standardize='auto')
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(train_data)
        
        # Return the trained model and status
        return model, "trained"
        
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise

def smooth_forecast(forecast_series, strength=0.7):
    """Apply Gaussian smoothing"""
    sigma = strength * 5
    smoothed = gaussian_filter1d(forecast_series, sigma=sigma)
    return smoothed

def fit_arima_trend(data, order=(2, 1, 2)):
    """Fit ARIMA model"""
    try:
        model = ARIMA(data, order=order, enforce_stationarity=False, enforce_invertibility=False)
        return model.fit()
    except:
        model = ARIMA(data, order=(1, 1, 1), enforce_stationarity=False)
        return model.fit()

def create_forecast_chart(stock_name, last_90_days, future_forecast, forecast_days):
    """Create interactive Plotly chart"""
    hist_dates = last_90_days['ds']
    hist_prices = last_90_days['y']
    hist_changes = hist_prices - hist_prices.iloc[0]
    hist_pct_changes = (hist_changes / hist_prices.iloc[0]) * 100
    
    forecast_dates = future_forecast['ds']
    forecast_prices = future_forecast['yhat']
    forecast_lower = future_forecast['yhat_lower']
    forecast_upper = future_forecast['yhat_upper']
    
    last_hist_price = hist_prices.iloc[-1]
    forecast_changes = forecast_prices - last_hist_price
    forecast_pct_changes = (forecast_changes / last_hist_price) * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=(
            f'{stock_name} - {forecast_days}-Day Forecast',
            'Daily Price Change (%)'
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=hist_dates, y=hist_prices,
            name='Historical Price',
            mode='lines',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='<b>Historical</b><br>Date: %{x|%d-%m-%Y}<br>Price: $%{y:.2f}<br>Change: $%{customdata[0]:.2f}<br>Change %: %{customdata[1]:.2f}%<br><extra></extra>',
            customdata=np.column_stack((hist_changes, hist_pct_changes))
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_dates, y=forecast_upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates, y=forecast_lower,
            mode='lines', line=dict(width=0),
            fillcolor='rgba(162, 59, 114, 0.2)',
            fill='tonexty',
            name='95% Confidence',
            showlegend=True, hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_dates, y=forecast_prices,
            name='Forecast',
            mode='lines',
            line=dict(color='#A23B72', width=4),
            hovertemplate='<b>Forecast</b><br>Date: %{x|%d-%m-%Y}<br>Price: $%{y:.2f}<br>Change: $%{customdata[0]:.2f}<br>Change %: %{customdata[1]:+.2f}%<br><extra></extra>',
            customdata=np.column_stack((forecast_changes, forecast_pct_changes))
        ),
        row=1, col=1
    )
    
    # Markers
    fig.add_trace(
        go.Scatter(
            x=[forecast_dates.iloc[0]], y=[forecast_prices.iloc[0]],
            mode='markers',
            marker=dict(size=12, color='#06D6A0', symbol='circle', line=dict(color='white', width=2)),
            name='Forecast Start',
            hovertemplate='<b>Forecast Start</b><br>Date: %{x|%d-%m-%Y}<br>Price: $%{y:.2f}<br><extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[forecast_dates.iloc[-1]], y=[forecast_prices.iloc[-1]],
            mode='markers',
            marker=dict(size=12, color='#EF476F', symbol='circle', line=dict(color='white', width=2)),
            name='Forecast End',
            hovertemplate='<b>Forecast End</b><br>Date: %{x|%d-%m-%Y}<br>Price: $%{y:.2f}<br>Total Change: %{customdata[0]:+.2f}%<br><extra></extra>',
            customdata=[[forecast_pct_changes.iloc[-1]]]
        ),
        row=1, col=1
    )
    
    # Daily changes
    hist_daily_changes = last_90_days['y'].pct_change() * 100
    colors_hist = ['#06D6A0' if x >= 0 else '#EF476F' for x in hist_daily_changes]
    
    fig.add_trace(
        go.Bar(
            x=hist_dates, y=hist_daily_changes,
            name='Historical Daily %',
            marker_color=colors_hist,
            hovertemplate='Date: %{x|%d-%m-%Y}<br>Daily Change: %{y:.2f}%<br><extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    forecast_daily_changes = forecast_prices.pct_change() * 100
    colors_forecast = ['#06D6A0' if x >= 0 else '#EF476F' for x in forecast_daily_changes]
    
    fig.add_trace(
        go.Bar(
            x=forecast_dates, y=forecast_daily_changes,
            name='Forecast Daily %',
            marker_color=colors_forecast,
            hovertemplate='Date: %{x|%d-%m-%Y}<br>Daily Change: %{y:.2f}%<br><extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Daily Change (%)", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

# Main app logic
def main():
    # Load data
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = load_stock_data(selected_stock)
    
    if df is None:
        st.error(f"❌ Data not found for {selected_stock}.")
        st.info("Make sure the dataset folder contains the stock data CSV files.")
        st.stop()
    
    # Prepare data (last 10 years)
    last_date = df['Date'].max()
    cutoff_date = last_date - pd.DateOffset(years=10)
    df = df[df['Date'] >= cutoff_date].reset_index(drop=True)
    
    # Add features
    df = add_technical_features(df)
    
    # Prepare training data
    train = pd.DataFrame({'ds': df['Date'], 'y': df['Close']})
    regressor_cols = ['EMA_7', 'EMA_21', 'EMA_50', 'Volatility_21', 'Momentum_14', 'Volume_Ratio']
    for col in regressor_cols:
        if col in df.columns:
            train[col] = df[col]
    
    # Load or train model
    with st.spinner(f"Loading/Training model for {selected_stock}..."):
        model_or_forecast, status = load_or_train_model(selected_stock, train)
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        if status == "loaded":
            # Use loaded forecast
            forecast = model_or_forecast
        else:
            # Generate new forecast from trained model
            model = model_or_forecast
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Extrapolate regressors
            for col in regressor_cols:
                if col in train.columns:
                    last_values = train[col].tail(30).values
                    trend = np.mean(np.diff(last_values))
                    last_value = train[col].iloc[-1]
                    future_values = [last_value + trend * i * 0.5 for i in range(1, forecast_days + 1)]
                    future.loc[future['ds'] > train['ds'].max(), col] = future_values
                    future.loc[future['ds'] <= train['ds'].max(), col] = train[col].values
            
            forecast = model.predict(future)
            
            # Apply hybrid ARIMA if selected
            if use_hybrid:
                try:
                    recent_data = train['y'].tail(365).values
                    arima_model = fit_arima_trend(recent_data, order=(1, 1, 1))
                    arima_forecast = arima_model.forecast(steps=forecast_days)
                    future_prophet = forecast[forecast['ds'] > train['ds'].max()]['yhat'].values
                    blended_forecast = 0.7 * future_prophet + 0.3 * arima_forecast
                    forecast.loc[forecast['ds'] > train['ds'].max(), 'yhat'] = blended_forecast
                except:
                    pass
            
            # Apply smoothing
            future_mask = forecast['ds'] > train['ds'].max()
            forecast.loc[future_mask, 'yhat'] = smooth_forecast(
                forecast.loc[future_mask, 'yhat'].values, 
                strength=smoothing_strength
            )
            forecast.loc[future_mask, 'yhat_lower'] = smooth_forecast(
                forecast.loc[future_mask, 'yhat_lower'].values, 
                strength=smoothing_strength * 0.8
            )
            forecast.loc[future_mask, 'yhat_upper'] = smooth_forecast(
                forecast.loc[future_mask, 'yhat_upper'].values, 
                strength=smoothing_strength * 0.8
            )
    
    # Display metrics
    current_price = df['Close'].iloc[-1]
    forecast_start = train['ds'].iloc[-1]
    future_forecast_only = forecast[forecast['ds'] > forecast_start].copy()
    
    if len(future_forecast_only) > 0:
        forecast_price = future_forecast_only['yhat'].iloc[min(forecast_days-1, len(future_forecast_only)-1)]
        price_change = forecast_price - current_price
        pct_change = (price_change / current_price) * 100
    else:
        forecast_price = current_price
        price_change = 0
        pct_change = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", help="Latest closing price")
    
    with col2:
        st.metric("Forecast Price", f"${forecast_price:.2f}", 
                 delta=f"{pct_change:+.2f}%",
                 help=f"Predicted price after {forecast_days} days")
    
    with col3:
        st.metric("Price Change", f"${price_change:+.2f}", help="Absolute price change")
    
    with col4:
        st.metric("Last Updated", last_date.strftime("%Y-%m-%d"), help="Latest data date")
    
    st.markdown("---")
    
    # Plot
    last_90_days = train.tail(90)
    
    fig = create_forecast_chart(selected_stock, last_90_days, future_forecast_only, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("### 📋 Forecast Details")
    forecast_table = future_forecast_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_table.columns = ['Date', 'Forecast Price', 'Lower Bound', 'Upper Bound']
    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
    forecast_table['Forecast Price'] = forecast_table['Forecast Price'].apply(lambda x: f"${x:.2f}")
    forecast_table['Lower Bound'] = forecast_table['Lower Bound'].apply(lambda x: f"${x:.2f}")
    forecast_table['Upper Bound'] = forecast_table['Upper Bound'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(forecast_table, use_container_width=True, height=400)
    
    # Download button
    csv = forecast_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"{selected_stock}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()