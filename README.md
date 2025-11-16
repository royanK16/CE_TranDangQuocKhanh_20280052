# US Stock Price Forecasting using Prophet and ARIMA

This project applies Machine Learning and Statistical methods to forecast the prices of major US stocks. It utilizes a hybrid approach combining Facebook's Prophet model with ARIMA to capture complex time-series patterns and provide smoothed, long-term forecasts.

## 🚀 Features

- **Hybrid Forecasting Model**: Combines the strengths of Prophet (for trends, seasonality, and holidays) and ARIMA (for autoregressive patterns in residuals).
- **Technical Feature Integration**: Enriches the model with technical indicators like Exponential Moving Averages (EMAs), Volatility, Momentum, and Volume Ratios.
- **Advanced Cross-Validation**: Uses Prophet's built-in cross-validation to robustly evaluate model performance over multiple time windows.
- **Performance Metrics**: Calculates key metrics including Accuracy, Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- **Interactive Visualizations**: Generates detailed, interactive charts for each stock using Plotly, showing historical data, the forecast, confidence intervals, and daily price changes.
- **Batch Processing**: Capable of running forecasts for a list of multiple stock data files in a single execution.
- **Configurable Parameters**: Easily adjust training data duration, forecast horizon, and model settings in the notebook's configuration section.

## 📂 Folder Structure

The project is structured as follows:

```
CE_TranDangQuocKhanh_20280052/
|--dataset/
├── Forecasting.ipynb         # The main Jupyter Notebook for running the forecasts.
└── README.md                 # This instruction file.
```

## ⚙️ Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Ensure you have Python 3.x installed. This project relies on several Python libraries.

### 2. Installation

1.  **Clone the repository or download the project files.**

2.  **Install the required libraries.** You can install them using pip. It is recommended to use a virtual environment.

    ```bash
    pip install pandas numpy matplotlib prophet scikit-learn statsmodels plotly nbformat ipywidgets
    ```

    *Note: `cmdstanpy` is a dependency of Prophet and should be installed automatically. If you encounter issues, refer to the official Prophet installation guide.*

### 3. Running the Forecast

1.  **Place Your Data**: Ensure your stock data CSV files are in the same directory as `Forecasting.ipynb`. The files should be pipe-separated (`|`) with `Date` and `Close` columns.

2.  **Open and Run the Notebook**:
    -   Launch Jupyter Notebook or JupyterLab.
    -   Open the `Forecasting.ipynb` file.
    -   You can run all cells by selecting `Cell` -> `Run All` from the menu.

3.  **View Results**: The notebook will output the training progress, cross-validation metrics, and an interactive forecast chart for each stock listed in the `df_files` array. A final summary table comparing the performance across all stocks will be printed at the end.

## 🔧 Configuration

You can customize the forecasting process by modifying the parameters in the **CONFIGURATION** cell of the `Forecasting.ipynb` notebook:

- `TRAIN_YEARS`: The number of years of historical data to use for training the model.
- `FORECAST_DAYS`: The number of days into the future to forecast.
- `CV_HORIZON`, `CV_PERIOD`, `CV_INITIAL`: Parameters for the cross-validation process.
- `SMOOTHING_STRENGTH`: Controls the degree of smoothing applied to the final forecast line.
- `USE_HYBRID`: Set to `True` to use the Prophet + ARIMA model, or `False` to use only Prophet.
- `USE_INTERACTIVE_CHARTS`: Set to `True` for Plotly charts or `False` for static Matplotlib charts.
- `df_files`: A Python list of the stock data CSV filenames to process.

## 🤖 Methodology

The forecasting process follows these key steps:

1.  **Data Loading & Preparation**: Stock data is loaded from a CSV file, and the date column is converted to the correct format. The dataset is trimmed to include only the number of years specified in `TRAIN_YEARS`.

2.  **Feature Engineering**: Several technical analysis indicators are calculated and added as regressors to the model to provide more context than price alone:
    -   Exponential Moving Averages (EMA_7, EMA_21, EMA_50)
    -   Smoothed Volatility
    -   Smoothed Momentum
    -   Volume Ratio (relative to its moving average)

3.  **Model Training**:
    -   An optimized **Prophet** model is initialized. It automatically detects yearly seasonality and US holidays. Custom monthly and quarterly seasonalities are also added.
    -   The model is trained on the historical data and the engineered features.
    -   If `USE_HYBRID` is enabled, an **ARIMA** model is trained on the residuals of the Prophet forecast to capture any remaining autocorrelation. The final trend is a blend of both models' predictions.

4.  **Forecasting & Smoothing**:
    -   The model predicts future values for the specified `FORECAST_DAYS`.
    -   A **Gaussian filter** is applied to the forecast (`yhat`) and its confidence intervals (`yhat_lower`, `yhat_upper`) to create a smoother, more realistic trend line, reducing daily noise.

5.  **Evaluation**:
    -   The model's performance is evaluated using **cross-validation**. The data is split into multiple training/testing sets to simulate forecasting over different historical periods.
    -   Accuracy is calculated as `100% - MAPE`, providing an intuitive measure of the model's correctness.

6.  **Visualization**:
    -   An interactive **Plotly** chart is generated for each stock, displaying the historical price, the smoothed forecast, the 95% confidence interval, and a bar chart of daily percentage changes.

---
