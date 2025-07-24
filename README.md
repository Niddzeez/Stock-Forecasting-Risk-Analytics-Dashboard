# 📊 Stock-Forecasting-Risk-Analytics-Dashboard

This project builds a comprehensive pipeline for **stock price forecasting** and **risk analytics** using time series models such as **ARIMA**, **SARIMAX**, and **Prophet**, enhanced with **exogenous technical indicators** and robust evaluation metrics.

---

## 🚀 Key Features

- 📥 **Real-time Data Ingestion** using `yfinance`
- 📈 **ARIMA & SARIMAX Grid Search** with AIC optimization and MAPE-based validation filtering
- ⛅ **Prophet Forecasting** with holiday effects and exogenous variables
- ⚙️ **Exogenous Features**:
  - Volume
  - Moving Averages (MA7, MA30)
  - Rolling Volatility
  - RSI (Relative Strength Index)
- 🧪 **Evaluation & Backtesting**:
  - RMSE
  - MAPE
- 🔮 **Forecast Horizons**:
  - 1-day
  - 7-day
  - 30-day ahead
- 📊 **Risk Layer**:
  - Rolling Volatility
  - Value at Risk (VaR)
- 📂 **Modular, clean, and extensible codebase**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/your-username/Stock-Forecasting-Risk-Analytics-Dashboard.git
cd Stock-Forecasting-Risk-Analytics-Dashboard
```

### 2. Create a Virtual Environment

```
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependecies
```
pip install -r requirements.txt
```

### 4. Run the Forecast Pipeline
```
streamlit run forecasting_pipeline.py
```

---

## Models Used

### 🧠 ARIMA & SARIMAX
- Full grid search over (p,d,q)x(P,D,Q,s) space
- AIC-based shortlisting and validation using MAPE
- Exogenous technical indicators included

### ⛅ Prophet
- U.S. holidays + custom regressors (volume, RSI, volatility)
- Clean, interpretable forecasts
- Rolling cross-validation

---

### 📊 Evaluation Metrics
- AIC: For selecting best ARIMA/SARIMAX structure
- MAPE: Mean Absolute Percentage Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error

---

### 🛡️ Risk Analytics Layer
- Rolling Volatility using standard deviation of log returns
- Value at Risk (VaR): Historical approach for 95% confidence
---

### 🧰 Tech Stack
- Python 3.10+
- yfinance, pandas, numpy
- statsmodels, prophet
- matplotlib, seaborn, scikit-learn
- tqdm, scipy

### 📝 License
This project is licensed under the MIT License.
Let me know if you'd like me to:
- Create `.gitignore`
- Generate `requirements.txt` from your current environment
- Help set up GitHub repo with proper structure


Happy shipping 🚀




