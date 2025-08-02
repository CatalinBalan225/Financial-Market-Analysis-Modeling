#  Financial Time Series Analysis with Python

This project demonstrates a complete time series analysis workflow using Python and historical financial data retrieved from Yahoo Finance. It includes data cleaning, exploratory analysis, time series modeling (ARIMA), machine learning (Linear Regression & KMeans), technical indicators (EMA, RSI), strategy backtesting, and risk analysis.

---

## Data Description

The dataset includes daily historical stock market data from May 2018 to April 2023 (business day frequency), with the following columns:

- `Date`
- `Open`, `High`, `Low`, `Close*`, `Adj Close**`
- `Volume`

The data is cleaned and preprocessed:

```python
# Read and clean data
df = pd.read_csv('yahoo_data.csv')
df = df.dropna().drop_duplicates()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('B')  # Business day frequency
```

Missing values: 7 rows had missing entries and were dropped.
No duplicates found
Data types: Converted columns like Open, High, Close*, etc. from strings with commas to float.

---

##  Project Workflow

### 1. Descriptive Statistics & Visual Analysis

We explore price dynamics and trading volume, and examine the relationships between variables:

```python
df.describe()
plt.plot(df.index, df['Close*'])
```
The data shows steady growth from 2018 to 2023, however a significant post-COVID impact is visible between 2020 and 2021.
Volume distribution shows typical trading spikes around major events


### 2. Correlation Heatmap

```python
sns.heatmap(df[['Open', 'High', 'Low', 'Close*', 'Adj Close**', 'Volume']].corr(), annot=True)
```

Strong positive correlations between Open, High, Low, Close*, and Adj Close** - expected in OHLC data.

Weaker correlation between prices and Volume.

### 3. Time Series Forecasting with ARIMA

```python
model = ARIMA(df['Close*'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
```
The ARIMA(5,1,0) model captures recent price momentum.

The residuals pass the Ljung-Box test (p=0.91), suggesting white noise.

Forecast Result (next 30 business days):

The forecast stabilizes around 34,149, suggesting a mean-reverting behavior.

Limited volatility predicted.


### 4. Technical Indicators (EMA, RSI, SMA)

```python
df['EMA12'] = df['Close*'].ewm(span=12).mean()
df['EMA26'] = df['Close*'].ewm(span=26).mean()

# RSI Calculation
delta = df['Close*'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
```

These are used later in the strategy and risk analysis steps.

### 5. Linear Regression (Lagged Feature)

```python
df['Lag1'] = df['Close*'].shift(1)
X = df[['Lag1']].dropna()
y = df['Close*'].dropna()
model_lr = LinearRegression().fit(X, y)
df['Pred_LR'] = model_lr.predict(X)
```

**Interpretation:**\
The linear regression captures short-term price inertia. Useful for basic prediction and error analysis.

### 6. KMeans Clustering

```python
features = StandardScaler().fit_transform(df[['Open', 'High', 'Low', 'Close*', 'Volume']])
df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(features)
```

Clusters differentiate between low, medium, and high volatility regimes.

### 7. EMA Crossover Strategy Backtest

```python
df['Signal'] = np.where(df['EMA12'] > df['EMA26'], 1, -1)
df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
```

This strategy simulates a basic momentum trade:

- Buy when EMA12 > EMA26
- Sell when EMA12 < EMA26

---

##  Visualizations

Here are some of the key visual outputs included in the notebook:

- **Price & Moving Averages**\
  Shows price evolution and EMA/SMA overlays

- **ARIMA Forecast vs Historical**\
  Visual comparison of forecasted values and historical data

- **Volume Histogram**\
  Highlights periods of intense trading activity

- **Regression vs Cluster Scatterplot**\
  Shows predicted vs actual prices, colored by KMeans clusters

- **Cumulative Return of Strategy**

```python
cum_returns = (1 + df['Strategy_Return'].dropna()).cumprod()
plt.plot(cum_returns)
```

**Observation:**\
The strategy underperforms during sideways or choppy markets but shows promise during trending periods - e.g., in the COVID recovery zone.

---

##  Risk Analysis

Risk metrics used:

```python
# Sharpe Ratio
sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

# Max Drawdown
cum_returns = (1 + df['Strategy_Return'].dropna()).cumprod()
drawdown = cum_returns / cum_returns.cummax() - 1
max_drawdown = drawdown.min()

# Value at Risk (95%)
var_95 = np.percentile(df['Daily_Return'].dropna(), 5)
```

### Results:

- **Sharpe Ratio:** Poor risk-adjusted performance (-0.06)
- **Max Drawdown:** Largest portfolio decline (-34.07%)
- **VaR (95%):** Daily loss at 5% worst case (-0.0195)

---

## Conclusions

The ARIMA model provides stable forecasts but lacks event sensitivity.

EMA crossover is trend-following and works best in momentum markets.

RSI and EMA give valuable signals, especially when filtered via clustering.

Linear regression on lagged prices captures short-term momentum well.

Risk analysis highlights strategy limitations during volatile or flat periods.

## Visualizations included

Close Price vs Time - shows long-term price dynamics.

ARIMA Forecast Plot - compares historical vs future estimates.

Volume Distribution - identifies high-activity days.

EMA Crossover Plot - overlays signals on price.

Cumulative Strategy Return - evaluates backtest performance.

Regression vs Real (Clustered) - visualizes prediction clusters.

