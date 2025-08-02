import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. Incarcare si curatare date

df = pd.read_csv('yahoo_data.csv')
print(df.head())

# Info general
print(df.info())

# Verificam valori lipsa
print("\nNumar de valori lipsa pe coloana:")
print(df.isnull().sum())

# Eliminam randurile cu valori lipsa
df = df.dropna()

# Verificam duplicate
duplicates = df.duplicated().sum()
print(f"\nNumar de duplicate: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()

print(f"\nDimensiune finala dataset: {df.shape}")


# 2. Conversie coloane la tipuri numerice/corecte


cols_to_numeric = ['Open', 'High', 'Low', 'Close*', 'Adj Close**']
for col in cols_to_numeric:
    df[col] = df[col].str.replace(',', '', regex=False).astype(float)

df['Volume'] = df['Volume'].str.replace(',', '', regex=False).astype(int)
df['Date'] = pd.to_datetime(df['Date'])

print("\nTipuri de date dupa conversie:")
print(df.dtypes)


# 3. Statistici descriptive si grafice de baza


print("\nStatistici descriptive:")
print(df.describe())

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close*'], label='Close Price', color='blue')
plt.title('Evolutia pretului de inchidere in timp')
plt.xlabel('Data')
plt.ylabel('Pret inchidere')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('evolutie_pret_inchidere.png')

# 4. Corelatii intre variabile


plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close*', 'Adj Close**', 'Volume']].corr(),
            annot=True, cmap='coolwarm')
plt.title('Matricea de corelatii')
plt.tight_layout()
plt.show()
plt.savefig('matrice_corelatii.png')


# 5. Distributia volumului


plt.figure(figsize=(10, 6))
sns.histplot(df['Volume'], bins=50, kde=True)
plt.title('Distributia volumului tranzactionat')
plt.xlabel('Volum')
plt.ylabel('Frecventa')
plt.tight_layout()
plt.show()
plt.savefig('distributie_volum.png')


# 6. Procesare date pentru analiza seriilor temporale


df = df.sort_values('Date')
df = df.set_index('Date')
df = df.asfreq('B')  # frecventa: business day

# Completam eventualele date lipsa cu forward fill
df['Close*'] = df['Close*'].ffill()


# 7. Indicatori suplimentari

df['Daily_Return'] = df['Close*'].pct_change()
df['Rolling_Mean_20'] = df['Close*'].rolling(window=20).mean()
df['Rolling_Std_20'] = df['Close*'].rolling(window=20).std()

print(df[['Close*', 'Daily_Return', 'Rolling_Mean_20', 'Rolling_Std_20']].head(25))


# 8. Plot pret inchidere si media mobila

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close*'], label='Pret inchidere', color='blue')
plt.plot(df.index, df['Rolling_Mean_20'], label='Media mobila 20 zile', color='orange')
plt.title('Pret inchidere si media mobila')
plt.xlabel('Data')
plt.ylabel('Pret')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('pret_inchidere_media_mobila.png')


# 9. Modelare ARIMA si forecast


df_model = df.dropna(subset=['Close*'])
print("\nTip index df_model:", type(df_model.index))

model = ARIMA(df_model['Close*'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=30)
print("\nForecast pe urmatoarele 30 de zile:")
print(forecast)


# 10. Plot istoric + forecast


plt.figure(figsize=(12, 6))
plt.plot(df_model.index, df_model['Close*'], label='Istoric', color='blue')
future_dates = pd.date_range(df_model.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
plt.plot(future_dates, forecast, label='Forecast', color='red')
plt.title('Forecast ARIMA pentru urmatoarele 30 de zile')
plt.xlabel('Data')
plt.ylabel('Close*')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('forecast_arima.png')


# 11. RSI, EMA, SMA


df['EMA12'] = df['Close*'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close*'].ewm(span=26, adjust=False).mean()
df['SMA20'] = df['Close*'].rolling(window=20).mean()

delta = df['Close*'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))


# 12. Linear Regression


df['Lag1'] = df['Close*'].shift(1)
df = df.dropna(subset=['Lag1', 'Close*'])
X = df[['Lag1']]
y = df['Close*']
model_lr = LinearRegression().fit(X, y)
df['Pred_LR'] = model_lr.predict(X)
df = df.dropna()



# 13. KMeans clustering


scaler = StandardScaler()
features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close*', 'Volume']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features)


# 14. Backtest strategie EMA cross


df['Signal'] = np.where(df['EMA12'] > df['EMA26'], 1, -1)
df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)


# 15. Risk metrics


var_95 = np.percentile(df['Daily_Return'].dropna(), 5)
cum_returns = (1 + df['Strategy_Return'].dropna()).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()
sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)

print(f"\nValue at Risk (95%): {var_95:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


# 16. Ploturi finale


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close*'], label='Close*', color='blue')
plt.plot(df.index, df['EMA12'], label='EMA12', color='green', linestyle='--')
plt.plot(df.index, df['EMA26'], label='EMA26', color='red', linestyle='--')
plt.title('Pret inchidere si EMA12/EMA26')
plt.xlabel('Data')
plt.ylabel('Pret')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('ema_cross_strategy.png')

plt.figure(figsize=(12, 6))
plt.plot(cum_returns, label='Cumulative Strategy Return', color='purple')
plt.title('Evolutie strategie EMA cross')
plt.xlabel('Data')
plt.ylabel('Randament cumulat')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('cumulative_strategy_return.png')

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Close*'], y=df['Pred_LR'], hue=df['Cluster'], palette='Set2')
plt.title('Predictie Linear Regression vs Clustere')
plt.xlabel('Close* real')
plt.ylabel('Close* prezis')
plt.tight_layout()
plt.show()
plt.savefig('linear_regression_clusters.png')
