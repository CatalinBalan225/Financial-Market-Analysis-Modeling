import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('yahoo_data.csv')
print(df.head())

#general info
print(df.info())

print("\n Numar de valori lipsa coloana")
print(df.isnull().sum())

df.dropna(inplace=True)

duplicates = df.duplicated().sum()
print(f"\nNumar de duplicate: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)

print(f"\n Final dimension of the dataset: {df.shape}")

#am eliminat un rand cu valori lipsa, ramanand cu toate coloanele de tip object. Acum trebuie sa transformam toate coloanele numerice din object in float / int

cols_to_numeric = ['Open', 'High', 'Low', 'Close*', 'Adj Close**']
for col in cols_to_numeric:
    df[col] = df[col].str.replace(',', '').astype(float)

df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
df['Date'] = pd.to_datetime(df['Date'])
print("\n Tipuri de date dupa conversie:")
print(df.dtypes)

# Statistici descriptive
print("\nStatistici descriptive:")
print(df.describe())

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close*'], label='Close Price', color='blue')
plt.title('Evolutia pretului de inchidere in timp')
plt.xlabel('Data')
plt.ylabel('Pret inchidere')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close*', 'Adj Close**', 'Volume']].corr(),
            annot=True, cmap='coolwarm')
plt.title('Matricea de corelatii')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Volume'], bins=50, kde=True)
plt.title('Distributia volumului tranzactionat')
plt.xlabel('Volum')
plt.ylabel('Frecventa')
plt.show()
