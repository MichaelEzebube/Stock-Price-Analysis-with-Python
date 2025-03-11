import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Step 1: Download Stock Data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data


# Step 2: Exploratory Data Analysis (EDA)
def plot_stock_price(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label=f'{ticker} Closing Price')
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.show()


# Step 3: Feature Engineering
def add_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].pct_change().rolling(window=10).std()
    data.dropna(inplace=True)
    return data


# Step 4: Build Predictive Model
def train_model(data):
    data['Tomorrow'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    X = data[['SMA_50', 'SMA_200', 'Volatility']]
    y = data['Tomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model Performance:")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")

    return model


# Run Analysis
ticker = "AAPL"  # Change to any stock symbol you want
start_date = "2020-01-01"
end_date = "2024-01-01"

data = get_stock_data(ticker, start_date, end_date)
plot_stock_price(data, ticker)
data = add_features(data)
model = train_model(data)

