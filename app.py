import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# --- Model Loading (Adjust if you have a saved model) ---
def load_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(4, 1))) # Input shape adjusted
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    # ... (Load weights if you have them) ... 
    return model

model = load_model() # Model ready to use

# --- App Structure ---
st.title("Stock Price Prediction Web App")

ticker = st.text_input("Enter Stock Ticker", "")  # Allow user to input any stock ticker
start_date = st.date_input("Start Date", datetime.date(2021, 1, 1)) 
end_date = st.date_input("End Date", datetime.date.today())

if st.button("Predict"):
    if ticker.strip() == "":
        st.warning("Please enter a valid stock ticker.")
    else:
        try:
            # --- Data Collection and Preparation ---
            today = date.today()
            d1 = today.strftime("%Y-%m-%d")
            d2 = date.today() - timedelta(days=50000)
            d2 = d2.strftime("%Y-%m-%d")

            data = yf.download(ticker, start=d2, end=d1, progress=False)
            if data.empty:
                st.error("No data found for the given ticker symbol. Please enter a valid ticker symbol.")
            else:
                data["Date"] = data.index
                data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                data.reset_index(drop=True, inplace=True)

                # --- Data Visualization ---
                figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                                    open=data["Open"],
                                                    high=data["High"],
                                                    low=data["Low"],
                                                    close=data["Close"])])
                figure.update_layout(title=f"{ticker} Stock Price Analysis", xaxis_rangeslider_visible=False)

                # --- Data Preprocessing ---
                x = data[["Open", "High", "Low", "Volume"]]
                y = data["Close"]
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
                xtrain = np.array(xtrain).reshape(xtrain.shape[0], xtrain.shape[1], 1)  # Adjusted reshaping
                xtest = np.array(xtest).reshape(xtest.shape[0], xtest.shape[1], 1)  # Adjusted reshaping
                ytrain = np.array(ytrain)
                ytest = np.array(ytest)

                # --- Prediction ---
                features = np.array([data.iloc[-1, 1:5]], dtype=np.float32)  # Adjusted feature selection and explicit casting
                features = features.reshape(1, features.shape[1], 1)  # Adjusted reshaping
                prediction = model.predict(features)

                # --- Display Results ---
                st.subheader(f"Predicted Price for {ticker}: {prediction[0][0]}")
                st.plotly_chart(figure)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
