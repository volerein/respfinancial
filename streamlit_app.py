import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st
import altair as alt

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def build_lstm_model(train_data, time_steps, epochs, batch_size):
    X_train, y_train = prepare_data(train_data, time_steps)
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data is insufficient for the given time steps.")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model

def make_future_predictions(model, data, time_steps, periods):
    last_sequence = data[-time_steps:]
    future_predictions = []

    for _ in range(periods):
        prediction = model.predict(last_sequence.reshape(1, time_steps, 1))[0, 0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = prediction

    return future_predictions

def load_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end, interval='1d')
    if stock_data.empty:
        raise ValueError("No data available for the selected stock symbol. Please enter a valid symbol.")
    
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_closing_prices = scaler.fit_transform(closing_prices)
    return stock_data, closing_prices, scaled_closing_prices, scaler

def add_moving_averages(stock_data, window_sma, window_ema):
    stock_data['SMA'] = stock_data['Close'].rolling(window=window_sma).mean()
    stock_data['EMA'] = stock_data['Close'].ewm(span=window_ema, adjust=False).mean()
    return stock_data

st.sidebar.header('Stock Price Prediction Dashboard')

stock_symbol = st.sidebar.text_input('Enter stock symbol (e.g., GOOGL for Alphabet):', 'GOOGL')
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2021-01-01'))
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2022-01-01'))

if st.sidebar.button('Predict'):
    try:
        stock_data, closing_prices, scaled_closing_prices, scaler = load_data(stock_symbol, start=start_date, end=end_date)
        
        stock_data = add_moving_averages(stock_data, window_sma=20, window_ema=20)

        train_size = int(len(scaled_closing_prices) * 0.8)
        train_data = scaled_closing_prices[:train_size]
        test_data = scaled_closing_prices[train_size:]

        time_steps = 30
        if len(train_data) <= time_steps:
            raise ValueError("Training data is insufficient for the given time steps.")
        
        epochs = 50
        batch_size = 32
        model = build_lstm_model(train_data, time_steps, epochs, batch_size)

        periods = 30
        future_preds = make_future_predictions(model, scaled_closing_prices, time_steps, periods)
        future_dates = pd.date_range(start=end_date, periods=periods + 1)[1:]

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
        })

        historical_df = pd.DataFrame({
            'Date': stock_data.index,
            'Close': closing_prices.flatten()
        })

        historical_chart = alt.Chart(historical_df).mark_line().encode(
            x='Date:T',
            y='Close:Q',
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            title='Historical Data and Future Predictions'
        ).interactive()

        sma_chart = alt.Chart(stock_data.reset_index()).mark_line(color='red').encode(
            x='Date:T',
            y='SMA:Q'
        )

        ema_chart = alt.Chart(stock_data.reset_index()).mark_line(color='orange').encode(
            x='Date:T',
            y='EMA:Q'
        )

        combined_chart = historical_chart + sma_chart + ema_chart

        st.write('### Historical Data and Future Predictions')
        st.altair_chart(combined_chart, use_container_width=True)

        future_chart = alt.Chart(future_df).mark_line(color='green').encode(
            x='Date:T',
            y='Predicted Close:Q'
        )

        st.altair_chart(future_chart, use_container_width=True)
        st.write(future_df)

    except ValueError as e:
        st.write(f"Error: {e}")
