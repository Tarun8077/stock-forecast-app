import streamlit as st
import matplotlib.pyplot as plt
from model_forecast import load_data, train_and_forecast

st.set_page_config(page_title="Stock Price Forecast", layout="wide")
st.title("📈 AAPL Stock Price Forecast (LSTM vs LR vs ARIMA)")

with st.spinner("Training models and forecasting..."):
    df = load_data()
    results = train_and_forecast(df)

st.subheader("Historical Closing Prices")
st.line_chart(df.set_index('Date')['Close'])

st.subheader("📊 120-Day Forecast")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(results["forecast_dates"], results["lstm"], label="LSTM", color='blue')
ax.plot(results["forecast_dates"], results["lr"], label="Linear Regression", color='green', linestyle='--')
ax.plot(results["forecast_dates"], results["arima"], label="ARIMA", color='red', linestyle=':')
ax.axvline(results["forecast_dates"][0], color='gray', linestyle='--')
ax.set_title("LSTM vs LR vs ARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.subheader("📁 Download Forecast")
forecast_df = results['forecast_dates'].to_frame(name='Date')
forecast_df['LSTM_Prediction'] = results['lstm']
forecast_df['LinearRegression_Prediction'] = results['lr']
forecast_df['ARIMA_Prediction'] = results['arima'].values

st.dataframe(forecast_df)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "forecast_120_days.csv", "text/csv")
