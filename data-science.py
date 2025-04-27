
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

model = joblib.load("AppleStockPricePredictionModel.pkl")

AppleData = pd.read_csv("AppleData.csv")
AppleData["Date"] = pd.to_datetime(AppleData["Date"], errors='coerce')
AppleData["Year"] = AppleData["Date"].dt.year
AppleData["Month"] = AppleData["Date"].dt.month
AppleData["Day"] = AppleData["Date"].dt.day

st.title("Apple Stock Price Analysis")

st.subheader("How did the 9/11 attacks affect the stock price of Apple?")

Apple2001 = AppleData[AppleData["Year"] == 2001]
TargetDate = pd.Timestamp("2001-09-11")
EndDate = TargetDate + pd.DateOffset(months=2)

WantedData = Apple2001[(Apple2001["Date"] >= TargetDate) & (Apple2001["Date"] <= EndDate)]

fig, Plot = plt.subplots(figsize=(12, 6))
Plot.plot(Apple2001["Date"], Apple2001["Open Price"], color='lightgray', label='Other Dates')
Plot.plot(WantedData["Date"], WantedData["Open Price"], color='blue', linewidth=2, label='Post 9/11 (2 Months)')
Plot.axvline(TargetDate, color='red', linestyle='--', linewidth=2, label='9/11')

Plot.set_xlabel("Date")
Plot.set_ylabel("Open Price")
Plot.set_title("Apple Open Prices: Highlighting 9/11 and Two Months After (2001)")
Plot.legend()
Plot.grid(True)

st.pyplot(fig)

st.write("As we can see, the event of 9/11 did affect Apple stocks in a negative way, but Apple managed to recover quickly.")

st.subheader("Is there a massive difference between the release of the first iPhone and the last iPhone?")

Iphone1Date = pd.Timestamp("2007-06-29")
Iphone16Date = pd.Timestamp("2024-09-01")

Iphone1PrevMonth = Iphone1Date - pd.DateOffset(months=1)
Iphone1NxtMonth = Iphone1Date + pd.DateOffset(months=1)

Iphone16PrevMonth = Iphone16Date - pd.DateOffset(months=1)
Iphone16NxtMonth = Iphone16Date + pd.DateOffset(months=1)

fig, Plots = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Apple Open Prices Around iPhone 1 and iPhone 16 Releases", fontsize=16)

Plot = Plots[0]
Apple2007 = AppleData[AppleData["Year"] == 2007]

Plot.plot(Apple2007["Date"], Apple2007["Open Price"], color='lightgray', label='Other Dates')

Iphone1Prev = Apple2007[(Apple2007["Date"] >= Iphone1PrevMonth) & (Apple2007["Date"] < Iphone1Date)]
Iphone1Nxt = Apple2007[(Apple2007["Date"] > Iphone1Date) & (Apple2007["Date"] <= Iphone1NxtMonth)]

Plot.plot(Iphone1Prev["Date"], Iphone1Prev["Open Price"], color='red', label='Month Before Release')
Plot.plot(Iphone1Nxt["Date"], Iphone1Nxt["Open Price"], color='blue', label='Month After Release')
Plot.axvline(Iphone1Date, color='black', linestyle='--', linewidth=2, label='iPhone 1 Release')

Plot.set_title("iPhone 1 Release (2007)")
Plot.set_xlabel("Date")
Plot.set_ylabel("Open Price")
Plot.legend()
Plot.grid(True)

Plot = Plots[1]
Apple2024 = AppleData[AppleData["Year"] == 2024]

Plot.plot(Apple2024["Date"], Apple2024["Open Price"], color='lightgray', label='Other Dates')

Iphone16Prev = Apple2024[(Apple2024["Date"] >= Iphone16PrevMonth) & (Apple2024["Date"] < Iphone16Date)]
Iphone16Nxt = Apple2024[(Apple2024["Date"] > Iphone16Date) & (Apple2024["Date"] <= Iphone16NxtMonth)]

Plot.plot(Iphone16Prev["Date"], Iphone16Prev["Open Price"], color='red', label='Month Before Release')
Plot.plot(Iphone16Nxt["Date"], Iphone16Nxt["Open Price"], color='blue', label='Month After Release')
Plot.axvline(Iphone16Date, color='green', linestyle='--', linewidth=2, label='iPhone 16 Release')

Plot.set_title("iPhone 16 Release (2024)")
Plot.set_xlabel("Date")
Plot.set_ylabel("Open Price")
Plot.legend()
Plot.grid(True)

plt.tight_layout()
st.pyplot(fig)

st.write("We can observe the impact around the releases of the first iPhone and the upcoming iPhone 16!")

st.subheader("Did COVID-19 affect the stock prices of Apple during quarantine?")

years = [2017, 2018, 2019, 2020, 2021, 2022]
Comparison = AppleData[AppleData["Year"].isin(years)]

fig, Plot = plt.subplots(figsize=(12, 6))

for year in years:
    DataYears = Comparison[Comparison["Year"] == year].sort_values("Date")
    Plot.plot(DataYears["Date"], DataYears["Open Price"], label=str(year))

Plot.set_xlabel("Date")
Plot.set_ylabel("Open Price")
Plot.set_title("Apple Open Prices (2017 - 2022)")
Plot.legend()
Plot.grid(True)

st.pyplot(fig)

st.write("COVID-19 severely affected the stock prices of Apple and caused long-term damage.")

st.subheader("Apple Stock Price Prediction")

Data = pd.read_csv("AppleData.csv")

X = Data[["Volume", "year", "month", "day"]]
Y = Data["Open Price"]

years = list(range(2007, 2051))
months = list(range(1, 13))
days = list(range(1, 32))

col1, col2, col3, col4 = st.columns(4)

with col1:
    DaySelect = st.selectbox("Select Day", days)
with col2:
    MonthSelect = st.selectbox("Select Month", months)
with col3:
    YearSelect = st.selectbox("Select Year", years)
with col4:
    VolumeInput = st.number_input("Enter Expected Volume", min_value=0, value=50000000, step=1000000)

try:
    selected_date = datetime.date(YearSelect, MonthSelect, DaySelect)
except ValueError:
    st.error("Invalid date selected. Please choose a valid date.")
    st.stop()

XInput = np.array([[VolumeInput, YearSelect, MonthSelect, DaySelect]])

prediction = model.predict(XInput)[0]

st.success(f"Open Price Prediction: ${prediction:.2f}")
st.write(f"Model Accuracy (R²): {r2_score(Y, model.predict(X)) * 100:.2f}%")
