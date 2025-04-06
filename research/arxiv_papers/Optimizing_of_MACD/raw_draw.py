import numpy as np
import pandas as pd

file_path = './data/NR.INE.xlsx'
data = pd.read_excel(file_path)
data = data.iloc[:-2]

import matplotlib.pyplot as plt

ema_short = data['收盘价(元)'].ewm(span=12, adjust=False).mean()
ema_long = data['收盘价(元)'].ewm(span=26, adjust=False).mean()
dif = ema_short - ema_long
dea = dif.ewm(span=9, adjust=False).mean()
macd_histogram = dif - dea

data['DIF'] = dif
data['DEA'] = dea
data['MACD'] = macd_histogram
red_bar = np.where(macd_histogram > 0, macd_histogram, 0)
blue_bar = np.where(macd_histogram < 0, macd_histogram, 0)

buy_signals = (data['DIF'] > data['MACD']) & (data['DIF'].shift(1) <= data['MACD'].shift(1))
sell_signals = (data['DIF'] < data['MACD']) & (data['DIF'].shift(1) >= data['MACD'].shift(1))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(data['日期'], data['收盘价(元)'], label='Close Price', color='blue')
ax1.plot(data['日期'][buy_signals], data['收盘价(元)'][buy_signals], '^', markersize=10, color='green', lw=0, label='Buy Signal')
ax1.plot(data['日期'][sell_signals], data['收盘价(元)'][sell_signals], 'v', markersize=10, color='red', lw=0, label='Sell Signal')
ax1.set_title('Close Price and Trade Signals')
ax1.set_ylabel('Price')
ax1.legend()

ax2.plot(data['日期'], data['DIF'], label='DIF')
ax2.plot(data['日期'], data['DEA'], label='DEA')
ax2.bar(data['日期'], red_bar, color="red")
ax2.bar(data['日期'], blue_bar, color="blue")
ax2.set_title('MACD Indicator')
ax2.set_ylabel('Value')
ax2.legend()

plt.show()
