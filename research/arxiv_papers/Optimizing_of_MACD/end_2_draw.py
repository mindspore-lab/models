import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

def find_local_extrema(data, window, greater=True):
    special_indices = []
    for i in range(window + 1, len(data)):
        window_data = data[i-window-1:i]
        if greater:
            if data[i-1] == window_data.max() and data[i] < data[i-1]:
                special_indices.append(i-1)
        else:
            if data[i-1] == window_data.min() and data[i] > data[i-1]:
                special_indices.append(i-1)

    return np.array(special_indices)

file_path = './data/510300.SH.xlsx'
data = pd.read_excel(file_path)
data = data.iloc[:500]

ema_short = data['收盘价(元)'].ewm(span=9, adjust=False).mean()
ema_long = data['收盘价(元)'].ewm(span=22, adjust=False).mean()
dif = ema_short - ema_long
dea = dif.ewm(span=25, adjust=False).mean()
macd_histogram = dif - dea
coeffs = pywt.wavedec(dif.values, 'coif5', level=4)

approximation = coeffs[0]

reconstructed_signal = pywt.waverec([approximation] + [np.zeros_like(coeff) for coeff in coeffs[1:]], 'coif5')
if len(reconstructed_signal) != len(dif):
    reconstructed_signal = reconstructed_signal[:len(dif)]
reconstructed_dif = pd.Series(reconstructed_signal, index=dif.index)

data['DIF'] = reconstructed_dif
data['DEA'] = dea
data['MACD'] = macd_histogram

buy_signals = (data['DIF'] > data['MACD']) & (data['DIF'].shift(1) <= data['MACD'].shift(1))
sell_signals = (data['DIF'] < data['MACD']) & (data['DIF'].shift(1) >= data['MACD'].shift(1))

order = 15

price_max_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=True)
price_min_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=False)
macd_max_idx = find_local_extrema(data['MACD'], window=order, greater=True)
macd_min_idx = find_local_extrema(data['MACD'], window=order, greater=False)

data['price_max'] = pd.Series(data['收盘价(元)'].iloc[price_max_idx].values, index=price_max_idx)
data['price_min'] = pd.Series(data['收盘价(元)'].iloc[price_min_idx].values, index=price_min_idx)
data['macd_max'] = pd.Series(data['MACD'].iloc[macd_max_idx].values, index=macd_max_idx)
data['macd_min'] = pd.Series(data['MACD'].iloc[macd_min_idx].values, index=macd_min_idx)

data['bull_divergence'] = False
data['bear_divergence'] = False

for i in range(1, len(price_max_idx)):
    current_idx = price_max_idx[i]
    previous_idx = price_max_idx[i-1]
    
    if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[current_idx] < data['MACD'].iloc[previous_idx]:
        data.loc[current_idx, 'bull_divergence'] = True

for i in range(1, len(macd_max_idx)):
    current_idx = macd_max_idx[i]
    previous_idx = macd_max_idx[i-1]
    
    if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[current_idx] < data['MACD'].iloc[previous_idx]:
        data.loc[current_idx, 'bull_divergence'] = True
        
for i in range(1, len(price_min_idx)):
    current_idx = price_min_idx[i]
    previous_idx = price_min_idx[i-1]
    
    if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[current_idx] > data['MACD'].iloc[previous_idx]:
        data.loc[current_idx, 'bear_divergence'] = True

for i in range(1, len(macd_min_idx)):
    current_idx = macd_min_idx[i]
    previous_idx = macd_min_idx[i-1]

if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[current_idx] > data['MACD'].iloc[previous_idx]:
    data.loc[current_idx, 'bear_divergence'] = True

red_bar = np.where(macd_histogram > 0, macd_histogram, 0)
blue_bar = np.where(macd_histogram < 0, macd_histogram, 0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 绘制牛背离点
bull_divergence_points = data['日期'][data['bull_divergence']]
bull_divergence_prices = data['收盘价(元)'][data['bull_divergence']]
ax1.plot(bull_divergence_points, bull_divergence_prices, 'v', markersize=10, color='red', lw=0)

# 绘制熊背离点
bear_divergence_points = data['日期'][data['bear_divergence']]
bear_divergence_prices = data['收盘价(元)'][data['bear_divergence']]
ax1.plot(bear_divergence_points, bear_divergence_prices, '^', markersize=10, color='green', lw=0)

ax1.plot(data['日期'], data['收盘价(元)'], label='Close Price', color='blue')
ax1.plot(data['日期'][buy_signals], data['收盘价(元)'][buy_signals], '^', markersize=10, color='green', lw=0, label='Buy Signal')
ax1.plot(data['日期'][sell_signals], data['收盘价(元)'][sell_signals], 'v', markersize=10, color='red', lw=0, label='Sell Signal')
ax1.set_title('9-22-25 Close Price and Trade Signals')
ax1.set_ylabel('Price')
ax1.legend()

ax2.plot(data['日期'], data['DIF'], label='DIF')
ax2.plot(data['日期'], data['DEA'], label='DEA')
ax2.bar(data['日期'], red_bar, color="red")
ax2.bar(data['日期'], blue_bar, color="blue")
ax2.set_title('9-22-25 MACD Indicator')
ax2.set_ylabel('Value')
ax2.legend()

plt.show()


initial_capital = 500000.0
capital = initial_capital
stocks_held = 0
portfolio_values = []

trades_tot = 0
buy_price = 0
trades = 0
gains = []
for i in range(len(data) - 1):
    if (buy_signals.iloc[i] or data['bear_divergence'][i]) and capital > 0:
        num_shares_to_buy = capital // data['收盘价(元)'].iloc[i]
        buy_price += num_shares_to_buy * data['收盘价(元)'].iloc[i]
        capital -= num_shares_to_buy * data['收盘价(元)'].iloc[i]
        stocks_held += num_shares_to_buy
        trades_tot += 1

    elif (sell_signals.iloc[i] or data['bull_divergence'][i]) and stocks_held > 0:
        num_shares_to_sell = stocks_held
        sell_price = num_shares_to_sell * data['收盘价(元)'].iloc[i]
        capital += sell_price
        stocks_held = 0
        trades += 1
        trades_tot += 1
        gains.append(sell_price - buy_price)
        buy_price = 0

    portfolio_value = capital + stocks_held * data['收盘价(元)'][i]
    portfolio_values.append(portfolio_value)

if stocks_held > 0:
    num_shares_to_sell = stocks_held
    capital += num_shares_to_sell * data['收盘价(元)'].iloc[len(data)-1]
    stocks_held = 0
    trades += 1
    trades_tot += 1
    gains.append(sell_price - buy_price)

portfolio_value = capital + stocks_held * data['收盘价(元)'][len(data)-1]
portfolio_values.append(portfolio_value)

gains = np.array(gains)
profits = gains[gains > 0]
losses = gains[gains < 0]

win_rate = len(profits) / trades if trades > 0 else 0
avg_gain = np.mean(profits) if len(profits) > 0 else 0
avg_loss = np.mean(losses) if len(losses) > 0 else 0
odds_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
trade_frequency = trades_tot / len(data) if len(data) > 0 else 0

total_return = capital - initial_capital
annual_return = ((total_return / initial_capital) + 1) ** (252 / len(data)) - 1
portfolio_values = np.array(portfolio_values)

# 假设的无风险年化回报率
annual_risk_free_rate = 2.653 / 100

# 将年化无风险回报率转换为日无风险回报率
daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/252) - 1
returns = np.diff(portfolio_values) / portfolio_values[:-1]
adjusted_returns = returns - daily_risk_free_rate
sharpe_ratio = (np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(len(data))
                if np.std(adjusted_returns) != 0 else 0)
max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values) if np.max(portfolio_values) > 0 else 0

print({
    "name": "AL.SHF",
    "win_rate": win_rate,
    "odds_ratio": odds_ratio,
    "trade_frequency": trade_frequency,
    "total_return": total_return,
    "annual_return": annual_return,
    "sharpe_ratio": sharpe_ratio,
    "max_drawdown": max_drawdown,
})