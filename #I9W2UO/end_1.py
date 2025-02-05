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

folder_path = 'data'
ans = []
for file in os.listdir(folder_path):
    if file.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_excel(file_path)

        data = data.iloc[:-2]

        base_name, _ = os.path.splitext(file)

        ema_short = data['收盘价(元)'].ewm(span=12, adjust=False).mean()
        ema_long = data['收盘价(元)'].ewm(span=26, adjust=False).mean()
        dif = ema_short - ema_long
        dea = dif.ewm(span=9, adjust=False).mean()
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

        # 设置order参数
        order = 15

        # 找到股价和MACD的局部极大值和极小值索引
        price_max_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=True)
        price_min_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=False)
        macd_max_idx = find_local_extrema(data['MACD'], window=order, greater=True)
        macd_min_idx = find_local_extrema(data['MACD'], window=order, greater=False)

        # 创建极值点的Series
        data['price_max'] = pd.Series(data['收盘价(元)'].iloc[price_max_idx].values, index=price_max_idx)
        data['price_min'] = pd.Series(data['收盘价(元)'].iloc[price_min_idx].values, index=price_min_idx)
        data['macd_max'] = pd.Series(data['MACD'].iloc[macd_max_idx].values, index=macd_max_idx)
        data['macd_min'] = pd.Series(data['MACD'].iloc[macd_min_idx].values, index=macd_min_idx)

        # 初始化背离标识
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

        initial_capital = 500000.0
        capital = initial_capital
        stocks_held = 0
        portfolio_values = []

        buy_price = 0
        trades = 0
        trades_tot = 0
        gains = []
        buy_queue = []
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

        annual_risk_free_rate = 2.653 / 100
        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1/252) - 1
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        adjusted_returns = returns - daily_risk_free_rate
        sharpe_ratio = (np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(len(data))
                        if np.std(adjusted_returns) != 0 else 0)
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values) if np.max(portfolio_values) > 0 else 0

        ans.append({
            "name": base_name,
            "win_rate": win_rate,
            "odds_ratio": odds_ratio,
            "trade_frequency": trade_frequency,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        })
ansdf = pd.DataFrame(ans)

file_path = "trading_strategies.xlsx"
ansdf.to_excel(file_path, index=False)