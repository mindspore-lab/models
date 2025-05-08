import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------
# 1. 获取 S&P500 股票代码
# ---------------------------
tickers_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(tickers_url)
sp500_table = tables[0]
tickers = sp500_table['Symbol'].tolist()

tickers = tickers[:3]

# ---------------------------
# 2. 下载股票历史数据 & 计算对数收益率
# ---------------------------
start_date = '2020-01-01'
end_date = '2022-12-31'

stock_data_list = []
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         progress=False, auto_adjust=False)
        if df.empty:
            continue
        print("下载成功:", ticker)
        df = df[['Adj Close']].copy()
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
        df['Return'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))
        df.dropna(inplace=True)
        df['Ticker'] = ticker
        df.index.name = 'Date'
        df = df.reset_index(drop=False)
        stock_data_list.append(df)
    except Exception as e:
        print(f"下载 {ticker} 出错: {e}")

all_stock_data = pd.concat(stock_data_list, ignore_index=True)

# ticker_counts = all_stock_data['Ticker'].value_counts()
# selected_tickers = ticker_counts[ticker_counts >= 100].index.tolist()
# all_stock_data = all_stock_data[all_stock_data['Ticker'].isin(selected_tickers)]

# ---------------------------
# 3. 透视 (pivot) 得到宽表：行=Date, 列=各Ticker的Return
# ---------------------------
pivot_df = all_stock_data.pivot_table(
    index='Date',
    columns='Ticker',
    values='Return',
    aggfunc='first'
)

if isinstance(pivot_df.columns, pd.MultiIndex):
    pivot_df.columns = pivot_df.columns.droplevel(0)

pivot_df.columns = [f"{col}_Return" for col in pivot_df.columns]
pivot_df = pivot_df.reset_index()
pivot_df = pivot_df[(pivot_df['Date'] >= '2020-01-01') & (pivot_df['Date'] <= '2022-12-31')]

# ---------------------------
# 4. 读取 Fama–French 因子数据
# ---------------------------
ff_path = 'data/F-F_Research_Data_5_Factors_2x3_daily.CSV'   # 注意先从 Kenneth French 网站下载并保存该文件
ff_data = pd.read_csv(ff_path)
ff_data.rename(columns={'Data': 'Date'}, inplace=True)
ff_data['Date'] = pd.to_datetime(ff_data['Date'].astype(str), format='%Y%m%d')
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
ff_factors = ff_data[["Date"] + factor_cols].copy()
ff_factors[factor_cols] = ff_factors[factor_cols].astype(float) / 100.0
# ---------------------------
# 5. 读取每日情感因子
# ---------------------------
sentiment_df = pd.read_csv('./data/daily_sentiment.csv', parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))
# ---------------------------
# 6. 合并：因子 & 情感因子 & 透视表
# ---------------------------
vix_path = 'data/VIX_History.csv'
vix_df = pd.read_csv(vix_path)
vix_df.columns = [col.strip().upper() for col in vix_df.columns]
vix_df['DATE'] = pd.to_datetime(vix_df['DATE'], format='%m/%d/%Y', errors='coerce')
vix_df = vix_df[['DATE', 'CLOSE']].rename(columns={'DATE': 'Date', 'CLOSE': 'VIX_Close'})
vix_df = vix_df[(vix_df['Date'] >= '2020-01-01') & (vix_df['Date'] <= '2022-12-31')]
dgs10_path = 'data/DGS10.csv'  # <-- 修改为您的实际文件路径
dgs10_df = pd.read_csv(dgs10_path)
dgs10_df['observation_date'] = pd.to_datetime(dgs10_df['observation_date'])
dgs10_df.rename(columns={'observation_date': 'Date', 'DGS10': 'DGS10_Yield'}, inplace=True)
factor_sentiment_df = pd.merge(ff_factors, sentiment_df, on='Date', how='left')
final_df = pd.merge(factor_sentiment_df, pivot_df, on='Date', how='left')
final_df = pd.merge(final_df, vix_df, on='Date', how='left')
final_df = pd.merge(final_df, dgs10_df, on='Date', how='left')
cols = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'DGS10_Yield', 'S_t', 'VIX_Close']
other_cols = [c for c in final_df.columns if c not in cols]
final_df = final_df[cols + other_cols]
final_df = final_df[(final_df['Date'] >= '2020-01-01') & (final_df['Date'] <= '2022-12-31')]

# ---------------------------
# 7. 保存结果
# ---------------------------
final_df.to_csv('WideFormat_FF5_Sentiment_VIX_Returns.csv', index=False)
print(final_df.head(10))
