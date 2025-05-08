import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from datetime import timedelta
df = pd.read_csv('WideFormat_FF5_Sentiment_VIX_Returns.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df['DGS10_diff'] = df['DGS10_Yield'].diff()
df['Excess'] = df['A_Return'] - df['RF']
df['HV'] = df['S_t'].rolling(window=21).std()
df = df.dropna(subset=['Excess','Mkt-RF','SMB','HML','RMW','CMA','S_t','HV','DGS10_diff'])

vars_for_stats = ['Excess','Mkt-RF','SMB','HML','RMW','CMA','S_t','HV','VIX_Close','DGS10_diff']
desc = df[vars_for_stats].describe().T[['mean','std','min','max','count']]
desc.columns = ['Mean','Std Dev','Min','Max','Obs']
print("Descriptive Statistics:")
print(desc)

y = df['Excess']

X_base = df[['Mkt-RF','SMB','HML','RMW','CMA','DGS10_diff']]
X_base = sm.add_constant(X_base)
model_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags':5})

X_sent = df[['Mkt-RF','SMB','HML','RMW','CMA','DGS10_diff','S_t']]
X_sent = sm.add_constant(X_sent)
model_sent = sm.OLS(y, X_sent).fit(cov_type='HAC', cov_kwds={'maxlags':5})

X_int = df[['Mkt-RF','SMB','HML','RMW','CMA','DGS10_diff','S_t','HV']].copy()
X_int['SxHV'] = X_int['S_t'] * X_int['HV']
X_int = sm.add_constant(X_int)
model_int = sm.OLS(y, X_int).fit(cov_type='HAC', cov_kwds={'maxlags':5})

def summarize_model(m, name):
    summary_df = pd.DataFrame({
        'coef': m.params,
        'std_err': m.bse,
        't': m.tvalues,
        'p': m.pvalues
    })
    summary_df['model'] = name
    return summary_df

results = pd.concat([
    summarize_model(model_base, 'Baseline'),
    summarize_model(model_sent, 'Sentiment'),
    summarize_model(model_int, 'Interaction')
])
results = results.reset_index().rename(columns={'index':'Variable'})
print("\nRegression Results (Newey-West SE):")
print(results)

vif_data = X_sent.copy()
vif = pd.DataFrame()
vif['Variable'] = vif_data.columns
vif['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
print("\nVIF for Sentiment-Augmented Model:")
print(vif)

window = 60
rolling_coefs = []
rolling_dates = []

for end in range(window, len(df)):
    sub = df.iloc[end-window:end]
    Xr = sub[['Mkt-RF','SMB','HML','RMW','CMA','DGS10_diff','S_t']]
    Xr = sm.add_constant(Xr)
    yr = sub['Excess']
    res = sm.OLS(yr, Xr).fit()
    rolling_coefs.append(res.params['S_t'])
    rolling_dates.append(sub.index[-1])

plt.figure(figsize=(12, 6))
plt.plot(rolling_dates, rolling_coefs, color='steelblue', linewidth=2, label='Sentiment Coef (60-day)')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Rolling 60-Day Sentiment Coefficient', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Coefficient on $S_t$', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


event_date = pd.Timestamp('2022-06-15')
loc = df.index.get_loc(event_date)

est_start = loc - 130
est_end = loc - 11
est_df = df.iloc[est_start:est_end+1]

X_ff = sm.add_constant(est_df[['Mkt-RF','SMB','HML']])
est_model_ff = sm.OLS(est_df['Excess'], X_ff).fit()
X_snt = sm.add_constant(est_df[['Mkt-RF','SMB','HML','S_t']])
est_model_snt = sm.OLS(est_df['Excess'], X_snt).fit()

evt_start = loc - 10
evt_end = loc + 10
evt_df = df.iloc[evt_start:evt_end+1].copy()

X_ff_evt = sm.add_constant(evt_df[['Mkt-RF','SMB','HML']])
evt_df['R_pred_ff'] = est_model_ff.predict(X_ff_evt)
X_snt_evt = sm.add_constant(evt_df[['Mkt-RF','SMB','HML','S_t']])
evt_df['R_pred_snt'] = est_model_snt.predict(X_snt_evt)

evt_df['AR_ff'] = evt_df['Excess'] - evt_df['R_pred_ff']
evt_df['AR_snt'] = evt_df['Excess'] - evt_df['R_pred_snt']

evt_df['CAR_ff'] = evt_df['AR_ff'].cumsum()
evt_df['CAR_snt'] = evt_df['AR_snt'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(evt_df.index, evt_df['CAR_ff'], label='Baseline CAR', color='darkorange', linewidth=2)
plt.plot(evt_df.index, evt_df['CAR_snt'], label='Sentiment-Augmented CAR', color='seagreen', linewidth=2)
plt.axvline(event_date, color='red', linestyle='--', label='Event Date', linewidth=1.5)
plt.title('Event Study: CAR around June 15, 2022 Rate Hike', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Abnormal Return', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
