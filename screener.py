import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def calc_sharpe(returns, rf=0):
    if len(returns) < 2 or returns.std() == 0: return 0
    return ((returns.mean() - rf) / returns.std()) * np.sqrt(252)

def calc_sortino(returns, rf=0):
    if len(returns) < 2: return 0
    downside = returns[returns < rf] - rf
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 0
    if downside_std == 0: return 0
    return ((returns.mean() - rf) / downside_std) * np.sqrt(252)

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50
    diffs = np.diff(closes)
    gains = diffs[diffs >= 0].sum()
    losses = -diffs[diffs < 0].sum()
    if losses == 0: return 100
    rs = (gains / period) / (losses / period)
    return 100 - (100 / (1 + rs))

print("Loading tickers and company names from CSV...")
# Reads your uploaded 2-column file
master_df = pd.read_csv('ticker.csv') 
tickers = [f"{str(t).strip()}.NS" for t in master_df['Ticker'].tolist()]

print("Fetching Nifty 50 Benchmark...")
nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False)['Close']
if isinstance(nifty, pd.DataFrame): nifty = nifty.squeeze()
nifty_returns = nifty.pct_change().dropna()

print(f"Fetching historical data for {len(tickers)} stocks...")
data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", threads=True, progress=False)

results = []

for ticker in tickers:
    try:
        if len(tickers) == 1:
            closes = data['Close'].dropna()
            volumes = data['Volume'].dropna()
        else:
            if ticker not in data.columns.levels[0]: continue
            closes = data[ticker]['Close'].dropna()
            volumes = data[ticker]['Volume'].dropna()
            
        if len(closes) < 200: continue 
        
        returns = closes.pct_change().dropna()
        
        # Time slices
        c_3m, c_6m, c_9m = closes.tail(63), closes.tail(126), closes.tail(189)
        r_3m, r_6m, r_9m = returns.tail(63), returns.tail(126), returns.tail(189)
        
        # Core Metrics
        w_sharpe = (calc_sharpe(r_3m)*0.4) + (calc_sharpe(r_6m)*0.3) + (calc_sharpe(r_9m)*0.2) + (calc_sharpe(returns)*0.1)
        w_sortino = (calc_sortino(r_3m)*0.4) + (calc_sortino(r_6m)*0.3) + (calc_sortino(r_9m)*0.2) + (calc_sortino(returns)*0.1)
        
        roc_3m = ((c_3m.iloc[-1] - c_3m.iloc[0]) / c_3m.iloc[0]) * 100
        roc_6m = ((c_6m.iloc[-1] - c_6m.iloc[0]) / c_6m.iloc[0]) * 100
        roc_9m = ((c_9m.iloc[-1] - c_9m.iloc[0]) / c_9m.iloc[0]) * 100
        roc_12m = ((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]) * 100
        w_roc = (roc_3m*0.4) + (roc_6m*0.3) + (roc_9m*0.2) + (roc_12m*0.1)
        
        # Blends
        sharpe_roc_blend = (w_sharpe + (w_roc / 10)) / 2
        sortino_roc_blend = (w_sortino + (w_roc / 10)) / 2
        master_blend = (w_sharpe + w_sortino + (w_roc / 10)) / 3
        
        # Technicals
        last_price = closes.iloc[-1]
        ma_200 = closes.tail(200).mean()
        trend = "Pass (Up)" if last_price > ma_200 else "Fail (Down)"
        max_52w = closes.max()
        dist_high = ((last_price - max_52w) / max_52w) * 100
        max_dd = ((closes - closes.cummax()) / closes.cummax()).min() * 100
        avg_turnover = (closes.tail(63) * volumes.tail(63)).mean() / 10000000
        rsi = calc_rsi(closes.values, 14)
        
        # Alpha & Beta
        aligned_returns, aligned_benchmark = returns.align(nifty_returns, join='inner')
        if len(aligned_returns) > 0 and aligned_benchmark.var() != 0:
            cov = np.cov(aligned_returns, aligned_benchmark)[0][1]
            beta = cov / aligned_benchmark.var()
            ann_ret_s = aligned_returns.mean() * 252
            ann_ret_b = aligned_benchmark.mean() * 252
            alpha = (ann_ret_s - (beta * ann_ret_b)) * 100
        else:
            beta, alpha = 1, 0

        # Math logic only - no Yahoo Finance text lookups
        results.append({
            "Ticker": ticker.replace('.NS', ''),
            "Weighted Sharpe": round(w_sharpe, 2),
            "Weighted Sortino": round(w_sortino, 2),
            "1-Year ROC %": round(roc_12m, 2),
            "Weighted ROC %": round(w_roc, 2),
            "Sharpe + ROC Blend": round(sharpe_roc_blend, 2),
            "Sortino + ROC Blend": round(sortino_roc_blend, 2),
            "Master Blend": round(master_blend, 2),
            "1-Year Max Drawdown %": round(max_dd, 2),
            "Last Close": round(last_price, 2),
            "200-Day MA": round(ma_200, 2),
            "Trend Check": trend,
            "Alpha (1Y)": round(alpha, 2),
            "Beta (1Y)": round(beta, 2),
            "Dist. from 52W High %": round(dist_high, 2),
            "Avg Turnover (Cr)": round(avg_turnover, 2),
            "14-Day RSI": round(rsi, 2)
        })
    except Exception as e:
        continue

# 3. Merge the math with your Company Names
calc_df = pd.DataFrame(results)
final_df = pd.merge(master_df, calc_df, on="Ticker", how="inner")

# Reorder so Company Name is right next to Ticker
cols = final_df.columns.tolist()
cols.insert(1, cols.pop(cols.index('Company Name')))
final_df = final_df[cols]

# 4. Save to CSV
final_df.to_csv('screener_results.csv', index=False)
print("Screener successfully completed and saved!")
