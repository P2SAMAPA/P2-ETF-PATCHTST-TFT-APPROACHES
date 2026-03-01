import pandas as pd
import numpy as np

def compute_strategy_logic(df, model_choice, yr_range, txn_cost, tsl_threshold, signals_df):
    """
    df: DataFrame with columns TLT, TBT, VNQ, SLV, GLD (prices)
    signals_df: DataFrame with index=date, column='signal' (ETF ticker or CASH)
    """
    # 1. Filter by year range
    mask = (df.index.year >= yr_range[0]) & (df.index.year <= yr_range[1])
    data = df.loc[mask].copy()
    if data.empty:
        return None

    # 2. Align signals with data dates
    common_dates = data.index.intersection(signals_df.index)
    if len(common_dates) == 0:
        return None
    data = data.loc[common_dates]
    signal = signals_df.loc[common_dates, 'signal'].values

    # 3. Model label (for display)
    # Adjust based on your radio button labels – here we check for "Option A" vs "Option B"
    if "Option A" in model_choice:
        label = "Transformer"   # or "PATCHTST" – match your app
    else:
        label = "TFT"

    # 4. Calculate asset returns
    asset_returns = data[["TLT", "TBT", "VNQ", "SLV", "GLD"]].pct_change().fillna(0)

    # 5. Strategy returns
    strat_rets = []
    for i in range(len(data)):
        pick = signal[i]
        if pick in asset_returns.columns:
            strat_rets.append(asset_returns[pick].iloc[i])
        else:
            strat_rets.append(0.0)  # CASH

    strat_rets = pd.Series(strat_rets, index=data.index)

    # 6. Apply trailing stop loss
    rolling_2d = strat_rets.rolling(2).sum()
    final_rets = np.where(rolling_2d < -(tsl_threshold / 100), 0, strat_rets)
    final_rets = pd.Series(final_rets, index=data.index)

    # 7. Transaction costs
    switches = pd.Series(signal).shift(1) != pd.Series(signal)
    final_rets = final_rets - (switches.values * (txn_cost / 100))

    # 8. Metrics
    cum_rets = (1 + final_rets).cumprod()
    sharpe = (final_rets.mean() / final_rets.std()) * np.sqrt(252) if final_rets.std() != 0 else 0

    return {
        "cum_rets": cum_rets,
        "sharpe": sharpe,
        "max_daily_val": final_rets.min(),
        "max_daily_date": final_rets.idxmin().strftime('%Y-%m-%d'),
        "max_p2t": ((cum_rets / cum_rets.cummax()) - 1).min(),
        "signal": signal,
        "daily_rets": final_rets,
        "label": label
    }
