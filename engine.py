import pandas as pd
import numpy as np

def compute_strategy_logic(df, model_choice, yr_range, txn_cost, tsl_threshold):
    """
    Simulates a trading strategy based on ETF signals.
    For demonstration, signals are random (replace with real PATCHTST/TFT predictions).
    """
    # 1. Filter by Year Slider
    mask = (df.index.year >= yr_range[0]) & (df.index.year <= yr_range[1])
    data = df.loc[mask].copy()
    if data.empty:
        return None

    # 2. Model Selection (different seeds for reproducibility)
    if "Option A" in model_choice:
        np.random.seed(42)
        label = "PATCHTST"
    else:
        np.random.seed(99)
        label = "TFT"

    # 3. Define Universe and generate mock signals
    tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
    # In production, replace with: signal = data['your_model_column']
    signal = np.random.choice(tickers + ["CASH"], size=len(data))

    # 4. Calculate daily returns for each asset
    asset_returns = data[tickers].pct_change().fillna(0)

    # 5. Build strategy returns from the picked assets
    strat_rets = []
    for i in range(len(data)):
        pick = signal[i]
        if pick in asset_returns.columns:
            strat_rets.append(asset_returns[pick].iloc[i])
        else:
            strat_rets.append(0.0)  # CASH

    strat_rets = pd.Series(strat_rets, index=data.index)

    # 6. Apply 2‑day trailing stop loss (TSL)
    rolling_2d = strat_rets.rolling(2).sum()
    final_rets = np.where(rolling_2d < -(tsl_threshold / 100), 0, strat_rets)
    final_rets = pd.Series(final_rets, index=data.index)

    # 7. Apply transaction costs when the predicted ETF changes
    switches = pd.Series(signal).shift(1) != pd.Series(signal)
    final_rets = final_rets - (switches.values * (txn_cost / 100))

    # 8. Calculate performance metrics
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
