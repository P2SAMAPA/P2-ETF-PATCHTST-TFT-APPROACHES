import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import gitlab
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

print("📦 Using darts version:", __import__('darts').__version__)

from darts.models import TFTModel, TransformerModel

# --- Configuration ---
GITLAB_URL = "https://gitlab.com"
PROJECT_ID = os.getenv('GITLAB_PROJECT_ID')
GL_TOKEN = os.getenv('GITLAB_API_TOKEN')
DATA_FILE = "master_data.csv"
TFT_SIGNALS_FILE = "signals_tft.csv"
TRANSFORMER_SIGNALS_FILE = "signals_transformer.csv"
TICKERS = ["TLT", "TBT", "VNQ", "SLV", "GLD"]

# Checkpoint files (store last processed date per model)
TFT_CHECKPOINT = "tft_checkpoint.txt"
TRANSFORMER_CHECKPOINT = "transformer_checkpoint.txt"

def fetch_file_from_gitlab(file_name):
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    try:
        file_info = project.files.get(file_path=file_name, ref='main')
        return file_info.decode().decode('utf-8')
    except:
        return None

def upload_to_gitlab(file_name, content):
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    try:
        f = project.files.get(file_path=file_name, ref='main')
        f.content = content
        f.save(branch='main', commit_message=f"Update {file_name} - {datetime.now().date()}")
    except:
        project.files.create({
            'file_path': file_name,
            'branch': 'main',
            'content': content,
            'commit_message': f"Add {file_name}"
        })

def read_checkpoint(model_name):
    """Return the last date (as Timestamp) processed for this model, or None."""
    fname = TFT_CHECKPOINT if model_name == "TFT" else TRANSFORMER_CHECKPOINT
    content = fetch_file_from_gitlab(fname)
    if content:
        try:
            return pd.to_datetime(content.strip())
        except:
            return None
    return None

def write_checkpoint(model_name, date):
    """Write the last processed date to a checkpoint file in GitLab."""
    fname = TFT_CHECKPOINT if model_name == "TFT" else TRANSFORMER_CHECKPOINT
    upload_to_gitlab(fname, date.strftime('%Y-%m-%d'))

def prepare_series(df, target_ticker):
    series = TimeSeries.from_dataframe(df, value_cols=target_ticker, fill_missing_dates=True, freq='B')
    covariates = datetime_attribute_timeseries(series, attribute="month", cyclic=True)
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="dayofweek", cyclic=True))
    return series, covariates

def predict_next_day(train_df, ticker, model_class, model_params):
    series, covariates = prepare_series(train_df, ticker)
    target_scaler = Scaler()
    scaled_series = target_scaler.fit_transform(series)
    covariate_scaler = Scaler()
    scaled_covariates = covariate_scaler.fit_transform(covariates)
    
    model = model_class(**model_params)
    model.fit(scaled_series, past_covariates=scaled_covariates, verbose=False)
    
    pred = model.predict(n=1, series=scaled_series, past_covariates=scaled_covariates)
    pred = target_scaler.inverse_transform(pred)
    pred_price = pred.values()[0][0]
    last_price = train_df.loc[train_df.index[-1], ticker]
    return (pred_price - last_price) / last_price

def process_model(model_name, model_class, model_params, df, start_date):
    """Process all dates >= start_date, resuming from last checkpoint."""
    signal_file = TFT_SIGNALS_FILE if model_name == "TFT" else TRANSFORMER_SIGNALS_FILE
    
    # Load existing signals if any
    existing_content = fetch_file_from_gitlab(signal_file)
    if existing_content:
        signals_df = pd.read_csv(StringIO(existing_content), index_col=0)
        signals_df.index = pd.to_datetime(signals_df.index)
    else:
        signals_df = pd.DataFrame(columns=['signal'], dtype=object)
    
    # Determine last processed date
    last_processed = read_checkpoint(model_name)
    if last_processed is None and not signals_df.empty:
        last_processed = signals_df.index[-1]
    
    # All dates in df that are >= start_date and after last_processed
    all_dates = df.index[df.index >= start_date]
    if last_processed:
        all_dates = all_dates[all_dates > last_processed]
    
    if len(all_dates) == 0:
        print(f"No new dates to process for {model_name}.")
        return
    
    min_train = 252 * 2  # need at least 2 years of history
    for test_date in all_dates:
        # Ensure enough training data
        train_end = df.index[df.index < test_date][-1]
        train_start_idx = df.index.get_loc(train_end) - min_train + 1
        if train_start_idx < 0:
            print(f"Not enough training data for {test_date.date()}, skipping.")
            continue
        
        train_df = df.loc[:train_end]
        print(f"{model_name}: processing {test_date.date()}...")
        
        pred_returns = {}
        for ticker in TICKERS:
            pred_returns[ticker] = predict_next_day(train_df, ticker, model_class, model_params)
        
        best_ticker = max(pred_returns, key=pred_returns.get)
        signals_df.loc[test_date] = best_ticker
        
        # After each day, upload updated signals and checkpoint
        upload_to_gitlab(signal_file, signals_df.to_csv())
        write_checkpoint(model_name, test_date)
        print(f"  -> {best_ticker} (saved)")

def main():
    print("📥 Fetching master data from GitLab...")
    df_content = fetch_file_from_gitlab(DATA_FILE)
    if df_content is None:
        raise Exception("master_data.csv not found in GitLab")
    df = pd.read_csv(StringIO(df_content), index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"Data shape: {df.shape}, last date: {df.index[-1].date()}")
    
    # Start from 2018 to include pre‑COVID years
    start_date = pd.to_datetime("2018-01-01")
    print(f"Processing from {start_date.date()} onward.")
    
    # Reduced model parameters for speed (still reasonable)
    tft_params = {
        'input_chunk_length': 20,
        'output_chunk_length': 1,
        'hidden_size': 32,
        'lstm_layers': 1,
        'num_attention_heads': 2,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 2,
        'optimizer_kwargs': {'lr': 1e-3},
        'add_relative_index': True,
        'force_reset': True,
        'random_state': 42
    }
    
    transformer_params = {
        'input_chunk_length': 20,
        'output_chunk_length': 1,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'batch_size': 16,
        'n_epochs': 2,
        'optimizer_kwargs': {'lr': 1e-3},
        'force_reset': True,
        'random_state': 42
    }
    
    print("\n🔮 Processing TFT...")
    process_model("TFT", TFTModel, tft_params, df, start_date)
    
    print("\n🔮 Processing Transformer...")
    process_model("Transformer", TransformerModel, transformer_params, df, start_date)
    
    print("✅ All processing complete.")

if __name__ == "__main__":
    main()
