import os
import gitlab
import yfinance as yf
import pandas as pd
from datetime import datetime

def run_seed():
    token = os.getenv('GITLAB_API_TOKEN')
    project_id = os.getenv('GITLAB_PROJECT_ID')
    
    symbols = ["TLT", "TBT", "VNQ", "GLD", "SLV", "SPY", "AGG", "^IRX"]
    
    print("📥 Pulling historical data from Yahoo Finance...")
    # Explicit end date to avoid pulling future data
    df = yf.download(symbols, start="2008-01-01", end=datetime.now().date())['Close']
    
    csv_name = "master_data.csv"
    df.to_csv(csv_name)
    
    print(f"📡 Connecting to GitLab Project {project_id}...")
    gl = gitlab.Gitlab("https://gitlab.com", private_token=token)
    project = gl.projects.get(project_id)
    
    with open(csv_name, 'r') as f:
        content = f.read()
    
    # Check if file already exists
    try:
        project.files.get(file_path=csv_name, ref='main')
        print("⚠️ File already exists. Use update_logic.py to update.")
    except gitlab.exceptions.GitlabGetError:
        # File does not exist, create it
        project.files.create({
            'file_path': csv_name,
            'branch': 'main',
            'content': content,
            'commit_message': 'Automated Initial Data Seed 2008-2026'
        })
        print("✅ SUCCESS: Data Lake populated!")

if __name__ == "__main__":
    run_seed()
