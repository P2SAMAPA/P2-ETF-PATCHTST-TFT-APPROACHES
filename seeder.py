import os
import gitlab
import yfinance as yf
import pandas as pd

def run_seed():
    # Retrieve secrets from the environment (GitHub Actions will provide these)
    token = os.getenv('GITLAB_API_TOKEN')
    project_id = os.getenv('GITLAB_PROJECT_ID')
    
    symbols = ["TLT", "TBT", "VNQ", "GLD", "SLV", "SPY", "AGG", "^IRX"]
    
    print("📥 Pulling historical data from Yahoo Finance...")
    df = yf.download(symbols, start="2008-01-01")['Close']
    
    # Save to a temporary CSV
    csv_name = "master_data.csv"
    df.to_csv(csv_name)
    
    print(f"📡 Connecting to GitLab Project {project_id}...")
    gl = gitlab.Gitlab("https://gitlab.com", private_token=token)
    project = gl.projects.get(project_id)
    
    with open(csv_name, 'r') as f:
        content = f.read()
        
    try:
        project.files.create({
            'file_path': csv_name,
            'branch': 'main',
            'content': content,
            'commit_message': 'Automated Initial Data Seed 2008-2026'
        })
        print("✅ SUCCESS: Data Lake populated!")
    except Exception as e:
        print(f"⚠️ Note: {e}. (If it says 400, the file already exists).")

if __name__ == "__main__":
    run_seed()
