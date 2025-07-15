
from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd
import os
from google.api_core.exceptions import Conflict

# load .env
load_dotenv()

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
DATASET_ID  = "my_dataset"
TABLE_ID    = "product_pairs"
CSV_PATH    = r"C:\Users\Andy\Downloads\dataset.csv"

# 1. read CSV
df = pd.read_csv(CSV_PATH)

# 2. BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# 3. make dataset if missing
full_dataset = f"{PROJECT_ID}.{DATASET_ID}"
try:
    client.create_dataset(full_dataset)
except Conflict:
    pass

# 4. load dataframe
job = client.load_table_from_dataframe(
    df,
    f"{full_dataset}.{TABLE_ID}",
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
)
job.result()

rows = client.get_table(f"{full_dataset}.{TABLE_ID}").num_rows
print(f"✓ Loaded {rows} rows into {DATASET_ID}.{TABLE_ID}")


b = 1
