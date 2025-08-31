'''
NOTE I found this dataset through kaggle URL below
https://www.kaggle.com/datasets/ziqizhang/product-data-miningentity-classificationlinking?resource=download

'''
from pathlib import Path
import time
start = time.time()


import os
import gzip
import json
import pandas as pd
import urllib.request

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
URL = (
    "http://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/swc/computers_train_xlarge.json.gz"
)
DATA_DIR = Path.home() / "datasets" / "mwpd"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILE_GZ = DATA_DIR / "computers_train_xlarge.json.gz"
MAX_ROWS = 100
# ---------------------------------------------------------------------
# 1. Download (if needed)
# ---------------------------------------------------------------------
if not FILE_GZ.exists():
    print("Downloading dataset …")
    urllib.request.urlretrieve(URL, FILE_GZ)
    print(f"Saved to: {FILE_GZ}")

# ---------------------------------------------------------------------
# 2. Stream‑load into DataFrame
# ---------------------------------------------------------------------
rows = []
with gzip.open(FILE_GZ, "rt", encoding="utf‑8") as fh:
    for i, line in enumerate(fh):
        rows.append(json.loads(line))
        if i == MAX_ROWS - 1:  # read only first n lines
            break

df = pd.DataFrame(rows)
end = time.time()
print(f'{(end - start)} seconds taken')
# ---------------------------------------------------------------------
# 3. Display basic info
# ---------------------------------------------------------------------
#print("\n=== Columns & dtypes ===")
#print(df.info())

#print("\n=== Head (first 5 rows) ===")
#print(df.head())
df.to_csv(fr"C:\Users\Andy\Downloads\dataset.csv", index = False)
b = 1
