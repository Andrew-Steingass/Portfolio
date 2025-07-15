import boto3
from dotenv import load_dotenv
import os
import psycopg2
import pandas as pd

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
s3.upload_file(
    Filename=r"C:\Users\Andy\Downloads\dataset.csv",
    Bucket=os.getenv('s3_bucketname'),
    Key="uploads/dataset.csv"
)
print("Uploaded to s3 bucket")


resp = s3.head_object(
    Bucket=os.getenv('s3_bucketname'),
    Key="uploads/dataset.csv",
)
print("✓ Found in S3 — size:", resp["ContentLength"], "bytes")
##########################################

df = pd.read_csv(fr"C:\Users\Andy\Downloads\dataset.csv")
print(df.dtypes)
conn = psycopg2.connect(
    host=os.getenv("redshift_db_host"),
    dbname = 'dev',
    user=os.getenv("redshift_db_user"),
    password=os.getenv("redshift_db_password"),
    port=5439,
)
cur = conn.cursor()

create_sql = """
CREATE TABLE IF NOT EXISTS product_pairs (
    id_left BIGINT,
    category_left VARCHAR(255),
    cluster_id_left BIGINT,
    id_right BIGINT,
    category_right VARCHAR(255),
    cluster_id_right BIGINT,
    label SMALLINT,
    pair_id VARCHAR(255),
    brand_left VARCHAR(255),
    brand_right VARCHAR(255),
    description_left VARCHAR(MAX),
    description_right VARCHAR(MAX),
    keyValuePairs_left VARCHAR(MAX),
    keyValuePairs_right VARCHAR(MAX),
    price_left VARCHAR(50),
    price_right VARCHAR(50),
    specTableContent_left VARCHAR(MAX),
    specTableContent_right VARCHAR(MAX),
    title_left VARCHAR(MAX),
    title_right VARCHAR(MAX)
);
"""
cur.execute(create_sql)
conn.commit()
print()
copy_sql = f"""
    COPY product_pairs
    FROM 's3://{os.getenv('s3_bucketname')}/uploads/dataset.csv'
    IAM_ROLE '{os.getenv("IAM_ROLE_ARN")}'
    REGION 'us-east-1'          
    CSV
    IGNOREHEADER 1
    DELIMITER ','
    QUOTE '"';
    """
try:
    cur.execute(copy_sql)
    conn.commit()
    b =1
except:
    conn.rollback()
    cur.execute("SELECT * FROM sys_load_error_detail ORDER BY start_time DESC LIMIT 10;")
    errors = cur.fetchall()
    for error in errors:
        print("Error details:", error)
    b = 1



cur.close()
conn.close()

print("COPY complete.")

