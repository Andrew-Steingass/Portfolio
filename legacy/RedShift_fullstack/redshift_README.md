
# ✅ Summary: Uploading a CSV to Amazon Redshift from Local Machine

## 🔧 Setup

### IAM Role Configuration
- Role type: AmazonRedshift-CommandsAccessRole
- Policies attached: AmazonS3ReadOnlyAccess
- Configure IAM_ROLE_ARN in .env file
- Grant role to your Redshift workgroup


## 🐍 Python S3 Upload

- `.env` stores AWS credentials:
  ```env
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  ```
- Python script used `boto3`:
  ```python
  s3.upload_file(
      Filename="C:/Users/Andy/Downloads/dataset.csv",
      Bucket="",
      Key="uploads/dataset.csv"
  )
  ```

## 🛢️ Redshift Setup

### 1. Workgroup: `msds-434`

- Public access **enabled**
- Noted host: `msds-434.<acct>.us-east-2.redshift-serverless.amazonaws.com`

### 2. Security Group

- Group ID: ``
- Inbound rule added:
  - Type: `Custom TCP`
  - Port: `5439`
  - Source: `My IP`

## 🐘 Python Redshift Load

- `.env` contains:
  ```env
  redshift_db_host=msds-434.<acct>.us-east-2.redshift-serverless.amazonaws.com
  redshift_db_user=admin
  redshift_db_password=...
  IAM_ROLE_ARN=arn:aws:iam::...:role/...
  ```
- Python uses `psycopg2` to:
  - Create table `product_pairs`
  - Load data using `COPY FROM s3`

## ✅ Status

- Upload to S3: **Success**
- COPY to Redshift: **In Progress** (connection now possible)

Let me know when you're ready for the NLP step.
