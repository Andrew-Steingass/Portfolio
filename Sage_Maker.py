# SageMaker AutoML Demo for Text Classification
# This is a demo of using AWS SageMaker Autopilot to automatically build 
# a text classification model. It takes two text columns and predicts if 
# they're a valid match (1) or not (0).

import os
import pandas as pd
import boto3
import sagemaker
import time
from sagemaker.automl.automl import AutoML
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

# Configuration - change these to match your setup
bucket_name = os.getenv('SAGEMAKER_BUCKET', 'your-bucket-name-here')
endpoint_name = os.getenv('SAGEMAKER_ENDPOINT', None)

print("Starting SageMaker AutoML demo...")

# Step 1: Load and prepare the data
print("Loading dataset...")
df = pd.read_csv('dataset.csv')
df = df[['label', 'title_right', 'title_left']]  # just the columns we need
print(f"Got {len(df)} rows of data")

# Save it as a CSV for AutoML
df.to_csv('autopilot_data.csv', index=False)

# Upload to S3 (AutoML needs data in S3)
print("Uploading data to S3...")
s3 = boto3.client('s3')
s3.upload_file('autopilot_data.csv', bucket_name, 'autopilot_data.csv')
data_location = f"s3://{bucket_name}/autopilot_data.csv"
print(f"Data uploaded to: {data_location}")

# Step 2: Set up SageMaker
print("Setting up SageMaker...")
session = sagemaker.Session()
role = sagemaker.get_execution_role()
print(f"Using execution role: {role}")

# Step 3: Create the AutoML job
output_location = f"s3://{bucket_name}/automl-results"

print("Creating AutoML job...")
automl_job = AutoML(
    role=role,
    target_attribute_name="label",  # what we want to predict
    output_path=output_location,
    max_candidates=1  # just one model for this demo
)

print("Starting AutoML training (this takes a while)...")
automl_job.fit(inputs=data_location, wait=False, logs=False)

# Get the job name so we can check on it
job_name = automl_job.latest_auto_ml_job.job_name
print(f"Job name: {job_name}")

# Step 4: Wait for the job to finish
print("Checking job status every minute...")
sm_client = boto3.client("sagemaker")

while True:
    job_info = sm_client.describe_auto_ml_job(AutoMLJobName=job_name)
    status = job_info["AutoMLJobStatus"]
    
    current_time = time.strftime("%H:%M:%S")
    print(f"{current_time} - Status: {status}")
    
    if status in ("Completed", "Failed", "Stopped"):
        print(f"Job finished with status: {status}")
        break
    
    time.sleep(60)  # wait a minute before checking again

# Step 5: If successful, get info about the best model
if status == "Completed":
    print("Getting details about the best model...")
    
    best_model = job_info["BestCandidate"]
    model_name = best_model["CandidateName"]
    accuracy = best_model["FinalAutoMLJobObjectiveMetric"]["Value"]
    metric_name = best_model["FinalAutoMLJobObjectiveMetric"]["MetricName"]
    
    print(f"Best model: {model_name}")
    print(f"{metric_name}: {accuracy}")
    
    # Step 6: Deploy the model
    print("Deploying model to an endpoint...")
    predictor = automl_job.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )
    
    deployed_endpoint = predictor.endpoint_name
    print(f"Model deployed to endpoint: {deployed_endpoint}")
    
    # Step 7: Test the model
    print("Testing the model with some sample data...")
    
    # Set up the predictor for CSV format (AutoML uses CSV, not JSON)
    test_predictor = Predictor(
        endpoint_name=deployed_endpoint,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer()
    )
    
    # Test with a few samples from our data
    test_data = df.drop(columns=["label"])  # remove the label column
    
    for i in range(min(3, len(df))):  # test up to 3 samples
        sample = test_data.iloc[i:i+1]
        actual = df.iloc[i]["label"]
        
        print(f"\nTest {i+1}:")
        print(f"Input: {sample.iloc[0].to_dict()}")
        print(f"Actual label: {actual}")
        
        prediction = test_predictor.predict(sample.values)
        print(f"Model prediction: {prediction}")
    
    print("\nDemo complete!")
    print(f"Your endpoint name is: {deployed_endpoint}")
    print("Don't forget to delete the endpoint when you're done to avoid charges:")
    print("predictor.delete_endpoint()")
    
else:
    print(f"AutoML job failed. Status: {status}")
    print("Check the SageMaker console for more details.")

# Utility function to clean up later
def cleanup():
    """Call this to delete the endpoint and avoid charges"""
    if 'predictor' in globals():
        predictor.delete_endpoint()
        print("Endpoint deleted")
    else:
        print("No predictor found to clean up")

# Function to list all your endpoints
def show_endpoints():
    """Show all your SageMaker endpoints"""
    sm = boto3.client("sagemaker")
    endpoints = sm.list_endpoints(SortBy="CreationTime", SortOrder="Descending")
    
    print("Your SageMaker endpoints:")
    for ep in endpoints["Endpoints"]:
        print(f"  {ep['EndpointName']} ({ep['EndpointStatus']})")

print("\nUse cleanup() to delete the endpoint when you're done")
print("Use show_endpoints() to see all your endpoints")