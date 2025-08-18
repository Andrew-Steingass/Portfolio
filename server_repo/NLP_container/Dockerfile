FROM python:3.12-slim
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY NLP.py .
COPY new_nlp_pull.py .

# Copy .env file
COPY .env .

# Create directory for results
RUN mkdir -p /app/results

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "NLP.py"]