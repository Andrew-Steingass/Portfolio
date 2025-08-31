FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files (note the trailing slash)
COPY *.py ./

# Copy templates directory
COPY templates/ templates/

EXPOSE 8000

CMD ["python", "app.py"]