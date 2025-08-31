# Ollama FastAPI Middleware

FastAPI service that adds callback support to Ollama. Send a prompt, get the response delivered to your callback URL.

## What it does

```
Your App → FastAPI → Ollama → FastAPI → Your Callback URL
```

## Quick Start

1. **Install:**
   ```bash
   pip install fastapi uvicorn httpx pydantic python-dotenv
   ```

2. **Configure (.env file):**
   ```bash
   OLLAMA_BASE_URL=your_ollama_url_here
   API_HOST=your_host_here
   API_PORT=your_port_here
   OLLAMA_REQUEST_TIMEOUT=your_timeout_here
   CALLBACK_TIMEOUT=your_callback_timeout_here
   ```

3. **Run:**
   ```bash
   uvicorn ollama_app_main:app --host 0.0.0.0 --port 5000
   ```

## Usage

**Send request:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write hello world in Python",
    "callback_url": "https://your-app.com/callback",
    "request_id": "123",
    "model": "gpt-oss:20b",
    "options": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }'
```

**Your callback URL receives:**
```json
{
  "request_id": "123",
  "prompt": "Write hello world in Python",
  "response": "print('Hello, World!')",
  "model": "gpt-oss:20b",
  "processing_time": 2.5,
  "timestamp": "2025-08-09T18:30:00",
  "status": "success"
}
```

## Endpoints

- `GET /health` - Check if service and Ollama are running
- `GET /docs` - Interactive API documentation  
- `POST /generate` - Generate text with callback

## Requirements

- Ollama running locally
- Model loaded in Ollama (e.g., `ollama pull gpt-oss:20b`)
- All environment variables in `.env` file (required - will crash if missing)