from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any
import httpx
import logging
import time
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama API Middleware",
    description="FastAPI middleware for Ollama LLM inference with callback support",
    version="1.0.0"
)

# Pydantic models for request/response validation
class GenerationOptions(BaseModel):
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=4000)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the LLM")
    callback_url: HttpUrl = Field(..., description="URL to send the response back to")
    request_id: str = Field(..., min_length=1, description="Unique identifier for this request")
    model: str = Field(..., description="Ollama model to use")
    options: Optional[GenerationOptions] = Field(default_factory=GenerationOptions)

class SuccessResponse(BaseModel):
    status: str
    request_id: str
    message: str

class CallbackResponse(BaseModel):
    request_id: str
    prompt: str
    response: str
    model: str
    processing_time: float
    timestamp: str
    status: str

class ErrorCallbackResponse(BaseModel):
    request_id: str
    error: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    timestamp: str

# Required configuration from environment variables (will crash if missing)
OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
API_HOST = os.environ["API_HOST"] 
API_PORT = int(os.environ["API_PORT"])
REQUEST_TIMEOUT = float(os.environ["OLLAMA_REQUEST_TIMEOUT"])
CALLBACK_TIMEOUT = float(os.environ["CALLBACK_TIMEOUT"])

@app.post("/generate", response_model=SuccessResponse)
async def generate(request: GenerationRequest):
    """
    Generate text using Ollama and send response to callback URL
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing request {request.request_id} for model {request.model}")
        
        # Prepare Ollama payload for chat format
        ollama_payload = {
            "model": request.model,
            "messages": [
                {"role": "user", "content": request.prompt}
            ],
            "stream": False
        }
        
        # Add options if provided
        if request.options:
            ollama_options = {}
            if request.options.temperature is not None:
                ollama_options["temperature"] = request.options.temperature
            if request.options.max_tokens is not None:
                ollama_options["num_predict"] = request.options.max_tokens
            
            if ollama_options:
                ollama_payload["options"] = ollama_options
        
        # Send request to Ollama using httpx (async HTTP client)
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=ollama_payload
                )
                response.raise_for_status()
                
            except httpx.TimeoutException:
                await send_error_callback(
                    request.callback_url,
                    request.request_id,
                    "Ollama request timeout"
                )
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail="Request timeout"
                )
                
            except httpx.RequestError as e:
                await send_error_callback(
                    request.callback_url,
                    request.request_id,
                    f"Ollama connection error: {str(e)}"
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Ollama service unavailable"
                )
        
        # Parse Ollama response
        ollama_result = response.json()
        
        if 'message' not in ollama_result or 'content' not in ollama_result['message']:
            await send_error_callback(
                request.callback_url,
                request.request_id,
                "Invalid response from Ollama"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid Ollama response"
            )
        
        # Prepare callback response
        processing_time = time.time() - start_time
        callback_data = CallbackResponse(
            request_id=request.request_id,
            prompt=request.prompt,
            response=ollama_result['message']['content'],
            model=request.model,
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
        # Send callback (don't wait for it to complete)
        asyncio.create_task(send_success_callback(request.callback_url, callback_data))
        
        return SuccessResponse(
            status="success",
            request_id=request.request_id,
            message="Response sent to callback URL"
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions (they're already handled)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error for request {request.request_id}: {str(e)}")
        
        # Try to send error callback
        try:
            await send_error_callback(
                request.callback_url,
                request.request_id,
                f"Internal server error: {str(e)}"
            )
        except Exception as callback_error:
            logger.error(f"Failed to send error callback: {str(callback_error)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

async def send_success_callback(callback_url: HttpUrl, data: CallbackResponse):
    """Send success response to callback URL"""
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            response = await client.post(
                str(callback_url),
                json=data.dict(),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Callback sent successfully to {callback_url}")
            
    except httpx.TimeoutException:
        logger.error(f"Callback timeout to {callback_url}")
        
    except httpx.RequestError as e:
        logger.error(f"Callback failed to {callback_url}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected callback error: {str(e)}")

async def send_error_callback(callback_url: HttpUrl, request_id: str, error_message: str):
    """Send error response to callback URL"""
    try:
        error_data = ErrorCallbackResponse(
            request_id=request_id,
            error=error_message,
            timestamp=datetime.now().isoformat()
        )
        
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            response = await client.post(
                str(callback_url),
                json=error_data.dict(),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Error callback sent to {callback_url}")
            
    except Exception as e:
        logger.error(f"Failed to send error callback: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        ollama_status = "unreachable"
    
    return HealthResponse(
        status="healthy",
        ollama_status=ollama_status,
        timestamp=datetime.now().isoformat()
    )

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to fetch models: {str(e)}"
        )

# Optional: Add CORS middleware if you need cross-origin requests
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False
    )