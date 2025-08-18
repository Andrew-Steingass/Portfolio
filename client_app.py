from flask import Flask, request, render_template_string, jsonify
import requests
import uuid
import threading
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Store responses in memory (in production, use a database)
responses = {}

# Configuration - change these to match your setup
OLLAMA_API_URL = f"http://{os.environ['ML_server_ip']}:5000"
CALLBACK_BASE_URL = f"http://{os.environ['my_ip']}:3000"
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { padding: 10px 20px; font-size: 16px; }
        .response { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .loading { color: #666; font-style: italic; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>AI Chat Interface</h1>
    
    <form onsubmit="sendPrompt(); return false;">
        <textarea id="prompt" placeholder="Enter your prompt here..."></textarea><br>
        <input type="text" id="model" value="gpt-oss:20b" placeholder="Model name"><br><br>
        <button type="submit">Send to AI</button>
    </form>
    
    <div id="responses"></div>
    
    <script>
        function sendPrompt() {
            const prompt = document.getElementById('prompt').value;
            const model = document.getElementById('model').value;
            
            if (!prompt.trim()) {
                alert('Please enter a prompt');
                return;
            }
            
            const requestId = 'req_' + Date.now();
            
            // Add loading message
            addResponse(requestId, prompt, 'Thinking...', 'loading');
            
            // Send request
            fetch('/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    prompt: prompt,
                    model: model,
                    request_id: requestId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateResponse(requestId, 'Error: ' + data.error, 'error');
                }
            })
            .catch(error => {
                updateResponse(requestId, 'Error sending request: ' + error, 'error');
            });
            
            // Clear the form
            document.getElementById('prompt').value = '';
        }
        
        function addResponse(requestId, prompt, response, className = '') {
            const div = document.createElement('div');
            div.className = 'response ' + className;
            div.id = 'response_' + requestId;
            div.innerHTML = `
                <strong>Prompt:</strong> ${prompt}<br>
                <strong>Response:</strong> <span id="resp_${requestId}">${response}</span>
            `;
            document.getElementById('responses').prepend(div);
        }
        
        function updateResponse(requestId, response, className = '') {
            const respElement = document.getElementById('resp_' + requestId);
            const divElement = document.getElementById('response_' + requestId);
            if (respElement && divElement) {
                respElement.textContent = response;
                divElement.className = 'response ' + className;
            }
        }
        
        // Poll for responses every 2 seconds
        setInterval(() => {
            fetch('/check_responses')
                .then(response => response.json())
                .then(data => {
                    Object.keys(data).forEach(requestId => {
                        const response = data[requestId];
                        updateResponse(requestId, response.response);
                    });
                });
        }, 2000);
    </script>
</body>
</html>
    ''')

@app.route('/send', methods=['POST'])
def send_to_ai():
    data = request.get_json()
    
    try:
        # Prepare request for your FastAPI server
        payload = {
            "prompt": data['prompt'],
            "callback_url": f"{CALLBACK_BASE_URL}/callback",
            "request_id": data['request_id'],
            "model": data['model'],
            "options": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        # Send to your FastAPI server
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json=payload,
            timeout=180
        )
        
        if response.status_code == 200:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": f"Server error: {response.status_code}"})
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Connection error: {str(e)}"})

@app.route('/callback', methods=['POST'])
def receive_callback():
    """Receive the AI response from your FastAPI server"""
    data = request.get_json()
    
    # Store the response
    request_id = data.get('request_id')
    if request_id:
        responses[request_id] = data
        print(f"Received response for {request_id}: {data.get('response', '')[:100]}...")
    
    return jsonify({"status": "received"})

@app.route('/check_responses')
def check_responses():
    """Return new responses and clear them"""
    current_responses = responses.copy()
    responses.clear()
    return jsonify(current_responses)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("Starting client app...")
    print(f"Make sure your FastAPI server is running at: {OLLAMA_API_URL}")
    print(f"This app will run at: http://localhost:3000")
    print("\nOpen http://localhost:3000 in your browser")
    
    app.run(host='0.0.0.0', port=3000, debug=True)