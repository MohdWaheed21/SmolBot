from flask import Flask, render_template, request, jsonify
import cv2
import base64
import requests
import threading
import time

app = Flask(__name__)

# Configuration
BASE_URL = "http://localhost:8080"
INTERVAL = 0.5  # seconds
INSTRUCTION = "What do you see?"
PROCESSING = False
CAMERA = None

def capture_frame():
    global CAMERA
    if CAMERA is None or not CAMERA.isOpened():
        return None
    
    ret, frame = CAMERA.read()
    if not ret:
        return None
    
    # Convert frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def send_to_smolvlm(instruction, image_base64):
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "ggml-org/SmolVLM-500M-Instruct-GGUF",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 100
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def processing_loop():
    global PROCESSING
    while PROCESSING:
        start_time = time.time()
        
        frame_base64 = capture_frame()
        if frame_base64:
            instruction = INSTRUCTION
            response = send_to_smolvlm(instruction, frame_base64)
            # In a real app, you'd send this to the frontend via WebSocket or similar
            
        # Sleep for the remaining interval time
        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_processing():
    global PROCESSING, CAMERA, INSTRUCTION, INTERVAL
    
    if not PROCESSING:
        CAMERA = cv2.VideoCapture(0)
        if not CAMERA.isOpened():
            return jsonify({"error": "Could not open camera"}), 400
        
        PROCESSING = True
        INSTRUCTION = request.json.get('instruction', INSTRUCTION)
        INTERVAL = float(request.json.get('interval', INTERVAL))
        
        thread = threading.Thread(target=processing_loop)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/stop', methods=['POST'])
def stop_processing():
    global PROCESSING, CAMERA
    
    if PROCESSING:
        PROCESSING = False
        if CAMERA:
            CAMERA.release()
            CAMERA = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})

@app.route('/frame', methods=['GET'])
def get_frame():
    frame_base64 = capture_frame()
    if frame_base64:
        return jsonify({"frame": frame_base64})
    return jsonify({"error": "Could not capture frame"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)