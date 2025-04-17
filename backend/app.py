from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys
import logging
import cv2
import numpy as np
import time
import json
import threading

# Add the yolov8_model directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Remove unused photo detection import
# from yolov8_model.detecting_images import detect_objects_in_photo 
# Remove YOLO class import
# from ultralytics import YOLO

# Import the MMDetection functions
from CCTV_GUN.detecting_images import detect_gun_in_image, process_cctv_gun_video_stream

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
sock = Sock(app)

# Configure folders
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BACKEND_DIR, 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
UPLOADS_DIR = os.path.join(IMAGES_DIR, 'uploads')
ANNOTATED_DIR = os.path.join(IMAGES_DIR, 'annotated')

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Print the actual paths for debugging
print(f"Static directory: {STATIC_DIR}")
print(f"Images directory: {IMAGES_DIR}")
print(f"Uploads directory: {UPLOADS_DIR}")
print(f"Annotated directory: {ANNOTATED_DIR}")

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', './yolov8_model/runs/detect/Normal_Compressed/weights/best.pt')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.3'))

# Store connected WebSocket clients
connected_clients = set()

# Add video path constant
VIDEO_PATH = "/Users/aryan98/threat-detection/backend/streaming/test/test_video_3.mov"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@sock.route('/ws')
def handle_websocket(ws):
    connected_clients.add(ws)
    try:
        while True:
            # Keep the connection alive
            ws.receive()
    except:
        connected_clients.remove(ws)

def notify_clients(data):
    """Notify all connected clients about new detection"""
    for client in connected_clients:
        try:
            client.send(json.dumps({
                'type': 'new_detection',
                'payload': data
            }))
        except:
            connected_clients.remove(client)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({"status": "healthy"}), 200

@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Handle image uploads and run object detection.
    Expects a multipart form with an 'image' field.
    Returns detection results and saves annotated image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOADS_DIR, filename)
        print(f"Saving uploaded file to: {upload_path}")
        file.save(upload_path)
        
        if not os.path.exists(upload_path):
            logger.error(f"Failed to save file at: {upload_path}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
        
        # Call the MMDetection function
        logger.info(f"Calling detect_gun_in_image for: {upload_path}")
        detection_result = detect_gun_in_image(upload_path)

        if detection_result is None:
            logger.error("Gun detection failed for image: {upload_path}")
            return jsonify({"error": "Detection processing failed."}), 500

        # Prepare response
        detections = detection_result.get('detections', [])
        relative_annotated_path = detection_result.get('annotated_image_path', None)

        # Get image shape (optional, could read image again or pass it to the function)
        try:
            image_shape = cv2.imread(upload_path).shape
        except Exception as e:
            logger.warning(f"Could not read image {upload_path} to get shape: {e}")
            image_shape = None

        response = {
            "message": "Detection completed successfully using MMDetection",
            "detections": detections,
            "image_info": {
                "shape": list(image_shape) if image_shape else None
            },
            "model_info": {
                "type": "MMDetection",
                "classes": {
                    "1": "gun"
                }
            },
            "original_image": os.path.join('images', 'uploads', filename),
            "annotated_image": os.path.join('images', relative_annotated_path) if relative_annotated_path else None
        }

        logger.info(f"Detection successful. Response: {response}")

        # Notify all connected clients
        notify_clients({
            "type": "new_detection",
            "payload": response
        })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in detect_objects endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/static/images/<path:filename>')
def serve_static(filename):
    """Serve static files from the images directory"""
    print(f"Serving static file: {filename} from {IMAGES_DIR}")
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except Exception as e:
        print(f"Error serving static file: {e}")
        return str(e), 404

# Add new WebSocket endpoint for video
@sock.route('/ws/video')
def handle_video_stream(ws):
    logger.info("New video WebSocket connection established")
    try:
        # Start MMDetection video processing in a separate thread
        # Pass the hardcoded VIDEO_PATH for now
        logger.info(f"Starting MMDetection video processing thread for: {VIDEO_PATH}")
        thread = threading.Thread(target=process_cctv_gun_video_stream, args=(VIDEO_PATH, ws))
        thread.daemon = True # Allow app to exit even if thread is running
        thread.start()

        # Keep the WebSocket connection alive simply by letting the thread run.
        # The connection will remain open until the client disconnects or the thread finishes/errors.
        # Removed the ws.receive() loop which was causing premature timeout.
        # while True:
        #     message = ws.receive(timeout=60) # Add a timeout
        #     if message is None: # Handle potential timeout or client disconnect
        #          logger.info("WebSocket keep-alive check: No message received, connection might be closing.")
        #          break
        #     # Optionally process client messages here if needed
        
        # Wait for the processing thread to finish (optional, but keeps the endpoint alive)
        # Or rely on the client disconnecting to terminate the context
        thread.join() # Wait here until the video processing thread completes
        logger.info("Video processing thread finished.")

    except Exception as e:
        # Catch specific WebSocket errors if possible, e.g., ConnectionClosed
        logger.error(f"Error in video WebSocket handler: {str(e)}", exc_info=True)
    finally:
        logger.info("Video WebSocket connection closed")

if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5001))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Run the app
    app.run(host=host, port=port, debug=debug) 