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
from yolov8_model.detecting_images import process_video_stream
import threading

# Add the yolov8_model directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolov8_model.detecting_images import detect_objects_in_photo
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

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
            print(f"Failed to save file at: {upload_path}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
        
        # Read and process the image
        image = cv2.imread(upload_path)
        if image is None:
            print(f"Failed to read image from: {upload_path}")
            return jsonify({"error": "Failed to read image"}), 400
        
        # Run detection
        results = model(image, conf=DETECTION_THRESHOLD)
        
        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    "bbox": {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2
                    },
                    "class": "guns" if cls == 0 else "knife",
                    "confidence": conf
                })
        
        # Save annotated image
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)
        print(f"Saving annotated image to: {annotated_path}")
        cv2.imwrite(annotated_path, results[0].plot())
        
        if not os.path.exists(annotated_path):
            print(f"Failed to save annotated image at: {annotated_path}")
            return jsonify({"error": "Failed to save annotated image"}), 500
        
        # Prepare response with relative paths
        response = {
            "message": "Detection completed successfully",
            "detections": detections,
            "image_info": {
                "shape": list(image.shape)
            },
            "model_info": {
                "classes": {
                    "0": "guns",
                    "1": "knife"
                }
            },
            "original_image": os.path.join('uploads', filename),
            "annotated_image": os.path.join('annotated', annotated_filename)
        }
        
        print(f"Detection successful. Response: {response}")
        
        # Notify all connected clients
        notify_clients(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
        # Start video processing in a separate thread
        thread = threading.Thread(target=process_video_stream, args=(VIDEO_PATH, ws))
        thread.start()
        
        # Keep the WebSocket connection alive
        while True:
            ws.receive()
            
    except Exception as e:
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