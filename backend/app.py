from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys
import logging
import cv2
# import numpy as np # Not directly used in this simplified version, cv2 uses it
import time
import json
import threading
# import base64 # Not used after removing video streaming logic temporarily

# --- Project Structure Setup ---
# Get the directory of the current script (app.py)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct path to the CCTV_GUN directory relative to app.py
CCTV_GUN_DIR = os.path.join(BACKEND_DIR, 'CCTV_GUN')
# Add CCTV_GUN directory to sys.path to allow imports
sys.path.insert(0, CCTV_GUN_DIR) # Insert at beginning to ensure it's checked first

try:
    # Import the MMDetection functions AFTER modifying sys.path
    from detecting_images import detect_gun_in_image, process_cctv_gun_video_stream
    print("Successfully imported detection functions from CCTV_GUN.")
except ImportError as e:
    print(f"Error importing detection functions from CCTV_GUN: {e}")
    print("Please ensure 'detecting_images.py' exists in the 'CCTV_GUN' directory adjacent to 'app.py'")
    print(f"and that all dependencies in the 'env_cc' conda environment are installed and the environment is active.")
    sys.exit(1) # Exit if core functionality cannot be imported
# --- ---

# Configure logging
# Using DEBUG for more verbose output during development
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (if any are needed for Flask config, API keys etc.)
load_dotenv()

# Initialize Flask app and WebSocket extension
app = Flask(__name__)
sock = Sock(app)

# --- Configure Folders ---
STATIC_DIR = os.path.join(BACKEND_DIR, 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
UPLOADS_DIR = os.path.join(IMAGES_DIR, 'uploads')
# Annotated images are saved relative to static/images by detecting_images.py

# Create directories if they don't exist
try:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    logger.info(f"Ensured uploads directory exists: {UPLOADS_DIR}")
except OSError as e:
    logger.error(f"Failed to create uploads directory {UPLOADS_DIR}: {e}", exc_info=True)
    sys.exit(1)

# Log the key paths for debugging setup issues
logger.debug(f"Backend directory: {BACKEND_DIR}")
logger.debug(f"Static directory: {STATIC_DIR}")
logger.debug(f"Uploads directory: {UPLOADS_DIR}")

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Test Video Path (Replace with dynamic handling later) ---
# !! IMPORTANT: This is hardcoded for testing. Needs replacement for production/flexibility !!
TEST_VIDEO_PATH = "/workspace/projects/threat-detector/backend/test/input/test_video_4.mov" # Updated path
logger.warning(f"Using HARDCODED video path for testing: {TEST_VIDEO_PATH}")
# --- ---


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- WebSocket Handling ---
# Store clients connected specifically for video streaming notifications
video_clients = set()

@sock.route('/ws/video_feed') # Changed route slightly for clarity
def handle_video_feed_ws(ws):
    """
    Handles WebSocket connection for streaming video detection results.
    Starts a background thread to process the video and send results back via this ws connection.
    """
    logger.info(f"New video feed WebSocket connection established: {ws}")
    video_clients.add(ws)
    processing_thread = None
    try:
        # --- Start Video Processing Thread ---
        video_to_process = TEST_VIDEO_PATH
        if not os.path.exists(video_to_process):
             logger.error(f"Video file specified for streaming not found: {video_to_process}")
             try:
                 ws.send(json.dumps({"type": "error", "payload": "Video file not found on server."}))
             except Exception as send_err:
                 logger.error(f"Failed to send error to WebSocket client: {send_err}")
             return

        logger.info(f"Starting CCTV_GUN video processing thread for: {video_to_process}")
        processing_thread = threading.Thread(
            target=process_cctv_gun_video_stream,
            args=(video_to_process, ws), # Pass ws object
            daemon=True
        )
        processing_thread.start()
        logger.info(f"Video processing thread started (Thread ID: {processing_thread.ident}).")

        # --- Keep Connection Alive ---
        while True:
            try:
                message = ws.receive(timeout=60)
                if message is None:
                    if processing_thread and not processing_thread.is_alive():
                         logger.warning("Processing thread is no longer alive. Closing WebSocket.")
                         break
                    logger.debug("Video feed WebSocket keep-alive check passed.")
                    continue
                else:
                    logger.debug(f"Received message on video feed ws: {message}")
            except Exception as receive_err:
                 logger.info(f"WebSocket receive error or client disconnected: {receive_err}")
                 break

    except Exception as e:
        logger.error(f"Video feed WebSocket handler error: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Video feed WebSocket connection closing: {ws} (Thread ID: {processing_thread.ident if processing_thread else 'N/A'}).")
        video_clients.discard(ws)


# --- REST API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested.")
    return jsonify({"status": "healthy"}), 200

@app.route('/detect', methods=['POST'])
def detect_objects_api():
    """
    API endpoint to handle single image uploads and run MMDetection gun detection.
    """
    logger.info("Received request for /detect endpoint.")
    if 'image' not in request.files:
        logger.warning("'/detect' call missing 'image' file part.")
        return jsonify({"error": "No image file part provided"}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("'/detect' call with empty filename.")
        return jsonify({"error": "No image selected"}), 400

    if not allowed_file(file.filename):
         logger.warning(f"'/detect' call with disallowed file type: {file.filename}")
         return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

    upload_path = None
    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOADS_DIR, filename)
        logger.info(f"Saving uploaded file to: {upload_path}")
        file.save(upload_path)

        if not os.path.exists(upload_path):
            logger.error(f"Failed to save file after call to save(): {upload_path}")
            return jsonify({"error": "Failed to save uploaded file"}), 500

        # --- Call MMDetection Function ---
        logger.info(f"Calling detect_gun_in_image (from CCTV_GUN) for: {upload_path}")
        detection_result = detect_gun_in_image(upload_path) # Pass absolute path
        # --- ---

        if detection_result is None:
            logger.error(f"Gun detection function returned None for image: {upload_path}")
            return jsonify({"error": "Detection processing failed."}), 500

        detections = detection_result.get('detections', [])
        relative_annotated_path = detection_result.get('annotated_image_path') # Path relative to static/images

        base_url = request.host_url.rstrip('/')
        original_image_url = f"{base_url}/static/images/uploads/{filename}" if filename else None
        annotated_image_url = f"{base_url}/static/images/{relative_annotated_path}" if relative_annotated_path else None

        response = {
            "message": "Detection completed successfully using CCTV_GUN (MMDetection)",
            "detections": detections,
            "original_image_url": original_image_url,
            "annotated_image_url": annotated_image_url,
             "model_info": {
                "type": "MMDetection (ConvNeXt based)",
                "classes": { "1": "gun" } # Based on current model assumption
            },
            "_debug_paths": { # For debugging, remove in production
                 "upload_abs": upload_path,
                 "annotated_rel": relative_annotated_path
             }
        }

        logger.info(f"Detection successful. Detections found: {len(detections)}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /detect endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred"}), 500

# --- Static File Serving ---
# Serve files from static/images (includes uploads and annotated images)
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve static files (images) from the static/images directory."""
    logger.debug(f"Request to serve static image: {filename}")
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except FileNotFoundError:
         logger.warning(f"Static file not found: {filename} in {IMAGES_DIR}")
         return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}", exc_info=True)
        return jsonify({"error": "Error serving file"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5001))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() in ['true', '1', 't']

    logger.info(f"Starting Flask server on {host}:{port} (Debug Mode: {debug_mode})")
    # Disable reloader if not in debug mode, as it can cause issues with threads/GPU context
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode) 