from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys
import logging
import json
import threading

# --- Import New Modules ---
# Assuming app.py is run from workspace root or PYTHONPATH is set correctly
# Changed to direct imports assuming script is run from backend/ directory
from yolo_model.yolo_processor import initialize_yolo
from predictions.dual_model_processor import DualModelProcessor

# --- Project Structure Setup ---
# Get the directory of the current script (app.py)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct path to the CCTV_GUN directory relative to app.py
CCTV_GUN_DIR = os.path.join(BACKEND_DIR, 'CCTV_GUN')
# Add CCTV_GUN directory to sys.path to allow imports (still needed for detect_gun_in_image and constants)
sys.path.insert(0, CCTV_GUN_DIR) # Insert at beginning to ensure it's checked first

try:
    # Import only what's needed by app.py now (detect_gun_in_image for API, constants for processor)
    from detecting_images import (
        detect_gun_in_image,
        DEVICE,
        CONFIG_PATH,
        CHECKPOINT_PATH
    )
    print("Successfully imported functions, DEVICE, paths from CCTV_GUN.")
except ImportError as e:
    print(f"Error importing from CCTV_GUN: {e}")
    print("Please ensure 'detecting_images.py' exists in the 'CCTV_GUN' directory adjacent to 'app.py',")
    print(f"that it defines DEVICE, CONFIG_PATH, CHECKPOINT_PATH and the function detect_gun_in_image,")
    print(f"and that all dependencies are installed and the environment is active.")
    sys.exit(1) # Exit if core functionality cannot be imported
# --- ---

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app and WebSocket extension
app = Flask(__name__)
sock = Sock(app)

# --- Initialize Models ---
YOLO_MODEL_PATH = os.path.join(BACKEND_DIR, "yolo_model", "yolo11m.pt")
logger.info("Attempting to initialize YOLO model via yolo_processor...")
# Ensure DEVICE is available from the import above
yolo_model = initialize_yolo(YOLO_MODEL_PATH, DEVICE)
if yolo_model is None:
    logger.error("YOLO model initialization failed. Video processing will be unavailable.")
else:
    logger.info("YOLO model successfully initialized via yolo_processor.")
# MMDet model is initialized within the DualModelProcessor instance per stream
# --- ---

# --- Processing Parameters ---
# These will be passed to the DualModelProcessor instance
HIGH_CONFIDENCE_THRESHOLD = 0.7
MMDET_FRAME_SKIP = 20
YOLO_IMG_SIZE = 320
# --- ---

# --- Configure Folders ---
STATIC_DIR = os.path.join(BACKEND_DIR, 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
UPLOADS_DIR = os.path.join(IMAGES_DIR, 'uploads')

# Create directories if they don't exist
try:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    logger.info(f"Ensured uploads directory exists: {UPLOADS_DIR}")
except OSError as e:
    logger.error(f"Failed to create uploads directory {UPLOADS_DIR}: {e}", exc_info=True)
    sys.exit(1)

# Log key paths
logger.debug(f"Backend directory: {BACKEND_DIR}")
logger.debug(f"Static directory: {STATIC_DIR}")
logger.debug(f"Uploads directory: {UPLOADS_DIR}")

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Test Video Path ---
TEST_VIDEO_PATH = "/workspace/projects/threat-detector/backend/test/input/test_video_4.mov"
logger.warning(f"Using HARDCODED video path for testing: {TEST_VIDEO_PATH}")
# --- ---

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- WebSocket Handling ---
video_clients = set()

@sock.route('/ws/video_feed')
def handle_video_feed_ws(ws):
    """
    Handles WebSocket connection. Instantiates DualModelProcessor and starts
    a background thread to process the video stream.
    """
    logger.info("--- WebSocket Connection Attempt Received ---")
    video_clients.add(ws)
    logger.info(f"WebSocket added. Current clients: {len(video_clients)}")
    processing_thread = None
    processor_instance = None

    try:
        # Check if YOLO model initialized successfully
        if yolo_model is None:
            logger.error("YOLO model not initialized. Cannot start video processing thread.")
            ws.send(json.dumps({"type": "error", "payload": "YOLO model failed to initialize on server."}))
            return

        video_to_process = TEST_VIDEO_PATH # Use the hardcoded path for now
        logger.info(f"Checking existence of video file: {video_to_process}")
        if not os.path.exists(video_to_process):
             logger.error(f"Video file specified for streaming not found: {video_to_process}")
             ws.send(json.dumps({"type": "error", "payload": "Video file not found on server."}))
             return

        logger.info(f"Video file found. Instantiating DualModelProcessor...")
        # Instantiate the processor, passing necessary configs
        # Ensure CONFIG_PATH, CHECKPOINT_PATH, DEVICE are available from import
        processor_instance = DualModelProcessor(
            yolo_model=yolo_model,
            static_dir=STATIC_DIR,
            high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
            mmdet_frame_skip=MMDET_FRAME_SKIP,
            yolo_img_size=YOLO_IMG_SIZE,
            mmdet_config_path=CONFIG_PATH,
            mmdet_checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE
        )
        logger.info(f"DualModelProcessor instantiated.")

        logger.info(f"Preparing to start processing thread for: {video_to_process}")
        processing_thread = threading.Thread(
            target=processor_instance.process_stream, # Target the instance method
            args=(video_to_process, ws), # Pass only video path and ws
            daemon=True
        )
        processing_thread.start()
        logger.info(f"Dual model video processing thread started (Thread ID: {processing_thread.ident}). Entering keep-alive loop.")

        # --- Keep Connection Alive ---
        while True:
            try:
                message = ws.receive(timeout=10) # Check periodically
                if message is None: # No message received
                    if processing_thread and not processing_thread.is_alive():
                        logger.info("Processing thread has finished. WebSocket closing check.")
                        break
                    continue
                else: # Message received (handle if needed)
                    logger.debug(f"Received message on video feed ws: {message}")
            except TimeoutError: # Expected timeout
                 if processing_thread and not processing_thread.is_alive():
                    logger.info("Processing thread has finished during timeout check. WebSocket closing check.")
                    break
                 continue
            except Exception as receive_err: # Other WebSocket errors
                 logger.info(f"WebSocket receive error or client disconnected: {receive_err}")
                 break

    except Exception as e:
        logger.error(f"Video feed WebSocket handler error: {str(e)}", exc_info=True)
        # Attempt to send error to client if possible
        try:
            if ws and hasattr(ws, 'connected') and ws.connected:
                 ws.send(json.dumps({"type": "error", "payload": f"WebSocket handler error: {str(e)}"}))
        except Exception as send_err:
             logger.error(f"Failed to send error to WebSocket client after handler error: {send_err}")

    finally:
        thread_id = processing_thread.ident if processing_thread else 'N/A'
        logger.info(f"Video feed WebSocket connection closing (Thread ID: {thread_id}).")
        video_clients.discard(ws)
        # Processor instance will be garbage collected

# --- REST API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested.")
    return jsonify({"status": "healthy"}), 200

@app.route('/detect', methods=['POST'])
def detect_objects_api():
    """
    API endpoint for single image gun detection using MMDetection.
    (Uses detect_gun_in_image imported from CCTV_GUN)
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

        # --- Call MMDetection Function --- (Still using the one from CCTV_GUN)
        logger.info(f"Calling detect_gun_in_image (from CCTV_GUN) for: {upload_path}")
        # Ensure detect_gun_in_image is available from import
        detection_result = detect_gun_in_image(upload_path)

        if detection_result is None:
            logger.error(f"Gun detection function returned None for image: {upload_path}")
            return jsonify({"error": "Detection processing failed."}), 500

        detections = detection_result.get('detections', [])
        relative_annotated_path = detection_result.get('annotated_image_path')

        base_url = request.host_url.rstrip('/')
        original_image_url = f"{base_url}/static/images/uploads/{filename}" if filename else None
        annotated_image_url = f"{base_url}/static/images/{relative_annotated_path}" if relative_annotated_path else None

        response = {
            "message": "Detection completed successfully using CCTV_GUN (MMDetection)",
            "detections": detections,
            "original_image_url": original_image_url,
            "annotated_image_url": annotated_image_url,
             "model_info": {
                "type": "MMDetection (ConvNeXt based)", # Keep this updated if model changes
                "classes": { "1": "gun" }
            },
            "_debug_paths": {
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

    if not debug_mode:
        app.logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Flask debug mode is OFF. Explicitly setting app logger level to DEBUG.")

    logger.info(f"Starting Flask server on {host}:{port} (Debug Mode: {debug_mode})")
    # Use reloader only if in debug mode
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
