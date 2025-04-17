from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys
import logging
import cv2
import time
import json
import threading
import numpy as np # Needed for IoU and model inference
from ultralytics import YOLO
import base64 # Needed for frame encoding

# --- Project Structure Setup ---
# Get the directory of the current script (app.py)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct path to the CCTV_GUN directory relative to app.py
CCTV_GUN_DIR = os.path.join(BACKEND_DIR, 'CCTV_GUN')
# Add CCTV_GUN directory to sys.path to allow imports
sys.path.insert(0, CCTV_GUN_DIR) # Insert at beginning to ensure it's checked first

try:
    # Import the MMDetection functions AND device info AFTER modifying sys.path
    from detecting_images import (
        detect_gun_in_image, 
        process_cctv_gun_video_stream, # Keep for reference?
        detect_gun_in_video_frame,   # Import frame-level MMDet function
        DEVICE, 
        CONFIG_PATH,                 # Need for MMDet init
        CHECKPOINT_PATH              # Need for MMDet init
    )
    from mmdet.apis import init_detector # Need for MMDet init
    print("Successfully imported functions, DEVICE, paths, and init_detector from CCTV_GUN.")
except ImportError as e:
    print(f"Error importing from CCTV_GUN: {e}")
    print("Please ensure 'detecting_images.py' exists in the 'CCTV_GUN' directory adjacent to 'app.py',")
    print(f"that it defines DEVICE, and that all dependencies are installed and the environment is active.")
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

# --- Initialize Models ---
# Define YOLO model path
YOLO_MODEL_PATH = os.path.join(BACKEND_DIR, "yolo_model", "yolo11m.pt")
yolo_model = None # Initialize as None
try:
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model file not found at {YOLO_MODEL_PATH}. Cannot initialize.")
        # Decide if the app should exit or continue without YOLO
        # sys.exit(1)
    else:
        logger.info(f"Initializing YOLOv11m model from {YOLO_MODEL_PATH} on device: {DEVICE}")
        # Note: YOLO automatically uses DEVICE if it's 'cuda:0' etc.
        yolo_model = YOLO(YOLO_MODEL_PATH)
        # Optional: Run a dummy inference to fully load model onto GPU if needed
        # yolo_model.predict(np.zeros((640, 640, 3)), verbose=False)
        logger.info("YOLO model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize YOLO model: {e}", exc_info=True)
    # Decide if the app should exit or continue without YOLO
    # sys.exit(1)
# Note: MMDetection model (for guns) is initialized within its processing functions for now.
# --- ---

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

# --- Helper Function for IoU ---
def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list or tuple): [xmin, ymin, xmax, ymax] for the first box.
        boxB (list or tuple): [xmin, ymin, xmax, ymax] for the second box.

    Returns:
        float: The IoU value (between 0.0 and 1.0).
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # Check if there is overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate the Union area
    union_area = float(boxA_area + boxB_area - intersection_area)

    # Compute the IoU
    iou = intersection_area / union_area
    return iou
# --- ---

# --- Dual Model Processing Function ---
def process_dual_model_video_stream(video_path, ws, yolo_model_instance):
    """
    Processes video using YOLO (persons every frame) and MMDet (guns/persons every N frames),
    associates guns with persons using IoU, maintains person_gun_state,
    triggers alerts only on first high-confidence gun detection per person,
    and sends annotated frames and results via WebSocket.
    """
    logger.info(f"[DUAL_MODEL] Started processing for video: {video_path}")
    if yolo_model_instance is None:
        logger.error("[DUAL_MODEL] YOLO model instance is None. Aborting.")
        ws.send(json.dumps({"type": "error", "payload": "YOLO model unavailable in processing thread."}))
        return

    mmdet_model = None
    frame_number = 0
    # State format: { "Person_ID": {"has_gun": bool, "first_detected_frame": int, "notification_sent": bool} }
    person_gun_state = {} # Initialize state dictionary
    state_file_path = os.path.join(STATIC_DIR, "person_gun_state.json")
    save_interval = 100 # Save state every 100 frames
    HIGH_CONFIDENCE_THRESHOLD = 0.7 # Confidence threshold for triggering alert
    MMDET_FRAME_SKIP = 20 # Run MMDet every N frames (adjust as needed)
    YOLO_IMG_SIZE = 320 # YOLO input resolution (e.g., 640, 416) - adjust for performance
    mmdet_gun_detected_first_time = False # Initialize flag for first MMDet gun detection

    try:
        # --- Initialize MMDet Model (within this thread) ---
        logger.info(f"[DUAL_MODEL] Initializing MMDet model from {CONFIG_PATH} on device: {DEVICE}")
        mmdet_model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        logger.info("[DUAL_MODEL] MMDet model initialized successfully.")

        # --- Start YOLO Tracking Stream (runs on every frame for tracking) ---
        logger.info(f"Starting YOLO tracking with imgsz={YOLO_IMG_SIZE}")
        yolo_results_generator = yolo_model_instance.track(
            source=video_path,
            classes=[0],              # Track only persons
            tracker="bytetrack.yaml",
            stream=True,
            conf=0.5,                 # Person confidence threshold
            iou=0.5,
            imgsz=YOLO_IMG_SIZE,      # Set YOLO image size for performance tuning
            verbose=False
        )

        # --- Process Frame by Frame ---
        total_processing_time = 0
        processed_frame_count = 0

        for yolo_result in yolo_results_generator:
            # Removed frame start log
            start_time = time.time() 
            frame_number += 1
            current_frame = yolo_result.orig_img
            if current_frame is None: continue
            annotated_frame = current_frame.copy() # Create frame copy for drawing

            # 1. Get YOLO Person Detections (Every Frame for Tracking ID continuity)
            yolo_persons_in_frame = [] # Store detected persons for later conditional drawing
            yolo_boxes = yolo_result.boxes
            if yolo_boxes is not None and yolo_boxes.id is not None:
                bboxes_xyxy = yolo_boxes.xyxy.cpu().numpy()
                track_ids = yolo_boxes.id.int().cpu().tolist()
                for bbox, track_id in zip(bboxes_xyxy, track_ids):
                    person_data = {
                        'id': track_id,
                        'bbox': list(map(int, bbox)) # [xmin, ymin, xmax, ymax]
                    }
                    yolo_persons_in_frame.append(person_data)

            # --- Conditional MMDet Processing --- 
            mmdet_guns = []
            mmdet_persons = []
            # Stores {yolo_id: [xmin, ymin, xmax, ymax] of matched MMDet person}
            yolo_id_to_matched_mmdet_person_bbox = {}
            validated_guns_for_frontend = [] # << INITIALIZE new list here
            run_mmdet_this_frame = (frame_number % MMDET_FRAME_SKIP == 1) 

            if run_mmdet_this_frame:
                original_frame_from_mmdet, mmdet_detections = detect_gun_in_video_frame(model=mmdet_model, frame=current_frame)
                mmdet_guns = [d for d in mmdet_detections if d.get('class') == 'gun']
                mmdet_persons = [d for d in mmdet_detections if d.get('class') == 'person']
                logger.info(f"[MMDET_RUN] Frame {frame_number}: Found {len(mmdet_guns)} guns, {len(mmdet_persons)} persons by MMDet.") # LOG 1

                # Log first gun detection event
                if not mmdet_gun_detected_first_time and len(mmdet_guns) > 0:
                    logger.info(f"[FIRST_MMDET_GUN] Frame {frame_number}: MMDet first detected a gun object (count: {len(mmdet_guns)}). Details: {mmdet_guns}") # LOG 2
                    mmdet_gun_detected_first_time = True

                # Stage 1: Match YOLO Persons to MMDet Persons
                person_iou_threshold = 0.6 
                available_mmdet_persons = list(mmdet_persons)
                # Clear map for this frame
                yolo_id_to_matched_mmdet_person_bbox.clear()
                for y_person in yolo_persons_in_frame: 
                    y_id = y_person['id']
                    y_bbox = y_person['bbox']
                    best_mm_match_idx = -1; max_iou = 0.0
                    for mm_idx, mm_person in enumerate(available_mmdet_persons):
                        mm_bbox_dict = mm_person['bbox']
                        mm_bbox = [mm_bbox_dict['xmin'], mm_bbox_dict['ymin'], mm_bbox_dict['xmax'], mm_bbox_dict['ymax']]
                        iou = calculate_iou(y_bbox, mm_bbox)
                        if iou > person_iou_threshold and iou > max_iou:
                            max_iou = iou; best_mm_match_idx = mm_idx
                    if best_mm_match_idx != -1:
                        matched_mm_person = available_mmdet_persons.pop(best_mm_match_idx) 
                        mm_bbox_dict = matched_mm_person['bbox']
                        mm_bbox = [mm_bbox_dict['xmin'], mm_bbox_dict['ymin'], mm_bbox_dict['xmax'], mm_bbox_dict['ymax']]
                        yolo_id_to_matched_mmdet_person_bbox[y_id] = mm_bbox 
                logger.info(f"[PERSON_MATCH] Frame {frame_number}: Matched {len(yolo_id_to_matched_mmdet_person_bbox)} YOLO IDs to MMDet persons.") # LOG 3
            else:
                # No logging for skipping MMDet
                pass 

            # 4. Stage 2: Associate MMDet Guns -> MMDet Persons -> YOLO ID & Update State
            current_frame_associations = {} 
            new_high_confidence_alerts_this_frame = [] 
            gun_mmdet_person_iou_threshold = 0.01 # Keep low threshold

            # Only proceed if MMDet ran and found guns
            if run_mmdet_this_frame and len(mmdet_guns) > 0:
                logger.info(f"[GUN_ASSOC_START] Frame {frame_number}: Starting association for {len(mmdet_guns)} guns vs {len(yolo_id_to_matched_mmdet_person_bbox)} matched persons.") # LOG 4

                processed_mmdet_persons = list(mmdet_persons) # Use a copy for matching

                for gun_idx, gun_det in enumerate(mmdet_guns):
                    gun_bbox = [gun_det['bbox']['xmin'], gun_det['bbox']['ymin'], gun_det['bbox']['xmax'], gun_det['bbox']['ymax']]
                    
                    # Stage 2a: Find best MMDet Person match for this gun
                    best_matching_mmdet_person_bbox = None; max_gun_person_iou = 0.0
                    for mm_person in processed_mmdet_persons: 
                        mm_person_bbox = [mm_person['bbox']['xmin'], mm_person['bbox']['ymin'], mm_person['bbox']['xmax'], mm_person['bbox']['ymax']]
                        iou = calculate_iou(gun_bbox, mm_person_bbox)
                        # logger.debug(f"  [GUN_MM_PERSON_CHECK] ... IoU={iou:.4f} ...") # DEBUG log commented
                        if iou > gun_mmdet_person_iou_threshold and iou > max_gun_person_iou:
                            max_gun_person_iou = iou; best_matching_mmdet_person_bbox = mm_person_bbox
                    
                    # Log Stage 2a result
                    if best_matching_mmdet_person_bbox:
                        logger.info(f"  [GUN_MM_PERSON_RESULT] Frame {frame_number}, Gun {gun_idx}: Found MMDet person match (IoU: {max_gun_person_iou:.4f})")
                    else:
                        logger.info(f"  [GUN_MM_PERSON_RESULT] Frame {frame_number}, Gun {gun_idx}: No nearby MMDet person found.")
                        continue # Skip if no MMDet person associated

                    # Stage 2b: Link MMDet person back to a YOLO ID
                    final_associated_yolo_id = None
                    for yolo_id, mapped_bbox in yolo_id_to_matched_mmdet_person_bbox.items():
                        if mapped_bbox == best_matching_mmdet_person_bbox:
                            final_associated_yolo_id = yolo_id
                            logger.info(f"    [YOLO_LINK] Frame {frame_number}, Gun {gun_idx}: Linked MMDet Person to YOLO ID {final_associated_yolo_id}.")
                            break
                    
                    if final_associated_yolo_id is None:
                        logger.info(f"    [YOLO_LINK] Frame {frame_number}, Gun {gun_idx}: Could not link matched MMDet person back to a YOLO ID.")
                        continue # Skip if no final link

                    # --- Logic dependent on successful final association --- 
                    gun_confidence = gun_det.get('confidence', 0.0) 
                    person_key = f"Person_{final_associated_yolo_id}"
                    # Only add gun to list sent to frontend IF it passes checks AND is high confidence
                    if gun_confidence >= HIGH_CONFIDENCE_THRESHOLD: # << ADD confidence check
                       validated_guns_for_frontend.append(gun_det)   # << ADD to new list
                       current_frame_associations[final_associated_yolo_id] = gun_det # Keep association map for context if needed

                    # Draw Red Gun Box (Active) - COMMENTING OUT FOR FRONTEND
                    # logger.info(f"  [DRAW_GUN_BOX] Frame {frame_number}: Drawing RED gun box for {person_key}") # LOG 5
                    # gx1, gy1, gx2, gy2 = gun_bbox
                    # cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2) # Red BGR
                    # cv2.putText(annotated_frame, f'GUN->{person_key}', (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # State Update & Notification Trigger
                    is_newly_detected_with_gun = person_key not in person_gun_state
                    if is_newly_detected_with_gun:
                        person_gun_state[person_key] = { "has_gun": True, "first_detected_frame": frame_number } 
                        # Log State Add (INFO)
                        logger.info(f"  [STATE_UPDATE] Frame {frame_number}: Added {person_key} to state.") # LOG 6
                        if gun_confidence >= HIGH_CONFIDENCE_THRESHOLD:
                            logger.info(f"  [ALERT_TRIGGER] Frame {frame_number}: First high-confidence ({gun_confidence:.2f}) gun alert for NEW person {person_key}.") # LOG 7
                            new_high_confidence_alerts_this_frame.append({"personId": person_key, "gunConfidence": gun_confidence})
                    else:
                        person_gun_state[person_key]["has_gun"] = True
                        # Log State Update (DEBUG)
                        # logger.debug(f"  [STATE_UPDATE] Frame {frame_number}: Updated {person_key} in state (already present).") # Keep commented

            # 5. Conditionally Draw YOLO Person Boxes (Blue - Remains Commented Out)
            for person_data in yolo_persons_in_frame:
                # ... (No logging here) ...
                track_id = person_data['id']
                person_key = f"Person_{track_id}"
                should_draw_blue_box = person_key in person_gun_state 
                
                if should_draw_blue_box: 
                    # Keep drawing lines commented out
                    # x1, y1, x2, y2 = person_data['bbox']
                    # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
                    # cv2.putText(annotated_frame, f'P_ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    pass # Add pass to prevent IndentationError

            # --- Performance Logging --- 
            # Keep FPS logging
            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            processed_frame_count += 1
            if frame_number % 30 == 0: 
                avg_fps = processed_frame_count / total_processing_time if total_processing_time > 0 else 0
                logger.info(f"[PERF] Frame {frame_number}: Avg FPS so far: {avg_fps:.2f} (Current frame time: {processing_time:.4f}s)")

            # 6. Periodically Save State
            if frame_number % save_interval == 0:
                try:
                    with open(state_file_path, 'w') as f:
                        json.dump(person_gun_state, f, indent=4)
                except Exception as save_err:
                    logger.warning(f"[STATE_UPDATE] Failed to save state to {state_file_path}: {save_err}")

            # --- Selective WebSocket Sending --- 
            send_update_this_frame = False; send_reason = ""
            if run_mmdet_this_frame: send_update_this_frame = True; send_reason = "MMDet Frame"
            if len(new_high_confidence_alerts_this_frame) > 0: send_update_this_frame = True; send_reason = "Alert Triggered"

            if send_update_this_frame:
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                try:
                    payload = { # Payload structure unchanged
                        "frame_number": frame_number, "image": frame_base64,
                        "yolo_persons": yolo_persons_in_frame, 
                        "mmdet_guns": validated_guns_for_frontend, # << USE filtered list
                        "associations": {f"Person_{pid}": g['confidence'] for pid, g in current_frame_associations.items()},
                        "person_gun_state": person_gun_state, 
                        "new_high_confidence_alerts": new_high_confidence_alerts_this_frame 
                    }
                    logger.info(f"[WEBSOCKET_SEND] Frame {frame_number}: Sending update ({send_reason}). Validated Guns: {len(validated_guns_for_frontend)}. Alerts: {new_high_confidence_alerts_this_frame}") # LOG 8 (Updated log)
                    ws.send(json.dumps({"type": "video_frame", "payload": payload}))
                except Exception as send_err:
                    logger.warning(f"[DUAL_MODEL] Frame {frame_number}: Failed to send frame data via WebSocket: {send_err}") 
                    if not ws.connected: logger.error("[DUAL_MODEL] WebSocket disconnected."); break
        
        # --- End of Loop Actions ---
        logger.info(f"[DUAL_MODEL] Finished processing {frame_number} frames.")
        if processed_frame_count > 0:
             final_avg_fps = processed_frame_count / total_processing_time
             logger.info(f"[PERF] Final Average FPS: {final_avg_fps:.2f}")
        # Final state save
        try:
            with open(state_file_path, 'w') as f:
                json.dump(person_gun_state, f, indent=4)
            logger.info(f"[STATE_UPDATE] Final person_gun_state saved to {state_file_path}")
        except Exception as save_err:
            logger.warning(f"[STATE_UPDATE] Failed to save final state: {save_err}")

        if ws.connected:
            ws.send(json.dumps({"type": "stream_end", "payload": {"frame_count": frame_number}}))

    except Exception as e:
        logger.error(f"[DUAL_MODEL] Error during video processing: {e}", exc_info=True)
        if ws.connected:
             try:
                 ws.send(json.dumps({"type": "error", "payload": f"Backend processing error: {str(e)}"}))
             except Exception as send_err:
                 logger.error(f"[DUAL_MODEL] Failed to send error message via WebSocket: {send_err}")
    finally:
        logger.info("[DUAL_MODEL] Exiting processing function.")
        # Release MMDet model
        if mmdet_model is not None:
            del mmdet_model
            if DEVICE.startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
# --- ---


# --- WebSocket Handling ---
# Store clients connected specifically for video streaming notifications
video_clients = set()

@sock.route('/ws/video_feed')
def handle_video_feed_ws(ws):
    """
    Handles WebSocket connection. Starts a background thread to process the video
    using the DUAL MODEL pipeline and send results back.
    """
    logger.info(f"--- WebSocket Connection Attempt Received ---")
    logger.info(f"WebSocket object: {ws}")
    logger.info(f"Attempting to add WebSocket to video_clients set.")
    video_clients.add(ws)
    logger.info(f"Successfully added WebSocket to video_clients set. Current clients: {len(video_clients)}")
    processing_thread = None
    try:
        # Check if YOLO model initialized successfully before starting thread
        if yolo_model is None:
            logger.error("YOLO model not initialized. Cannot start video processing thread.")
            ws.send(json.dumps({"type": "error", "payload": "YOLO model failed to initialize on server."}))
            return

        video_to_process = TEST_VIDEO_PATH
        logger.info(f"Checking existence of video file: {video_to_process}")
        if not os.path.exists(video_to_process):
             logger.error(f"Video file specified for streaming not found: {video_to_process}")
             try:
                 logger.info(f"Sending 'video not found' error to WebSocket client: {ws}")
                 ws.send(json.dumps({"type": "error", "payload": "Video file not found on server."}))
             except Exception as send_err:
                 logger.error(f"Failed to send error to WebSocket client: {send_err}")
             return

        logger.info(f"Video file found. Preparing to start DUAL MODEL video processing thread for: {video_to_process}")
        processing_thread = threading.Thread(
            target=process_dual_model_video_stream,
            args=(video_to_process, ws, yolo_model),
            daemon=True
        )
        logger.info(f"Processing thread object created: {processing_thread}")
        processing_thread.start()
        logger.info(f"Dual model video processing thread started (Thread ID: {processing_thread.ident}). Entering keep-alive loop.")

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

    # Ensure app logger level is DEBUG regardless of Flask's debug setting
    if not debug_mode:
        app.logger.setLevel(logging.DEBUG)
        # Also set the root logger? Might be redundant if basicConfig worked, but safer.
        logging.getLogger().setLevel(logging.DEBUG) 
        logger.info("Flask debug mode is OFF. Explicitly setting app logger level to DEBUG.")

    logger.info(f"Starting Flask server on {host}:{port} (Debug Mode: {debug_mode})")
    # Disable reloader if not in debug mode, as it can cause issues with threads/GPU context
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode) 