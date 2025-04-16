import os
import cv2
import mmcv
import torch
import numpy as np
import logging
from mmdet.apis import init_detector, inference_detector
import time
import argparse
import json
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define constants for model paths and device
# Construct absolute paths based on the script's directory
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'configs/gun_detection/convnext.py')
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'work_dirs/convnext/epoch_3.pth')

# --- Device Selection ---
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    logger.info("CUDA available. Using GPU: cuda:0")
else:
    DEVICE = 'cpu'
    logger.warning("CUDA not available. Using CPU. This will be very slow for video.")
# --- ---

# Define output directory relative to the backend static folder
# Assumes this script is in backend/CCTV_GUN
ANNOTATED_DIR_REL_STATIC = 'annotated/cctv_gun'
# Go up two levels from SCRIPT_DIR (CCTV_GUN) to reach backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ANNOTATED_DIR_ABS = os.path.join(BACKEND_DIR, 'static', 'images', ANNOTATED_DIR_REL_STATIC)

# Ensure the output directory exists
try:
    os.makedirs(ANNOTATED_DIR_ABS, exist_ok=True)
    logger.info(f"Ensured annotated directory exists: {ANNOTATED_DIR_ABS}")
except OSError as e:
    logger.error(f"Failed to create annotated directory {ANNOTATED_DIR_ABS}: {e}", exc_info=True)
    # Depending on the use case, you might want to exit here
    # exit(1)

def detect_gun_in_image(image_path):
    """
    Detects guns in a single image using the MMDetection model.

    Args:
        image_path (str): The absolute path to the input image.

    Returns:
        dict or None: A dictionary containing detection results and the
                      relative path to the annotated image, or None if an error occurs.
                      Format: {
                          'detections': [{'bbox': {...}, 'class': 'gun', 'confidence': ...}],
                          'annotated_image_path': 'relative/path/to/annotated_image.jpg'
                      }
    """
    logger.info(f"Starting gun detection for image: {image_path}")

    # --- Input Validation ---
    if not os.path.exists(image_path):
        logger.error(f"Input image not found: {image_path}")
        return None
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return None
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint file not found: {CHECKPOINT_PATH}")
        return None

    # --- Model Initialization ---
    model = None # Initialize model to None
    try:
        logger.info(f"Initializing detector with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH} on device: {DEVICE}")
        model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        logger.info("Detector initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing detector: {e}", exc_info=True)
        # Explicitly release model if initialization fails partially?
        # del model
        # if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Image Reading ---
    img = None
    try:
        logger.info(f"Reading image: {image_path}")
        img = mmcv.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            # Release model before returning
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return None
        logger.info(f"Image read successfully. Shape: {img.shape}")
        vis_img = img.copy() # Create a copy for visualization
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}", exc_info=True)
        # Release model before returning
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Inference ---
    result = None
    try:
        logger.info("Running inference...")
        start_time = time.time()
        result = inference_detector(model, img)
        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        # Release model before returning
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    # --- Result Processing ---
    detections_list = []
    try:
        logger.info(f"Processing detection results. Type: {type(result)}")
        confidence_threshold = 0.3 # Set confidence threshold
        # Define class IDs and names
        target_classes = {
            0: "person",
            1: "gun"
        }
        # Define colors for bounding boxes (BGR)
        class_colors = {
            "person": (255, 0, 0), # Blue
            "gun": (0, 255, 0)   # Green
        }

        # Determine the structure of the result (list vs pred_instances)
        is_pred_instances_format = hasattr(result, 'pred_instances')

        if is_pred_instances_format:
            logger.info("Processing pred_instances-based result format.")
            predictions = result.pred_instances
            bboxes = predictions.bboxes.cpu().numpy()
            scores = predictions.scores.cpu().numpy()
            labels = predictions.labels.cpu().numpy()
            logger.info(f"Found {len(bboxes)} potential detections in pred_instances format.")

            for i in range(len(bboxes)):
                bbox = bboxes[i]
                score = scores[i]
                label = labels[i]

                if label in target_classes and score >= confidence_threshold:
                    class_name = target_classes[label]
                    color = class_colors.get(class_name, (0, 0, 255)) # Default to Red if class unknown
                    x1, y1, x2, y2 = map(int, bbox)
                    detection_data = {
                        "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                        "class": class_name,
                        "confidence": float(score)
                    }
                    detections_list.append(detection_data)
                    logger.debug(f"Detected {class_name} with confidence {score:.2f} at bbox [{x1}, {y1}, {x2}, {y2}]")
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_img, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif isinstance(result, list):
            logger.info("Processing list-based result format (may be older MMDetection).")
            # Assuming the list index corresponds to the class ID
            for label, class_results in enumerate(result):
                if label in target_classes:
                    class_name = target_classes[label]
                    color = class_colors.get(class_name, (0, 0, 255))
                    logger.info(f"Found {len(class_results)} potential {class_name} detections.")
                    for detection in class_results:
                        if len(detection) == 5:
                            bbox = detection[:4]
                            score = detection[4]
                            if score >= confidence_threshold:
                                x1, y1, x2, y2 = map(int, bbox)
                                detection_data = {
                                    "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                                    "class": class_name,
                                    "confidence": float(score)
                                }
                                detections_list.append(detection_data)
                                logger.debug(f"Detected {class_name} with confidence {score:.2f} at bbox [{x1}, {y1}, {x2}, {y2}]")
                                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(vis_img, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            logger.warning(f"Unexpected detection format in list for class {label}: {detection}")
                else:
                    # Optional: Log if results for non-target classes are found
                    # logger.debug(f"Skipping results for class ID {label} (not in target_classes)")
                    pass

        else:
            # Handle cases where result might be None or an unexpected type
            if result is None:
                logger.warning("Inference result was None.")
            else:
                logger.error(f"Unrecognized result format: {type(result)}")

        logger.info(f"Processed results. Found {len(detections_list)} target detections (persons/guns) above threshold {confidence_threshold}.")

    except Exception as e:
        logger.error(f"Error processing detection results: {e}", exc_info=True)
        # Continue to save the image (possibly without annotations if processing failed early)

    # --- Save Annotated Image ---
    relative_output_path = None
    absolute_output_path = None # Initialize to None
    try:
        original_filename = os.path.basename(image_path)
        # Add a unique identifier or timestamp if multiple runs use the same input name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = f"annotated_cctv_gun_{timestamp}_{original_filename}"
        absolute_output_path = os.path.join(ANNOTATED_DIR_ABS, output_filename)
        # Ensure forward slashes for web paths/URLs
        relative_output_path = os.path.join(ANNOTATED_DIR_REL_STATIC, output_filename).replace(os.sep, '/')

        logger.info(f"Saving annotated image to: {absolute_output_path}")
        if not mmcv.imwrite(vis_img, absolute_output_path):
             logger.error(f"mmcv.imwrite failed to save image to {absolute_output_path}")
             relative_output_path = None
        else:
             logger.info(f"Annotated image saved successfully.")

    except Exception as e:
        logger.error(f"Error saving annotated image to {absolute_output_path}: {e}", exc_info=True)
        relative_output_path = None

    # --- Return Results ---
    # Release GPU memory after use in single image detection
    del model
    if torch.cuda.is_available():
         torch.cuda.empty_cache()
         logger.debug("Cleared CUDA cache after single image detection.")

    return {
        "detections": detections_list,
        "annotated_image_path": relative_output_path
    }

# --- Helper function for processing a single video frame ---
def detect_gun_in_video_frame(model, frame):
    """
    Detects guns in a single video frame using a pre-loaded MMDetection model.

    Args:
        model: The initialized MMDetection model.
        frame (np.ndarray): The input video frame.

    Returns:
        tuple: A tuple containing:
            - annotated_frame (np.ndarray): The frame with bounding boxes drawn.
            - detections_list (list): List of detection dictionaries for the frame.
    """
    vis_frame = frame.copy()
    detections_list = []
    result = None # Initialize result
    try:
        # logger.debug("Running inference on video frame...") # Optional: Can be verbose
        result = inference_detector(model, frame)
        # logger.debug("Inference complete for frame.")

        confidence_threshold = 0.3
        # Define class IDs and names
        target_classes = {
            0: "person",
            1: "gun"
        }
        # Define colors for bounding boxes (BGR)
        class_colors = {
            "person": (255, 0, 0), # Blue
            "gun": (0, 255, 0)   # Green
        }

        # Determine the structure of the result (list vs pred_instances)
        is_pred_instances_format = hasattr(result, 'pred_instances')

        if is_pred_instances_format:
            # logger.debug("Processing pred_instances format for video frame.")
            predictions = result.pred_instances
            bboxes = predictions.bboxes.cpu().numpy()
            scores = predictions.scores.cpu().numpy()
            labels = predictions.labels.cpu().numpy()
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                score = scores[i]
                label = labels[i]
                if label in target_classes and score >= confidence_threshold:
                    class_name = target_classes[label]
                    color = class_colors.get(class_name, (0, 0, 255))
                    x1, y1, x2, y2 = map(int, bbox)
                    detection_data = { "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}, "class": class_name, "confidence": float(score) }
                    detections_list.append(detection_data)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_frame, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif isinstance(result, list):
            # logger.debug("Processing list format for video frame.")
            for label, class_results in enumerate(result):
                if label in target_classes:
                    class_name = target_classes[label]
                    color = class_colors.get(class_name, (0, 0, 255))
                    for detection in class_results:
                        if len(detection) == 5:
                            bbox = detection[:4]
                            score = detection[4]
                            if score >= confidence_threshold:
                                x1, y1, x2, y2 = map(int, bbox)
                                detection_data = { "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}, "class": class_name, "confidence": float(score) }
                                detections_list.append(detection_data)
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(vis_frame, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Optional: Log if results for non-target classes are found
                    # logger.debug(f"Skipping results for class ID {label} (not in target_classes)")
                    pass

        else:
            # Handle cases where result might be None or an unexpected type
            if result is None:
                logger.warning("Inference result was None.")
            else:
                logger.error(f"Unrecognized result format: {type(result)}")

        logger.info(f"Processed results. Found {len(detections_list)} target detections (persons/guns) above threshold {confidence_threshold}.")

    except Exception as e:
        # Log error less frequently for video to avoid spamming logs
        # Maybe log only the first error or sample errors
        # logger.error(f"Error processing video frame: {e}", exc_info=True) # Too verbose
        pass # Suppress error for now, return original frame
        # Return original frame and empty list if error occurs during inference/processing
        return frame, []

    return vis_frame, detections_list


# --- Main function for processing video stream ---
def process_cctv_gun_video_stream(video_path, ws=None, frames_to_skip=15, display=False):
    """
    Process video stream using MMDetection. Optionally sends frames via WebSocket
    or displays locally.

    Args:
        video_path (str): Path to the video file.
        ws (WebSocket, optional): WebSocket connection for sending results. Defaults to None.
        frames_to_skip (int): Process every Nth frame. Defaults to 15.
        display (bool): Whether to display the processed video locally. Defaults to False.
    """
    cap = None
    model = None
    processing_successful = False # Flag to track overall success
    try:
        logger.info(f"[CCTV_GUN] Starting video processing for: {video_path}")
        logger.info(f"Processing 1 frame every {frames_to_skip} frames.")

        # --- Input Validation ---
        if not os.path.exists(video_path):
            logger.error(f"[CCTV_GUN] Video file not found: {video_path}")
            if ws: ws.send(json.dumps({"type": "error", "payload": "Video file not found."}))
            return False
        if not os.path.exists(CONFIG_PATH):
            logger.error(f"[CCTV_GUN] Config file not found: {CONFIG_PATH}")
            if ws: ws.send(json.dumps({"type": "error", "payload": "Model config file not found."}))
            return False
        if not os.path.exists(CHECKPOINT_PATH):
            logger.error(f"[CCTV_GUN] Checkpoint file not found: {CHECKPOINT_PATH}")
            if ws: ws.send(json.dumps({"type": "error", "payload": "Model checkpoint file not found."}))
            return False

        # --- Model Initialization (Load once) ---
        try:
            logger.info(f"[CCTV_GUN] Initializing detector on device {DEVICE}...")
            model_init_start = time.time()
            model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
            model_init_end = time.time()
            logger.info(f"[CCTV_GUN] Detector initialized successfully in {model_init_end - model_init_start:.2f} seconds.")
        except Exception as model_e:
            logger.error(f"[CCTV_GUN] Error initializing detector: {model_e}", exc_info=True)
            if ws: ws.send(json.dumps({"type": "error", "payload": "Failed to initialize detection model."}))
            return False

        # --- Open Video File ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[CCTV_GUN] Failed to open video file: {video_path}")
            if ws: ws.send(json.dumps({"type": "error", "payload": "Failed to open video file."}))
            del model # Clean up model if video fails to open
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return False
        logger.info("[CCTV_GUN] Video file opened successfully.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning(f"[CCTV_GUN] Video FPS reported as {fps}, defaulting to 30.")
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"[CCTV_GUN] Video properties - W: {width}, H: {height}, FPS: {fps:.2f}, Total Frames: {total_frames if total_frames > 0 else 'Unknown'}")
        if ws: # Send video info to client
            ws.send(json.dumps({"type": "video_info", "payload": {"width": width, "height": height, "fps": fps, "total_frames": total_frames if total_frames > 0 else -1}}))


        # --- Process Frames ---
        frame_count = 0
        processed_frame_count = 0
        total_detection_time = 0
        start_process_time = time.time()
        # last_detections = [] # Not currently used

        while True:
            # Check WebSocket state before reading frame (optional, maybe too much overhead)
            # if ws and ws.closed:
            #      logger.info("[CCTV_GUN] WebSocket connection closed by client.")
            #      break

            read_start = time.time()
            ret, frame = cap.read()
            read_end = time.time()

            if not ret:
                logger.info("[CCTV_GUN] End of video stream reached.")
                if ws: ws.send(json.dumps({"type": "stream_end", "payload": {"frame_count": frame_count}}))
                break

            frame_count += 1
            processing_this_frame = (frame_count % frames_to_skip == 1)

            annotated_frame = frame # Start with original frame
            detections = []       # Default to no detections for this message

            if processing_this_frame:
                processed_frame_count += 1
                detect_start = time.time()
                # Pass the model and the original frame to the detection helper
                annotated_frame, detections = detect_gun_in_video_frame(model, frame)
                detect_end = time.time()
                frame_detect_time = detect_end - detect_start
                total_detection_time += frame_detect_time
                # logger.debug(f"Frame {frame_count} detection time: {frame_detect_time:.4f}s") # Verbose

            # --- WebSocket Sending (if ws is provided) ---
            if ws:
                try:
                    # Encode the potentially annotated frame
                    encode_start = time.time()
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]) # Adjust quality
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    encode_end = time.time()

                    current_avg_fps = processed_frame_count / total_detection_time if total_detection_time > 0 else 0

                    message = {
                        "type": "video_frame",
                        "payload": {
                            "frame": frame_base64,
                            "detections": detections, # Send current frame's detections
                            "frame_number": frame_count,
                            "total_frames": total_frames if total_frames > 0 else -1,
                            "processing_fps": round(current_avg_fps, 2)
                        }
                    }
                    ws.send(json.dumps(message))

                except Exception as send_e:
                    # Handle specific WebSocket errors if possible
                    logger.error(f"[CCTV_GUN] Error encoding or sending frame {frame_count} via WebSocket: {str(send_e)}")
                    # Attempt to inform client before breaking
                    try:
                        ws.send(json.dumps({"type": "error", "payload": f"Error processing frame {frame_count}. Stopping stream."}))
                    except:
                        pass # Ignore error if sending error message fails
                    break # Stop processing loop if WebSocket fails

            # --- Local Display (if display is True) ---
            if display:
                display_fps = processed_frame_count / total_detection_time if total_detection_time > 0 else 0
                cv2.putText(annotated_frame, f"Proc. FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("CCTV Gun Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit key pressed. Stopping display.")
                    break # Stop main loop if 'q' is pressed locally


        end_process_time = time.time()
        total_time = end_process_time - start_process_time
        avg_fps = processed_frame_count / total_detection_time if total_detection_time > 0 else 0
        logger.info(f"[CCTV_GUN] Video processing completed.")
        logger.info(f"Total time: {total_time:.2f} seconds.")
        logger.info(f"Total frames processed: {processed_frame_count} frames out of {frame_count} read.")
        logger.info(f"Average Detection FPS (on processed frames): {avg_fps:.2f}")
        processing_successful = True # Mark as successful if loop completed naturally

    except KeyboardInterrupt:
        logger.info("[CCTV_GUN] Processing interrupted by user (KeyboardInterrupt).")
        if ws: ws.send(json.dumps({"type": "stream_interrupted", "payload": "Processing stopped by server."}))
        processing_successful = False
    except Exception as e:
        logger.error(f"[CCTV_GUN] Critical error in process_cctv_gun_video_stream: {str(e)}", exc_info=True)
        if ws: ws.send(json.dumps({"type": "error", "payload": f"Critical server error: {str(e)}"}))
        processing_successful = False
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("[CCTV_GUN] Video capture released.")
        if display:
            cv2.destroyAllWindows()
            logger.info("[CCTV_GUN] Display windows closed.")
        # Release GPU memory
        if model is not None: # Check if model was successfully initialized
             del model
             model = None # Ensure reference is cleared
             if torch.cuda.is_available():
                  torch.cuda.empty_cache()
                  logger.info("[CCTV_GUN] Cleared CUDA cache.")
        # Ensure WebSocket is closed if this function terminates unexpectedly?
        # Flask-Sock might handle this when the context ends.

    return processing_successful


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect guns in a video file using MMDetection.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--skip", type=int, default=15, help="Process every Nth frame.")
    parser.add_argument("--display", action='store_true', help="Display the processed video locally.")
    args = parser.parse_args()

    video_file_path = args.video

    logger.info(f"--- Running Standalone Video Test --- ")
    logger.info(f"Input video: {video_file_path}")
    logger.info(f"Frames to skip: {args.skip}")
    logger.info(f"Display locally: {args.display}")

    if not os.path.exists(video_file_path):
         logger.error(f"Video file not found at: {video_file_path}. Please provide a valid path.")
    else:
        # Resolve to absolute path for consistency
        absolute_video_path = os.path.abspath(video_file_path)
        logger.info(f"Using absolute video path: {absolute_video_path}")

        # Call the video processing function
        success = process_cctv_gun_video_stream(
            video_path=absolute_video_path,
            ws=None, # No WebSocket for standalone run
            frames_to_skip=args.skip,
            display=args.display # Pass display flag
        )

        if success:
            print("\n--- Video Processing Finished Successfully ---")
        else:
            print("\n--- Video Processing Failed --- ")
            print("Check logs above for details.")

    logger.info(f"--- Standalone Video Test Finished --- ") 