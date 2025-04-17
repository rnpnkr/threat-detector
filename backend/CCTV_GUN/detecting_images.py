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
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory where this script (detecting_images.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define constants for model paths and device (hardcoded for now)
# Construct absolute paths based on the script's directory
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'configs/gun_detection/convnext.py')
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'work_dirs/convnext/epoch_3.pth')
DEVICE = 'cpu'

# Define output directory relative to the backend static folder
# This assumes app.py is in 'backend' and static files are served from 'backend/static'
ANNOTATED_DIR_REL_STATIC = 'annotated/cctv_gun'
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Path to backend dir
ANNOTATED_DIR_ABS = os.path.join(BACKEND_DIR, 'static', 'images', ANNOTATED_DIR_REL_STATIC)

# Ensure the output directory exists
os.makedirs(ANNOTATED_DIR_ABS, exist_ok=True)
logger.info(f"Ensured annotated directory exists: {ANNOTATED_DIR_ABS}")

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
    # Use the absolute paths for checking
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found at absolute path: {CONFIG_PATH}")
        return None
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint file not found at absolute path: {CHECKPOINT_PATH}")
        return None

    # --- Model Initialization ---
    try:
        # Use the absolute paths for initialization
        logger.info(f"Initializing detector with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
        model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        logger.info("Detector initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing detector: {e}", exc_info=True)
        return None

    # --- Image Reading ---
    try:
        logger.info(f"Reading image: {image_path}")
        img = mmcv.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        logger.info(f"Image read successfully. Shape: {img.shape}")
        vis_img = img.copy() # Create a copy for visualization
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}", exc_info=True)
        return None

    # --- Inference ---
    try:
        logger.info("Running inference...")
        start_time = time.time()
        result = inference_detector(model, img)
        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return None

    # --- Result Processing ---
    detections_list = []
    try:
        logger.info(f"Processing detection results. Type: {type(result)}")
        confidence_threshold = 0.3 # Set confidence threshold
        gun_label = 1 # Assuming label 1 is 'gun'

        if isinstance(result, list):
            # Handle list format (older MMDetection versions)
            logger.info("Processing list-based result format.")
            if len(result) > gun_label: # Check if gun label index exists
                gun_results = result[gun_label] # Get results for the gun class
                logger.info(f"Found {len(gun_results)} potential gun detections in list format.")
                for detection in gun_results:
                    if len(detection) == 5: # Expected format [x1, y1, x2, y2, score]
                        bbox = detection[:4]
                        score = detection[4]
                        if score >= confidence_threshold:
                            x1, y1, x2, y2 = map(int, bbox)
                            detection_data = {
                                "bbox": {
                                    "xmin": x1,
                                    "ymin": y1,
                                    "xmax": x2,
                                    "ymax": y2
                                },
                                "class": "gun",
                                "confidence": float(score)
                            }
                            detections_list.append(detection_data)
                            logger.info(f"Detected gun with confidence {score:.2f} at bbox [{x1}, {y1}, {x2}, {y2}]")

                            # Draw on visualization image
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                         logger.warning(f"Unexpected detection format in list: {detection}")
            else:
                logger.warning(f"Result list does not contain index for gun_label {gun_label}")

        elif hasattr(result, 'pred_instances'):
            # Handle pred_instances format (newer MMDetection versions)
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

                if label == gun_label and score >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    detection_data = {
                        "bbox": {
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2
                        },
                        "class": "gun",
                        "confidence": float(score)
                    }
                    detections_list.append(detection_data)
                    logger.info(f"Detected gun with confidence {score:.2f} at bbox [{x1}, {y1}, {x2}, {y2}]")

                    # Draw on visualization image
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            logger.error(f"Unrecognized result format: {type(result)}")

        logger.info(f"Processed results. Found {len(detections_list)} guns above threshold {confidence_threshold}.")

    except Exception as e:
        # Catch the AttributeError specifically if you want, but general Exception is fine
        logger.error(f"Error processing detection results: {e}", exc_info=True)
        # Decide if partial results should be returned or None
        # For now, let's return what we have, but log the error

    # --- Save Annotated Image ---
    relative_output_path = None
    try:
        # Create filename
        original_filename = os.path.basename(image_path)
        output_filename = f"annotated_cctv_gun_{original_filename}"
        # Use the predefined absolute directory path
        absolute_output_path = os.path.join(ANNOTATED_DIR_ABS, output_filename)
        # Generate the relative path for the response
        relative_output_path = os.path.join(ANNOTATED_DIR_REL_STATIC, output_filename)

        logger.info(f"Saving annotated image to: {absolute_output_path}")
        mmcv.imwrite(vis_img, absolute_output_path)
        logger.info(f"Annotated image saved successfully.")

    except Exception as e:
        logger.error(f"Error saving annotated image to {absolute_output_path}: {e}", exc_info=True)
        relative_output_path = None # Indicate saving failed

    # --- Return Results ---
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
    try:
        # logger.debug("Running inference on video frame...") # Optional: Can be verbose
        result = inference_detector(model, frame)
        # logger.debug("Inference complete for frame.")

        # Process results (similar to image function)
        confidence_threshold = 0.3
        gun_label = 1

        if isinstance(result, list):
            if len(result) > gun_label:
                gun_results = result[gun_label]
                for detection in gun_results:
                    if len(detection) == 5:
                        bbox = detection[:4]
                        score = detection[4]
                        if score >= confidence_threshold:
                            x1, y1, x2, y2 = map(int, bbox)
                            detection_data = {
                                "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                                "class": "gun",
                                "confidence": float(score)
                            }
                            detections_list.append(detection_data)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_frame, f'Gun: {score:.2f}', (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif hasattr(result, 'pred_instances'):
            predictions = result.pred_instances
            bboxes = predictions.bboxes.cpu().numpy()
            scores = predictions.scores.cpu().numpy()
            labels = predictions.labels.cpu().numpy()
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                score = scores[i]
                label = labels[i]
                if label == gun_label and score >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    detection_data = {
                        "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                        "class": "gun",
                        "confidence": float(score)
                    }
                    detections_list.append(detection_data)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f'Gun: {score:.2f}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # else: # Don't log error for every frame if format is unrecognized
        #    pass

    except Exception as e:
        logger.error(f"Error processing video frame: {e}", exc_info=True)
        # Return original frame and empty detections on error for this frame
        return frame, []

    return vis_frame, detections_list


# --- Main function for processing video stream --- 
def process_cctv_gun_video_stream(video_path, ws):
    """
    Process video stream using MMDetection and send frames via WebSocket.
    Args:
        video_path: Path to the video file
        ws: WebSocket connection
    """
    cap = None # Initialize cap to None for error handling
    try:
        logger.info(f"[CCTV_GUN] Starting video processing for: {video_path}")

        # --- Model Initialization (Load once) --- 
        if not os.path.exists(CONFIG_PATH) or not os.path.exists(CHECKPOINT_PATH):
            logger.error(f"[CCTV_GUN] Config ({CONFIG_PATH}) or Checkpoint ({CHECKPOINT_PATH}) not found.")
            # Optionally send an error message via WebSocket
            # ws.send(json.dumps({"type": "error", "payload": "Model files not found."}))
            return False # Indicate failure
        
        try:
            logger.info(f"[CCTV_GUN] Initializing detector...")
            model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
            logger.info("[CCTV_GUN] Detector initialized successfully.")
        except Exception as model_e:
            logger.error(f"[CCTV_GUN] Error initializing detector: {model_e}", exc_info=True)
            # Optionally send an error message via WebSocket
            return False

        # --- Open Video File --- 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[CCTV_GUN] Failed to open video file: {video_path}")
            # Optionally send an error message via WebSocket
            return False

        logger.info("[CCTV_GUN] Video file opened successfully")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Check for invalid FPS
            logger.warning(f"[CCTV_GUN] Video FPS reported as {fps}, defaulting to 30.")
            fps = 30 # Set a default FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"[CCTV_GUN] Video properties - W: {width}, H: {height}, FPS: {fps}, Total Frames: {total_frames}")

        # --- Process Frames --- 
        frame_count = 0
        start_process_time = time.time()
        last_detections = [] # Store detections from the last processed frame
        frames_to_skip = 15 # Process 1 frame every 15 frames

        while True:
            read_start = time.time()
            ret, frame = cap.read()
            read_end = time.time()
            logger.debug(f"Frame read time: {read_end - read_start:.4f}s")

            if not ret:
                logger.info("[CCTV_GUN] End of video stream reached")
                break

            frame_count += 1
            logger.debug(f"[CCTV_GUN] Processing frame {frame_count}/{total_frames}")

            # --- Frame Skipping Logic --- 
            if frame_count % frames_to_skip == 1: # Process frame 1, 16, 31, etc.
                logger.debug(f"Running detection on frame {frame_count}")
                detect_start = time.time()
                annotated_frame, detections = detect_gun_in_video_frame(model, frame)
                detect_end = time.time()
                logger.debug(f"Frame detection time: {detect_end - detect_start:.4f}s")
                last_detections = detections # Update last known detections
            else:
                # For skipped frames, use the raw frame and the last known detections
                # Or send empty detections: detections = []
                logger.debug(f"Skipping detection for frame {frame_count}")
                annotated_frame = frame # Use the original frame
                detections = [] # Send empty detections for skipped frames
            # --- End Frame Skipping --- 

            # Convert frame to base64
            encode_start = time.time()
            try:
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Add quality setting
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as encode_err:
                 logger.error(f"[CCTV_GUN] Error encoding frame {frame_count}: {encode_err}")
                 continue # Skip this frame
            encode_end = time.time()
            logger.debug(f"Frame encode time: {encode_end - encode_start:.4f}s")

            # Prepare WebSocket message
            message = {
                "type": "video_frame",
                "payload": {
                    "frame": frame_base64,
                    "detections": detections,
                    "frame_number": frame_count,
                    "total_frames": total_frames
                }
            }

            # Send frame via WebSocket
            send_start = time.time()
            try:
                ws.send(json.dumps(message))
                logger.debug(f"[CCTV_GUN] Frame {frame_count} sent via WebSocket")
            except Exception as e:
                logger.error(f"[CCTV_GUN] Error sending frame {frame_count} via WebSocket: {str(e)}")
                break # Stop sending if WebSocket error occurs
            send_end = time.time()
            logger.debug(f"Frame send time: {send_end - send_start:.4f}s")

            # Control frame rate (consider processing time)
            elapsed_time = time.time() - read_start
            sleep_time = max(0, (1 / fps) - elapsed_time)
            logger.debug(f"Frame elapsed: {elapsed_time:.4f}s, Sleep: {sleep_time:.4f}s")
            time.sleep(sleep_time)

        end_process_time = time.time()
        logger.info(f"[CCTV_GUN] Video processing completed in {end_process_time - start_process_time:.2f} seconds.")
        return True

    except Exception as e:
        logger.error(f"[CCTV_GUN] Error in process_cctv_gun_video_stream: {str(e)}", exc_info=True)
        return False # Indicate failure
    finally:
        # Ensure video capture is released
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("[CCTV_GUN] Video capture released.")


if __name__ == "__main__":
    # --- Standalone Testing with Command-Line Argument ---
    parser = argparse.ArgumentParser(description="Detect guns in an image using MMDetection.")
    parser.add_argument("image_path", help="Path to the input image file.")
    args = parser.parse_args()

    test_image_path = args.image_path

    logger.info(f"--- Running Standalone Test --- ")
    if not os.path.exists(test_image_path):
         logger.error(f"Test image not found at: {test_image_path}. Please provide a valid path.")
    else:
        # Make sure to pass the absolute path if the script needs it
        # If the input path might be relative, resolve it:
        absolute_test_image_path = os.path.abspath(test_image_path)
        logger.info(f"Using absolute image path: {absolute_test_image_path}")
        detection_result = detect_gun_in_image(absolute_test_image_path)

        if detection_result:
            print("\n--- Detection Results ---")
            print(f"Detections: {detection_result['detections']}")
            print(f"Annotated image saved at relative path: {detection_result['annotated_image_path']}")
            # You can construct the absolute path if needed:
            if detection_result['annotated_image_path']:
                 full_annotated_path = os.path.join(BACKEND_DIR, 'static', 'images', detection_result['annotated_image_path'])
                 print(f"Full annotated image path: {full_annotated_path}")
            else:
                print("Annotated image was not saved due to an error.")
        else:
            print("\n--- Detection Failed --- ")
            print("Check logs for details.")
    logger.info(f"--- Standalone Test Finished --- ") 