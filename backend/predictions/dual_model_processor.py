import os
import logging
import cv2
import time
import json
import base64
import numpy as np
import torch # Required for torch.cuda.empty_cache()

# Assuming these are in the correct path relative to this file
# If imports fail, adjust sys.path in app.py before importing this module
# Changed to import from sibling directory within backend
from CCTV_GUN.detecting_images import (
    detect_gun_in_video_frame, 
    DEVICE, 
    CONFIG_PATH, 
    CHECKPOINT_PATH
)
from mmdet.apis import init_detector

logger = logging.getLogger(__name__)

# --- Helper Function for IoU --- (Moved from app.py)
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

class DualModelProcessor:
    def __init__(self, yolo_model, static_dir, 
                 high_confidence_threshold=0.7, mmdet_frame_skip=20, yolo_img_size=320,
                 # Pass MMDet details instead of assuming globals from import
                 mmdet_config_path=CONFIG_PATH, mmdet_checkpoint_path=CHECKPOINT_PATH, device=DEVICE):
        """Initialize the processor with models and parameters."""
        self.yolo_model = yolo_model
        self.static_dir = static_dir
        self.high_confidence_threshold = high_confidence_threshold
        self.mmdet_frame_skip = mmdet_frame_skip
        self.yolo_img_size = yolo_img_size
        self.mmdet_config_path = mmdet_config_path
        self.mmdet_checkpoint_path = mmdet_checkpoint_path
        self.device = device
        
        # Internal state
        self.person_gun_state = {} 
        self.frame_number = 0
        self.mmdet_gun_detected_first_time = False
        self.state_file_path = os.path.join(self.static_dir, "person_gun_state.json")
        self.save_interval = 100 # Can be made configurable

    def process_stream(self, video_path, ws):
        """
        Processes video using YOLO (persons every frame) and MMDet (guns/persons every N frames),
        associates guns with persons using IoU, maintains person_gun_state,
        triggers alerts only on first high-confidence gun detection per person,
        filters guns sent to frontend, and sends results via WebSocket.
        (Logic moved from app.py's process_dual_model_video_stream)
        """
        logger.info(f"[DUAL_MODEL_PROCESSOR] Started processing for video: {video_path}")
        if self.yolo_model is None:
            logger.error("[DUAL_MODEL_PROCESSOR] YOLO model instance is None. Aborting.")
            ws.send(json.dumps({"type": "error", "payload": "YOLO model unavailable in processing thread."}))
            return

        mmdet_model = None
        self.frame_number = 0 # Reset frame count for this stream
        self.person_gun_state = {} # Reset state for this stream
        self.mmdet_gun_detected_first_time = False # Reset flag for this stream

        try:
            # --- Initialize MMDet Model (within this thread/processor) ---
            logger.info(f"[DUAL_MODEL_PROCESSOR] Initializing MMDet model from {self.mmdet_config_path} on device: {self.device}")
            mmdet_model = init_detector(self.mmdet_config_path, self.mmdet_checkpoint_path, device=self.device)
            logger.info("[DUAL_MODEL_PROCESSOR] MMDet model initialized successfully.")

            # --- Start YOLO Tracking Stream (runs on every frame for tracking) ---
            logger.info(f"Starting YOLO tracking with imgsz={self.yolo_img_size}")
            yolo_results_generator = self.yolo_model.track(
                source=video_path,
                classes=[0],              # Track only persons
                tracker="bytetrack.yaml",
                stream=True,
                conf=0.5,                 # Person confidence threshold (make configurable?)
                iou=0.5,                  # IoU threshold for tracker (make configurable?)
                imgsz=self.yolo_img_size,
                verbose=False
            )

            # --- Process Frame by Frame ---
            total_processing_time = 0
            processed_frame_count = 0

            for yolo_result in yolo_results_generator:
                start_time = time.time() 
                self.frame_number += 1
                current_frame = yolo_result.orig_img
                if current_frame is None: continue
                annotated_frame = current_frame.copy() # Create frame copy for drawing (if re-enabled)

                # 1. Get YOLO Person Detections (Every Frame for Tracking ID continuity)
                yolo_persons_in_frame = []
                yolo_boxes = yolo_result.boxes
                if yolo_boxes is not None and yolo_boxes.id is not None:
                    bboxes_xyxy = yolo_boxes.xyxy.cpu().numpy()
                    track_ids = yolo_boxes.id.int().cpu().tolist()
                    for bbox, track_id in zip(bboxes_xyxy, track_ids):
                        person_data = {
                            'id': track_id,
                            'bbox': list(map(int, bbox))
                        }
                        yolo_persons_in_frame.append(person_data)

                # --- Conditional MMDet Processing --- 
                mmdet_guns_raw = [] # All guns detected by MMDet this frame
                mmdet_persons = []
                yolo_id_to_matched_mmdet_person_bbox = {}
                validated_guns_for_frontend = [] # Guns passing filters for WS
                run_mmdet_this_frame = (self.frame_number % self.mmdet_frame_skip == 1) 

                if run_mmdet_this_frame:
                    original_frame_from_mmdet, mmdet_detections = detect_gun_in_video_frame(model=mmdet_model, frame=current_frame)
                    mmdet_guns_raw = [d for d in mmdet_detections if d.get('class') == 'gun']
                    mmdet_persons = [d for d in mmdet_detections if d.get('class') == 'person']
                    logger.info(f"[MMDET_RUN] Frame {self.frame_number}: Found {len(mmdet_guns_raw)} raw guns, {len(mmdet_persons)} persons by MMDet.") # LOG 1 (using raw count)

                    # Log first gun detection event
                    if not self.mmdet_gun_detected_first_time and len(mmdet_guns_raw) > 0:
                        logger.info(f"[FIRST_MMDET_GUN] Frame {self.frame_number}: MMDet first detected raw gun object(s) (count: {len(mmdet_guns_raw)}). Details: {mmdet_guns_raw}") # LOG 2
                        self.mmdet_gun_detected_first_time = True

                    # Stage 1: Match YOLO Persons to MMDet Persons
                    person_iou_threshold = 0.6 
                    available_mmdet_persons = list(mmdet_persons)
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
                    logger.info(f"[PERSON_MATCH] Frame {self.frame_number}: Matched {len(yolo_id_to_matched_mmdet_person_bbox)} YOLO IDs to MMDet persons.") # LOG 3
                else:
                    pass 

                # 4. Stage 2: Associate MMDet Guns -> MMDet Persons -> YOLO ID & Update State
                current_frame_associations = {} 
                new_high_confidence_alerts_this_frame = [] 
                gun_mmdet_person_iou_threshold = 0.01

                # Only proceed if MMDet ran and found guns
                if run_mmdet_this_frame and len(mmdet_guns_raw) > 0:
                    logger.info(f"[GUN_ASSOC_START] Frame {self.frame_number}: Starting association for {len(mmdet_guns_raw)} raw guns vs {len(yolo_id_to_matched_mmdet_person_bbox)} matched persons.") # LOG 4

                    processed_mmdet_persons = list(mmdet_persons) # Use a copy for matching

                    for gun_idx, gun_det in enumerate(mmdet_guns_raw):
                        gun_bbox = [gun_det['bbox']['xmin'], gun_det['bbox']['ymin'], gun_det['bbox']['xmax'], gun_det['bbox']['ymax']]
                        
                        # Stage 2a: Find best MMDet Person match for this gun
                        best_matching_mmdet_person_bbox = None; max_gun_person_iou = 0.0
                        for mm_person in processed_mmdet_persons: 
                            mm_person_bbox = [mm_person['bbox']['xmin'], mm_person['bbox']['ymin'], mm_person['bbox']['xmax'], mm_person['bbox']['ymax']]
                            iou = calculate_iou(gun_bbox, mm_person_bbox)
                            if iou > gun_mmdet_person_iou_threshold and iou > max_gun_person_iou:
                                max_gun_person_iou = iou; best_matching_mmdet_person_bbox = mm_person_bbox
                        
                        if best_matching_mmdet_person_bbox:
                            logger.info(f"  [GUN_MM_PERSON_RESULT] Frame {self.frame_number}, Raw Gun {gun_idx}: Found MMDet person match (IoU: {max_gun_person_iou:.4f})")
                        else:
                            logger.info(f"  [GUN_MM_PERSON_RESULT] Frame {self.frame_number}, Raw Gun {gun_idx}: No nearby MMDet person found.")
                            continue 

                        # Stage 2b: Link MMDet person back to a YOLO ID
                        final_associated_yolo_id = None
                        for yolo_id, mapped_bbox in yolo_id_to_matched_mmdet_person_bbox.items():
                            if mapped_bbox == best_matching_mmdet_person_bbox:
                                final_associated_yolo_id = yolo_id
                                logger.info(f"    [YOLO_LINK] Frame {self.frame_number}, Raw Gun {gun_idx}: Linked MMDet Person to YOLO ID {final_associated_yolo_id}.")
                                break
                        
                        if final_associated_yolo_id is None:
                            logger.info(f"    [YOLO_LINK] Frame {self.frame_number}, Raw Gun {gun_idx}: Could not link matched MMDet person back to a YOLO ID.")
                            continue 

                        # --- Logic dependent on successful final association --- 
                        gun_confidence = gun_det.get('confidence', 0.0) 
                        person_key = f"Person_{final_associated_yolo_id}"
                        
                        # Filter guns sent to frontend (Added previously)
                        if gun_confidence >= self.high_confidence_threshold:
                            validated_guns_for_frontend.append(gun_det)
                            current_frame_associations[final_associated_yolo_id] = gun_det # Keep association map

                        # State Update & Notification Trigger
                        is_newly_detected_with_gun = person_key not in self.person_gun_state
                        if is_newly_detected_with_gun:
                            self.person_gun_state[person_key] = { "has_gun": True, "first_detected_frame": self.frame_number }
                            logger.info(f"  [STATE_UPDATE] Frame {self.frame_number}: Added {person_key} to state.") # LOG 6
                            if gun_confidence >= self.high_confidence_threshold:
                                logger.info(f"  [ALERT_TRIGGER] Frame {self.frame_number}: First high-confidence ({gun_confidence:.2f}) gun alert for NEW person {person_key}.") # LOG 7
                                new_high_confidence_alerts_this_frame.append({"personId": person_key, "gunConfidence": gun_confidence})
                        else:
                            # Only update has_gun if not already true? Or always update?
                            # Current logic always sets to True if associated again. Might be fine.
                            self.person_gun_state[person_key]["has_gun"] = True 
                            # logger.debug(...) # Keep debug log commented

                # 5. Conditionally Draw YOLO Person Boxes (Remains Commented Out)
                for person_data in yolo_persons_in_frame:
                    track_id = person_data['id']
                    person_key = f"Person_{track_id}"
                    should_draw_blue_box = person_key in self.person_gun_state and self.person_gun_state[person_key].get("has_gun", False)
                    
                    if should_draw_blue_box: 
                        pass # Drawing logic removed/commented in source

                # --- Performance Logging --- 
                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                processed_frame_count += 1
                if self.frame_number % 30 == 0: 
                    avg_fps = processed_frame_count / total_processing_time if total_processing_time > 0 else 0
                    logger.info(f"[PERF] Frame {self.frame_number}: Avg FPS so far: {avg_fps:.2f} (Current frame time: {processing_time:.4f}s)")

                # 6. Periodically Save State
                if self.frame_number % self.save_interval == 0:
                    try:
                        with open(self.state_file_path, 'w') as f:
                            json.dump(self.person_gun_state, f, indent=4)
                    except Exception as save_err:
                        logger.warning(f"[STATE_UPDATE] Failed to save state to {self.state_file_path}: {save_err}")

                # --- Selective WebSocket Sending --- 
                send_update_this_frame = False; send_reason = ""
                if run_mmdet_this_frame: send_update_this_frame = True; send_reason = "MMDet Frame"
                # Also send if an alert was triggered *this frame* 
                if len(new_high_confidence_alerts_this_frame) > 0: send_update_this_frame = True; send_reason = "Alert Triggered"

                if send_update_this_frame:
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    try:
                        payload = {
                            "frame_number": self.frame_number, "image": frame_base64,
                            "yolo_persons": yolo_persons_in_frame, 
                            "mmdet_guns": validated_guns_for_frontend, # USE filtered list
                            "associations": {f"Person_{pid}": g['confidence'] for pid, g in current_frame_associations.items()}, 
                            "person_gun_state": self.person_gun_state, # Send current state
                            "new_high_confidence_alerts": new_high_confidence_alerts_this_frame 
                        }
                        logger.info(f"[WEBSOCKET_SEND] Frame {self.frame_number}: Sending update ({send_reason}). Validated Guns: {len(validated_guns_for_frontend)}. Alerts: {len(new_high_confidence_alerts_this_frame)}") # LOG 8 (Updated)
                        ws.send(json.dumps({"type": "video_frame", "payload": payload}))
                    except Exception as send_err:
                        logger.warning(f"[DUAL_MODEL_PROCESSOR] Frame {self.frame_number}: Failed to send frame data via WebSocket: {send_err}") 
                        if hasattr(ws, 'connected') and not ws.connected:
                            logger.error("[DUAL_MODEL_PROCESSOR] WebSocket disconnected."); break
                        elif not hasattr(ws, 'connected'): # Simple check if it's a basic socket
                             # Attempt a simple check, might need refinement based on actual WS library 
                             try: 
                                 ws.ping() # Example check 
                             except: 
                                 logger.error("[DUAL_MODEL_PROCESSOR] WebSocket appears disconnected."); break
        
            # --- End of Loop Actions --- (MOVE INSIDE TRY BLOCK) --->
            logger.info(f"[DUAL_MODEL_PROCESSOR] Finished processing {self.frame_number} frames.")
            if processed_frame_count > 0:
                    final_avg_fps = processed_frame_count / total_processing_time
                    logger.info(f"[PERF] Final Average FPS: {final_avg_fps:.2f}")
            # Final state save
            try:
                with open(self.state_file_path, 'w') as f:
                    json.dump(self.person_gun_state, f, indent=4)
                logger.info(f"[STATE_UPDATE] Final person_gun_state saved to {self.state_file_path}")
            except Exception as save_err:
                logger.warning(f"[STATE_UPDATE] Failed to save final state: {save_err}")

            if hasattr(ws, 'connected') and ws.connected:
                ws.send(json.dumps({"type": "stream_end", "payload": {"frame_count": self.frame_number}}))
            elif not hasattr(ws, 'connected'): # Simple check 
                logger.info("[DUAL_MODEL_PROCESSOR] WebSocket already closed before sending stream_end.")
            
        # <--- END OF TRY BLOCK
        except Exception as e:
            logger.error(f"[DUAL_MODEL_PROCESSOR] Error during video processing: {e}", exc_info=True)
            if hasattr(ws, 'connected') and ws.connected:
                    try:
                        ws.send(json.dumps({"type": "error", "payload": f"Backend processing error: {str(e)}"}))
                    except Exception as send_err:
                        logger.error(f"[DUAL_MODEL_PROCESSOR] Failed to send error message via WebSocket: {send_err}")
            elif not hasattr(ws, 'connected'):
                logger.error("[DUAL_MODEL_PROCESSOR] WebSocket already closed when trying to send processing error.")
            
        finally:
            logger.info("[DUAL_MODEL_PROCESSOR] Exiting processing function.")
            # Release MMDet model
            if mmdet_model is not None:
                del mmdet_model
                # Only clear cache if using CUDA
                if self.device.startswith('cuda'):
                    try:
                        torch.cuda.empty_cache()
                        logger.info("[DUAL_MODEL_PROCESSOR] Cleared CUDA cache.")
                    except Exception as cache_err:
                        logger.warning(f"[DUAL_MODEL_PROCESSOR] Failed to clear CUDA cache: {cache_err}")
