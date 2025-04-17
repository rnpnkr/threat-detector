# System Patterns: Real-time Threat Detection and Alerting

## Architecture Overview

A modular system likely consisting of:

1.  **Video Input Module:** Handles connection to and frame grabbing from various video sources (IP cameras, video files, webcams).
2.  **Threat Detection Module:** Runs the primary object detection model (e.g., YOLOv8, optimized ConvNeXt/TensorRT) on incoming frames.
3.  **Profiling Module:** Runs secondary models (e.g., age estimation, height estimation) on detected regions of interest.
4.  **Alert Formatting Module:** Constructs the alert message and performs language translation (potentially using an LLM or translation service).
5.  **Alert Dispatch Module:** Interfaces with the WhatsApp API (or a gateway service) to send messages.
6.  **(Optional) API/Control Plane:** A potential Flask API (as previously explored) could manage configuration, display status, or serve results.

## Key Technical Decisions (In Progress/Explored)

-   **Threat Detection Model:** Explored MMDetection (ConvNeXt) and YOLOv8. Performance is critical, leading towards optimization (TensorRT) or lighter models (YOLO series).
-   **Real-time Processing:** Requires efficient frame processing pipelines and optimized models to achieve sufficient FPS.
-   **Deployment Environment:** Currently exploring containerized deployment (e.g., Docker on RunPod) requiring management of system dependencies.

## Component Relationships

Video Input -> Threat Detection -> (if threat) -> Profiling -> Alert Formatting -> Alert Dispatch

API/Control Plane interacts potentially with all modules for configuration and status.

# System Patterns: Threat Detector

## Architecture Overview
- **Backend API:** Flask application (`app.py`) serving REST endpoints and WebSocket connections.
- **Detection Engine:** Currently uses MMDetection (via `CCTV_GUN/detecting_images.py`) for object detection (guns, persons).
- **Real-time Processing:** WebSocket handler in `app.py` spawns a thread to process video streams frame-by-frame using the detection engine.
- **Static Analysis:** `/detect` endpoint handles image uploads and calls the detection engine.
- **Frontend:** (In Development) React application interacting with the backend API and WebSocket.

## Key Technical Decisions
- **Python/Flask:** Chosen for backend due to strong ML library support and rapid development.
- **MMDetection:** Selected as the initial detection framework (ConvNeXt model for guns).
- **WebSocket:** Used for efficient real-time communication of video frames and detection results between backend and frontend.
- **Hardcoded Video Path:** Currently used for testing the video streaming feature (`TEST_VIDEO_PATH` in `app.py`).

## Proposed Enhancement: Dual-Model Tracking & Association
- **Problem:** The current MMDet-only approach exhibits flickering detections, making person tracking and gun-person association unreliable.
- **Solution:** Introduce a dual-model system:
    1.  **YOLO Model:** Utilize a YOLO model (optimized for CCTV person tracking) to detect and consistently track all persons (up to ~10) across frames, assigning unique tracking IDs.
    2.  **MMDetection Model:** Continue using the existing MMDet model specifically for accurate gun detection.
    3.  **Association Logic:** Implement Intersection over Union (IoU) matching between YOLO person bounding boxes and MMDet gun bounding boxes within each frame.
        - *Implementation likely started in `backend/predictions/dual_model_processor.py`.*
    4.  **State Management:** Maintain a `person_gun_state` dictionary, updating it based on the tracked person IDs and their associated gun detections (via IoU matching). This dictionary should primarily contain only the persons currently associated with a gun.
        - *State likely managed within the processor or `app.py`.*
- **Status:** Initial integration in progress. Code exists in `backend/yolo_model/` and `backend/predictions/`. `app.py` and `requirements.txt` modified.
- **Goal:** Achieve stable tracking of individuals and reliable association of detected guns, focusing on accurately identifying the consistent state of persons who have guns. 