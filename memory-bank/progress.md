# Progress: Threat Detector

## What Works
- Backend Flask API (`app.py`) is running.
- Health check endpoint (`/health`) responds.
- Static image upload and detection (`/detect`) using MMDet works for guns and persons (verified via `curl`).
- Detection results include bounding boxes, class names ('gun', 'person'), and confidence scores.
- Annotated images are generated and saved for static detection.
- Basic WebSocket setup (`/ws/video_feed`) exists in `app.py` using Flask-Sock.
- Video processing function (`process_cctv_gun_video_stream`) exists and uses MMDet (now configured for guns & persons).
- Conda environment (`env_cc`) setup on RunPod is functional.
- Initial integration of YOLO model (code added in `backend/yolo_model/` and `backend/predictions/`).

## What's In Progress
- **Dual-Model Tracking/Association:** Implementing the YOLO + MMDet approach to address flickering and improve gun-person association reliability.
    - Integrating YOLO-based person tracker.
    - **Refining YOLO Integration:** Ensuring the tracker runs correctly in the pipeline.
    - **Implementing/Refining IoU matching logic** within the `DualModelProcessor` or similar.
    - Developing stable state management (`person_gun_state`) based on combined model outputs.
- **Frontend Development:** Basic React UI for interacting with the backend.
- **WebSocket Video Streaming:** Currently blocked by ngrok bandwidth limits, preventing frontend testing. Requires ngrok issue resolution or alternative testing method.

## What's Left to Build
- Profiling system integration (age, height, etc.).
- WhatsApp alert mechanism.
- Robust error handling and edge case management.
- Configuration for different video sources (beyond hardcoded path).
- Deployment strategy and configuration.

## Known Issues
- **Detection Flickering:** MMDet-only detection in video streams is inconsistent frame-to-frame.
- **Unreliable Tracking/Association:** Difficulty in consistently tracking persons and associating detected guns with the correct individuals due to flickering.
- **Ngrok Bandwidth Limit:** Current ngrok free tier bandwidth exceeded, blocking external access (including WebSocket tests from frontend).
- **Hardcoded Paths:** Test video path is hardcoded in `app.py`. 