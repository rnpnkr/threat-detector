# Active Context: Threat Detector

## Current Focus
**Addressing Detection Flickering and Improving Tracking/Association:** The primary focus is to resolve the inconsistent detection (flickering) issue observed with the current MMDet-only approach in video streams. This involves implementing a dual-model system as outlined in `systemPatterns.md`:
1. Integrate a YOLO-based person tracker for stable person IDs.
2. Continue using MMDet for gun detection.
3. Develop IoU-based logic to associate MMDet guns with YOLO person tracks.
4. Implement state management (`person_gun_state`) to reliably track only those persons currently associated with a gun.

## Recent Changes
- **Integrated YOLO Model:** Added `backend/yolo_model/` for YOLOv8 related code and `backend/predictions/` potentially containing the `DualModelProcessor` logic.
- Updated `backend/requirements.txt` likely adding `ultralytics` or other YOLO dependencies.
- Modified `backend/app.py` to potentially incorporate the new prediction logic.
- Modified `detecting_images.py` (both `detect_gun_in_image` and `detect_gun_in_video_frame` functions) to detect and visualize both persons (class 0) and guns (class 1).
- Investigated WebSocket connection issues, identifying an ngrok bandwidth limit as the likely cause for recent frontend connection failures.
- Added detailed logging to the WebSocket handler (`handle_video_feed_ws`) in `app.py`.
- Experimented with different Git commits to find stable points.

## Next Steps
1.  **(Verify YOLO Integration):** Confirm the YOLO tracker is correctly processing frames within the video pipeline (`process_cctv_gun_video_stream` or equivalent).
2.  **Implement/Refine IoU Association:** Code or refine the logic to match MMDet gun detections with YOLO person tracks based on bounding box IoU within the `DualModelProcessor` or relevant module.
3.  **Develop State Management:** Create and manage the `person_gun_state` dictionary based on tracking and association results, ensuring it accurately reflects persons holding guns.
4.  **Test and Refine:** Evaluate the dual-model system's performance on the test video, focusing on tracking stability and accurate gun-person association.
5.  **(Resolve Ngrok Issue):** Address the ngrok bandwidth limit to re-enable frontend testing.

## Open Questions/Considerations
- Which specific YOLO model and tracking library offers the best performance/ease of integration for this use case?
- How to handle cases where a gun is detected but not clearly associated with a tracked person (e.g., low IoU)?
- Performance implications of running two models (YOLO and MMDet) concurrently in the video stream.