# Active Context

## Current Focus
- **Performance Optimization:** Preparing for deployment to a cloud GPU instance to address the CPU inference bottleneck (~2.4s/frame) in MMDetection video streaming.
- **Documentation Update:** Synchronizing Memory Bank files with the recent switch to MMDetection and the performance findings.
- **Code Cleanup:** Planning removal of unused YOLOv8 code and directories after GPU deployment validation.
- **Ongoing Features:** Continuing development of image capture logic, profiling system, and WhatsApp alert system.

## Current Implementation Status

### Backend
✅ Implemented:
- Flask API with endpoints for health (`/health`), image detection (`/detect`), static files (`/static/images/<path>`), and video streaming (`/ws/video`).
- MMDetection (ConvNext from `CCTV_GUN`) integration for gun detection in both image and video streams.
- Frame skipping (1/15 frames) implemented for video streaming to mitigate CPU bottleneck.
- Absolute path resolution for model config/checkpoints.
- Basic error handling and logging (including DEBUG level).

🔄 In Progress:
- Setup and configuration for cloud GPU instance deployment.
- High-confidence image capture logic.
- Profiling system implementation.
- WhatsApp and OpenAI integration for alerts.

### Frontend
✅ Implemented:
- Image upload and display.
- Detection results visualization with bounding boxes.
- WebSocket connection for video stream display.
- Demo options UI (video selection, language, etc.).

🔄 In Progress:
- Potentially enhancing video display smoothness or handling of skipped frames.
- Integration of profiling results display (when backend is ready).
- Multi-language alert configuration UI (when backend is ready).

## Next Steps
1. **Backend Deployment:**
   - Provision a cloud GPU instance (GCP/AWS/Azure).
   - Set up the `env_cc` environment on the instance (Python, Conda, PyTorch-GPU, MMCV-GPU, MMDetection, Flask dependencies).
   - Deploy the backend code.
   - Modify `detecting_images.py` to use `DEVICE = 'cuda:0'`.
   - Configure a process manager (e.g., `systemd`, `gunicorn`) to run `app.py` persistently.
2. **Performance Testing & Tuning:**
   - Test `/detect` and `/ws/video` endpoints on the GPU instance.
   - Measure inference times.
   - Adjust or potentially remove frame skipping based on performance.
3. **Code Cleanup:**
   - Delete the `backend/yolov8_model` directory and any remaining unused code related to it.
   - Review `/detect` endpoint for potential optimization (e.g., loading model once).
4. **Continue Feature Development:** Progress on image capture logic, profiling, and alerts.

## Current Considerations
1. **Performance:** Primary focus is overcoming the CPU inference bottleneck via GPU deployment.
2. **Cloud Costs:** Evaluating cost-effective GPU instance options (Free credits, Spot/Preemptible instances).
3. **Deployment Complexity:** Setting up the GPU environment, drivers, CUDA, and persistent service requires careful configuration.
4. **Feature Integration:** Planning how new features (profiling, alerts) will integrate with the MMDetection results.
5. **Environment Consistency:** Ensuring dependencies are managed correctly between local (CPU) and remote (GPU) environments. 