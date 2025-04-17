# Technology Context: Real-time Threat Detection and Alerting

## Core Technologies

-   **Language:** Python
-   **AI/ML Frameworks:** PyTorch (core), MMDetection (model training/structure), OpenCV (image/video processing)
-   **Inference Optimization:** TensorRT (target for optimization), MMDeploy (tool for conversion)
-   **Potential Web Framework:** Flask (explored for API)
-   **Containerization:** Docker
-   **Deployment Platform:** RunPod (currently used)
-   **Environment Management:** Conda

## Key Dependencies (Specific versions may vary)

-   `torch`, `torchvision`
-   `mmcv-full` (specific version)
-   `mmdetection`
-   `opencv-python`
-   `numpy`
-   `conda` (for environment setup)
-   Potentially `tensorrt`, `mmdeploy`
-   Libraries for profiling models (TBD)
-   Libraries for language translation (TBD, e.g., `transformers`, cloud service SDK)
-   Libraries for WhatsApp API interaction (TBD)

## Development Setup

-   Requires Conda environment setup (`env.yml` used previously).
-   Requires installation of system libraries (`gcc`, `libgl1`, `git`) within the container environment.
-   Requires NVIDIA GPU with appropriate drivers and CUDA toolkit for accelerated inference.

## Technical Constraints

-   Inference speed is paramount for real-time processing.
-   Container environment restarts require re-installation of system dependencies.
-   Compatibility between CUDA, PyTorch, TensorRT, and NVIDIA driver versions is critical.

## Backend
- **Language:** Python 3.10
- **Framework:** Flask 2.3.3
- **WebSocket:** Flask-Sock
- **Detection:** MMDetection (using ConvNeXt-based model for guns/persons from `CCTV_GUN` module)
- **Dependencies:** See `backend/requirements.txt` (includes `Flask`, `Flask-Sock`, `ultralytics`, `torch`, `torchvision`, `opencv-python`, `mmcv-full`, etc.)
- **Environment:** Conda environment (`env_cc`) or Python virtual environment (`venv`). Activation required.

## Planned Additions
- **Person Tracking:** YOLOv8 model integrated (`backend/yolo_model/`) for robust real-time person tracking, complementing MMDet's gun detection.
- **Tracking Algorithm:** YOLOv8's built-in tracking capabilities likely used initially.
- **Association Logic:** Custom Intersection over Union (IoU) implementation (likely in `backend/predictions/`) for linking YOLO tracks with MMDet gun detections.
- **Alerting:** WhatsApp integration (library TBD).
- **Profiling:** Libraries like DeepFace (partially added), potentially others for height/other estimations.

## Frontend (In Development)
- **Framework:** React
- **Language:** TypeScript
- **UI:** Shadcn UI
- **State Management:** (TBD, likely Zustand or Context API)
- **WebSocket Client:** Native browser WebSocket API.

## Infrastructure & Deployment
- **Development:** Local machine, potentially Docker.
- **Testing/Staging:** RunPod GPU instances.
- **Tunneling:** ngrok for exposing local dev server.
- **Production:** (TBD)

## Key Paths & Configs
- **MMDet Model:** `backend/CCTV_GUN/work_dirs/convnext/epoch_3.pth`
- **MMDet Config:** `backend/CCTV_GUN/configs/gun_detection/convnext.py`
- **YOLO Model/Code:** `backend/yolo_model/`
- **Prediction Logic:** `backend/predictions/`
- **Test Video:** `backend/test/input/test_video_4.mov` (Hardcoded in `app.py`)
- **Environment Vars:** Managed via `.env` file (e.g., `FLASK_HOST`, `FLASK_PORT`). 