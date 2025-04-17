# Technical Context

## Technology Stack

### Backend
- **Framework**: Flask 2.3.3
- **AI Model**: MMDetection (ConvNext from `backend/CCTV_GUN`)
- **Image Processing**: OpenCV (via mmcv)
- **Real-time Communication**: flask-sock 0.6.0
- **Environment Management**: Conda (env_cc environment)
- **Python Version**: 3.10 (defined in env.yml)

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/ui
- **Real-time**: WebSocket Client

## Development Environment

### Backend Setup
```bash
# Environment Setup (using Conda, based on env_cc)
# 1. Create environment from file (if needed):
#    cd backend/CCTV_GUN
#    conda env create -f requirements/env.yml
#    cd ../..
# 2. Activate environment:
conda activate env_cc

# 3. Install Core Dependencies (PyTorch/MMCV with specific versions for GPU/CPU):
#    Example GPU (CUDA 11.7 - Adjust based on actual hardware):
#    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
#    Example CPU:
#    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch -y
pip install openmim --upgrade
#    Use MIM to install mmcv-full matching installed PyTorch (GPU/CPU aware)
mim install mmcv-full==1.7.0

# 4. Install MMDetection Model Dependencies:
pip install -r backend/CCTV_GUN/requirements.txt

# 5. Install MMDetection Package:
cd backend/CCTV_GUN
pip install -e .
cd ../..

# 6. Install Flask App Dependencies:
pip install -r backend/requirements.txt

# Run Server (from backend directory)
cd backend
python app.py
```

### Frontend Setup
```bash
# Dependencies
npm install

# Development Server
npm run dev
```

## Configuration

### Environment Variables
```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development # Set to 'production' for deployment
FLASK_DEBUG=1         # Set to 0 for production
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# Model Configuration (Currently handled within backend/CCTV_GUN/detecting_images.py)
# CONFIG_PATH = backend/CCTV_GUN/configs/gun_detection/convnext.py
# CHECKPOINT_PATH = backend/CCTV_GUN/work_dirs/convnext/epoch_3.pth
# DEVICE = cpu # Change to cuda:0 for GPU
DETECTION_THRESHOLD=0.3
```

### Port Configuration
- Backend API: 5001 (avoiding AirPlay conflicts)
- Frontend Dev Server: 8080 # Assuming default Vite port
- WebSocket: 5001 (shared with API)

## Dependencies

### Backend Dependencies (`env_cc` Environment)
- **Core ML:** PyTorch (~1.13), torchvision (~0.14), mmcv-full (1.7.0), MMDetection (local source) - Installation managed via Conda/MIM for hardware compatibility.
- **MMDetection Specific:** See `backend/CCTV_GUN/requirements.txt` (includes addict, numpy, opencv-python, pycocotools, etc.)
- **Flask App Specific:** See `backend/requirements.txt` (includes flask, python-dotenv, Werkzeug, twilio, flask-sock, etc.)
- **Note:** Overlapping dependencies like `opencv-python`, `numpy`, `pillow` should use versions compatible with the MMDetection setup.

### Frontend Dependencies
```json
{
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "tailwindcss": "^3.0.0",
    "@radix-ui/react-alert-dialog": "^1.0.0",
    "lucide-react": "^0.0.0"
  }
}
```

## API Endpoints

### REST Endpoints
1. `POST /detect`
   - Upload image for detection
   - Returns detection results and paths

2. `GET /static/images/<path>`
   - Serve static images
   - Handles both original and annotated images

### WebSocket Endpoints
1. `ws://localhost:5001/ws`
   - Real-time detection notifications
   - Client connection management

## File Storage

### Directory Structure
```
backend/
└── static/
    └── images/
        ├── uploads/     # Original images
        └── annotated/   # Processed images
```

### File Naming
- Original: `filename.jpg`
- Annotated: `annotated_filename.jpg`

## Detection Model

### MMDetection Configuration (ConvNext)
- **Source:** `backend/CCTV_GUN/`
- **Framework:** MMDetection
- **Backbone:** ConvNext
- **Config File:** `backend/CCTV_GUN/configs/gun_detection/convnext.py`
- **Checkpoint:** `backend/CCTV_GUN/work_dirs/convnext/epoch_3.pth`
- **Classes:** `gun` (label 1)
- **Confidence Threshold:** 0.3
- **Inference Device:** Configurable (`cpu` or `cuda:0`), currently set to `cpu`.

### Processing Pipeline (Image - `/detect`)
1. Image Upload via POST request.
2. Image saved to `backend/static/images/uploads/`.
3. Absolute path passed to `detect_gun_in_image` function.
4. MMDetection model loaded (if not already loaded, though currently loaded per-call).
5. MMDetection inference run on the image.
6. Results processed (filtering for 'gun', thresholding).
7. Bounding boxes drawn on image copy.
8. Annotated image saved to `backend/static/images/annotated/cctv_gun/`.
9. Detection list and annotated image path returned.
10. JSON response sent to client.

### Processing Pipeline (Video - `/ws/video`)
1. WebSocket connection established.
2. Background thread started, targeting `process_cctv_gun_video_stream`.
3. MMDetection model loaded **once** within the thread.
4. Video file opened frame by frame.
5. **Frame Skipping:** Only every Nth (e.g., 15th) frame is processed by detection.
6. For processed frames: `detect_gun_in_video_frame` called with pre-loaded model.
7. For processed frames: Inference run, results processed, bounding boxes drawn.
8. For skipped frames: Original frame used, empty detections sent.
9. Annotated/Original frame encoded to Base64 JPEG.
10. JSON message (frame, detections, frame number) sent via WebSocket.
11. Loop continues until video ends or WebSocket closes.

## Development Tools

### Required Tools
- Python 3.9+
- Node.js 16+
- npm/yarn
- Git

### Recommended Tools
- VS Code
- Python extension
- ESLint
- Prettier

## Testing

### Backend Testing
- Unit tests (to be implemented)
- Integration tests (to be implemented)
- API tests (to be implemented)

### Frontend Testing
- Component tests (to be implemented)
- Integration tests (to be implemented)
- E2E tests (to be implemented)

## Deployment

### Requirements
- Python 3.9+
- Node.js 16+
- npm/yarn
- Git
- OpenCV dependencies

### Process
1. Clone repository
2. Install dependencies
3. Configure environment
4. Build frontend
5. Start backend server

## Security Considerations

### Input Validation
- File type checking
- Size limits
- Path validation

### Output Sanitization
- File path sanitization
- Response data cleaning
- Error message sanitization

### Resource Protection
- Rate limiting (to be implemented)
- File size limits
- Connection limits

## Performance Optimization

### Current
- **Frame Skipping:** Video stream processes only 1 in every 15 frames for detection to improve perceived framerate, sending raw frames in between.
- **Model Loaded Once (Video):** MMDetection model is loaded only once per video stream connection.

### Identified Bottlenecks
- **CPU Inference:** MMDetection model takes ~2.4 seconds per frame on CPU, severely limiting real-time detection frequency.

### Planned
- **GPU Acceleration:** Deploy backend to a cloud instance with an NVIDIA GPU (e.g., T4) and configure inference device to `cuda:0`.
- **Further Frame Rate Tuning:** Adjust frame skipping interval after GPU acceleration.
- **Batch Processing (Future):** Explore batch inference if multiple streams/requests need handling simultaneously.

## Technical Constraints

### 1. System Requirements
- Python 3.8+
- Node.js 16+
- Virtual environment
- Sufficient disk space for model and images
- Memory for model inference

### 2. File Handling
- Supported formats: PNG, JPG, JPEG
- Upload size limits: To be implemented
- Storage cleanup: To be implemented

### 3. Performance Considerations
- Model loading time
- Image processing time
- Memory usage with large images
- Concurrent request handling
- Frontend bundle size optimization

## Development Tools

### 1. Version Control
- Git
- .gitignore configured for:
  - Virtual environment
  - Uploaded files
  - Environment variables
  - Python cache
  - Node modules
  - Build artifacts

### 2. Testing Tools
- Manual testing with curl
- Postman (optional)
- Basic error logging
- React Testing Library (frontend)

### 3. Monitoring
- Health check endpoint
- Basic error tracking
- Performance monitoring (to be implemented) 