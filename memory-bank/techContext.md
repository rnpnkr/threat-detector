# Technical Context

## Core Technologies

### 1. Backend Framework
- **Flask 2.3.3**
  - Lightweight web framework
  - RESTful API implementation
  - Running on port 5001
  - Debug mode enabled
  - Current endpoint: http://localhost:5001

### 2. Frontend Framework
- **React 18.3.1**
  - Vite as build tool
  - TypeScript support
  - React Router for navigation
  - React Query for data fetching
  - Shadcn UI components
  - Tailwind CSS for styling
  - Current endpoint: http://localhost:8080

### 3. Machine Learning
- **YOLOv8**
  - Model version: 8.0.196
  - Detection threshold: 0.3
  - Model path: ./yolov8_model/runs/detect/Normal_Compressed/weights/best.pt
  - Integration: Direct Python integration
  - Available classes: guns, knife

### 4. Python Dependencies
```python
# Core dependencies
Flask==2.3.3
ultralytics==8.0.196
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
python-dotenv==1.0.0
```

### 5. Frontend Dependencies
```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "react-router-dom": "^6.26.2",
  "@tanstack/react-query": "^5.56.2",
  "tailwindcss": "^3.4.11",
  "typescript": "^5.5.3",
  "vite": "^5.4.1"
}
```

## Development Setup

### 1. Environment Configuration
```bash
# Backend Virtual Environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
cd backend
python app.py  # Runs on http://localhost:5001

# Frontend Setup
cd frontend
npm install
npm run dev  # Runs on http://localhost:8080
```

### 2. Directory Structure
```
threat-detection/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env
│   ├── uploads/
│   ├── outputs/
│   └── yolov8_model/
│       ├── detecting-images.py
│       ├── runs/
│       │   └── detect/
│       │       └── Normal_Compressed/
│       │           └── weights/
│       │               └── best.pt
│       └── imgs/
│           └── Test/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── pages/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
└── venv/
```

### 3. Environment Variables
```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# Model Configuration
MODEL_PATH=./yolov8_model/runs/detect/Normal_Compressed/weights/best.pt
DETECTION_THRESHOLD=0.3

# File Storage
UPLOAD_FOLDER=./uploads
OUTPUT_FOLDER=./yolov8_model/imgs/Test
```

## API Endpoints

### 1. Health Check
```http
GET http://localhost:5001/health
Response: {"status": "healthy"}
```

### 2. Detection
```http
POST http://localhost:5001/detect
Content-Type: multipart/form-data
Body: image=<file>

Response: {
    "message": "Detection completed successfully",
    "original_image": "<path>",
    "annotated_image": "<path>",
    "detections": [
        {
            "class": "guns",
            "confidence": 0.40,
            "bbox": {
                "xmin": 752,
                "ymin": 940,
                "xmax": 836,
                "ymax": 1046
            }
        }
    ],
    "image_info": {
        "shape": [1600, 1001, 3],
        "original_path": "<path>"
    },
    "model_info": {
        "path": "<model_path>",
        "classes": {
            "0": "guns",
            "1": "knife"
        }
    }
}
```

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