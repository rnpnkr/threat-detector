# Technical Context

## Technology Stack

### Backend
- **Framework**: Flask 2.3.3
- **AI Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV
- **Real-time Communication**: flask-sock 0.6.0
- **Environment Management**: python-dotenv
- **Python Version**: 3.9+

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
# Virtual Environment
python -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Run Server
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
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# Model Configuration
MODEL_PATH=./yolov8_model/runs/detect/Normal_Compressed/weights/best.pt
DETECTION_THRESHOLD=0.3
```

### Port Configuration
- Backend API: 5001 (avoiding AirPlay conflicts)
- Frontend Dev Server: 8080
- WebSocket: 5001 (shared with API)

## Dependencies

### Backend Dependencies
```txt
flask==2.3.3
ultralytics==8.0.196
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
python-dotenv==1.0.0
flask-sock==0.6.0
```

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

### YOLOv8 Configuration
- Model: YOLOv8
- Weights: `best.pt`
- Classes: guns (0), knife (1)
- Confidence Threshold: 0.3

### Processing Pipeline
1. Image Upload
2. YOLOv8 Detection
3. OpenCV Annotation
4. Result Storage
5. Client Notification

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

### Image Processing
- Efficient annotation
- Memory management
- Batch processing (future)

### WebSocket
- Connection pooling
- Message batching
- Reconnection handling

### Frontend
- Image caching
- State management
- UI updates optimization

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