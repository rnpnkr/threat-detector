# Threat Detection System

A real-time threat detection system using YOLOv8 and Flask, with a modern React frontend for monitoring and alerts.

## Features

- 🎯 Real-time weapon detection using YOLOv8
- 🖥️ Modern React frontend with real-time updates
- 🚀 Flask backend with WebSocket support
- 📸 Image processing and annotation
- ⚡ Instant threat alerts and notifications
- 🔍 Detection confidence scoring
- 📊 Threat severity classification

## Tech Stack

### Backend
- Flask 2.3.3
- YOLOv8 (Ultralytics)
- OpenCV
- WebSocket (flask-sock)
- Python 3.9+

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- Shadcn/ui Components
- WebSocket Client

## Project Structure
```
threat-detection/
├── backend/
│   ├── app.py                 # Flask API
│   ├── requirements.txt       # Python dependencies
│   └── static/
│       └── images/
│           ├── uploads/       # Original uploaded images
│           └── annotated/     # Images with detection boxes
├── frontend/
│   ├── src/
│   │   ├── components/       
│   │   │   └── CameraFeed.tsx # Main camera feed component
│   │   └── ...
│   ├── package.json
│   └── ...
└── yolov8_model/
    ├── runs/
    │   └── detect/
    │       └── Normal_Compressed/
    │           └── weights/
    │               └── best.pt # YOLOv8 model weights
    └── imgs/
        └── Test/              # Test images
```

## Setup Instructions

### Backend Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5001`

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:8080`

## API Endpoints

### `/detect` (POST)
- Upload an image for threat detection
- Returns detection results and annotated image
- Example:
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5001/detect
```

### `/static/images/<path>` (GET)
- Serves static images (both original and annotated)

### `/ws` (WebSocket)
- Real-time updates for new detections

## Detection Classes

The system currently detects:
- Guns (Class 0)
- Knives (Class 1)

## Environment Variables

Create a `.env` file in the backend directory:
```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
MODEL_PATH=./yolov8_model/runs/detect/Normal_Compressed/weights/best.pt
DETECTION_THRESHOLD=0.3
```

## Current Status

✅ Implemented:
- Backend API with YOLOv8 integration
- Image upload and processing
- Real-time WebSocket notifications
- Frontend camera feed display
- Threat detection visualization
- Alert system for high-severity threats

🚧 In Progress:
- Multiple camera support
- Video stream processing
- Detection history logging
- Advanced alert configurations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.