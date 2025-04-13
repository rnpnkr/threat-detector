# Real-time Threat Detection System

A comprehensive real-time threat detection system designed for CCTV surveillance in Nagpur, India. The system uses YOLOv8 for weapon detection, with a modern React frontend for monitoring and instant alerts to concerned officers.

## Project Overview

### Problem Statement
The system addresses the need for real-time weapon detection in public spaces through CCTV cameras, providing instant alerts to law enforcement when potential threats are detected.

### Solution
- Real-time weapon detection using YOLOv8
- Instant WhatsApp alerts to concerned officers
- Profiling system for threat identification
- Multi-language support for alerts
- Modern web interface for monitoring

## Features

### Current Implementation
- 🎯 Static image weapon detection using YOLOv8
- 🖥️ Modern React frontend with real-time updates
- 🚀 Flask backend with WebSocket support
- 📸 Image processing and annotation
- 🔍 Detection confidence scoring
- 📊 Threat severity classification

### In Progress
- 📹 Real-time video feed processing
- 📱 WhatsApp alert system
- 👤 Threat profiling system
- 🌐 Multi-language support

### Future Features
- 🔄 Multiple camera support
- 📈 Advanced analytics
- 🔐 Enhanced security features
- 📱 Mobile application

## Tech Stack

### Backend
- Flask 2.3.3
- YOLOv8 (Ultralytics)
- OpenCV
- WebSocket (flask-sock)
- Python 3.9+
- Twilio (WhatsApp API)

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- Shadcn/ui Components
- WebSocket Client
- React Query

## Project Structure
```
threat-detection/
├── backend/
│   ├── app.py                 # Flask API
│   ├── requirements.txt       # Python dependencies
│   ├── .env                  # Environment variables
│   ├── static/
│   │   └── images/
│   │       ├── uploads/      # Original uploaded images
│   │       └── annotated/    # Images with detection boxes
│   └── yolov8_model/
│       ├── detecting-images.py # Detection logic
│       └── runs/detect/Normal_Compressed/weights/best.pt
├── frontend/
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── layouts/         # Layout components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── utils/           # Utility functions
│   │   ├── types/           # TypeScript types
│   │   ├── styles/          # Global styles
│   │   ├── assets/          # Static assets
│   │   └── api/             # API integration
│   ├── public/              # Public assets
│   └── tests/               # Test files
└── memory-bank/             # Project documentation
    ├── projectbrief.md      # Project overview
    ├── productContext.md    # Product context
    ├── systemPatterns.md    # System architecture
    ├── techContext.md       # Technical details
    ├── activeContext.md     # Current focus
    └── progress.md          # Project progress
```

## Local Development Setup

### Prerequisites
- Python 3.9+
- Node.js 16+
- npm or yarn
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/threat-detection.git
cd threat-detection
```

### Step 2: Backend Setup

1. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Create .env file
cp .env.example .env
# Edit .env with your configuration
```

4. Start the Flask server:
```bash
python app.py
```

The backend server will start on `http://localhost:5001`

### Step 3: Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:8080`

### Step 4: Test the Setup

1. Test the API:
```bash
curl -X POST -F "image=@backend/yolov8_model/imgs/Test/concept_terrorist_2.jpg" http://localhost:5001/detect
```

2. Verify the frontend:
- Open `http://localhost:8080` in your browser
- Check if the camera feed is displayed
- Verify detection results

## API Documentation

### Endpoints

#### `/detect` (POST)
- **Purpose**: Upload image for threat detection
- **Request**: Multipart form with 'image' field
- **Response**: JSON with detection results
- **Example**:
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5001/detect
```

#### `/static/images/<path>` (GET)
- **Purpose**: Serve static images
- **Access**: Public access to processed images

#### `/ws` (WebSocket)
- **Purpose**: Real-time updates for new detections
- **Events**: New detection notifications

### Detection Classes
- Guns (Class 0)
- Knives (Class 1)

## Environment Variables

### Backend (.env)
```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# YOLOv8 Model Configuration
MODEL_PATH=./yolov8_model/runs/detect/Normal_Compressed/weights/best.pt
DETECTION_THRESHOLD=0.3

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

## Development Guidelines

### Backend
- Follow PEP 8 style guide
- Use type hints
- Document all functions
- Handle errors gracefully
- Implement proper logging

### Frontend
- Follow React best practices
- Use TypeScript for type safety
- Implement proper error handling
- Follow component structure guidelines
- Maintain consistent styling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.