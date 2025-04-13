# Standalone Gun Detection System

This package contains a standalone gun detection system that works with video feeds.

## Setup Instructions

1. Install Python 3.10 if not already installed
2. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the model files:
   - Put the config file in the `configs` folder
   - Put the checkpoint file in the `checkpoints` folder

## Usage

Run the video detection script:
```bash
python scripts/video_gun_detector.py
```

Optional arguments:
- `--config`: Path to config file (default: configs/gun_detection.py)
- `--checkpoint`: Path to checkpoint file (default: checkpoints/epoch_3.pth)
- `--device`: Device to use (default: cuda:0)
- `--camera`: Camera device number (default: 0)
- `--threshold`: Detection threshold (default: 0.5)

Example with custom settings:
```bash
python scripts/video_gun_detector.py --camera 1 --threshold 0.7
```

## Controls
- Press 'q' to quit the application

## Features
- Real-time gun detection from video feed
- FPS counter
- Detection count display
- Confidence scores for each detection
- Bounding box visualization 