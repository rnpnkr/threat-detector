# Local Threat Detection Demo

This project is a proof-of-concept for a threat detection system, designed as a demo in Nagpur, India. It:
1. Allows uploading an image locally through a web-based UI.
2. Uses a YOLOv8-based AI model to detect weapons (guns/knives) in the uploaded image.
3. Sends a WhatsApp alert with a screenshot of the detection output when a threat is detected.

This is the initial phase, running entirely on a local machine. Future phases will incorporate mobile uploads and live video streaming (potentially tested on Android for better quality). The solution is designed to be general-purpose for later expansion.


---

## Prerequisites

- **Local Machine**: A computer (Windows, macOS, or Linux) with Python 3.8+ installed.
- **Python Environment**: For running the YOLOv8 model and Flask server.
- **WhatsApp Account**: For sending alerts (via Twilio API).
- **Image Source**: Local images (e.g., from your filesystem) for testing.

---

## Project Structure

This phase focuses on local image upload via a web UI and weapon detection. Mobile uploads and streaming are planned for later.


## Important
Make sure all requirements and packages are executed via a new virtual envioronment "threat-ml"