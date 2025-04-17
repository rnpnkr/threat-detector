# Product Context: Threat Detector

## Problem Solved
Standard CCTV systems often require constant human monitoring to detect threats. This system aims to automate the detection of visible threats like weapons in real-time, allowing security personnel to react faster and more effectively. It also provides an API for analyzing static images for threats.

## Target Users
- Security personnel monitoring CCTV feeds.
- Law enforcement or security analysts reviewing footage/images.

## How it Should Work
- **Real-time:** Connects to CCTV streams, processes frames, detects threats (guns/knives), and potentially people.
- **Alerting:** When a threat is detected (especially a weapon associated with a person), generates an alert (initially via logs/console, eventually WhatsApp).
- **Static Analysis:** Accepts image uploads via API, runs detection, returns results including bounding boxes and confidence scores.
- **Visualization:** Displays bounding boxes on images/video frames for detected objects.

## User Experience Goals
- **Reliability:** Detection should be accurate with minimal false positives/negatives.
- **Timeliness:** Real-time alerts should be delivered promptly.
- **Clarity:** Visualizations and alert information should be clear and easy to understand.
- **Stability:** **(Current Challenge)** Detections, especially for tracked individuals and associated weapons, should be consistent across frames, avoiding confusing flickering or dropped tracks. The goal is to reliably identify and track the specific individuals associated with threats (e.g., the 2 persons with guns in a crowd). 