# Project Brief: Threat Detector

## Core Goal
Develop a system capable of detecting potential threats (initially weapons like guns and knives) in real-time video feeds (CCTV) and static images, providing timely alerts.

## Key Requirements
- Real-time detection from video streams.
- Static image analysis via API.
- Initial focus on weapon detection (guns/knives).
- Bounding box visualization for detected objects.
- Basic API for integration (health check, detection endpoint).
- **NEW:** Improve detection stability and tracking consistency, especially for associating weapons with specific individuals, to address flickering issues.

## Scope
- Backend API (Flask)
- Detection Model Integration (MMDetection initially, potential for others)
- Basic Frontend for Upload/Display (React - In Progress)
- WebSocket for real-time video feed results.
- Alerting Mechanism (WhatsApp - Planned)
- Profiling System (Planned)

## Future Considerations
- LLM integration for multi-language support.
- Enhanced profiling capabilities.
- Expansion to other threat categories.
- Cloud deployment strategy (e.g., RunPod). 