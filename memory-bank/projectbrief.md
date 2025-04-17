# Project Brief: Real-time Threat Detection System

## Core Requirements
- Real-time video feed processing from CCTV cameras in Nagpur, India
- MMDetection-based AI model (ConvNext from `CCTV_GUN`) for weapon detection (guns)
- WhatsApp alert system for threat notifications
- Profiling system for threat identification
- Multi-language support via LLM integration

## Project Goals
1. Create a real-time threat detection system for CCTV cameras
2. Demonstrate weapon detection capabilities in live video feeds
3. Implement automated alert system with profiling
4. Build foundation for multi-language support

## Project Scope
### Phase 1 (Current)
- Basic image upload and detection via web UI
- Weapon detection using MMDetection (ConvNext) model
- Display of detection results with bounding boxes
- WebSocket video streaming with frame skipping (due to CPU limits)
- Local CPU-based testing and deployment

### Phase 2 (In Progress)
- Deployment to Cloud GPU Instance
- Performance optimization on GPU
- Integration with CCTV cameras (pending)
- Profiling system implementation
- WhatsApp alert system

### Future Phases
- LLM integration for multi-language support
- Enhanced profiling capabilities
- System optimization and scaling
- Additional threat detection categories

## Success Criteria
- Successful real-time weapon detection in video feeds
- Accurate profiling of detected threats
- Timely WhatsApp notifications to concerned officers
- Reliable system operation in production environment
- Scalable architecture for future features 