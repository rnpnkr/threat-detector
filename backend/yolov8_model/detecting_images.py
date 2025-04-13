import cv2
from ultralytics import YOLO
import os
import logging
import base64
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter out YOLO's logs
class YOLOLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith('ultralytics')

# Apply the filter to the root logger
logging.getLogger().addFilter(YOLOLogFilter())

def detect_objects_in_photo(image_path):
    """
    Detect objects in a photo using YOLOv8.
    """
    try:
        # Read the image
        logger.info(f"Reading image from: {image_path}")
        image_orig = cv2.imread(image_path)
        if image_orig is None:
            logger.error(f"Could not read image {image_path}")
            return None
        
        logger.info(f"Image loaded successfully: {image_orig.shape}")
        
        # Check if model exists
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 'runs/detect/Normal_Compressed/weights/best.pt')
        logger.info(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
            
        logger.info(f"Model found at: {model_path}")
            
        # Initialize YOLO model
        logger.info("Initializing YOLO model...")
        yolo_model = YOLO(model_path)
        logger.info("Model loaded successfully, running detection...")
        
        # Run detection
        results = yolo_model(image_orig)
        logger.info("Detection completed")
        logger.info(f"Number of detections: {len(results)}")

        # Process results
        detections_data = []
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            logger.info(f"Found {len(detections)} potential detections")
            logger.info(f"Available classes: {classes}")
            
            for pos, detection in enumerate(detections):
                confidence = float(conf[pos])
                class_id = int(cls[pos])
                class_name = classes[class_id]
                logger.info(f"Detection {pos + 1}: Class={class_name}, Confidence={confidence:.2f}")
                
                if confidence >= 0.3:  # Lowered threshold to 0.3 (30%)
                    xmin, ymin, xmax, ymax = detection
                    label = f"{class_name} {confidence:.2f}" 
                    color = (0, 255, 0)  # Changed to green for better visibility
                    cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    logger.info(f"Drawing box at coordinates: ({int(xmin)}, {int(ymin)}) to ({int(xmax)}, {int(ymax)})")
                    
                    # Add detection data to list
                    detections_data.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "xmin": int(xmin),
                            "ymin": int(ymin),
                            "xmax": int(xmax),
                            "ymax": int(ymax)
                        }
                    })

        # Save results
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  'imgs/Test/detected_test.jpg')
        logger.info(f"Saving results to: {result_path}")
        cv2.imwrite(result_path, image_orig)
        logger.info(f"Detection results saved to: {result_path}")
        
        # Return both the result path and detection data
        return {
            "result_path": result_path,
            "detections": detections_data,
            "image_info": {
                "shape": image_orig.shape,
                "original_path": image_path
            },
            "model_info": {
                "path": model_path,
                "classes": classes
            }
        }
        
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}", exc_info=True)
        return None

def detect_objects_in_video(video_path):
    try:
        model_path = './runs/detect/Normal_Compressed/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        yolo_model = YOLO(model_path)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
        
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = "detected_objects_video2.avi"
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
    video_capture.release()
    out.release()

    return result_video_path

def detect_objects_and_plot(path_orig):
    image_orig = cv2.imread(path_orig)
    if image_orig is None:
        print(f"Error: Could not read image {path_orig}")
        return
    
    try:
        model_path = './runs/detect/Normal_Compressed/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        yolo_model = YOLO(model_path)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imshow("Test Detection", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video_stream(video_path, ws):
    """
    Process video stream in real-time and send frames via WebSocket.
    Args:
        video_path: Path to the video file
        ws: WebSocket connection
    """
    try:
        logger.info(f"Starting video processing for: {video_path}")
        
        # Load YOLO model using the correct path
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 'runs/detect/Normal_Compressed/weights/best.pt')
        logger.info(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
            
        logger.info("Loading YOLO model...")
        # Create YOLO model with verbosity set to False
        yolo_model = YOLO(model_path)
        # Set verbosity using the correct method
        yolo_model.verbose = False
        logger.info("Model loaded successfully")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return None
            
        logger.info("Video file opened successfully")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached")
                break
                
            frame_count += 1
            logger.debug(f"Processing frame {frame_count}/{total_frames}")
            
            # Run detection with verbose=False
            results = yolo_model(frame, conf=0.3, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        "bbox": {
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2
                        },
                        "class": "guns" if cls == 0 else "knife",
                        "confidence": conf
                    })
            
            # Draw annotations on frame
            annotated_frame = results[0].plot()
            
            # Convert frame to base64 for WebSocket transmission
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare WebSocket message
            message = {
                "type": "video_frame",
                "payload": {
                    "frame": frame_base64,
                    "detections": detections,
                    "frame_number": frame_count,
                    "total_frames": total_frames
                }
            }
            
            # Send frame via WebSocket
            try:
                ws.send(json.dumps(message))
                logger.debug(f"Frame {frame_count} sent via WebSocket")
            except Exception as e:
                logger.error(f"Error sending frame {frame_count}: {str(e)}")
                break
            
            # Add small delay to control frame rate
            time.sleep(1/fps)
        
        cap.release()
        logger.info("Video processing completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in process_video_stream: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    # Test image detection
    test_image = "./imgs/Test/concept-terrorist.jpg"
    print(f"Testing detection on image: {test_image}")
    result = detect_objects_in_photo(test_image)
    if result:
        print(f"Detection completed. Results saved to: {result}")
        # Display the result
        result_img = cv2.imread(result)
        if result_img is not None:
            cv2.imshow("Test Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Detection failed")