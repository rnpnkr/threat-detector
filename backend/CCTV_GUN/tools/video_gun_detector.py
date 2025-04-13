import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import torch
import argparse
import time
import os

def filter_gun_detections(bbox, score, img_shape):
    """Filter detections based on size and aspect ratio constraints"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    # Get image dimensions
    img_height, img_width = img_shape[:2]
    bbox_area = width * height
    img_area = img_width * img_height
    
    # Filter conditions for guns:
    # 1. Reasonable aspect ratio for guns (typically between 1.5 and 4.0)
    # 2. Size constraints (not too large compared to image)
    # 3. Higher confidence threshold
    is_valid_ratio = 1.5 < aspect_ratio < 4.0
    is_valid_size = bbox_area < 0.3 * img_area
    is_valid_score = score > 0.5
    
    return is_valid_ratio and is_valid_size and is_valid_score

def process_frame(model, frame):
    """Process a single frame with gun detection"""
    # Run inference
    result = inference_detector(model, frame)
    
    # Create a copy for visualization
    vis_img = frame.copy()
    detection_count = 0
    detections = []
    
    # Handle different result formats
    if isinstance(result, list):
        for class_id, class_result in enumerate(result):
            if len(class_result) > 0:
                for detection in class_result:
                    bbox = detection[:4]
                    score = detection[4]
                    
                    if filter_gun_detections(bbox, score, frame.shape):
                        detection_count += 1
                        x1, y1, x2, y2 = map(int, bbox)
                        detections.append((x1, y1, x2, y2, score))
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        predictions = result.pred_instances
        for i in range(len(predictions.bboxes)):
            bbox = predictions.bboxes[i].cpu().numpy()
            score = predictions.scores[i].cpu().numpy()
            label = predictions.labels[i].cpu().numpy()
            
            if label == 0 and filter_gun_detections(bbox, score, frame.shape):
                detection_count += 1
                x1, y1, x2, y2 = map(int, bbox)
                detections.append((x1, y1, x2, y2, score))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add detection count to frame
    cv2.putText(vis_img, f'Guns Detected: {detection_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis_img, detection_count, detections

def main():
    parser = argparse.ArgumentParser(description='Gun detection in video feed')
    parser.add_argument('--config', default='configs/gun_detection/convnext.py', help='Config file path')
    parser.add_argument('--checkpoint', default='work_dirs/convnext/epoch_3.pth', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--input', default='0', help='Input source. Use 0 for webcam or path to video file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--output', help='Output video file path (optional)')
    args = parser.parse_args()
    
    # Initialize the model
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # Open video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Starting video processing. Press 'q' to quit.")
    if not args.input.isdigit():
        print(f"Total frames: {total_frames}")
    
    frame_count = 0
    total_detections = 0
    processing_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        start_time = time.time()
        result_frame, num_detections, detections = process_frame(model, frame)
        process_time = time.time() - start_time
        
        # Update statistics
        frame_count += 1
        total_detections += num_detections
        processing_times.append(process_time)
        
        # Add FPS info
        fps = 1.0 / process_time
        cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add progress info for video files
        if not args.input.isdigit():
            progress = (frame_count / total_frames) * 100
            cv2.putText(result_frame, f'Progress: {progress:.1f}%', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Gun Detection', result_frame)
        
        # Save frame if writer exists
        if writer:
            writer.write(result_frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print statistics
    avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {total_detections}")
    print(f"Average FPS: {avg_fps:.1f}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 