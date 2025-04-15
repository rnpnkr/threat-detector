import argparse
import cv2
import mmcv
import torch
import numpy as np
import os
import time  # Import the time module
from mmdet.apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector on a single image')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--image', help='path to input image')
    parser.add_argument('--output', help='path to output image')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Build the model from a config file and a checkpoint file
    print("Initializing the model...")
    model_init_start = time.time()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model_init_end = time.time()
    print(f"Model initialization took: {model_init_end - model_init_start:.4f} seconds")
    
    # Read the image
    print(f"Reading image: {args.image}")
    img = mmcv.imread(args.image)
    
    # Run inference and time it
    print("Running inference...")
    start_time = time.time()
    result = inference_detector(model, img)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Inference processing time: {processing_time:.4f} seconds") # Print the processing time
    
    # Create a copy of the image for visualization
    print("Visualizing results...")
    vis_img = img.copy()
    
    # Handle different result formats
    if isinstance(result, list):
        # For older MMDetection versions
        for class_id, class_result in enumerate(result):
            if len(class_result) > 0:
                # The format is [x1, y1, x2, y2, score]
                for detection in class_result:
                    # check if the class is gun
                    if class_id == 1:
                        bbox = detection[:4]
                        score = detection[4]
                        if score > 0.3:  # Confidence threshold
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # For newer MMDetection versions
        predictions = result.pred_instances
        for i in range(len(predictions.bboxes)):
            bbox = predictions.bboxes[i].cpu().numpy()
            score = predictions.scores[i].cpu().numpy()
            label = predictions.labels[i].cpu().numpy()
            
            # Only draw if it's a gun (assuming class 0 is gun)
            if label == 1 and score > 0.1:  # You can adjust the confidence threshold
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Ensure output file has a valid extension
    if not args.output.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        if os.path.isdir(args.output):
            # If output is a directory, create a file name based on the input image
            input_filename = os.path.basename(args.image)
            output_filename = os.path.splitext(input_filename)[0] + '_result.jpg'
            args.output = os.path.join(args.output, output_filename)
        else:
            args.output = args.output + '.jpg'
    
    # Save the result
    print(f"Saving result to {args.output}")
    mmcv.imwrite(vis_img, args.output)
    print(f'Result saved to {args.output}')

if __name__ == '__main__':
    main() 