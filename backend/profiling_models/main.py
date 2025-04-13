import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
# import openpose  # Requires OpenPose Python API
import torch
from torchvision.models import resnet18

# Initialize models
yolo = YOLO("yolov8m.pt")
# openpose = openpose.OpenPose()  # Configure per OpenPose docs
complexion_model = resnet18(pretrained=False)  # Load FairFace-trained weights
complexion_model.eval()

# Load image
img = cv2.imread("person.jpeg")
original_height, original_width = img.shape[:2]
results = []

# Human detection
detections = yolo(img)
for r in detections:
    box = r.boxes.xyxy[0].numpy()
    x1, y1, x2, y2 = map(int, box)
    person = img[y1:y2, x1:x2]
    person_data = {}

    # Age estimation
    try:
        result = DeepFace.analyze(person, actions=['age'], enforce_detection=True)
        person_data["age"] = f"{result['age']}-{result['age']+10}"
    except:
        person_data["age"] = "Unknown"

    # Height estimation (disabled - requires OpenPose)
    # keypoints = openpose.forward(person)
    # if keypoints:
    #     head_y = keypoints[0][1]  # Head
    #     feet_y = max(keypoints[15][1], keypoints[16][1])  # Ankles
    #     pixel_height = feet_y - head_y
    #     # Assume reference object (e.g., 1m = 200px)
    #     real_height = (pixel_height / 200) * 100  # Adjust with calibration
    #     person_data["height_cm"] = int(real_height)
    # else:
    #     person_data["height_cm"] = "Unknown"
    person_data["height_cm"] = "Not available"  # Placeholder since height estimation is disabled

    # Complexion (placeholder)
    try:
        face = person  # Assume MTCNN crop
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            complexion_pred = complexion_model(face_tensor)
            # Get the predicted class index
            pred_idx = complexion_pred.argmax().item()
            # Map the prediction to a complexion category
            complexion_categories = ["Light", "Medium", "Dark"]
            if pred_idx < len(complexion_categories):
                person_data["complexion"] = complexion_categories[pred_idx]
            else:
                person_data["complexion"] = "Unknown"
    except Exception as e:
        print(f"Complexion detection error: {str(e)}")
        person_data["complexion"] = "Unknown"

    # Annotate image
    label = f"Age: {person_data['age']}, H: {person_data['height_cm']}, C: {person_data['complexion']}"
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 6)  # Increased line thickness
    
    # Improved text visibility
    font_scale = 3
    font_thickness = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Calculate text background coordinates
    text_x = x1
    text_y = y1 - 30  # Increased padding from top
    bg_x1 = text_x - 10  # Added padding
    bg_y1 = text_y - text_height - baseline - 10  # Added padding
    bg_x2 = text_x + text_width + 10  # Added padding
    bg_y2 = text_y + baseline + 10  # Added padding
    
    # Draw text background with thicker outline
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 200, 0), 6)  # Outline
    cv2.rectangle(img, (bg_x1+3, bg_y1+3), (bg_x2-3, bg_y2-3), (0, 255, 0), -1)  # Fill
    
    # Draw text
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    
    results.append(person_data)

# Save results
cv2.imwrite("annotated_image.jpg", img)
with open("results.json", "w") as f:
    import json
    json.dump(results, f)

# Display with proper window management
window_name = "Profiling"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Calculate the display size while maintaining aspect ratio
screen_height = 800  # Maximum height for display
scale = screen_height / original_height
display_width = int(original_width * scale)
display_height = screen_height

# Resize window to the calculated dimensions
cv2.resizeWindow(window_name, display_width, display_height)

# Show the image
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()