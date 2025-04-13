import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import io
from mmdet.apis import init_detector, inference_detector
import torch
import win32clipboard
from io import BytesIO

class GunDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gun Detection GUI")
        
        # Initialize model
        config_file = 'configs/gun_detection/convnext.py'
        checkpoint_file = 'work_dirs/convnext/epoch_3.pth'
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        
        # Create GUI elements
        self.create_widgets()
        
        # Bind paste event
        self.root.bind('<Control-v>', self.paste_image)
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Instructions label
        ttk.Label(main_frame, text="Press Ctrl+V to paste image from clipboard").grid(row=0, column=0, columnspan=2, pady=5)
        
        # Left panel for image
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, padx=5)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(left_panel, width=800, height=600)
        self.canvas.pack(pady=5)
        
        # Right panel for results
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, padx=5, sticky='n')
        
        # Results label
        ttk.Label(right_panel, text="Detection Results:", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Create text widget for results
        self.results_text = tk.Text(right_panel, width=40, height=20, font=('Courier', 10))
        self.results_text.pack(pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
    def filter_gun_detections(self, bbox, score, img_shape):
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
    
    def get_clipboard_image(self):
        """Get image from clipboard"""
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
                win32clipboard.CloseClipboard()
                
                # Convert clipboard data to image
                stream = BytesIO(data)
                image = Image.open(stream)
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            win32clipboard.CloseClipboard()
        except:
            return None
        return None
    
    def process_image(self, img):
        """Process image with gun detection model"""
        # Run inference
        result = inference_detector(self.model, img)
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Detected Guns:\n\n")
        
        detection_count = 0
        
        # Handle different result formats
        if isinstance(result, list):
            for class_id, class_result in enumerate(result):
                if len(class_result) > 0:
                    for detection in class_result:
                        bbox = detection[:4]
                        score = detection[4]
                        
                        if self.filter_gun_detections(bbox, score, img.shape):
                            detection_count += 1
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Add to results text
                            self.results_text.insert(tk.END, f"Gun {detection_count}:\n")
                            self.results_text.insert(tk.END, f"Confidence: {score:.2%}\n")
                            self.results_text.insert(tk.END, f"Location: ({x1}, {y1}) to ({x2}, {y2})\n\n")
        else:
            predictions = result.pred_instances
            for i in range(len(predictions.bboxes)):
                bbox = predictions.bboxes[i].cpu().numpy()
                score = predictions.scores[i].cpu().numpy()
                label = predictions.labels[i].cpu().numpy()
                
                if label == 0 and self.filter_gun_detections(bbox, score, img.shape):
                    detection_count += 1
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f'Gun: {score:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add to results text
                    self.results_text.insert(tk.END, f"Gun {detection_count}:\n")
                    self.results_text.insert(tk.END, f"Confidence: {score:.2%}\n")
                    self.results_text.insert(tk.END, f"Location: ({x1}, {y1}) to ({x2}, {y2})\n\n")
        
        if detection_count == 0:
            self.results_text.insert(tk.END, "No guns detected in the image.")
        
        return vis_img
    
    def paste_image(self, event=None):
        """Handle paste event"""
        # Get image from clipboard
        img = self.get_clipboard_image()
        if img is None:
            self.status_label.config(text="No image in clipboard!")
            return
        
        # Process image
        self.status_label.config(text="Processing image...")
        self.root.update()
        
        try:
            # Run detection
            result_img = self.process_image(img)
            
            # Convert to RGB for display
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas while maintaining aspect ratio
            height, width = result_img.shape[:2]
            canvas_width = 800
            canvas_height = 600
            scale = min(canvas_width/width, canvas_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            result_img_resized = cv2.resize(result_img_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(result_img_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference
            
            self.status_label.config(text="Detection complete!")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = GunDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 