import os
import logging
from ultralytics import YOLO
import numpy as np # Often needed alongside YOLO

logger = logging.getLogger(__name__)

# Define YOLO model path relative to this file's directory might be tricky.
# Let's assume the path is passed in, or construct it relative to backend_dir passed in.
# For now, define it as it was in app.py, assuming app.py passes the correct backend_dir
# BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ../
# YOLO_MODEL_PATH = os.path.join(BACKEND_DIR, "yolo_model", "yolo11m.pt")

def initialize_yolo(model_path, device):
    """Initializes the YOLO model."""
    yolo_model = None
    try:
        if not os.path.exists(model_path):
            logger.error(f"YOLO model file not found at {model_path}. Cannot initialize.")
            return None
        else:
            logger.info(f"Initializing YOLO model from {model_path} on device: {device}")
            # Note: YOLO automatically uses DEVICE if it's 'cuda:0' etc.
            yolo_model = YOLO(model_path)
            # Optional: Run a dummy inference to fully load model onto GPU if needed
            # Test with a small black image
            # try:
            #     dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            #     yolo_model.predict(dummy_img, verbose=False, device=device)
            #     logger.info("YOLO model dummy inference successful.")
            # except Exception as dummy_e:
            #      logger.warning(f"YOLO dummy inference failed (non-critical): {dummy_e}")

            logger.info("YOLO model initialized successfully.")
            return yolo_model
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model at {model_path}: {e}", exc_info=True)
        return None
