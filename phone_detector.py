
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import torch
import cv2
import numpy as np

from models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.dataloaders import letterbox

class PhoneDetector:
    def __init__(self, model_path='yolov5s.pt', device='cpu', conf_thres=0.25, iou_thres=0.45):
        self.device = device
        self.model = DetectMultiBackend(model_path, device=device)
        self.model.eval()
        self.names = self.model.names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame):
        # Resize and pad image
        img = letterbox(frame, new_shape=640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        # Convert to torch
        img = torch.from_numpy(img).to(self.device).float() / 255.0  # Normalize to 0-1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=False, visualize=False)

            # Handle model outputs
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        # Check if any 'cell phone' class is detected
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                if self.names[int(cls)] == 'cell phone':
                    return True

        return False
