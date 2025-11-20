"""Detect crosswalk/traffic-light classes using YOLOv5 PyTorch (.pt) weights."""

import os
import cv2
import torch
import numpy as np

from config import crosswalk_class_names, crosswalk_yolov5_pt
from modules.LensOpticCalculator import RatioProportionCalculator, LimitVal, SafetyLevel

_crosswalk_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _crosswalk_model
    if _crosswalk_model is not None:
        return _crosswalk_model

    repo_dir = os.path.join(os.path.dirname(__file__), "..", "yolov5")
    repo_dir = os.path.abspath(repo_dir)
    _crosswalk_model = torch.hub.load(repo_dir, "custom", path=crosswalk_yolov5_pt, source="local", force_reload=False)
    _crosswalk_model.to(_device)
    _crosswalk_model.eval()
    return _crosswalk_model


def FindCrosswalkObjects(img, target_object_depth_val, monocular_depth_val):
    """
    Detects crosswalk/traffic-light objects and overlays distance estimation based on the
    existing target object (license plate) reference depth.
    """
    if not bool(target_object_depth_val):
        # Need target object to establish reference distance.
        return

    reference_distance = target_object_depth_val[0]
    reference_point = target_object_depth_val[1]

    model = _load_model()

    # YOLOv5 expects RGB uint8 images; img is already RGB upstream.
    results = model(img, size=640)
    preds = results.xyxy[0].detach().cpu().numpy() if hasattr(results, "xyxy") else np.empty((0, 6))

    hT, wT, _ = img.shape

    for det in preds:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        if conf < 0.25:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)

        xcoord = (x1 + x2) / 2
        ycoord = (y1 + y2) / 2

        xcoord = LimitVal(xcoord, wT)
        ycoord = LimitVal(ycoord, hT)

        object_depthmap_val = monocular_depth_val[int(ycoord), int(xcoord)]
        computed_distance = RatioProportionCalculator(object_depthmap_val, reference_distance, reference_point)
        safety = SafetyLevel(computed_distance)

        color = safety[1]
        label = f"{crosswalk_class_names[cls_id].upper()}- {computed_distance:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(img, (int(xcoord), int(ycoord)), 3, color, -1)
