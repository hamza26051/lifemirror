import os
from .base import BaseTool, ToolInput, ToolResult

class DetectTool(BaseTool):
    name = 'detect'

    def run(self, input: ToolInput) -> ToolResult:
        mode = os.getenv("LIFEMIRROR_MODE", "mock")

        if mode == "mock":
            return ToolResult(
                success=True,
                data={
                    "detections": [
                        {"label": "person", "score": 0.98, "bbox": [0.1, 0.1, 0.8, 0.9]},
                        {"label": "shirt", "score": 0.88, "bbox": [0.2, 0.3, 0.6, 0.5]}
                    ]
                }
            )

        # --- PROD MODE ---
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            import requests
            
            model = YOLO("yolov8n.pt")  # small model for speed, can switch to yolov8m/l
            
            # Handle different input types
            image = None
            if input.url.startswith('http'):
                # Download image from URL
                response = requests.get(input.url, timeout=10)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    return ToolResult(success=False, data={}, error=f"Failed to download image from {input.url}")
            else:
                # Local file path
                if os.path.exists(input.url):
                    image = cv2.imread(input.url)
                else:
                    return ToolResult(success=False, data={}, error=f"File not found: {input.url}")
            
            if image is None:
                return ToolResult(success=False, data={}, error="Failed to load image")
            
            # Run YOLO detection on the single image
            results = model(image, verbose=False)  # Disable verbose output
            detections = []
            
            if results and len(results) > 0:
                r = results[0]  # Get first (and only) result
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_name = r.names[int(box.cls[0])]
                        score = float(box.conf[0])
                        xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                        detections.append({"label": cls_name, "score": score, "bbox": xywh})
            
            return ToolResult(success=True, data={"detections": detections})

        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))
