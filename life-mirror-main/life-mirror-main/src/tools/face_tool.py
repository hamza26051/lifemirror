import os
import math
import tempfile
import requests
import numpy as np
from .base import BaseTool, ToolInput, ToolResult
# S3 upload removed - using local storage instead

MODE = os.getenv("LIFEMIRROR_MODE", "mock")
USE_DEEPFACE = os.getenv("FACE_USE_DEEPFACE", "false").lower() in ("1", "true", "yes")

mp = None
cv2 = None
deepface = None

def _ensure_deps():
    global mp, cv2, deepface
    if mp is None:
        try:
            import mediapipe as mp_pkg
            mp = mp_pkg
        except Exception as e:
            # Handle MediaPipe DLL loading issues on Windows
            if "DLL load failed" in str(e) or "_framework_bindings" in str(e):
                print(f"Warning: MediaPipe DLL loading failed: {e}")
                print("Falling back to mock mode for face detection")
                # Set a flag to indicate MediaPipe is unavailable
                mp = "unavailable"
            else:
                raise e
    if cv2 is None:
        try:
            import cv2 as cv_pkg
            cv2 = cv_pkg
        except Exception as e:
            print(f"Warning: OpenCV loading failed: {e}")
            cv2 = "unavailable"
    if USE_DEEPFACE and deepface is None:
        try:
            import deepface as df_pkg
            deepface = df_pkg
        except Exception:
            deepface = None

def _download_image_to_np(url_or_path: str) -> np.ndarray:
    _ensure_deps()
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        # Check if this is a localhost URL pointing to our own storage
        if "localhost:8000/storage/media/" in url_or_path:
            # Extract the local file path from the URL
            # URL format: http://localhost:8000/storage/media/{media_id}/{filename}
            url_parts = url_or_path.split("/storage/media/")
            if len(url_parts) == 2:
                local_path = f"uploads/media/{url_parts[1]}"
                if os.path.exists(local_path):
                    img = cv2.imread(local_path)
                    if img is None:
                        raise ValueError(f"Unable to read local image path: {local_path}")
                    return img
        
        # Fallback to HTTP request for external URLs
        resp = requests.get(url_or_path, timeout=20)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from url")
        return img
    else:
        img = cv2.imread(url_or_path)
        if img is None:
            raise ValueError("Unable to read local image path")
        return img

def _landmarks_to_xy(landmarks, image_width, image_height):
    return [[float(lm.x * image_width), float(lm.y * image_height)] for lm in landmarks]

class FaceTool(BaseTool):
    name = "face"

    def run(self, input: ToolInput) -> ToolResult:

        mode = os.getenv("LIFEMIRROR_MODE", "mock")
        
        # Check dependencies first
        _ensure_deps()
        
        # Fall back to mock mode if MediaPipe or OpenCV are unavailable
        if mode == "mock" or mp == "unavailable" or cv2 == "unavailable":
            if mp == "unavailable" or cv2 == "unavailable":
                print("Face detection falling back to mock mode due to dependency issues")
            return ToolResult(
                success=True,
                data={
                    "faces": [
                        {
                            "bbox": [100, 50, 80, 80],
                            "landmarks": {"left_eye": [110, 70], "right_eye": [150, 70]},
                            "crop_url": input.url,
                            "attributes": {"gender": None, "age": None, "expression": None}
                        }
                    ]
                }
            )

        try:
            # Try FaceDetection first for better detection
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            img = _download_image_to_np(input.url)
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # First detect faces with FaceDetection
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
                detection_results = face_detection.process(img_rgb)
                print(f"FaceDetection results: {detection_results.detections is not None}")
                if detection_results.detections:
                    print(f"Number of faces detected by FaceDetection: {len(detection_results.detections)}")
                
            # Then use FaceMesh for landmarks
            with mp_face_mesh.FaceMesh(static_image_mode=True,
                                       max_num_faces=5,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.3) as face_mesh:
                results = face_mesh.process(img_rgb)
                faces_out = []
                print(f"FaceMesh results: {results.multi_face_landmarks is not None}")
                if results.multi_face_landmarks:
                    print(f"Number of faces detected by FaceMesh: {len(results.multi_face_landmarks)}")
                
                # If FaceMesh fails but FaceDetection succeeded, use detection results
                if not results.multi_face_landmarks and detection_results.detections:
                    print("Using FaceDetection results since FaceMesh failed")
                    for i, detection in enumerate(detection_results.detections):
                        bbox_rel = detection.location_data.relative_bounding_box
                        x_min = bbox_rel.xmin * w
                        y_min = bbox_rel.ymin * h
                        width = bbox_rel.width * w
                        height = bbox_rel.height * h
                        bbox = [x_min, y_min, width, height]
                        
                        # Create basic landmarks from bounding box
                        landmarks = {
                            "left_eye": [x_min + width * 0.3, y_min + height * 0.4],
                            "right_eye": [x_min + width * 0.7, y_min + height * 0.4],
                            "nose_tip": [x_min + width * 0.5, y_min + height * 0.6],
                        }
                        
                        # Crop image
                        x0, y0 = max(int(x_min), 0), max(int(y_min), 0)
                        x1, y1 = min(int(x_min + width), w), min(int(y_min + height), h)
                        crop = img[y0:y1, x0:x1]
                        _, buf = cv2.imencode(".jpg", crop)
                        
                        # Save locally
                        os.makedirs("uploads/faces", exist_ok=True)
                        crop_filename = f"face_crop_{os.path.basename(input.url)}_{i}.jpg"
                        crop_path = os.path.join("uploads/faces", crop_filename)
                        with open(crop_path, "wb") as f:
                            f.write(buf.tobytes())
                        crop_url = f"http://localhost:8000/storage/faces/{crop_filename}"
                        
                        attributes = {"gender": None, "age": None, "expression": None}
                        if USE_DEEPFACE and deepface is not None:
                            try:
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                analysis = deepface.DeepFace.analyze(
                                    img_path=crop_rgb,
                                    actions=['age', 'gender', 'emotion'],
                                    enforce_detection=False
                                )
                                print(f"DeepFace analysis result: {analysis}")
                                if isinstance(analysis, list) and len(analysis) > 0:
                                    analysis = analysis[0]
                                attributes["age"] = int(analysis.get("age")) if analysis.get("age") else None
                                attributes["gender"] = analysis.get("dominant_gender") or analysis.get("gender")
                                if isinstance(analysis.get("emotion"), dict):
                                    top_emotion = max(analysis["emotion"].items(), key=lambda x: x[1])[0]
                                    attributes["expression"] = top_emotion
                                elif analysis.get("dominant_emotion"):
                                    attributes["expression"] = analysis.get("dominant_emotion")
                            except Exception as e:
                                print(f"DeepFace analysis failed: {e}")
                                pass
                        
                        faces_out.append({
                            "bbox": bbox,
                            "landmarks": landmarks,
                            "crop_url": crop_url,
                            "attributes": attributes
                        })
                    
                    return ToolResult(success=True, data={"faces": faces_out})
                
                if not results.multi_face_landmarks:
                    print("No faces detected by either method")
                    return ToolResult(success=True, data={"faces": []})

                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    pts = _landmarks_to_xy(face_landmarks.landmark, w, h)
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    x_min, x_max = float(min(xs)), float(max(xs))
                    y_min, y_max = float(min(ys)), float(max(ys))
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    def _get_lm(idx):
                        lm = face_landmarks.landmark[idx]
                        return [float(lm.x * w), float(lm.y * h)]
                    landmarks = {
                        "left_eye": _get_lm(33),
                        "right_eye": _get_lm(263),
                        "nose_tip": _get_lm(1),
                    }

                    # Crop image
                    x0, y0 = max(int(x_min), 0), max(int(y_min), 0)
                    x1, y1 = min(int(x_max), w), min(int(y_max), h)
                    crop = img[y0:y1, x0:x1]
                    _, buf = cv2.imencode(".jpg", crop)
                    
                    # Save locally instead of S3 upload
                    os.makedirs("uploads/faces", exist_ok=True)
                    crop_filename = f"face_crop_{os.path.basename(input.url)}_{i}.jpg"
                    crop_path = os.path.join("uploads/faces", crop_filename)
                    with open(crop_path, "wb") as f:
                        f.write(buf.tobytes())
                    crop_url = f"http://localhost:8000/storage/faces/{crop_filename}"

                    attributes = {"gender": None, "age": None, "expression": None}
                    if USE_DEEPFACE and deepface is not None:
                        try:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            analysis = deepface.DeepFace.analyze(
                                img_path=crop_rgb,
                                actions=['age', 'gender', 'emotion'],
                                enforce_detection=False
                            )
                            attributes["age"] = int(analysis.get("age")) if analysis.get("age") else None
                            attributes["gender"] = analysis.get("gender")
                            if isinstance(analysis.get("emotion"), dict):
                                top_emotion = max(analysis["emotion"].items(), key=lambda x: x[1])[0]
                                attributes["expression"] = top_emotion
                        except Exception as e:
                            print(f"DeepFace analysis failed: {e}")
                            pass

                    faces_out.append({
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "crop_url": crop_url,
                        "attributes": attributes
                    })

                return ToolResult(success=True, data={"faces": faces_out})

        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))
