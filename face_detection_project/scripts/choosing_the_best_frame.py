import cv2
import mediapipe as mp
import os
import numpy as np
import shutil

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def evaluate_frontalness(image_rgb):
    """
    Evaluates how directly the face is looking at the camera based on the distances from the nose to the outer eye corners.
    Returns a value from 0 to 1, where 1 means a perfectly frontal face.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return 0
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        left_distance = abs(nose_tip.x - left_eye.x)
        right_distance = abs(right_eye.x - nose_tip.x)
        if max(left_distance, right_distance) < 1e-6:
            return 0
        ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
        return ratio

def evaluate_eyes_open(image_rgb):
    """
    Evaluates how open the eyes are using the eye aspect ratio (EAR).
    Returns a value from 0 to 1, where 1 means fully open eyes.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return 0
        landmarks = results.multi_face_landmarks[0].landmark

        # Left eye
        left_corner = landmarks[33]
        right_corner = landmarks[133]
        top_left = landmarks[159]
        bottom_left = landmarks[145]
        
        # Right eye
        right_corner_r = landmarks[263]
        left_corner_r = landmarks[362]
        top_right = landmarks[386]
        bottom_right = landmarks[374]
        
        # EAR calculation
        left_horizontal = np.sqrt((left_corner.x - right_corner.x)**2 + (left_corner.y - right_corner.y)**2)
        left_vertical = np.sqrt((top_left.x - bottom_left.x)**2 + (top_left.y - bottom_left.y)**2)
        left_ear = left_vertical / left_horizontal if left_horizontal > 0 else 0

        right_horizontal = np.sqrt((right_corner_r.x - left_corner_r.x)**2 + (right_corner_r.y - left_corner_r.y)**2)
        right_vertical = np.sqrt((top_right.x - bottom_right.x)**2 + (top_right.y - bottom_right.y)**2)
        right_ear = right_vertical / right_horizontal if right_horizontal > 0 else 0

        ear_avg = (left_ear + right_ear) / 2
        normalized_ear = np.clip((ear_avg - 0.2) / 0.2, 0, 1)
        return normalized_ear

def evaluate_face_quality(image_path):
    """
    Computes the final score of an image based on:
      1. Sharpness.
      2. Brightness.
      3. Face coverage.
      4. Centering.
      5. Frontalness.
      6. Eye openness.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(image_rgb)
        if not results.detections:
            return 0
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_x, face_y = bbox.xmin * w, bbox.ymin * h
        face_w, face_h = bbox.width * w, bbox.height * h
        face_area = face_w * face_h
        coverage = face_area / (w * h)

        # Centering
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        center_distance = np.sqrt((face_center_x - w / 2) ** 2 + (face_center_y - h / 2) ** 2)
        center_distance_norm = center_distance / np.sqrt(w ** 2 + h ** 2)
    
    # Quality metrics
    sharpness = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    brightness_score = 1 - abs(brightness - 128) / 128
    frontalness = evaluate_frontalness(image_rgb)
    eyes_open = evaluate_eyes_open(image_rgb)

    # Weights for each criterion
    w_sharp, w_bright, w_cov, w_center, w_frontal, w_eye = 0.20, 0.20, 0.15, 0.15, 0.15, 0.15
    
    # Sharpness normalization
    sharpness_norm = np.clip(sharpness / 1000.0, 0, 1)
    
    # Final score
    final_score = (w_sharp * sharpness_norm +
                   w_bright * brightness_score +
                   w_cov * coverage +
                   w_center * (1 - center_distance_norm) +
                   w_frontal * frontalness +
                   w_eye * eyes_open)
    
    return final_score

def select_best_image_in_folder(folder_path):
    """
    Selects the best image in a folder based on the highest score.
    """
    best_score = -1
    best_image = None
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            score = evaluate_face_quality(image_path)
            if score > best_score:
                best_score = score
                best_image = image_path
    return best_image, best_score

def process_all_folders(parent_folder, output_folder):
    """
    Processes all folders and copies the best photo to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            best_image, score = select_best_image_in_folder(folder_path)
            if best_image:
                output_subfolder = os.path.join(output_folder, folder)
                os.makedirs(output_subfolder, exist_ok=True)
                output_path = os.path.join(output_subfolder, f"{folder}_best.jpg")
                shutil.copy(best_image, output_path)
                print(f"Best photo in '{folder}': {os.path.basename(best_image)}, score {score:.2f} -> saved to {output_path}")
            else:
                print(f"No suitable images found in '{folder}'")

if __name__ == "__main__":
    parent_folder = "/content/clustered_faces"
    output_folder = "/content/best_faces"
    process_all_folders(parent_folder, output_folder)

