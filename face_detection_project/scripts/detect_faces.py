import cv2
import os
import tempfile
import concurrent.futures
import gc
import multiprocessing
from ultralytics import YOLO

def get_optimal_threads():
    """Get optimal thread count (max 8)."""
    return min(multiprocessing.cpu_count(), 8)

def process_video_part(video_path, start_frame, end_frame, frame_skip=5, output_dir='./faces_part'):
    """Process video segment, detect and save faces."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n-face.pt")  
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  

    frame_id = start_frame
    temp_files = []

    while cap.isOpened() and frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_skip == 0:
            results = model(frame)  

            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]  

                temp_file = tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix='.jpg')
                cv2.imwrite(temp_file.name, face)  
                print(f"Saved: {temp_file.name}")
                
                temp_files.append(temp_file.name)

        frame_id += 1
        del frame
        gc.collect()

    cap.release()

def split_video_into_parts(video_path, num_parts):
    """Split video into parts."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    part_length = total_frames // num_parts
    return [(i * part_length, (i + 1) * part_length if i < num_parts - 1 else total_frames) for i in range(num_parts)]

def process_video_in_parts(video_path, frame_skip=5, output_dir='./faces_part'):
    """Run parallel face detection on video."""
    num_parts = get_optimal_threads()  
    print(f"Using {num_parts} threads")

    frame_ranges = split_video_into_parts(video_path, num_parts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parts) as executor:
        futures = [executor.submit(process_video_part, video_path, start, end, frame_skip, output_dir) for start, end in frame_ranges]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

# Run the function
video_path = "/content/8877996-hd_1920_1080_25fps.mp4"
process_video_in_parts(video_path, frame_skip=5, output_dir='./faces_part')

