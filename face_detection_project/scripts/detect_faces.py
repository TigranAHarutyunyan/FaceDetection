import cv2
import os
import tempfile
import concurrent.futures
import gc
from ultralytics import YOLO

def process_video_part(video_path, start_frame, end_frame, frame_skip=5, output_dir='./faces_part'):
    # Create a temporary directory to save faces
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n-face.pt")  # Use the YOLOv8 model for face detection on GPU
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Move to the required starting frame

    frame_id = start_frame
    temp_files = []

    while cap.isOpened() and frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every N-th frame
        if frame_id % frame_skip == 0:
            results = model(frame)  # Detect faces on the frame

            # Save each detected face
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)  # Get face coordinates
                face = frame[y1:y2, x1:x2]  # Crop the face

                # Create a temporary file to save the face
                temp_file = tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix='.jpg')
                cv2.imwrite(temp_file.name, face)  # Save the face to a file
                print(f"Saved face to {temp_file.name}")
                
                temp_files.append(temp_file.name)

        frame_id += 1

        # Clean up memory after processing the frame
        del frame
        gc.collect()

    cap.release()
"""
    # Delete temporary files after use
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Deleted temporary file: {temp_file}")
        except Exception as e:
            print(f"Error deleting file {temp_file}: {e}")
"""
def split_video_into_parts(video_path, num_parts):
    # Get the total number of frames in the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Split the video into parts
    frame_range = []
    part_length = total_frames // num_parts
    for i in range(num_parts):
        start_frame = i * part_length
        end_frame = start_frame + part_length if i < num_parts - 1 else total_frames
        frame_range.append((start_frame, end_frame))
    
    return frame_range

def process_video_in_parts(video_path, num_parts=4, frame_skip=5, output_dir='./faces_part'):
    # Split the video into parts
    frame_ranges = split_video_into_parts(video_path, num_parts)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_video_part, video_path, start_frame, end_frame, frame_skip, output_dir)
            for start_frame, end_frame in frame_ranges
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result and handle errors
            except Exception as e:
                print(f"Error processing video part: {e}")

# Example of calling the function for one video, splitting it into 4 parts
video_path = "/content/8877996-hd_1920_1080_25fps.mp4"
process_video_in_parts(video_path, num_parts=4, frame_skip=5, output_dir='./faces_part')

