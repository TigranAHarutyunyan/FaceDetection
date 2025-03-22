import os
import cv2
import gc
import torch
import multiprocessing
from ultralytics import YOLO

# Set device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_range(video_path, start_frame, end_frame, frame_skip, output_dir, model_path):
    """
    Process a range of video frames and save detected faces.
    
    Args:
        video_path (str): Path to the video file.
        start_frame (int): Starting frame number for this worker.
        end_frame (int): Ending frame number for this worker.
        frame_skip (int): Process every 'frame_skip'-th frame.
        output_dir (str): Directory to save the detected face images.
        model_path (str): Path to the YOLO model file.
    """
    # Initialize the YOLO model within this worker process
    model = YOLO(model_path).to(device)
    if device == "cuda":
        model.fuse()
        model.half()

    print(f"[{multiprocessing.current_process().name}] Processing frames {start_frame}–{end_frame}")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_id = start_frame
    while cap.isOpened() and frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            try:
                # Run the model on the frame
                results = model(frame)  # YOLO automatically handles dtype conversion
                h, w, _ = frame.shape
                for box in results[0].boxes.xyxy:
                    # Convert bounding box coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    # Add 5% padding to each side of the bounding box
                    padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                    x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                    x2, y2 = min(w, x2 + padding_x), min(h, y2 + padding_y)
                    # Crop the face region from the frame
                    face = frame[y1:y2, x1:x2]
                    # Save the face image to the output directory
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_face.jpg"), face)
            except Exception as e:
                print(f"[ERROR] Frame {frame_id}: {e}")

        frame_id += 1
        del frame
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU cache if using CUDA

    cap.release()

class VideoProcessor:
    def __init__(self, video_path, model_path, frame_skip=5, output_dir='./faces', max_gpu_workers=2):
        """
        Initialize the VideoProcessor.
        
        Args:
            video_path (str): Path to the video file.
            model_path (str): Path to the YOLO model file.
            frame_skip (int): Process every 'frame_skip'-th frame.
            output_dir (str): Directory where output face images will be saved.
            max_gpu_workers (int): Maximum number of workers per GPU (if available).
        """
        self.video_path = video_path
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        self.device = device
        self.num_workers = self._choose_worker_count(max_gpu_workers)
        os.makedirs(self.output_dir, exist_ok=True)

    def _choose_worker_count(self, max_gpu_workers):
        """
        Determine the number of worker processes based on available CPU and GPU cores.
        
        Args:
            max_gpu_workers (int): Maximum workers allowed per GPU core.
        
        Returns:
            int: Number of worker processes to use.
        """
        cpu_cores = multiprocessing.cpu_count()
        gpu_cores = torch.cuda.device_count() if device == "cuda" else 0
        return min(cpu_cores, gpu_cores * max_gpu_workers if gpu_cores else 8)

    def _split_video_ranges(self):
        """
        Split the video into frame ranges for each worker process.
        
        Returns:
            list of tuples: Each tuple contains (start_frame, end_frame) for a worker.
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        part_len = total_frames // self.num_workers
        return [
            (i * part_len, (i + 1) * part_len if i < self.num_workers - 1 else total_frames)
            for i in range(self.num_workers)
        ]

    def process(self):
        """
        Process the video using multiprocessing.Pool.
        """
        print(f"[INFO] Device: {self.device.upper()}, Workers: {self.num_workers}")
        frame_ranges = self._split_video_ranges()

        # Use multiprocessing.Pool to parallelize the processing across processes
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            args = [
                (self.video_path, start, end, self.frame_skip, self.output_dir, self.model_path)
                for start, end in frame_ranges
            ]
            pool.starmap(process_range, args)

        print("[INFO] Processing complete.")

if __name__ == "__main__":
    # Set the video and model paths (adjust these paths to your environment)
    video_path = "/content/drive/MyDrive/7119947-hd_1080_1920_25fps (1).mp4"
    model_path = "/content/drive/MyDrive/yolov8n-face.pt"

    # Initialize the VideoProcessor and start processing the video
    processor = VideoProcessor(video_path, model_path)
    processor.process()

