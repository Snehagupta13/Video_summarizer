import os
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def convert_video(input_video):
    """Convert input video to MP4 format."""
    try:
        if not os.path.exists(input_video):
            raise FileNotFoundError("Input video file not found.")

        mp4_path = os.path.splitext(input_video)[0] + "_converted.mp4"
        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            raise Exception("Error: Unable to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        return mp4_path
    except Exception as e:
        print(f"Error in convert_video: {e}")
        return None

def extract_frames(video_path, frame_interval=30):
    """Extract frames from video at a specified interval."""
    try:
        output_folder = "frames"
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_frames = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)

            frame_count += 1

        cap.release()
        return extracted_frames
    except Exception as e:
        print(f"Error in extract_frames: {e}")
        return []

def generate_captions(frames, similarity_threshold=0.9):
    """Generate unique captions for extracted frames using BLIP."""
    from difflib import SequenceMatcher

    def is_similar(a: str, b: str, threshold: float = 0.9) -> bool:
        return SequenceMatcher(None, a, b).ratio() > threshold

    captions = []
    prev_caption = ""

    if not frames:
        print("[⚠️] No frames provided for captioning")
        return []

    for frame in frames:
        try:
            if not os.path.exists(frame):
                print(f"[⚠️] Frame not found: {frame}")
                continue

            image = cv2.imread(frame)
            if image is None:
                print(f"[⚠️] Failed to read frame: {frame}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(images=image, return_tensors="pt").to(device)
            caption_ids = model.generate(**inputs)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)

            if not caption.strip():
                print(f"[⚠️] Empty caption for frame: {frame}")
                continue

            # Compare with previous caption
            if not is_similar(caption, prev_caption, similarity_threshold):
                captions.append((frame, caption))
                prev_caption = caption

        except Exception as e:
            print(f"[❌] Error processing {frame}: {e}")
            continue

    if not captions:
        print("[❌] No valid captions generated from any frames")
    
    return captions