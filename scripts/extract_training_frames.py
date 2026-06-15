import os
import cv2
import argparse
from datetime import datetime

def extract_frames(source, output_dir, interval=1, duration=None, fps=None):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0 # fallback

    # Calculate frame skip
    if fps:
        skip_frames = int(max(1, video_fps / fps))
    else:
        skip_frames = int(video_fps * interval)

    max_frames = float('inf')
    if duration:
        max_frames = duration * video_fps

    frame_count = 0
    saved_count = 0
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Extracting frames from: {source}")
    print(f"Video FPS: {video_fps:.2f} | Extracting 1 frame every {skip_frames} frames.")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
            break

        if frame_count % skip_frames == 0:
            saved_count += 1
            filename = os.path.join(output_dir, f"{prefix}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            if saved_count % 50 == 0:
                print(f"Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Extraction complete! Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training frames from video/RTSP.")
    parser.add_argument("--video", type=str, help="Path to local video file")
    parser.add_argument("--rtsp", type=str, help="RTSP URL stream")
    parser.add_argument("--fps", type=float, default=None, help="Extract exactly N frames per second")
    parser.add_argument("--interval", type=float, default=1.0, help="Extract 1 frame every N seconds")
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds of video length")
    parser.add_argument("--output", type=str, default="../datasets/ccms_heads/raw_frames", help="Output directory")

    args = parser.parse_args()

    source = args.video if args.video else args.rtsp
    if not source:
        parser.print_help()
        print("\nError: You must provide either --video or --rtsp")
        exit(1)

    extract_frames(source, args.output, args.interval, args.duration, args.fps)
