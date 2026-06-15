import cv2
import time
import argparse
from ultralytics import YOLO

def compare_models(video_path, old_model_path, new_model_path, max_frames=100):
    print("="*50)
    print("⚔️ MODEL COMPARISON BENCHMARK")
    print("="*50)
    
    print(f"Loading Old Model: {old_model_path}")
    old_model = YOLO(old_model_path)
    print(f"Loading New Model: {new_model_path}")
    new_model = YOLO(new_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    metrics = {
        "old": {"frames": 0, "total_heads": 0, "total_conf": 0.0, "time": 0.0},
        "new": {"frames": 0, "total_heads": 0, "total_conf": 0.0, "time": 0.0}
    }
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
            
        frame_count += 1
        
        # Test Old
        t0 = time.time()
        res_old = old_model.predict(frame, imgsz=1280, conf=0.10, max_det=2000, verbose=False)
        metrics["old"]["time"] += time.time() - t0
        metrics["old"]["frames"] += 1
        
        if len(res_old) > 0 and res_old[0].boxes:
            metrics["old"]["total_heads"] += len(res_old[0].boxes)
            metrics["old"]["total_conf"] += float(res_old[0].boxes.conf.sum())
            
        # Test New
        t0 = time.time()
        res_new = new_model.predict(frame, imgsz=1280, conf=0.10, max_det=2000, verbose=False)
        metrics["new"]["time"] += time.time() - t0
        metrics["new"]["frames"] += 1
        
        if len(res_new) > 0 and res_new[0].boxes:
            metrics["new"]["total_heads"] += len(res_new[0].boxes)
            metrics["new"]["total_conf"] += float(res_new[0].boxes.conf.sum())
            
        print(f"Processed frame {frame_count}/{max_frames}", end="\r")

    cap.release()
    
    print("\n\n" + "="*40)
    print("RESULTS COMPARISON")
    print("="*40)
    
    for name, data in metrics.items():
        frames = max(1, data["frames"])
        heads = data["total_heads"]
        avg_heads = heads / frames
        avg_conf = data["total_conf"] / max(1, heads)
        fps = frames / max(0.001, data["time"])
        
        print(f"--- {name.upper()} MODEL ---")
        print(f"Total Detections:  {heads}")
        print(f"Avg Heads/Frame:   {avg_heads:.1f}")
        print(f"Avg Confidence:    {avg_conf:.3f}")
        print(f"Inference Speed:   {fps:.1f} FPS\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Old vs New Head Detector")
    parser.add_argument("--video", type=str, required=True, help="Test video file")
    parser.add_argument("--frames", type=int, default=100, help="Frames to benchmark")
    parser.add_argument("--old", type=str, default="/Volumes/tridelMac/Projects/Dev/ccms/ccms/models/best_head.pt")
    parser.add_argument("--new", type=str, default="/Volumes/tridelMac/Projects/Dev/ccms/ccms/models/best_head_custom.pt")
    
    args = parser.parse_args()
    compare_models(args.video, args.old, args.new, args.frames)
